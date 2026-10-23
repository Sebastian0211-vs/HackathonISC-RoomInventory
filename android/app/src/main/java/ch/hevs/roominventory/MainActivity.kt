package ch.hevs.roominventory

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.RectF
import android.graphics.Typeface
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.File
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.hypot

/**
 * Camera, on-device YOLOE boxes, and a live answer to "what is this and what does it sell for?" for the
 * object under the reticle, all on the phone:
 *
 *   YOLOE boxes -> the box in focus is named with YOLOE's product vocabulary -> JEV (Tev1 on llama.cpp)
 *   puts it in a tutti.ch category -> tutti is searched straight from the phone -> listing thumbnails are
 *   compared with the object (DINOv2) -> product names from the closest titles are searched in turn ->
 *   the best name and the median price of its closest listings are shown, refined as long as you keep aiming.
 *
 * Items are remembered by appearance (DINOv2) between frames and launches, with what was found about them,
 * so an object seen before gets its answer back at once. Tapping a box locks the focus on it (tap again to
 * release) and saves a photo of it in Pictures/RoomInventory, as before.
 *
 * The pause button stops everything: the camera (the last frame stays on screen), the models, JEV (cut off
 * mid-question) and every tutti search and download. Leaving the app pauses the searches too.
 */
class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var hud: TextView
    private lateinit var card: ItemCard
    private lateinit var menu: MenuPanel
    private lateinit var menuButton: TextView
    private lateinit var pauseButton: PauseButton
    private lateinit var frozen: ImageView
    private lateinit var thumbs: MatchThumbs
    private var cameraProvider: ProcessCameraProvider? = null

    @Volatile private var paused = false   // the pause button
    @Volatile private var stopped = false  // the app is in the background

    /** Models, detection, re-identification and thumbnail descriptors all run here (the GPU needs one thread). */
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()
    /** Photo decoding, cropping and saving, so detection never waits for it. */
    private val saver: ExecutorService = Executors.newSingleThreadExecutor()
    private var detector: YoloeDetector? = null
    private var embedder: Embedder? = null
    private var vocab: YoloeVocab? = null
    @Volatile private var engine: ProductEngine? = null
    private var imageCapture: ImageCapture? = null
    private val tracker = Tracker()
    private val memory = ItemMemory()                 // worker thread only
    @Volatile private var itemsView: List<ItemMemory.Item> = emptyList() // copy for the UI thread
    private val taps = ConcurrentLinkedQueue<Int>()   // track ids tapped on screen, handled by the worker
    private val itemsFile by lazy { File(filesDir, "items.json") }

    // focus (worker thread)
    private var candidate = 0
    private var candidateFrames = 0
    private var focusTrack = 0
    private var pinnedTrack = 0
    private var pinnedItem = 0
    private var pinnedLostAt = 0L
    @Volatile private var focusedItem: ItemMemory.Item? = null
    private var itemsDirty = false
    private var lastSave = 0L

    @Volatile private var minScore = 0.50f
    private var fps = 0f
    private var lastFrameNs = 0L
    private var frameIndex = 0
    private var thumbsPerFrame = 0f

    // UI thread
    private var cardItem: ItemMemory.Item? = null
    private var cardResult: ItemResult? = null
    private var cardLinger = 0
    private var cardTick = 0

    private val askCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else hud.text = "Camera permission is needed."
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        thumbs = MatchThumbs(this)
        buildUi()
        overlayView.labelOf = { id -> itemsView.firstOrNull { it.id == id }?.result?.tag }
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (menu.isOpen) menu.close() else finish()
            }
        })
        overlayView.onTap = { item ->
            if (paused) {
                Toast.makeText(this, "Paused: press ▶ to carry on", Toast.LENGTH_SHORT).show()
            } else {
                overlayView.flash(item.id)
                taps += item.id
            }
        }

        worker.execute { loadModels() }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) startCamera() else askCamera.launch(Manifest.permission.CAMERA)
    }

    override fun onStart() {
        super.onStart()
        stopped = false
        applyPause()
    }

    override fun onStop() {
        super.onStop()
        stopped = true
        applyPause()
        worker.execute { saveItems() }
    }

    /** The engine works only while the app is on screen and not paused. */
    private fun applyPause() {
        engine?.setPaused(paused || stopped)
    }

    /** UI thread: the pause button. */
    private fun setUserPaused(p: Boolean) {
        if (p == paused) return
        paused = p
        pauseButton.paused = p
        if (p) {
            // keep the last frame (and its boxes) on screen, and turn the camera off
            frozen.setImageBitmap(previewView.bitmap)
            frozen.visibility = View.VISIBLE
            cameraProvider?.unbindAll()
            imageCapture = null
            hud.text = "Paused"
            worker.execute { saveItems() }
        } else {
            frozen.visibility = View.GONE
            frozen.setImageDrawable(null)
            worker.execute {
                lastFrameNs = 0L
                fps = 0f
            }
            hud.text = "Resuming..."
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            }
        }
        applyPause()
        refreshUi()
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.execute {
            saveItems()
            engine?.close()
            detector?.close()
            embedder?.close()
            engine = null
            detector = null
            embedder = null
        }
        worker.shutdown()
        saver.shutdown()
        thumbs.close()
    }

    private fun loadModels() {
        EmbedderProbe.run(this) // debugging aid, only when files/embedder_probe.tflite was pushed
        val t0 = SystemClock.elapsedRealtime()
        try {
            detector = YoloeDetector(this)
        } catch (e: Throwable) {
            Log.e(YoloeDetector.TAG, "cannot load detector", e)
            runOnUiThread { hud.text = "Cannot load model: ${e.message}" }
            return
        }
        val emb = try {
            Embedder(this)
        } catch (e: Throwable) {
            Log.e(YoloeDetector.TAG, "cannot load embedder", e)
            runOnUiThread { hud.text = "Cannot load the embedder: ${e.message}" }
            return
        }
        embedder = emb
        vocab = try {
            YoloeVocab(this).takeIf { detector!!.featDim == it.dim }
        } catch (e: Throwable) {
            Log.e(YoloeDetector.TAG, "no YOLOE vocabulary, objects will not be named", e)
            null
        }
        val categories = JevClassifier.loadCategories(this)
        runCatching { memory.load(itemsFile, categories, emb.stamp) }.onFailure { Log.w(YoloeDetector.TAG, "cannot read saved items", it) }
        itemsView = memory.items.toList()
        engine = ProductEngine(this, categories, ListingStore(this, emb.stamp), CategoryRefs.load(this)) { _ ->
            runOnUiThread { refreshUi() }
        }.also { it.setPaused(paused || stopped) }
        val ms = SystemClock.elapsedRealtime() - t0
        Log.i(YoloeDetector.TAG, "models ready in $ms ms, ${memory.items.size} items remembered")
        runOnUiThread {
            if (!paused) hud.text = "${detector?.backend} ready in $ms ms" + if (vocab == null) " (no names: re-export yoloe)" else ""
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider
            if (paused) return@addListener
            // Same aspect ratio everywhere, so preview, analysis and photo share one field of view.
            val ratio = AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY

            val preview = Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(ratio).build())
                .build()
            preview.setSurfaceProvider(previewView.surfaceProvider)

            // Small frames are enough for detection: the model sees them at its own input size anyway.
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(ratio)
                        .setResolutionStrategy(
                            ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                        )
                        .build()
                )
                // Drop stale frames instead of queueing them, so boxes match what's on screen now.
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(worker, ::analyze)

            // Full-resolution photos for the crops.
            val capture = ImageCapture.Builder()
                .setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(ratio).build())
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            provider.unbindAll()
            try {
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture)
                imageCapture = capture
            } catch (e: Exception) {
                Log.e(YoloeDetector.TAG, "camera cannot run preview + analysis + photo together", e)
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ------------------------------------------------------------------ worker thread

    private fun analyze(image: ImageProxy) {
        val det = detector
        if (det == null || embedder == null || paused) {
            image.close()
            return
        }
        val rotation = image.imageInfo.rotationDegrees
        val bitmap = try {
            image.toBitmap()
        } finally {
            image.close()
        }
        val frameW = if (rotation % 180 == 0) bitmap.width else bitmap.height
        val frameH = if (rotation % 180 == 0) bitmap.height else bitmap.width
        frameIndex++

        val raw = det.detect(bitmap, rotation, minScore)
        val kept = nms(raw.filter { it.box.width() * it.box.height() >= MIN_AREA }, NMS_IOU)
        val tracks = tracker.update(kept).sortedByDescending { it.score }.take(MAX_OBJECTS)
        var thumbs = 0
        try {
            handleTaps(bitmap, rotation, tracks)
            reidentify(bitmap, rotation, tracks)
            val focus = chooseFocus(tracks)
            focusTrack = focus?.id ?: 0
            focusedItem = focus?.let { follow(bitmap, rotation, it, raw, det) }
            engine?.setActive(focusedItem)
            thumbs = embedThumbnails()
        } catch (e: Exception) {
            Log.e(YoloeDetector.TAG, "frame processing failed", e)
        }
        bitmap.recycle()
        if (itemsDirty && SystemClock.elapsedRealtime() - lastSave > SAVE_EVERY_MS) saveItems()

        val now = SystemClock.elapsedRealtimeNanos()
        if (lastFrameNs != 0L) {
            val inst = 1e9f / (now - lastFrameNs)
            fps = if (fps == 0f) inst else 0.9f * fps + 0.1f * inst
        }
        lastFrameNs = now
        thumbsPerFrame = 0.95f * thumbsPerFrame + 0.05f * thumbs
        val items = tracks.map {
            OverlayView.Item(it.id, RectF(it.box), it.itemId, it.id == focusTrack, it.id == pinnedTrack && pinnedTrack != 0)
        }
        if (frameIndex % 30 == 0) {
            Log.i(YoloeDetector.TAG, "%.1f fps, %.0f ms model, %.0f ms embed, %.1f thumbs/frame, %d shown, %d items"
                .format(fps, det.lastInferMs, embedder?.lastMs ?: 0f, thumbsPerFrame, items.size, memory.items.size))
        }
        val status = "%s %.0f fps %.0f+%.0f ms".format(det.backend, fps, det.lastInferMs, embedder?.lastMs ?: 0f) +
            if (thumbsPerFrame > 0.05f) " | %.1f thumbs/frame".format(thumbsPerFrame) else ""
        runOnUiThread {
            if (paused) return@runOnUiThread // the frozen frame keeps its boxes
            overlayView.show(items, frameW, frameH)
            hud.text = status
            refreshCard(false)
        }
    }

    /** Taps lock the focus on a box (tap it again to release) and save a photo of it, as before. */
    private fun handleTaps(frame: Bitmap, rotation: Int, tracks: List<Tracker.Track>) {
        while (true) {
            val id = taps.poll() ?: break
            val t = tracks.firstOrNull { it.id == id } ?: continue
            if (pinnedTrack == t.id) {
                pinnedTrack = 0
                pinnedItem = 0
                continue
            }
            val item = ensureItem(frame, rotation, t, tracks)
            pinnedTrack = t.id
            pinnedItem = item.id
            item.saves++
            itemsDirty = true
            val box = RectF(t.box)
            val view = item.saves
            runOnUiThread { takePhoto(item.id, view, box) }
        }
    }

    /** The box to identify: the locked one, else the one under the reticle once it has been there a moment. */
    private fun chooseFocus(tracks: List<Tracker.Track>): Tracker.Track? {
        if (pinnedTrack != 0 || pinnedItem != 0) {
            val t = tracks.firstOrNull { it.id == pinnedTrack }
                ?: tracks.firstOrNull { pinnedItem != 0 && it.itemId == pinnedItem }?.also { pinnedTrack = it.id }
            if (t != null) {
                pinnedLostAt = 0L
                return t
            }
            val now = SystemClock.elapsedRealtime()
            if (pinnedLostAt == 0L) pinnedLostAt = now
            if (now - pinnedLostAt < PIN_GRACE_MS) return null
            pinnedTrack = 0
            pinnedItem = 0
        }
        val under = tracks.filter { it.box.contains(0.5f, 0.5f) }.minByOrNull { it.box.width() * it.box.height() }
            ?: tracks.minByOrNull { hypot(it.box.centerX() - 0.5f, it.box.centerY() - 0.5f) }
                ?.takeIf { hypot(it.box.centerX() - 0.5f, it.box.centerY() - 0.5f) < NEAR_CENTER }
        if (under == null) {
            candidate = 0
            candidateFrames = 0
            return null
        }
        if (under.id == candidate) candidateFrames++ else {
            candidate = under.id
            candidateFrames = 1
        }
        if (candidateFrames >= FOCUS_FRAMES) return under
        // keep the current focus while the reticle moves between boxes
        return tracks.firstOrNull { it.id == focusTrack && it.box.contains(0.5f, 0.5f) }
    }

    /** The item shown in [t], created if it is new. */
    private fun ensureItem(frame: Bitmap, rotation: Int, t: Tracker.Track, tracks: List<Tracker.Track>): ItemMemory.Item {
        memory[t.itemId]?.let { return it }
        val emb = embedder!!
        val e = emb.embed(frame, rotation, t.box)
        t.lastCheck = frameIndex
        val busy = tracks.filter { it.itemId != 0 && it !== t }.map { it.itemId }.toSet()
        val match = memory.match(e, busy)
        val item = if (match != null) {
            tracker.all().forEach { if (it !== t && it.itemId == match.first.id) it.itemId = 0 }
            Log.i(YoloeDetector.TAG, "track ${t.id} is item ${match.first.id} (%.2f)".format(match.second))
            match.first
        } else {
            memory.create(e).also {
                it.thumb = emb.lastCrop()
                itemsView = memory.items.toList()
                Log.i(YoloeDetector.TAG, "track ${t.id} is a new item ${it.id}")
            }
        }
        t.itemId = item.id
        itemsDirty = true
        return item
    }

    /** The item in focus: more views of it, and its names from YOLOE's vocabulary at its anchor. */
    private fun follow(frame: Bitmap, rotation: Int, t: Tracker.Track, raw: List<Detection>, det: YoloeDetector): ItemMemory.Item {
        val tracks = tracker.all()
        val item = ensureItem(frame, rotation, t, tracks)
        item.lastSeen = System.currentTimeMillis()
        val emb = embedder!!
        if (frameIndex - t.lastCheck >= VIEW_EVERY) {
            t.lastCheck = frameIndex
            val e = emb.embed(frame, rotation, t.box)
            val s = memory.similarity(item, e)
            if (s < ItemMemory.KEEP && item.views.size >= 2) {
                Log.i(YoloeDetector.TAG, "track ${t.id} no longer looks like item ${item.id} (%.2f)".format(s))
                t.itemId = 0
            } else if (s >= ItemMemory.NEW_VIEW && memory.addView(item, e)) {
                itemsDirty = true
            }
        }
        val v = vocab
        if (v != null && item.names.isEmpty() && frameIndex % NAME_EVERY == 0) {
            // anchors that produced this box: the best scoring one overlapping it most
            val anchor = raw.filter { iou(it.box, t.box) >= 0.5f }.maxByOrNull { it.score * iou(it.box, t.box) }?.anchor ?: -1
            val feats = if (anchor >= 0) det.features() else null
            if (feats != null) {
                val p = v.probabilities(feats, anchor * v.dim, anchor)
                val top = p.indices.sortedByDescending { p[it] }.take(8).map { v.names[it] to p[it] }
                memory.addNames(item, top)
                if (item.names.isNotEmpty()) {
                    itemsDirty = true
                    Log.i(YoloeDetector.TAG, "item ${item.id} named ${item.names}")
                }
            }
        }
        return item
    }

    /**
     * Recognises known items in boxes that appear, and unlinks a box whose tracker drifted onto something else.
     * At most [BUDGET] appearance checks per frame, oldest first.
     */
    private fun reidentify(frame: Bitmap, rotation: Int, tracks: List<Tracker.Track>) {
        val emb = embedder ?: return
        if (memory.items.isEmpty()) return
        val busy = tracks.filter { it.itemId != 0 }.map { it.itemId }.toMutableSet()
        val due = tracks
            .filter { it.id != focusTrack && frameIndex - it.lastCheck >= if (it.itemId == 0) RECHECK_NEW else RECHECK_KNOWN }
            .sortedBy { it.lastCheck }
            .take(BUDGET)
        for (t in due) {
            t.lastCheck = frameIndex
            val e = emb.embed(frame, rotation, t.box)
            val own = memory[t.itemId]
            if (own != null) {
                if (memory.similarity(own, e) < ItemMemory.KEEP) {
                    busy -= own.id
                    t.itemId = 0
                }
            } else {
                memory.match(e, busy)?.let { (item, _) ->
                    // a hidden older track of the same object must not show it a second time
                    tracker.all().forEach { if (it !== t && it.itemId == item.id) it.itemId = 0 }
                    t.itemId = item.id
                    busy += item.id
                }
            }
        }
    }

    /** Descriptors of downloaded tutti thumbnails, within a time budget per frame. */
    private fun embedThumbnails(): Int {
        val eng = engine ?: return 0
        val emb = embedder ?: return 0
        val t0 = SystemClock.elapsedRealtime()
        var n = 0
        while (n < THUMBS_MAX && SystemClock.elapsedRealtime() - t0 < THUMBS_BUDGET_MS) {
            val job = eng.pollEmbedJob() ?: break
            val bmp = BitmapFactory.decodeByteArray(job.jpeg, 0, job.jpeg.size)
            if (bmp == null) {
                eng.onEmbedded(job, null)
                continue
            }
            val e = emb.embed(bmp)
            bmp.recycle()
            eng.onEmbedded(job, e)
            n++
        }
        return n
    }

    private fun saveItems() {
        if (memory.items.isEmpty() && !itemsFile.exists()) return
        try {
            memory.save(itemsFile, embedder?.stamp ?: return)
            itemsDirty = false
            lastSave = SystemClock.elapsedRealtime()
        } catch (e: Exception) {
            Log.w(YoloeDetector.TAG, "cannot save items", e)
        }
    }

    private fun forgetItems() = worker.execute {
        memory.clear()
        tracker.all().forEach { it.itemId = 0 }
        pinnedTrack = 0
        pinnedItem = 0
        focusedItem = null
        itemsView = emptyList()
        itemsFile.delete()
        engine?.forget()
        runOnUiThread { refreshUi() }
    }

    // ------------------------------------------------------------------ UI thread

    /** UI thread: take the photo now, crop and save it on the saver thread. */
    private fun takePhoto(itemId: Int, view: Int, box: RectF) {
        val cap = imageCapture ?: return
        cap.takePicture(saver, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val result = try {
                    CropSaver.save(this@MainActivity, image, box, itemId, view)
                } catch (e: Exception) {
                    Log.e(YoloeDetector.TAG, "cannot save crop", e)
                    null
                } finally {
                    image.close()
                }
                runOnUiThread {
                    val what = if (result == null) "Could not save the photo" else "Locked on #$itemId, photo saved"
                    Toast.makeText(this@MainActivity, what, Toast.LENGTH_SHORT).show()
                }
            }

            override fun onError(e: ImageCaptureException) {
                Log.e(YoloeDetector.TAG, "capture failed", e)
            }
        })
    }

    private fun refreshUi() {
        refreshCard(true)
        if (menu.isOpen) renderMenu()
    }

    /** The card shows the item in focus, and the last one for a moment after the reticle leaves it. */
    private fun refreshCard(force: Boolean) {
        val focused = focusedItem
        val item = when {
            focused != null -> focused.also { cardLinger = 0 }
            cardItem != null && cardLinger++ < CARD_LINGER_FRAMES -> cardItem
            else -> null
        }
        val r = item?.result
        cardTick++
        if (!force && item === cardItem && r === cardResult && cardTick % 15 != 0) return
        cardItem = item
        cardResult = r
        card.show(item, engine?.jevStatus.orEmpty(), paused)
    }

    private fun renderMenu() {
        val det = detector
        val eng = engine
        val status = listOfNotNull(
            det?.let { "Detector on ${it.backend}" + if (vocab != null) ", ${vocab!!.names.size} product names" else ", no names" },
            embedder?.let { "DINOv2 on ${it.backend}" },
            eng?.jevStatus,
            eng?.summary(),
        ).joinToString("\n")
        menu.render(itemsView, status)
    }

    private fun pickCategory(item: ItemMemory.Item) {
        val eng = engine ?: return
        ItemCard.pickCategory(this, eng.categories, eng.rankedCategories(item)) { key ->
            eng.setCategory(item, key)
            Toast.makeText(this, if (key == null) "JEV decides" else "Searching ${eng.categories.category(key)?.label}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun buildUi() {
        val dp = resources.displayMetrics.density
        previewView = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        frozen = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_XY // PreviewView.getBitmap is exactly what was on screen
            visibility = View.GONE
        }
        overlayView = OverlayView(this)

        hud = TextView(this).apply {
            setTextColor(Color.rgb(120, 255, 120))
            textSize = 11f
            typeface = Typeface.MONOSPACE
            text = "Loading models..."
            setShadowLayer(3f, 0f, 0f, Color.BLACK)
            gravity = Gravity.END
            maxLines = 2
        }
        card = ItemCard(this, thumbs).apply { onPickCategory = ::pickCategory }

        menu = MenuPanel(this, thumbs, minScore,
            onScore = { minScore = it },
            onItem = { menu.close(); pickCategory(it) },
            onClearCache = {
                engine?.clearCaches()
                Toast.makeText(this, "tutti cache cleared", Toast.LENGTH_SHORT).show()
            },
            onForgetItems = { forgetItems() },
        )
        menuButton = TextView(this).apply {
            text = "☰"
            textSize = 20f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            minWidth = (48 * dp).toInt()
            minHeight = (48 * dp).toInt()
            setPadding((14 * dp).toInt(), 0, (14 * dp).toInt(), 0)
            background = android.graphics.drawable.GradientDrawable().apply {
                setColor(0x99000000.toInt())
                cornerRadius = 24 * dp
            }
            setOnClickListener {
                renderMenu()
                menu.open()
            }
        }
        pauseButton = PauseButton(this).apply {
            setOnClickListener { setUserPaused(!paused) }
        }

        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            addView(previewView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
            addView(frozen, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
            addView(overlayView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
            addView(card, FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.BOTTOM))
            addView(hud, FrameLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT, Gravity.TOP or Gravity.END))
            addView(menuButton, FrameLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT, Gravity.TOP or Gravity.START))
            addView(pauseButton, FrameLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT, Gravity.TOP or Gravity.START))
            addView(menu, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        }
        setContentView(root)

        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val i = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            val m = (12 * dp).toInt()
            (menuButton.layoutParams as FrameLayout.LayoutParams).apply {
                leftMargin = m + i.left
                topMargin = m + i.top
            }
            (pauseButton.layoutParams as FrameLayout.LayoutParams).apply {
                leftMargin = m + i.left + (56 * dp).toInt()
                topMargin = m + i.top
            }
            (hud.layoutParams as FrameLayout.LayoutParams).apply {
                leftMargin = m + i.left + (112 * dp).toInt() // right of the menu and pause buttons
                rightMargin = m + i.right
                topMargin = m + i.top
            }
            (card.layoutParams as FrameLayout.LayoutParams).apply {
                leftMargin = m + i.left
                rightMargin = m + i.right
                bottomMargin = m + i.bottom
            }
            menuButton.requestLayout()
            pauseButton.requestLayout()
            hud.requestLayout()
            card.requestLayout()
            menu.setInsets(i.left, i.top, i.bottom)
            insets
        }
    }

    companion object {
        private const val MIN_AREA = 0.003f   // hide boxes smaller than this fraction of the frame
        private const val NMS_IOU = 0.5f
        private const val MAX_OBJECTS = 25
        private const val BUDGET = 1          // appearance checks per frame for boxes out of focus
        private const val RECHECK_NEW = 12    // frames between checks of a box that is not an item (yet)
        private const val RECHECK_KNOWN = 30  // frames between checks of a box showing an item
        private const val FOCUS_FRAMES = 10   // frames under the reticle before a box is in focus
        private const val NEAR_CENTER = 0.12f // or this close to the center when no box covers it
        private const val PIN_GRACE_MS = 4000L
        private const val VIEW_EVERY = 8      // frames between new views of the item in focus
        private const val NAME_EVERY = 3      // frames between namings of the item in focus
        private const val THUMBS_MAX = 8      // thumbnail descriptors per frame...
        private const val THUMBS_BUDGET_MS = 14L // ... within this time
        private const val SAVE_EVERY_MS = 10_000L
        private const val CARD_LINGER_FRAMES = 45
    }
}
