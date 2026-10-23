package ch.hevs.roominventory

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * The items seen so far and what they look like: a few descriptors per item (one per view).
 * A new box is recognised as an item when it is clearly closer to that item than to any other.
 *
 * Thresholds come from the crops saved on the Fold4: the same object seen again scored 0.88 to 0.97
 * (0.70 for blurry shoe crops), two different open laptops up to 0.86, unrelated objects about 0.3.
 *
 * Views and naming are changed on the camera worker thread only; other threads read [Item.snapshot],
 * [Item.names] and [Item.result].
 */
class ItemMemory {

    class Item(val id: Int) {
        val views = ArrayList<FloatArray>()
        /** Copy of [views] for other threads, replaced on every change. */
        @Volatile var snapshot: List<FloatArray> = emptyList()
            private set
        var saves = 0

        /** YOLOE vocabulary scores summed over the frames it was named in (worker thread). */
        val nameScores = HashMap<String, Float>()
        var namedFrames = 0
        /** Settled names, best first (empty until [namedFrames] reaches NAME_FRAMES). */
        @Volatile var names: List<String> = emptyList()
        /** Small crop of the item, for the UI. */
        @Volatile var thumb: Bitmap? = null
        /** What the matcher found so far (set by ProductEngine). */
        @Volatile var result: ItemResult? = null
        @Volatile var lastSeen = 0L

        fun publishViews() {
            snapshot = views.toList()
        }
    }

    val items = ArrayList<Item>()
    private var nextId = 1

    fun create(e: FloatArray?): Item = Item(nextId++).also {
        if (e != null) it.views += e
        it.publishViews()
        items += it
    }

    operator fun get(id: Int): Item? = items.firstOrNull { it.id == id }

    fun similarity(item: Item, e: FloatArray): Float = item.views.maxOfOrNull { dot(it, e) } ?: 0f

    /**
     * Best item for [e], or null. Items in [busy] are already on screen in another box:
     * they still count as competitors, so an ambiguous box is left alone rather than mislabelled.
     */
    fun match(e: FloatArray, busy: Set<Int>): Pair<Item, Float>? {
        var best: Item? = null
        var bestSim = -1f
        var second = 0f
        for (it in items) {
            val s = similarity(it, e)
            if (s > bestSim) {
                second = maxOf(second, bestSim)
                best = it
                bestSim = s
            } else if (s > second) second = s
        }
        val b = best ?: return null
        return if (b.id !in busy && bestSim >= MATCH && bestSim - second >= MARGIN) b to bestSim else null
    }

    /** Remembers another view of [item], unless it is a near duplicate or the item is full. Returns true if added. */
    fun addView(item: Item, e: FloatArray): Boolean {
        if (item.views.size >= MAX_VIEWS) return false
        if (similarity(item, e) >= DUPLICATE) return false
        item.views += e
        item.publishViews()
        return true
    }

    /** Adds one frame of vocabulary scores; settles [Item.names] after NAME_FRAMES frames. */
    fun addNames(item: Item, scores: List<Pair<String, Float>>) {
        for ((n, p) in scores) item.nameScores[n] = (item.nameScores[n] ?: 0f) + p
        item.namedFrames++
        if (item.namedFrames >= NAME_FRAMES && item.names.isEmpty()) {
            item.names = item.nameScores.entries.sortedByDescending { it.value }.take(NAMES).map { it.key }
        }
    }

    // ---------------------------------------------------------------- persistence (files/items.json)

    /** [stamp] identifies the embedder: views from another model are meaningless, [load] drops them. */
    fun save(file: File, stamp: String) {
        val arr = JSONArray()
        for (it in items) {
            val o = JSONObject().put("id", it.id).put("saves", it.saves)
            o.put("views", JSONArray(it.views.map { v -> Base64.encodeToString(ListingStore.toBytes(v), Base64.NO_WRAP) }))
            o.put("names", JSONArray(it.names))
            it.thumb?.let { t ->
                val out = ByteArrayOutputStream()
                t.compress(Bitmap.CompressFormat.JPEG, 80, out)
                o.put("thumb", Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP))
            }
            it.result?.let { r -> o.put("result", r.toJson()) }
            arr.put(o)
        }
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(JSONObject().put("next", nextId).put("embedder", stamp).put("items", arr).toString())
        tmp.renameTo(file)
    }

    fun load(file: File, categories: Categories, stamp: String) {
        if (!file.isFile) return
        val root = JSONObject(file.readText())
        if (root.optString("embedder") != stamp) return
        val arr = root.getJSONArray("items")
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val item = Item(o.getInt("id"))
            item.saves = o.optInt("saves")
            val views = o.getJSONArray("views")
            for (j in 0 until views.length()) item.views += ListingStore.fromBytes(Base64.decode(views.getString(j), Base64.NO_WRAP))
            item.publishViews()
            val names = o.optJSONArray("names")
            if (names != null) item.names = (0 until names.length()).map { names.getString(it) }
            o.optString("thumb").takeIf { it.isNotEmpty() }?.let {
                val b = Base64.decode(it, Base64.NO_WRAP)
                item.thumb = BitmapFactory.decodeByteArray(b, 0, b.size)
            }
            o.optJSONObject("result")?.let { item.result = ItemResult.fromJson(it, item.id, categories) }
            items += item
        }
        nextId = maxOf(root.optInt("next", 1), (items.maxOfOrNull { it.id } ?: 0) + 1)
    }

    fun clear() {
        items.clear()
        nextId = 1
    }

    companion object {
        const val MATCH = 0.75f    // minimum similarity to call it the same item
        const val MARGIN = 0.03f   // and that much closer than to any other item
        const val KEEP = 0.55f     // a tracked box that drops below this is no longer that item (tracker drift)
        const val NEW_VIEW = 0.65f // views this similar or more are trusted enough to add to the item
        private const val DUPLICATE = 0.97f
        private const val MAX_VIEWS = 20
        const val NAME_FRAMES = 5  // frames of vocabulary scores before the names are settled
        const val NAMES = 4        // names kept, the first one is what the item is called

        fun dot(a: FloatArray, b: FloatArray): Float {
            var s = 0f
            for (i in a.indices) s += a[i] * b[i]
            return s
        }
    }
}
