package ch.hevs.roominventory

import android.content.Context
import android.os.SystemClock
import android.util.Log
import java.io.Closeable
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Finds what the item in focus is and what it sells for, on the phone, step by step while it stays in view:
 *
 *   names    YOLOE's product vocabulary names the object (MainActivity, camera worker)
 *   sort     JEV (Tev1-0.8B on llama.cpp) ranks the tutti categories from those names; a visual check
 *            against example listings of every category (CategoryRefs) is mixed in when the app has one
 *   seed     the newest listings of the likeliest categories are compared with the item (DINOv2, thumbnails
 *            embedded on the camera worker's GPU between frames)
 *   refine   product names found in the closest titles are searched (FinderCore, port of find_product.py),
 *            each search sharpening the next; the winning name and the median price of its closest
 *            listings are the answer
 *
 * Only the item in focus makes progress: looking at it longer refines the answer, looking away pauses it.
 * [setPaused] stops everything: searches and thumbnail downloads wait, a running JEV question is cut off
 * (and asked again after the pause).
 * Everything is cached (JEV answers, searches, listing descriptors in ListingStore, item state in ItemMemory),
 * so an item seen before comes back almost at once.
 *
 * All session state lives on one thread ([brain]); the public methods are thread-safe.
 */
class ProductEngine(
    private val context: Context,
    val categories: Categories,
    private val store: ListingStore,
    private val refs: CategoryRefs?,
    private val listener: (ItemResult) -> Unit,
) : Closeable {

    /** A downloaded thumbnail waiting for its descriptor (computed by the camera worker). */
    class EmbedJob(val itemId: Int, val generation: Int, val listingId: String, val jpeg: ByteArray)

    class Ranked(val category: Categories.Category, val p: Float, val pJev: Float, val pVisual: Float)

    private val brain = Executors.newSingleThreadScheduledExecutor { r -> Thread(r, "engine") }
    private val api = Executors.newFixedThreadPool(API_THREADS) { r -> Thread(r, "tutti") }
    private val downloads = Executors.newFixedThreadPool(DOWNLOADS) { r -> Thread(r, "thumbs") }
    private val jevThread = Executors.newSingleThreadExecutor { r -> Thread(r, "jev") }
    private val embedQueue = ConcurrentLinkedDeque<EmbedJob>()

    @Volatile private var jev: JevClassifier? = null   // used on the jev thread only
    @Volatile var jevStatus = "JEV: loading..."
        private set
    @Volatile private var jevFailed = false
    @Volatile private var active: ItemMemory.Item? = null
    @Volatile private var closed = false
    @Volatile var paused = false
        private set

    private val sessions = LinkedHashMap<Int, Session>(16, 0.75f, true) // brain thread, access order
    private val rankedById = java.util.concurrent.ConcurrentHashMap<Int, List<Ranked>>()

    private inner class Session(val item: ItemMemory.Item) {
        var generation = 0
        var core = FinderCore()
        var stage = ItemResult.Stage.LOOKING
        var jevNames: List<String>? = null
        var jevChoices: List<JevClassifier.Choice>? = null
        var jevBusy = false
        var visual: Map<String, Float>? = null
        var visualViews = 0
        var manual: Categories.Category? = null
        var ranked: List<Ranked> = emptyList()
        val seeded = ArrayList<String>()
        val embs = HashMap<String, FloatArray>()
        val newEmbs = HashMap<String, FloatArray>()
        val pending = ArrayDeque<Listing>()
        val queued = HashSet<String>()
        var inFlight = 0
        var embedding = 0
        var running = 0
        var lastLaunch = 0L
        var viewsScored = 0
        var dirty = true
        var lastPublish = 0L
        var detail = "Looking..."
        var failed = 0

        fun reset() {
            generation++
            core = FinderCore()
            seeded.clear()
            embs.clear()
            newEmbs.clear()
            pending.clear()
            queued.clear()
            inFlight = 0
            embedding = 0
            running = 0
            viewsScored = 0
            dirty = true
        }
    }

    init {
        jevThread.execute {
            try {
                val j = JevClassifier(context, categories)
                j.load { msg -> jevStatus = msg }
                jev = j
                jevStatus = j.status
            } catch (e: Throwable) {
                jevFailed = true
                jevStatus = "JEV unavailable: ${e.message}"
                Log.e(TAG, "cannot load JEV", e)
            }
        }
        brain.scheduleWithFixedDelay({
            try {
                tick()
            } catch (e: Throwable) {
                Log.e(TAG, "engine tick failed", e)
            }
        }, 300, TICK_MS, TimeUnit.MILLISECONDS)
    }

    // ------------------------------------------------------------------ called from other threads

    /**
     * UI: stops (or restarts) all work. Searches and downloads wait between pages and pictures, and the JEV
     * question being answered stops within milliseconds; nothing is lost, it all carries on after the pause.
     */
    fun setPaused(p: Boolean) {
        paused = p
        if (!jevFailed) runCatching { JevNative.nativeSetAbort(p) }
        post { sessions.values.forEach { it.dirty = true } }
    }

    /** Camera worker, every frame: the item in focus (null when nothing is). */
    fun setActive(item: ItemMemory.Item?) {
        active = item
    }

    /** Camera worker: next thumbnail to embed, the item in focus first. */
    fun pollEmbedJob(): EmbedJob? {
        val id = active?.id
        if (id != null) {
            val it = embedQueue.iterator()
            while (it.hasNext()) {
                val j = it.next()
                if (j.itemId == id) {
                    it.remove()
                    return j
                }
            }
        }
        return embedQueue.pollFirst()
    }

    /** Camera worker: [job]'s descriptor, or null if the thumbnail could not be decoded. */
    fun onEmbedded(job: EmbedJob, emb: FloatArray?) = post {
        val s = sessions[job.itemId] ?: return@post
        if (s.generation != job.generation) return@post
        s.embedding--
        if (emb == null) {
            s.failed++
            return@post
        }
        s.embs[job.listingId] = emb
        s.newEmbs[job.listingId] = emb
        score(s, job.listingId)
    }

    /** UI: use this category for the item instead of JEV's (null: back to JEV). */
    fun setCategory(item: ItemMemory.Item, key: String?) = post {
        val s = session(item)
        s.manual = key?.let { categories.category(it) }
        s.reset()
        s.ranked = rank(s)
        s.detail = "Category set by hand"
    }

    /** UI: forget cached searches and descriptors, and restart every item's search. */
    fun clearCaches() = post {
        store.clear()
        for (s in sessions.values) s.reset()
    }

    /** Items were forgotten (their IDs will be reused): drop every session. */
    fun forget() = post {
        sessions.values.forEach { it.reset() }
        sessions.clear()
        rankedById.clear()
        embedQueue.clear()
    }

    fun summary(): String {
        val (listings, searches) = store.counts()
        return "$listings listings and $searches searches cached" + (refs?.let { ", ${it.size} category references" } ?: "")
    }

    private fun post(block: () -> Unit) {
        if (closed) return
        brain.execute {
            try {
                block()
            } catch (e: Throwable) {
                Log.e(TAG, "engine task failed", e)
            }
        }
    }

    // ------------------------------------------------------------------ brain thread

    private fun session(item: ItemMemory.Item): Session = sessions.getOrPut(item.id) {
        Session(item).also { evict() }
    }

    private fun evict() {
        if (sessions.size <= MAX_SESSIONS) return
        val activeId = active?.id
        val victim = sessions.keys.firstOrNull { it != activeId } ?: return
        sessions.remove(victim)
    }

    private fun tick() {
        val now = SystemClock.elapsedRealtime()
        val focus = active
        val s = if (paused) null else focus?.let { session(it) }
        if (s != null) advance(s, now)
        for (other in sessions.values) {
            if (other.newEmbs.isNotEmpty()) {
                store.putEmbeddings(HashMap(other.newEmbs))
                other.newEmbs.clear()
            }
            if (other !== s && other.stage != ItemResult.Stage.REFINED && other.stage != ItemResult.Stage.PAUSED &&
                other.stage != ItemResult.Stage.FAILED) {
                other.stage = ItemResult.Stage.PAUSED
                other.dirty = true
            }
            if (other.dirty && now - other.lastPublish >= PUBLISH_MS) publish(other, now)
        }
    }

    /** One step for the item in focus. */
    private fun advance(s: Session, now: Long) {
        val item = s.item
        // 1. names (from the camera)
        if (item.names.isEmpty() && s.manual == null) {
            setStage(s, ItemResult.Stage.NAMING, "Naming the object...")
            return
        }
        // 2. categories: JEV, plus the visual check
        if (s.manual == null && s.jevNames != item.names && !s.jevBusy && jev != null) requestJev(s)
        if (refs != null && item.snapshot.size != s.visualViews && s.seeded.isEmpty()) {
            s.visual = refs.probabilities(item.snapshot)
            s.visualViews = item.snapshot.size
            s.ranked = rank(s)
        }
        if (s.ranked.isEmpty()) {
            when {
                s.jevBusy || (jev == null && !jevFailed) -> setStage(s, ItemResult.Stage.SORTING, "Sorting \"${item.names.first()}\" with JEV...")
                else -> setStage(s, ItemResult.Stage.FAILED, "No category: pick one in the item card")
            }
            return
        }
        // 3. seeds: the likeliest categories, more when nothing close has turned up
        val seedsDone = s.core.searches.filter { it.isSeed }.all { it.done }
        val bestScore = s.core.bestId()?.let { s.core.scores[it] } ?: 0f
        val wanted = when {
            s.manual != null -> 1
            s.seeded.isEmpty() -> s.ranked.take(MAX_FIRST_SEEDS).count { it === s.ranked[0] || it.p >= SECOND_SEED_P }
            seedsDone && bestScore < ItemResult.LIKELY && s.seeded.size < MAX_SEEDS -> s.seeded.size + 1
            else -> s.seeded.size
        }
        while (s.seeded.size < minOf(wanted, s.ranked.size)) {
            val c = s.ranked[s.seeded.size].category
            s.seeded += c.key
            val limit = if (s.seeded.size == 1) SEED_LIMIT else SEED_LIMIT_MORE
            launch(s, FinderCore.Search(null, true, "newest in ${c.label}", c.key), Tutti.Sort.TIMESTAMP, limit)
        }
        // 4. thumbnails
        while (s.inFlight < DOWNLOADS && s.pending.isNotEmpty()) download(s, s.pending.removeFirst())
        // 5. rescore when the item got new views
        if (item.snapshot.size != s.viewsScored) {
            s.viewsScored = item.snapshot.size
            for (id in s.embs.keys) score(s, id)
        }
        // 6. name searches
        val names = s.core.refined()
        val ready = s.core.scores.size >= MIN_COMPARED || (seedsDone && s.core.scores.size >= 10)
        if (ready && now - s.lastLaunch >= LAUNCH_EVERY_MS && names.size < MAX_SEARCHES &&
            names.count { !it.done } < PARALLEL) {
            s.core.nextName()?.let { next ->
                val cat = s.core.categoryOf(next.source) ?: s.core.categoryOf(s.core.bestId()) ?: s.seeded.first()
                launch(s, FinderCore.Search(next.name, false, next.why, cat), Tutti.Sort.RELEVANCE, NAME_LIMIT)
                s.lastLaunch = now
            }
        }
        // 7. stage
        val busy = s.running > 0 || s.pending.isNotEmpty() || s.inFlight > 0 || s.embedding > 0
        if (busy) {
            val running = s.core.searches.filter { !it.done }.joinToString { it.text ?: "newest in ${categories.category(it.category ?: "")?.label}" }
            setStage(s, ItemResult.Stage.SEARCHING, if (running.isNotEmpty()) "Searching $running" else "Comparing listings")
        } else if (s.core.scores.isNotEmpty()) {
            val more = names.size < MAX_SEARCHES && s.core.nextName() != null
            if (!more && now - s.lastLaunch >= LAUNCH_EVERY_MS) setStage(s, ItemResult.Stage.REFINED, "Done: nothing left to search")
            else setStage(s, ItemResult.Stage.SEARCHING, "Choosing the next name")
        }
    }

    private fun setStage(s: Session, stage: ItemResult.Stage, detail: String) {
        if (s.stage != stage || s.detail != detail) {
            s.stage = stage
            s.detail = detail
            s.dirty = true
        }
    }

    private fun requestJev(s: Session) {
        val names = s.item.names
        s.jevNames = names
        s.jevBusy = true
        s.dirty = true
        jevThread.execute {
            val j = jev
            fun apply(result: JevClassifier.Result, done: Boolean) = post {
                s.jevChoices = result.choices
                // the first ranking starts the search; later ones only add categories not browsed yet
                s.ranked = if (s.seeded.isEmpty()) rank(s) else s.ranked + rank(s).filter { r -> s.ranked.none { it.category == r.category } }
                s.detail = "JEV: " + result.choices.take(2).joinToString { "${it.category.label} %.0f%%".format(it.p * 100) } +
                    when {
                        result.cached -> " (cached)"
                        done -> " in %.1f s".format(result.ms / 1000.0)
                        else -> ", still sorting..."
                    }
                s.dirty = true
            }
            val result = try {
                j?.classify(names) { partial -> apply(partial, false) }
            } catch (e: java.util.concurrent.CancellationException) {
                Log.i(TAG, "JEV stopped by the pause, asked again after it")
                post { if (s.jevNames == names) s.jevNames = null }
                null
            } catch (e: Throwable) {
                Log.e(TAG, "JEV failed", e)
                null
            }
            if (result != null) apply(result, true)
            post { s.jevBusy = false }
        }
    }

    /** Categories ranked by JEV's probability, mixed with the visual check when there is one. */
    private fun rank(s: Session): List<Ranked> {
        s.manual?.let { return listOf(Ranked(it, 1f, 1f, 0f)) }
        val jevP = s.jevChoices?.associate { it.category.key to it.p }
        val visP = s.visual
        if (jevP == null && visP == null) return emptyList()
        val wJ = if (jevP == null) 0f else if (visP == null) 1f else JEV_WEIGHT
        val wV = 1f - wJ
        return categories.all.map { c ->
            val pj = jevP?.get(c.key) ?: 0f
            val pv = visP?.get(c.key) ?: 0f
            Ranked(c, wJ * pj + wV * pv, pj, pv)
        }.filter { it.p > 0f }.sortedByDescending { it.p }
    }

    private fun launch(s: Session, search: FinderCore.Search, sort: Tutti.Sort, limit: Int) {
        s.core.add(search)
        s.running++
        s.dirty = true
        val generation = s.generation
        api.execute { runSearch(s, generation, search, sort, limit) }
    }

    /** Tutti thread: one search, from the cache or page by page. */
    private fun runSearch(s: Session, generation: Int, search: FinderCore.Search, sort: Tutti.Sort, limit: Int) {
        val key = "${search.text.orEmpty()}|${search.category.orEmpty()}|$sort|$limit"
        try {
            val cached = store.search(key)
            if (cached != null) {
                val listings = store.listings(cached)
                val embs = store.embeddings(cached)
                post { if (s.generation == generation) onListings(s, search, listings, embs) }
            } else {
                val ids = ArrayList<String>()
                var offset = 0
                while (offset < limit && offset <= Tutti.MAX_OFFSET && !closed && s.generation == generation) {
                    awaitResume()
                    if (s.generation != generation) break
                    val page = Tutti.search(search.text, search.category, sort, offset, minOf(Tutti.PAGE_SIZE, limit - offset))
                    if (page.listings.isEmpty()) break
                    store.putListings(page.listings)
                    val embs = store.embeddings(page.listings.map { it.id })
                    post { if (s.generation == generation) onListings(s, search, page.listings, embs) }
                    ids += page.listings.map { it.id }
                    offset += page.listings.size
                    if (offset >= page.total) break
                    Thread.sleep(PAGE_PAUSE_MS)
                }
                if (s.generation == generation) store.putSearch(key, ids)
            }
        } catch (e: Exception) {
            search.error = e.message ?: e.javaClass.simpleName
            Log.w(TAG, "search \"${search.label}\" in ${search.category} failed", e)
        }
        post {
            search.done = true
            if (s.generation == generation) {
                s.running--
                s.dirty = true
            }
        }
    }

    private fun onListings(s: Session, search: FinderCore.Search, listings: List<Listing>, embs: Map<String, FloatArray>) {
        for (l in listings) {
            s.core.addListing(search, l)
            val e = s.embs[l.id] ?: embs[l.id]
            if (e != null) {
                s.embs[l.id] = e
                score(s, l.id)
            } else if (l.thumbUrl != null && s.queued.add(l.id)) {
                s.pending.addLast(l)
            }
        }
        s.dirty = true
    }

    private fun download(s: Session, l: Listing) {
        s.inFlight++
        val generation = s.generation
        val itemId = s.item.id
        downloads.execute {
            val bytes = try {
                awaitResume()
                Tutti.download(l.thumbUrl!!)
            } catch (e: Exception) {
                null
            }
            if (bytes != null) embedQueue.addLast(EmbedJob(itemId, generation, l.id, bytes))
            post {
                if (s.generation != generation) return@post
                s.inFlight--
                if (bytes != null) s.embedding++ else s.failed++
            }
        }
    }

    /** Worker threads (tutti, thumbs): waits while the engine is paused. */
    private fun awaitResume() {
        while (paused && !closed) Thread.sleep(200)
    }

    private fun score(s: Session, id: String) {
        val e = s.embs[id] ?: return
        val views = s.item.snapshot
        if (views.isEmpty()) return
        var best = -1f
        for (v in views) best = maxOf(best, ItemMemory.dot(v, e))
        s.core.setScore(id, best)
        s.dirty = true
    }

    private fun publish(s: Session, now: Long) {
        s.dirty = false
        s.lastPublish = now
        val v = s.core.verdict()
        val cat = s.core.categoryOf(v.best?.id)?.let { categories.category(it) } ?: s.ranked.firstOrNull()?.category
        val p = s.ranked.firstOrNull { it.category == cat }?.p ?: 0f
        val r = ItemResult(
            itemId = s.item.id,
            stage = s.stage,
            names = s.item.names,
            category = cat,
            categoryP = p,
            guess = v.name,
            price = v.price,
            priced = v.priced,
            best = v.best,
            bestScore = v.bestScore,
            compared = v.compared,
            searches = s.core.searches.size,
            detail = s.detail,
        )
        s.item.result = r
        rankedById[s.item.id] = s.ranked
        listener(r)
    }

    /** Ranked categories of [item], for the item card (UI thread; a snapshot, may be stale). */
    fun rankedCategories(item: ItemMemory.Item): List<Ranked> = rankedById[item.id].orEmpty()

    override fun close() {
        closed = true
        brain.shutdownNow()
        api.shutdownNow()
        downloads.shutdownNow()
        jevThread.execute { jev?.close() }
        jevThread.shutdown()
    }

    companion object {
        private const val TAG = "RoomInventory"
        private const val TICK_MS = 250L
        private const val PUBLISH_MS = 400L
        private const val API_THREADS = 2
        private const val DOWNLOADS = 6
        private const val MAX_SESSIONS = 12
        private const val JEV_WEIGHT = 0.5f
        private const val MAX_FIRST_SEEDS = 2      // categories browsed at first
        private const val SECOND_SEED_P = 0.12f    // ... the second one only if it is that likely
        private const val MAX_SEEDS = 4            // more categories, one at a time, while nothing looks close
        private const val SEED_LIMIT = 300
        private const val SEED_LIMIT_MORE = 150
        private const val NAME_LIMIT = 100
        private const val MIN_COMPARED = 60        // compared listings before the first name search
        private const val LAUNCH_EVERY_MS = 2500L
        private const val MAX_SEARCHES = 6
        private const val PARALLEL = 2
        private const val PAGE_PAUSE_MS = 200L
    }
}
