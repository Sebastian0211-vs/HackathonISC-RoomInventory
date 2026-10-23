package ch.hevs.roominventory

import android.content.Context
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileNotFoundException
import kotlin.math.exp

/** JNI side of JEV (src/main/cpp/jev_jni.cpp, llama.cpp). One handle, one thread at a time. */
object JevNative {
    init {
        System.loadLibrary("jev")
    }

    @JvmStatic external fun nativeLoad(libDir: String, path: String, threads: Int, ctx: Int): Long
    @JvmStatic external fun nativeOptionLogits(handle: Long, prompt: String, options: Array<String>): FloatArray
    /** While true, JEV's forward passes stop at once and nativeOptionLogits throws CancellationException. */
    @JvmStatic external fun nativeSetAbort(abort: Boolean)
    @JvmStatic external fun nativeLastPromptTokens(handle: Long): Int
    @JvmStatic external fun nativeDescribe(handle: Long): String
    @JvmStatic external fun nativeFree(handle: Long)
}

/**
 * Puts what the camera saw into a tutti category with JEV, running on the phone (Tev1-0.8B, a Qwen3.5
 * decision model, through llama.cpp). Two questions, like jev_classifier.py: the group (15 options), then the
 * category inside the likeliest groups. Returns the categories ranked by p(group) x p(category | group), so the
 * matcher can try the second one when the first holds nothing that looks like the object.
 *
 * Answers are cached (files/jev_cache.json, kept between launches), so the same YOLOE names never run the
 * model twice: the key is the main name plus the other names in any order. An object whose main name was
 * already answered confidently reuses that answer at once; otherwise the old answer starts the search while
 * JEV answers for the new names. Blocking; create, use and close on one thread.
 */
class JevClassifier(private val context: Context, val categories: Categories) {

    class Choice(val category: Categories.Category, val group: Categories.Group, val p: Float, val pGroup: Float)

    class Result(val state: String, val choices: List<Choice>, val ms: Long, val cached: Boolean)

    private var handle = 0L
    private val cacheFile = File(context.filesDir, "jev_cache.json")
    private val cache = HashMap<String, List<Pair<String, Float>>>()
    /** Main name -> the cache key of its latest answer. */
    private val byMain = HashMap<String, String>()

    /** Human-readable model state, for the menu. */
    @Volatile var status = "JEV not loaded"
        private set

    val loaded get() = handle != 0L

    init {
        runCatching {
            val root = JSONObject(cacheFile.readText())
            for (k in root.keys()) {
                val arr = root.getJSONArray(k)
                cache[k] = (0 until arr.length()).map { i ->
                    arr.getJSONObject(i).let { it.getString("key") to it.getDouble("p").toFloat() }
                }
                mainName(k)?.let { byMain[it] = k }
            }
        }
    }

    /**
     * Unpacks the bundled model to app storage on first launch (llama.cpp needs a file) and loads it.
     * A jev.gguf pushed to the app's external files folder wins, to try another export without rebuilding:
     *   adb push jev.gguf /sdcard/Android/data/ch.hevs.roominventory/files/
     */
    fun load(progress: (String) -> Unit) {
        if (handle != 0L) return
        val t0 = SystemClock.elapsedRealtime()
        val pushed = context.getExternalFilesDir(null)?.let { File(it, MODEL) }?.takeIf { it.isFile }
        val file = pushed ?: unpack(progress)
        progress("Loading JEV...")
        handle = JevNative.nativeLoad(context.applicationInfo.nativeLibraryDir, file.absolutePath, THREADS, CONTEXT)
        val desc = JevNative.nativeDescribe(handle)
        status = "JEV ready: $desc (${if (pushed != null) "pushed" else "bundled"}), " +
            "loaded in ${SystemClock.elapsedRealtime() - t0} ms"
        Log.i(TAG, status)
    }

    private fun unpack(progress: (String) -> Unit): File {
        val out = File(context.filesDir, MODEL)
        val stamp = File(context.filesDir, "$MODEL.stamp")
        val version = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
        if (out.isFile && stamp.isFile && stamp.readText() == version) return out
        val size = try {
            context.assets.openFd(MODEL).use { it.length }
        } catch (e: FileNotFoundException) {
            // built without the model (gradlew -PnoJev, for quick installs): keep the copy unpacked before
            if (out.isFile) return out
            throw IllegalStateException("no $MODEL in the app: run tools/export_jev.py before building")
        }
        val tmp = File(context.filesDir, "$MODEL.part")
        context.assets.open(MODEL).use { input ->
            tmp.outputStream().use { output ->
                val buf = ByteArray(1 shl 20)
                var done = 0L
                var lastShown = -1L
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    output.write(buf, 0, n)
                    done += n
                    val pct = done * 100 / maxOf(size, 1L)
                    if (pct / 10 != lastShown) {
                        lastShown = pct / 10
                        progress("Unpacking JEV model, $pct%")
                    }
                }
            }
        }
        if (!tmp.renameTo(out)) {
            out.delete()
            check(tmp.renameTo(out)) { "cannot write $out" }
        }
        stamp.writeText(version)
        return out
    }

    /**
     * Categories for an object the camera named [names] (best first), likeliest first. [partial] gets the
     * ranking after each group's categories are known, so the search can start before JEV is done.
     */
    fun classify(names: List<String>, partial: (Result) -> Unit = {}): Result {
        require(names.isNotEmpty())
        val main = names.first()
        // the other names in a fixed order, so the same names seen in another order hit the cache
        val state = JevPrompt.state(main, names.drop(1).sorted())
        cache[state]?.let { hit ->
            return Result(state, hit.mapNotNull { (k, p) -> choice(k, p, p) }, 0, true)
        }
        val sameMain = byMain[main]?.let { cache[it] }
        if (sameMain != null) {
            val choices = sameMain.mapNotNull { (k, p) -> choice(k, p, p) }
            if (choices.isNotEmpty() && choices.first().p >= SAME_NAME_SURE) {
                Log.i(TAG, "JEV $names: reusing the answer for \"$main\" (%.2f)".format(choices.first().p))
                remember(state, main, sameMain)
                return Result(state, choices, 0, true)
            }
            if (choices.isNotEmpty()) partial(Result(state, choices, 0, true)) // search meanwhile
        }
        check(handle != 0L) { "JEV is not loaded" }
        val t0 = SystemClock.elapsedRealtime()

        val (gPrompt, gKeys) = JevPrompt.groupPrompt(categories, state)
        val gProbs = ask(gPrompt, gKeys.size)
        val groups = gKeys.indices.sortedByDescending { gProbs[it] }
        // Second question for the likeliest groups, until they cover most of the probability.
        val choices = ArrayList<Choice>()
        var covered = 0f
        for ((rank, gi) in groups.withIndex()) {
            if (rank >= MAX_GROUPS || (rank > 0 && covered >= COVER)) break
            covered += gProbs[gi]
            val group = categories.groups[gi]
            val (cPrompt, cKeys) = JevPrompt.categoryPrompt(categories, state, group.key)
            val cProbs = ask(cPrompt, cKeys.size)
            cKeys.forEachIndexed { ci, key ->
                choices += Choice(categories.category(key)!!, group, gProbs[gi] * cProbs[ci], gProbs[gi])
            }
            if (rank + 1 < MAX_GROUPS && covered < COVER) {
                partial(Result(state, choices.sortedByDescending { it.p }, SystemClock.elapsedRealtime() - t0, false))
            }
        }
        choices.sortByDescending { it.p }
        val ms = SystemClock.elapsedRealtime() - t0
        Log.i(TAG, "JEV ${names.take(3)}: " + choices.take(3).joinToString { "${it.category.key} %.2f".format(it.p) } + " in $ms ms")

        remember(state, main, choices.take(KEEP).map { it.category.key to it.p })
        return Result(state, choices.take(KEEP), ms, false)
    }

    private fun remember(state: String, main: String, answer: List<Pair<String, Float>>) {
        cache[state] = answer
        byMain[main] = state
        save()
    }

    /** The main name in a state written by [JevPrompt.state]. */
    private fun mainName(state: String): String? =
        state.lineSequence().first().takeIf { it.startsWith(SEEN) }?.removePrefix(SEEN)

    private fun choice(key: String, p: Float, pGroup: Float): Choice? {
        val c = categories.category(key) ?: return null
        return Choice(c, categories.groupOf(c), p, pGroup)
    }

    /** Softmax over the option letters' logits, the decision's probabilities. */
    private fun ask(prompt: String, n: Int): FloatArray {
        val letters = Array(n) { JevPrompt.LETTERS[it].toString() }
        val t0 = SystemClock.elapsedRealtime()
        val logits = JevNative.nativeOptionLogits(handle, prompt, letters)
        val ms = SystemClock.elapsedRealtime() - t0
        val tokens = JevNative.nativeLastPromptTokens(handle)
        Log.i(TAG, "JEV question: $tokens tokens in $ms ms (%.0f tokens/s)".format(tokens * 1000f / maxOf(ms, 1L)))
        val max = logits.max()
        val e = FloatArray(n) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum = e.sum()
        return FloatArray(n) { e[it] / sum }
    }

    private fun save() {
        val root = JSONObject()
        for ((k, v) in cache) {
            root.put(k, JSONArray(v.map { (key, p) -> JSONObject().put("key", key).put("p", p.toDouble()) }))
        }
        runCatching { cacheFile.writeText(root.toString()) }
    }

    fun close() {
        if (handle != 0L) JevNative.nativeFree(handle)
        handle = 0L
    }

    companion object {
        private const val TAG = "RoomInventory"
        const val MODEL = "jev.gguf"
        private const val THREADS = 4
        private const val CONTEXT = 2048
        private const val MAX_GROUPS = 2   // second question for at most this many groups
        private const val COVER = 0.9f     // ... or fewer, once they hold this much probability
        private const val KEEP = 6         // categories remembered per object
        private const val SAME_NAME_SURE = 0.8f // reuse an answer for the same main name when it was this sure
        private const val SEEN = "Object seen by the camera: "

        fun loadCategories(context: Context): Categories =
            Categories.parse(context.assets.open("jev_categories.json").bufferedReader().use { it.readText() })
    }
}
