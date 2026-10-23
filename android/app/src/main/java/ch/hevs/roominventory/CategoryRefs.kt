package ch.hevs.roominventory

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * Visual check of the category: DINOv2 descriptors of recent tutti listings in every category
 * (assets/category_refs.bin, written by tools/build_category_refs.py). An item is compared with all of them;
 * a category whose listings look like it gets a high probability. It catches the cases where the camera's
 * name for an object is wrong (YOLOE calls a GameCube an "arcade machine"), so JEV's category is too.
 *
 *   "CREF", version 1, n, dim, categories, then the category keys (uint16 length + UTF-8),
 *   then n uint16 category indices, then n x dim float16 descriptors.
 */
class CategoryRefs private constructor(private val keys: List<String>, private val cats: ShortArray,
                                       private val embs: FloatArray, private val dim: Int) {

    val size get() = cats.size

    /** Probability of each category key for an item seen as [views] (descriptors, L2-normalised). */
    fun probabilities(views: List<FloatArray>): Map<String, Float> {
        if (views.isEmpty()) return emptyMap()
        val n = cats.size
        val best = FloatArray(n)
        for (i in 0 until n) {
            var m = -1f
            val off = i * dim
            for (v in views) {
                var s = 0f
                for (d in 0 until dim) s += v[d] * embs[off + d]
                if (s > m) m = s
            }
            best[i] = m
        }
        // per category: mean of its TOP closest references
        val perCat = HashMap<Int, MutableList<Float>>()
        for (i in 0 until n) perCat.getOrPut(cats[i].toInt()) { ArrayList() } += best[i]
        val score = perCat.mapValues { (_, v) -> v.sortedDescending().take(TOP).average().toFloat() }
        val max = score.values.max()
        val e = score.mapValues { (_, s) -> exp(((s - max) / TEMPERATURE).toDouble()).toFloat() }
        val sum = e.values.sum()
        return e.entries.associate { (k, v) -> keys[k] to v / sum }
    }

    companion object {
        private const val TOP = 3
        private const val TEMPERATURE = 0.03f
        const val ASSET = "category_refs.bin"

        /** The bundled references, or null when the app was built without them. */
        fun load(context: Context): CategoryRefs? {
            val bytes = try {
                context.assets.open(ASSET).use { it.readBytes() }
            } catch (e: java.io.IOException) {
                return null
            }
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4).also { buf.get(it) }
            require(String(magic) == "CREF") { "not a category reference file" }
            require(buf.int == 1) { "unknown category reference version" }
            val n = buf.int
            val dim = buf.int
            val nCats = buf.int
            val keys = (0 until nCats).map {
                val len = buf.short.toInt() and 0xFFFF
                val b = ByteArray(len).also { buf.get(it) }
                String(b, Charsets.UTF_8)
            }
            val cats = ShortArray(n) { buf.short }
            val embs = FloatArray(n * dim) { YoloeVocab.half(buf.short) }
            return CategoryRefs(keys, cats, embs, dim)
        }
    }
}
