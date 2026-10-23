package ch.hevs.roominventory

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/**
 * YOLOE's prompt-free vocabulary (4585 names), scored on the phone for one anchor at a time: the detector
 * outputs each anchor's class embedding, this turns it into names (tools/export_tflite.py writes both files).
 *
 *   yoloe_vocab.bin   "YVOC", version, levels, classes, dim, then per level W [classes, dim] and b [classes]
 *                     as float16, then the anchor count of each level
 *   yoloe_names.txt   one name per line
 */
class YoloeVocab(context: Context) {

    val names: List<String> = context.assets.open("yoloe_names.txt").bufferedReader().readLines().filter { it.isNotEmpty() }
    val dim: Int
    private val classes: Int
    private val weights: Array<FloatArray>   // per level, classes * dim
    private val biases: Array<FloatArray>    // per level, classes
    private val levelStart: IntArray         // first anchor of each level

    init {
        val bytes = context.assets.open("yoloe_vocab.bin").use { it.readBytes() }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { buf.get(it) }
        require(String(magic) == "YVOC") { "not a YOLOE vocabulary file" }
        require(buf.int == 1) { "unknown vocabulary version" }
        val levels = buf.int
        classes = buf.int
        dim = buf.int
        require(classes == names.size) { "vocabulary has $classes classes, names file ${names.size}" }
        // per level: W, then b
        val w = ArrayList<FloatArray>()
        val b = ArrayList<FloatArray>()
        repeat(levels) {
            w += FloatArray(classes * dim) { half(buf.short) }
            b += FloatArray(classes) { half(buf.short) }
        }
        weights = w.toTypedArray()
        biases = b.toTypedArray()
        val counts = IntArray(levels) { buf.int }
        levelStart = IntArray(levels)
        for (i in 1 until levels) levelStart[i] = levelStart[i - 1] + counts[i - 1]
    }

    private fun level(anchor: Int): Int {
        var l = 0
        while (l + 1 < levelStart.size && anchor >= levelStart[l + 1]) l++
        return l
    }

    /** Probability of every name for the class embedding [e] of [anchor], as YOLOE computes it. */
    fun probabilities(e: FloatArray, offset: Int, anchor: Int): FloatArray {
        val l = level(anchor)
        val w = weights[l]
        val b = biases[l]
        val out = FloatArray(classes)
        var k = 0
        for (c in 0 until classes) {
            var s = b[c]
            for (d in 0 until dim) s += w[k++] * e[offset + d]
            out[c] = (1.0 / (1.0 + exp(-s.toDouble()))).toFloat()
        }
        return out
    }

    companion object {
        /** IEEE 754 half -> float. */
        fun half(h: Short): Float {
            val bits = h.toInt() and 0xFFFF
            val sign = bits ushr 15
            val exp = (bits ushr 10) and 0x1F
            val mant = bits and 0x3FF
            val f = when (exp) {
                0 -> if (mant == 0) 0 else {
                    var m = mant
                    var e = -14
                    while (m and 0x400 == 0) {
                        m = m shl 1
                        e--
                    }
                    ((e + 127) shl 23) or ((m and 0x3FF) shl 13)
                }
                31 -> (0xFF shl 23) or (mant shl 13)
                else -> ((exp - 15 + 127) shl 23) or (mant shl 13)
            }
            return Float.fromBits((sign shl 31) or f)
        }
    }
}
