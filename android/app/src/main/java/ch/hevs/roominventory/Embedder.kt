package ch.hevs.roominventory

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import kotlin.math.max

/**
 * Appearance descriptor (DINOv2-small, exported by tools/export_embedder.py) of a box in a camera frame or of
 * a whole picture (a tutti thumbnail). Output is L2-normalised, so similarity is the dot product.
 *
 * Must be created, used and closed on the same thread.
 */
class Embedder(context: Context) : Closeable {

    var backend: String
        private set
    /** Changes with the model file, so descriptors cached on disk are recomputed after an update. */
    val stamp: String
    var lastMs = 0f
        private set

    private val env = Environment.create()
    private var model: CompiledModel
    private var inputs: List<TensorBuffer>
    private var outputs: List<TensorBuffer>
    private val size: Int
    private val floats: FloatArray
    private val pixels: IntArray
    private val crop: Bitmap
    private val canvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()

    init {
        val apkStamp = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        stamp = "dinov2s-" + context.assets.openFd(MODEL).use { it.length } + "-v2"
        fun gpu(precision: CompiledModel.GpuOptions.Precision) = CompiledModel.Options(Accelerator.GPU).apply {
            gpuOptions = CompiledModel.GpuOptions(
                precision = precision,
                serializationDir = context.codeCacheDir.absolutePath,
                modelCacheKey = "embedder-$precision-$apkStamp",
                serializeProgramCache = true,
            )
        }
        val cpu = CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
        }
        // The CPU is always right, so it runs first and is the reference: a GPU backend is used only if it
        // gives the same descriptors (the stock DINOv2 graph did not on a Fold4's GPU, see dinov2_gpu.py)
        // and is faster. FP16 first, it is the fastest when it is accurate enough.
        val cpuModel = CompiledModel.create(context.assets, MODEL, cpu, env)
        val cpuIo = cpuModel.createInputBuffers() to cpuModel.createOutputBuffers()
        val inputType = try {
            cpuModel.getInputTensorType("args_0", SIGNATURE)
        } catch (e: Exception) {
            cpuModel.getInputTensorType(SIGNATURE, "args_0")
        }
        val s = inputType.layout!!.dimensions[1]
        val ref = selfCheck(cpuModel, cpuIo.first, cpuIo.second, s)
        Log.i(YoloeDetector.TAG, "embedder on CPU: %.1f ms, two unrelated patterns score %.3f".format(ref.ms, ref.similarity))
        var chosen: Triple<String, CompiledModel, Pair<List<TensorBuffer>, List<TensorBuffer>>> = Triple("CPU", cpuModel, cpuIo)
        for ((name, options) in listOf(
            "GPU FP16" to gpu(CompiledModel.GpuOptions.Precision.FP16),
            "GPU FP32" to gpu(CompiledModel.GpuOptions.Precision.FP32),
        )) {
            val m = try {
                CompiledModel.create(context.assets, MODEL, options, env)
            } catch (e: Exception) {
                Log.w(YoloeDetector.TAG, "embedder: $name unavailable", e)
                continue
            }
            val io = m.createInputBuffers() to m.createOutputBuffers()
            val c = selfCheck(m, io.first, io.second, s)
            val agree = minOf(ItemMemory.dot(c.a, ref.a), ItemMemory.dot(c.b, ref.b))
            val ok = agree >= AGREE_MIN && c.similarity < SELF_CHECK_MAX && c.ms < ref.ms
            Log.i(YoloeDetector.TAG, "embedder on $name: %.1f ms, agrees with the CPU %.4f, two unrelated patterns score %.3f%s"
                .format(c.ms, agree, c.similarity, if (ok) "" else ", not used"))
            if (ok) {
                chosen = Triple(name, m, io)
                cpuIo.first.forEach { it.close() }
                cpuIo.second.forEach { it.close() }
                cpuModel.close()
                break
            }
            io.first.forEach { it.close() }
            io.second.forEach { it.close() }
            m.close()
        }
        val (name, m, io) = chosen
        model = m
        inputs = io.first
        outputs = io.second
        backend = name
        size = s
        floats = FloatArray(size * size * 3)
        pixels = IntArray(size * size)
        crop = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        canvas = Canvas(crop)
        Log.i(YoloeDetector.TAG, "embedder loaded on $name, input ${size}x$size")
    }

    /** Descriptors of two unrelated test patterns (a smooth gradient, colour blocks) and the time of one run. */
    private class Check(val a: FloatArray, val b: FloatArray, val ms: Float) {
        /** Close to 1 means broken: every input gets the same descriptor. */
        val similarity get() = ItemMemory.dot(a, b)
    }

    private fun selfCheck(m: CompiledModel, ins: List<TensorBuffer>, outs: List<TensorBuffer>, s: Int): Check {
        fun run(pixel: (Int, Int) -> Int): FloatArray {
            val f = FloatArray(s * s * 3)
            var j = 0
            for (y in 0 until s) for (x in 0 until s) {
                val p = pixel(x, y)
                f[j++] = (p shr 16 and 0xFF) * INV255
                f[j++] = (p shr 8 and 0xFF) * INV255
                f[j++] = (p and 0xFF) * INV255
            }
            ins[0].writeFloat(f)
            m.run(ins, outs)
            return outs[0].readFloat()
        }
        val a = run { x, y -> val g = (x + y) * 255 / (2 * s); (g shl 16) or (g shl 8) or g }
        val colors = intArrayOf(0xE53935, 0x1E88E5, 0x43A047, 0xFDD835, 0x8E24AA, 0x212121)
        val blocks = { x: Int, y: Int -> colors[((x / 21) * 7 + (y / 21) * 3) % colors.size] }
        val b = run(blocks)
        val t0 = SystemClock.elapsedRealtimeNanos()
        repeat(3) { run(blocks) }
        return Check(a, b, (SystemClock.elapsedRealtimeNanos() - t0) / 3e6f)
    }

    /**
     * Descriptor of [box] (normalized, upright frame, a small margin is added) in [frame]
     * (sensor orientation, [rotation] = its rotationDegrees). The box is letterboxed.
     */
    fun embed(frame: Bitmap, rotation: Int, box: RectF): FloatArray {
        val upright = rotation % 180 == 0
        val upW = if (upright) frame.width else frame.height
        val upH = if (upright) frame.height else frame.width
        val mx = box.width() * MARGIN
        val my = box.height() * MARGIN
        val x0 = (box.left - mx).coerceIn(0f, 1f) * upW
        val y0 = (box.top - my).coerceIn(0f, 1f) * upH
        val x1 = (box.right + mx).coerceIn(0f, 1f) * upW
        val y1 = (box.bottom + my).coerceIn(0f, 1f) * upH
        val bw = max(x1 - x0, 1f)
        val bh = max(y1 - y0, 1f)
        val s = size / max(bw, bh)

        matrix.reset()
        matrix.postTranslate(-frame.width / 2f, -frame.height / 2f) // sensor -> centered
        matrix.postRotate(rotation.toFloat())                       // -> upright, centered
        matrix.postTranslate(upW / 2f - (x0 + x1) / 2f, upH / 2f - (y0 + y1) / 2f) // box center at origin
        matrix.postScale(s, s)
        matrix.postTranslate(size / 2f, size / 2f)
        canvas.drawColor(PAD_COLOR)
        canvas.save()
        canvas.clipRect((size - bw * s) / 2f, (size - bh * s) / 2f, (size + bw * s) / 2f, (size + bh * s) / 2f)
        canvas.drawBitmap(frame, matrix, paint)
        canvas.restore()
        return run()
    }

    /** Descriptor of a whole picture (a listing thumbnail), letterboxed like a box. */
    fun embed(picture: Bitmap): FloatArray {
        val s = size.toFloat() / max(picture.width, picture.height)
        matrix.reset()
        matrix.postScale(s, s)
        matrix.postTranslate((size - picture.width * s) / 2f, (size - picture.height * s) / 2f)
        canvas.drawColor(PAD_COLOR)
        canvas.drawBitmap(picture, matrix, paint)
        return run()
    }

    /** The current crop, to show which object a descriptor came from. */
    fun lastCrop(): Bitmap = crop.copy(Bitmap.Config.ARGB_8888, false)

    private fun run(): FloatArray {
        crop.getPixels(pixels, 0, size, 0, 0, size, size)
        var j = 0
        for (p in pixels) {
            floats[j++] = (p shr 16 and 0xFF) * INV255
            floats[j++] = (p shr 8 and 0xFF) * INV255
            floats[j++] = (p and 0xFF) * INV255
        }
        val t0 = SystemClock.elapsedRealtimeNanos()
        inputs[0].writeFloat(floats)
        model.run(inputs, outputs)
        val e = outputs[0].readFloat()
        lastMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6f
        return e
    }

    override fun close() {
        inputs.forEach { it.close() }
        outputs.forEach { it.close() }
        model.close()
        env.close()
    }

    companion object {
        private const val MODEL = "embedder.tflite"
        private const val SIGNATURE = "serving_default"
        private const val MARGIN = 0.06f
        private const val INV255 = 1f / 255f
        private const val PAD_COLOR = 0xFF727272.toInt() // rgb(114, 114, 114)
        private const val SELF_CHECK_MAX = 0.9f // healthy DINOv2 scores these patterns far lower
        private const val AGREE_MIN = 0.98f     // GPU descriptors must be this close to the CPU's
    }
}
