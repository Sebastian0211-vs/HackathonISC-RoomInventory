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
import com.google.ai.edge.litert.TensorType
import java.io.Closeable
import java.io.File
import kotlin.math.min

/** A box in normalized coordinates (0..1) of the upright camera frame, its objectness, and its model anchor. */
data class Detection(val box: RectF, val score: Float, val anchor: Int = -1)

/**
 * Runs the YOLOE export (tools/export_tflite.py) on camera frames.
 *   output [1, anchors, 5]    x0, y0, x1, y1 normalized to the model input, objectness 0..1
 *   output [1, anchors, 128]  class embedding of each anchor (optional: older exports only have boxes)
 * [features] reads the class embeddings of the last frame, for [YoloeVocab] to name a box.
 *
 * Must be created, used and closed on the same thread.
 */
class YoloeDetector(context: Context) : Closeable {

    // A model pushed to the app's files folder wins over the bundled one, to try exports without rebuilding:
    //   adb push yoloe.tflite /sdcard/Android/data/ch.hevs.roominventory/files/
    private val override = context.getExternalFilesDir(null)?.let { File(it, MODEL) }?.takeIf { it.isFile }

    val source = if (override != null) "files" else "assets"
    val backend: String
    var lastInferMs = 0f
        private set

    private val env = Environment.create()
    private val model: CompiledModel
    private val inputs: List<TensorBuffer>
    private val outputs: List<TensorBuffer>

    private val inH: Int
    private val inW: Int
    private val anchors: Int
    private val boxOut: Int
    /** Output index of the class embeddings, -1 when the model has none. */
    private val featOut: Int
    /** Length of one anchor's class embedding (0 without them). */
    val featDim: Int
    private val floats: FloatArray
    private val pixels: IntArray
    private val letterbox: Bitmap
    private val canvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()

    init {
        // Changes whenever the model file or the app is updated, so a stale GPU program cache is never used.
        val stamp = override?.let { "${it.length()}-${it.lastModified()}" }
            ?: context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
        val gpu = CompiledModel.Options(Accelerator.GPU).apply {
            gpuOptions = CompiledModel.GpuOptions(
                precision = CompiledModel.GpuOptions.Precision.FP16,
                serializationDir = context.codeCacheDir.absolutePath,
                modelCacheKey = "yoloe-$stamp",
                serializeProgramCache = true,
            )
        }
        val cpu = CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
        }

        var m: CompiledModel? = null
        var name = "CPU"
        try {
            m = open(context, gpu)
            name = "GPU"
        } catch (e: Exception) {
            Log.w(TAG, "GPU unavailable, using CPU", e)
        }
        if (m == null) m = open(context, cpu)
        model = m
        backend = name

        val inShape = dims("args_0", true) // [1, h, w, 3]
        inH = inShape[1]
        inW = inShape[2]
        outputs = model.createOutputBuffers()
        val shapes = outputs.indices.map { i -> dims("output_$i", false) }
        boxOut = shapes.indexOfFirst { it.last() == 5 }
        require(boxOut >= 0) { "expected an output with 5 values per anchor, got $shapes" }
        anchors = shapes[boxOut][1]
        featOut = shapes.indexOfFirst { it.last() != 5 && it.getOrNull(1) == anchors }
        featDim = if (featOut >= 0) shapes[featOut].last() else 0

        inputs = model.createInputBuffers()
        floats = FloatArray(inH * inW * 3)
        pixels = IntArray(inH * inW)
        letterbox = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888)
        canvas = Canvas(letterbox)
        Log.i(TAG, "loaded from $source on $name, input ${inW}x$inH, outputs $shapes")
    }

    private fun open(context: Context, options: CompiledModel.Options): CompiledModel =
        override?.let { CompiledModel.create(it.absolutePath, options, env) }
            ?: CompiledModel.create(context.assets, MODEL, options, env)

    private fun dims(tensor: String, input: Boolean): List<Int> {
        val t: TensorType = try {
            if (input) model.getInputTensorType(tensor, SIGNATURE) else model.getOutputTensorType(tensor, SIGNATURE)
        } catch (e: Exception) { // argument order differs between LiteRT releases
            if (input) model.getInputTensorType(SIGNATURE, tensor) else model.getOutputTensorType(SIGNATURE, tensor)
        }
        return t.layout!!.dimensions
    }

    /** [frame] as delivered by the camera (sensor orientation), [rotation] its rotationDegrees. */
    fun detect(frame: Bitmap, rotation: Int, minScore: Float): List<Detection> {
        val upright = rotation % 180 == 0
        val upW = if (upright) frame.width else frame.height
        val upH = if (upright) frame.height else frame.width

        // Rotate and letterbox in a single draw, like Ultralytics' LetterBox (gray 114 padding).
        val s = min(inW.toFloat() / upW, inH.toFloat() / upH)
        val dx = (inW - upW * s) / 2f
        val dy = (inH - upH * s) / 2f
        matrix.reset()
        matrix.postTranslate(-frame.width / 2f, -frame.height / 2f)
        matrix.postRotate(rotation.toFloat())
        matrix.postScale(s, s)
        matrix.postTranslate(inW / 2f, inH / 2f)
        canvas.drawColor(PAD_COLOR)
        canvas.drawBitmap(frame, matrix, paint)

        letterbox.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        var j = 0
        for (p in pixels) {
            floats[j++] = (p shr 16 and 0xFF) * INV255
            floats[j++] = (p shr 8 and 0xFF) * INV255
            floats[j++] = (p and 0xFF) * INV255
        }

        val t0 = SystemClock.elapsedRealtimeNanos()
        inputs[0].writeFloat(floats)
        model.run(inputs, outputs)
        val out = outputs[boxOut].readFloat()
        lastInferMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6f

        val dets = ArrayList<Detection>()
        var o = 0
        for (a in 0 until anchors) {
            val score = out[o + 4]
            if (score >= minScore) {
                // input (normalized) -> input pixels -> upright frame, normalized
                val x0 = ((out[o] * inW - dx) / s / upW).coerceIn(0f, 1f)
                val y0 = ((out[o + 1] * inH - dy) / s / upH).coerceIn(0f, 1f)
                val x1 = ((out[o + 2] * inW - dx) / s / upW).coerceIn(0f, 1f)
                val y1 = ((out[o + 3] * inH - dy) / s / upH).coerceIn(0f, 1f)
                if (x1 > x0 && y1 > y0) dets += Detection(RectF(x0, y0, x1, y1), score, a)
            }
            o += 5
        }
        return dets
    }

    /** Class embeddings of the last [detect] call, [anchors] x [featDim], or null if the model has none. */
    fun features(): FloatArray? = if (featOut >= 0) outputs[featOut].readFloat() else null

    override fun close() {
        inputs.forEach { it.close() }
        outputs.forEach { it.close() }
        model.close()
        env.close()
    }

    companion object {
        const val TAG = "RoomInventory"
        private const val MODEL = "yoloe.tflite"
        private const val SIGNATURE = "serving_default"
        private const val INV255 = 1f / 255f
        private const val PAD_COLOR = 0xFF727272.toInt() // rgb(114, 114, 114)
    }
}
