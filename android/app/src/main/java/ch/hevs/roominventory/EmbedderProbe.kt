package ch.hevs.roominventory

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File
import kotlin.math.sqrt

/**
 * Debugging aid for the embedder on the GPU (tools/export_embedder_probe.py): for every
 * files/embedder_probe*.tflite, runs it on the GPU (FP16 and FP32) and on the CPU with the same
 * picture and logs how close every output is, layer by layer. Nothing happens without the file.
 */
object EmbedderProbe {

    fun run(context: Context) {
        val files = context.getExternalFilesDir(null)?.listFiles { f -> f.name.startsWith("embedder_probe") && f.name.endsWith(".tflite") }
        files?.sortedBy { it.name }?.forEach { run(it) }
    }

    private fun run(file: File) {
        val env = Environment.create()
        try {
            val cpu = outputs(file, CompiledModel.Options(Accelerator.CPU), env)
            for (p in listOf(CompiledModel.GpuOptions.Precision.FP16, CompiledModel.GpuOptions.Precision.FP32)) {
                val gpu = try {
                    outputs(file, CompiledModel.Options(Accelerator.GPU).apply { gpuOptions = CompiledModel.GpuOptions(precision = p) }, env)
                } catch (e: Exception) {
                    Log.w(TAG, "probe: GPU $p failed", e)
                    continue
                }
                // createOutputBuffers follows the signature, whose names sort as text: output_0, output_1, output_10...
                val names = (0 until cpu.size).map { "output_$it" }.sorted()
                for (i in cpu.indices) {
                    val g = gpu[i]
                    val c = cpu[i]
                    var dot = 0.0
                    var ng = 0.0
                    var nc = 0.0
                    for (k in c.indices) {
                        dot += g[k] * c[k]
                        ng += g[k] * g[k]
                        nc += c[k] * c[k]
                    }
                    Log.i(TAG, "probe ${file.name} $p ${names[i]}: cos %.4f, norm gpu %.2f cpu %.2f".format(dot / sqrt(ng * nc + 1e-12), sqrt(ng), sqrt(nc)))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "probe failed", e)
        } finally {
            env.close()
        }
    }

    private fun outputs(file: File, options: CompiledModel.Options, env: Environment): List<FloatArray> {
        val m = CompiledModel.create(file.absolutePath, options, env)
        try {
            val ins = m.createInputBuffers()
            val outs = m.createOutputBuffers()
            val s = 168
            val colors = intArrayOf(0xE53935, 0x1E88E5, 0x43A047, 0xFDD835, 0x8E24AA, 0x212121)
            val f = FloatArray(s * s * 3)
            var j = 0
            for (y in 0 until s) for (x in 0 until s) {
                val p = colors[((x / 21) * 7 + (y / 21) * 3) % colors.size]
                f[j++] = (p shr 16 and 0xFF) / 255f
                f[j++] = (p shr 8 and 0xFF) / 255f
                f[j++] = (p and 0xFF) / 255f
            }
            ins[0].writeFloat(f)
            m.run(ins, outs)
            val r = outs.map { it.readFloat() }
            ins.forEach { it.close() }
            outs.forEach { it.close() }
            return r
        } finally {
            m.close()
        }
    }

    private const val TAG = "RoomInventory"
}
