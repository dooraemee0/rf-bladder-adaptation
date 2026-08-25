package org.dgist.bladderbenchmark

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.dgist.bladderbenchmark.preprocessing.PreprocessingMetadata
import android.os.SystemClock
import java.io.Closeable
import java.nio.FloatBuffer

class OrtModelRunner(
    context: Context,
    modelAsset: String = "model_8bit.onnx",
) : Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private val sessionOptions: OrtSession.SessionOptions
    private val session: OrtSession
    private val inputName: String

    val assetReadTimeNs: Long
    val sessionCreateTimeNs: Long

    init {
        val readStart = SystemClock.elapsedRealtimeNanos()
        val modelBytes = context.assets.open(modelAsset).use { it.readBytes() }
        assetReadTimeNs = SystemClock.elapsedRealtimeNanos() - readStart

        sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val sessionStart = SystemClock.elapsedRealtimeNanos()
        session = environment.createSession(modelBytes, sessionOptions)
        sessionCreateTimeNs = SystemClock.elapsedRealtimeNanos() - sessionStart

        require(session.inputNames.size == 1) { "Expected one model input: ${session.inputNames}" }
        inputName = session.inputNames.first()
    }

    fun createTensor(values: FloatArray): OnnxTensor {
        require(values.size == PreprocessingMetadata.LOGICAL_CHANNELS * PreprocessingMetadata.MODEL_SAMPLES)
        return OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(values),
            longArrayOf(
                1,
                PreprocessingMetadata.LOGICAL_CHANNELS.toLong(),
                PreprocessingMetadata.MODEL_SAMPLES.toLong(),
            ),
        )
    }

    fun runInput(values: FloatArray): Float = createTensor(values).use { tensor -> runTensor(tensor) }

    fun runTensor(tensor: OnnxTensor): Float {
        session.run(mapOf(inputName to tensor)).use { result ->
            @Suppress("UNCHECKED_CAST")
            val output = result[0].value as Array<FloatArray>
            return output[0][0]
        }
    }

    override fun close() {
        session.close()
        sessionOptions.close()
    }
}
