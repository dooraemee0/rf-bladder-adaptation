package org.dgist.bladderbenchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Debug
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.dgist.bladderbenchmark.preprocessing.PreprocessingMetadata
import org.dgist.bladderbenchmark.preprocessing.WearableRfPreprocessor
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

@RunWith(AndroidJUnit4::class)
class DeploymentProfilingTest {
    @Test
    fun measureRepeatedInProcessInitialization() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val testContext = instrumentation.context
        val iterations = JSONArray()
        repeat(10) { iteration ->
            val totalStart = SystemClock.elapsedRealtimeNanos()

            val environmentStart = SystemClock.elapsedRealtimeNanos()
            val environment = OrtEnvironment.getEnvironment()
            val environmentEnd = SystemClock.elapsedRealtimeNanos()

            val assetStart = SystemClock.elapsedRealtimeNanos()
            val modelBytes = context.assets.open("model_8bit.onnx").use { it.readBytes() }
            val assetEnd = SystemClock.elapsedRealtimeNanos()

            val optionsStart = SystemClock.elapsedRealtimeNanos()
            val options = createSessionOptions()
            val optionsEnd = SystemClock.elapsedRealtimeNanos()

            val sessionStart = SystemClock.elapsedRealtimeNanos()
            val session = environment.createSession(modelBytes, options)
            val sessionEnd = SystemClock.elapsedRealtimeNanos()

            val inputStart = SystemClock.elapsedRealtimeNanos()
            val input = loadValidationModelInput(testContext)
            val tensor = OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(input),
                longArrayOf(1, 7, 400),
            )
            val inputEnd = SystemClock.elapsedRealtimeNanos()
            val totalEnd = inputEnd

            assertTrue(input.all { it.isFinite() })
            iterations.put(
                JSONObject().apply {
                    put("iteration", iteration + 1)
                    put("environment_acquisition_ms", milliseconds(environmentEnd - environmentStart))
                    put("model_asset_read_ms", milliseconds(assetEnd - assetStart))
                    put("session_options_ms", milliseconds(optionsEnd - optionsStart))
                    put("session_create_ms", milliseconds(sessionEnd - sessionStart))
                    put("input_asset_decode_and_tensor_create_ms", milliseconds(inputEnd - inputStart))
                    put("total_initialization_ms", milliseconds(totalEnd - totalStart))
                }
            )
            tensor.close()
            session.close()
            options.close()
        }

        writeJson(
            context,
            "cold_start_android.json",
            JSONObject().apply {
                put("scope", "Repeated fresh OrtSession initialization within one instrumentation process")
                put("true_process_cold_start_measured", false)
                put("clock", "SystemClock.elapsedRealtimeNanos")
                put("iterations", iterations)
            },
        )
    }

    @Test
    fun captureRuntimeMemoryPointObservations() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val observations = JSONArray()
        forceGc()
        observations.put(memoryObservation("baseline_before_model"))

        val runner = OrtModelRunner(context)
        observations.put(memoryObservation("after_onnx_session_creation"))

        val sample = LocalRawRfAssetLoader.load(context).first()
        observations.put(memoryObservation("after_raw_rf_buffer_loading"))

        val preprocessor = WearableRfPreprocessor()
        val modelInput = preprocessor.process(sample.physicalTraces)
        observations.put(memoryObservation("after_preprocessing_point_observation"))

        var output = Float.NaN
        repeat(100) { output = runner.runInput(modelInput) }
        assertTrue(output.isFinite())
        observations.put(memoryObservation("after_100_repeated_inferences"))

        runner.close()
        forceGc()
        observations.put(memoryObservation("after_session_close_and_gc"))

        writeJson(
            context,
            "memory_profile_android.json",
            JSONObject().apply {
                put("scope", "Process memory point observations in a separate instrumentation run")
                put("peak_memory_measured", false)
                put("continuous_sampling_used", false)
                put("units_for_pss_and_memory_stats", "kB")
                put("model_file_size_bytes", 6_651_648)
                put("observations", observations)
            },
        )
    }

    private fun createSessionOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

    private fun loadValidationModelInput(context: Context): FloatArray {
        val valuesPerSample = PreprocessingMetadata.LOGICAL_CHANNELS *
            PreprocessingMetadata.MODEL_SAMPLES
        val requiredBytes = valuesPerSample * Float.SIZE_BYTES
        val bytes = context.assets
            .open("golden_validation/50_30/10_padded_7x400.f32le.bin")
            .use { input ->
            val output = ByteArray(requiredBytes)
            var offset = 0
            while (offset < output.size) {
                val read = input.read(output, offset, output.size - offset)
                require(read >= 0) { "validation golden input ended before one complete model input" }
                offset += read
            }
            output
        }
        val values = FloatArray(valuesPerSample)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values)
        return values
    }

    private fun memoryObservation(stage: String): JSONObject {
        val memoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memoryInfo)
        val runtime = Runtime.getRuntime()
        val memoryStats = JSONObject()
        val keys = listOf(
            "summary.java-heap",
            "summary.native-heap",
            "summary.code",
            "summary.stack",
            "summary.graphics",
            "summary.private-other",
            "summary.system",
            "summary.total-pss",
            "summary.total-swap",
        )
        for (key in keys) {
            memoryStats.put(key, memoryInfo.getMemoryStat(key) ?: JSONObject.NULL)
        }
        return JSONObject().apply {
            put("stage", stage)
            put("elapsed_realtime_ns", SystemClock.elapsedRealtimeNanos())
            put("total_pss_kb", memoryInfo.totalPss)
            put("dalvik_pss_kb", memoryInfo.dalvikPss)
            put("native_pss_kb", memoryInfo.nativePss)
            put("other_pss_kb", memoryInfo.otherPss)
            put("runtime_java_heap_used_bytes", runtime.totalMemory() - runtime.freeMemory())
            put("runtime_java_heap_committed_bytes", runtime.totalMemory())
            put("native_heap_allocated_bytes", Debug.getNativeHeapAllocatedSize())
            put("memory_stats", memoryStats)
        }
    }

    private fun forceGc() {
        Runtime.getRuntime().gc()
        SystemClock.sleep(200)
    }

    private fun writeJson(context: Context, name: String, value: JSONObject) {
        val outputDirectory = context.getExternalFilesDir(null) ?: context.filesDir
        File(outputDirectory, name).writeText(value.toString(2) + "\n")
    }

    private fun milliseconds(nanoseconds: Long): Double = nanoseconds / 1e6
}
