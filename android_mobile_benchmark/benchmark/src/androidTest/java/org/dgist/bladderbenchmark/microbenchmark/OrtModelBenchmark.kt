package org.dgist.bladderbenchmark.microbenchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import androidx.benchmark.junit4.BenchmarkRule
import androidx.benchmark.junit4.measureRepeated
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.After
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

@RunWith(AndroidJUnit4::class)
class OrtModelBenchmark {
    @get:Rule
    val benchmarkRule = BenchmarkRule()

    private lateinit var environment: OrtEnvironment
    private lateinit var sessionOptions: OrtSession.SessionOptions
    private lateinit var session: OrtSession
    private lateinit var inputName: String
    private lateinit var tensors: List<OnnxTensor>

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val inputs = loadInputs(context)
        environment = OrtEnvironment.getEnvironment()
        sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val model = context.assets.open("model_8bit.onnx").use { it.readBytes() }
        session = environment.createSession(model, sessionOptions)
        inputName = session.inputNames.single()
        tensors = inputs.map {
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(it), longArrayOf(1, 7, 400))
        }
        repeat(30) { index -> runTensor(tensors[index % tensors.size]) }
    }

    @After
    fun tearDown() {
        tensors.forEach(OnnxTensor::close)
        session.close()
        sessionOptions.close()
    }

    @Test
    fun ortSessionRunBatch1SingleThread() {
        var index = 0
        benchmarkRule.measureRepeated {
            runTensor(tensors[index])
            index = (index + 1) % tensors.size
        }
    }

    private fun runTensor(tensor: OnnxTensor): Float {
        session.run(mapOf(inputName to tensor)).use { result ->
            @Suppress("UNCHECKED_CAST")
            val output = result[0].value as Array<FloatArray>
            return output[0][0]
        }
    }

    private fun loadInputs(context: Context): List<FloatArray> {
        val manifest = JSONObject(
            context.assets.open("test_manifest.json").bufferedReader().use { it.readText() }
        )
        val shape = manifest.getJSONArray("shape")
        val count = shape.getInt(0)
        require(shape.getInt(1) == 7 && shape.getInt(2) == 400)
        val valuesPerSample = 7 * 400
        val bytes = context.assets.open("test_inputs.bin").use { it.readBytes() }
        val values = FloatArray(count * valuesPerSample)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values)
        return List(count) { index ->
            values.copyOfRange(index * valuesPerSample, (index + 1) * valuesPerSample)
        }
    }
}
