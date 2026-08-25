package org.dgist.bladderbenchmark.microbenchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.benchmark.junit4.BenchmarkRule
import androidx.benchmark.junit4.measureRepeated
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.dgist.bladderbenchmark.preprocessing.PreprocessingMetadata
import org.dgist.bladderbenchmark.preprocessing.WearableRfPreprocessor
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.FloatBuffer

@RunWith(AndroidJUnit4::class)
class RawRfToVolumeBenchmark {
    @get:Rule
    val benchmarkRule = BenchmarkRule()

    private lateinit var rawRf: Array<FloatArray>
    private lateinit var preprocessor: WearableRfPreprocessor
    private lateinit var environment: OrtEnvironment
    private lateinit var sessionOptions: OrtSession.SessionOptions
    private lateinit var session: OrtSession
    private lateinit var inputName: String
    private var resultSink = 0.0f

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().context
        rawRf = BenchmarkRawRfAssets.loadPredeclaredValidationSample(context)
        preprocessor = WearableRfPreprocessor()
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
    }

    @After
    fun tearDown() {
        session.close()
        sessionOptions.close()
    }

    @Test
    fun sixChannelRawRfToScalarVolume() {
        benchmarkRule.measureRepeated {
            val modelInput = preprocessor.process(rawRf)
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(modelInput),
                longArrayOf(
                    1,
                    PreprocessingMetadata.LOGICAL_CHANNELS.toLong(),
                    PreprocessingMetadata.MODEL_SAMPLES.toLong(),
                ),
            ).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val output = result[0].value as Array<FloatArray>
                    resultSink = output[0][0]
                }
            }
        }
        assertTrue(resultSink.isFinite())
    }
}
