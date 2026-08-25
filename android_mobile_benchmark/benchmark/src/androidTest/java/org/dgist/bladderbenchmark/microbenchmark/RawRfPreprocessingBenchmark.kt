package org.dgist.bladderbenchmark.microbenchmark

import androidx.benchmark.junit4.BenchmarkRule
import androidx.benchmark.junit4.measureRepeated
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.dgist.bladderbenchmark.preprocessing.WearableRfPreprocessor
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RawRfPreprocessingBenchmark {
    @get:Rule
    val benchmarkRule = BenchmarkRule()

    private lateinit var rawRf: Array<FloatArray>
    private lateinit var preprocessor: WearableRfPreprocessor
    private var resultSink = 0.0f

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().context
        rawRf = BenchmarkRawRfAssets.loadPredeclaredValidationSample(context)
        preprocessor = WearableRfPreprocessor()
    }

    @Test
    fun sixChannelRawRfToModelTensor() {
        benchmarkRule.measureRepeated {
            val output = preprocessor.process(rawRf)
            resultSink = output[0]
        }
        assertTrue(resultSink.isFinite())
    }
}
