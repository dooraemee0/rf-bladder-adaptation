package org.dgist.bladderbenchmark

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.dgist.bladderbenchmark.preprocessing.PreprocessingMetadata
import org.dgist.bladderbenchmark.preprocessing.PreprocessingStages
import org.dgist.bladderbenchmark.preprocessing.WearableRfPreprocessor
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class PreprocessingParityTest {
    private val sampleDirectories = listOf("50_30", "150_30", "300_30")

    @Test
    fun compareEveryPreprocessingStageAndFinalPrediction() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val testContext = instrumentation.context
        val targetContext = instrumentation.targetContext
        val preprocessor = WearableRfPreprocessor()
        val sampleReports = JSONArray()

        OrtModelRunner(targetContext).use { runner ->
            for (sampleDirectory in sampleDirectories) {
                val manifest = loadManifest(testContext, sampleDirectory)
                val raw = loadMatrix(
                    context = testContext,
                    asset = "golden_validation/$sampleDirectory/01_parsed_six_channel_raw.f32le.bin",
                    channels = PreprocessingMetadata.PHYSICAL_CHANNELS,
                    samples = PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL,
                )
                val actualStages = preprocessor.processWithStages(raw)
                val stageReports = JSONArray()

                for ((stageName, actual) in actualStages.namedStages()) {
                    val expectedMetadata = manifest.getJSONObject("stages").getJSONObject(stageName)
                    val shape = expectedMetadata.getJSONArray("shape")
                    val expectedChannels = shape.getInt(0)
                    val expectedSamples = shape.getInt(1)
                    assertEquals(expectedChannels, actual.size)
                    assertTrue(actual.all { channel -> channel.size == expectedSamples })
                    val expected = loadMatrix(
                        context = testContext,
                        asset = "golden_validation/$sampleDirectory/${expectedMetadata.getString("file")}",
                        channels = expectedChannels,
                        samples = expectedSamples,
                    )
                    stageReports.put(compareStage(stageName, expected, actual))
                }

                val actualPrediction = runner.runInput(actualStages.modelInputFlat())
                val expectedPrediction = manifest
                    .getJSONObject("predictions_ml")
                    .getDouble("u8u8")
                assertTrue(actualPrediction.isFinite())
                sampleReports.put(
                    JSONObject().apply {
                        put("sample_id", manifest.getString("sample_id"))
                        put("partition", manifest.getString("partition"))
                        put("target_ml", manifest.getDouble("target_ml"))
                        put("stages", stageReports)
                        put("android_u8u8_prediction_ml", actualPrediction.toDouble())
                        put("python_u8u8_prediction_ml", expectedPrediction)
                        put(
                            "prediction_absolute_difference_ml",
                            abs(actualPrediction.toDouble() - expectedPrediction),
                        )
                    }
                )
            }
        }

        val report = JSONObject().apply {
            put("report_type", "Android-Python wearable RF preprocessing parity")
            put("partition", "wearable validation only")
            put("held_out_test_used", false)
            put("numeric_tolerance_predeclared", false)
            put("comparison_policy", "Record observed differences; assert only shape and finiteness")
            put("physical_channel_order", JSONArray(PreprocessingMetadata.physicalChannelOrder))
            put("logical_channel_order", JSONArray(PreprocessingMetadata.logicalChannelOrder))
            put("samples", sampleReports)
        }
        val outputDirectory = targetContext.getExternalFilesDir(null) ?: targetContext.filesDir
        File(outputDirectory, "preprocessing_parity_android.json").writeText(report.toString(2) + "\n")
    }

    private fun compareStage(
        stageName: String,
        expected: Array<FloatArray>,
        actual: Array<FloatArray>,
    ): JSONObject {
        var maximumAbsoluteDifference = 0.0
        var sumAbsoluteDifference = 0.0
        var sumSquaredDifference = 0.0
        var sumSquaredReference = 0.0
        var expectedNonFinite = 0
        var actualNonFinite = 0
        var valueCount = 0
        val channelReports = JSONArray()

        for (channelIndex in expected.indices) {
            var channelMaximum = 0.0
            var channelAbsoluteSum = 0.0
            var channelSquaredSum = 0.0
            val expectedChannel = expected[channelIndex]
            val actualChannel = actual[channelIndex]
            for (sampleIndex in expectedChannel.indices) {
                val reference = expectedChannel[sampleIndex].toDouble()
                val observed = actualChannel[sampleIndex].toDouble()
                if (!reference.isFinite()) expectedNonFinite += 1
                if (!observed.isFinite()) actualNonFinite += 1
                if (reference.isFinite() && observed.isFinite()) {
                    val difference = abs(observed - reference)
                    channelMaximum = max(channelMaximum, difference)
                    channelAbsoluteSum += difference
                    channelSquaredSum += difference * difference
                    maximumAbsoluteDifference = max(maximumAbsoluteDifference, difference)
                    sumAbsoluteDifference += difference
                    sumSquaredDifference += difference * difference
                    sumSquaredReference += reference * reference
                    valueCount += 1
                }
            }
            channelReports.put(
                JSONObject().apply {
                    put("channel_index", channelIndex)
                    put("maximum_absolute_difference", channelMaximum)
                    put("mean_absolute_difference", channelAbsoluteSum / expectedChannel.size)
                    put("rmse", sqrt(channelSquaredSum / expectedChannel.size))
                }
            )
        }

        assertEquals(0, actualNonFinite)
        return JSONObject().apply {
            put("stage", stageName)
            put("expected_shape", JSONArray(listOf(expected.size, expected[0].size)))
            put("actual_shape", JSONArray(listOf(actual.size, actual[0].size)))
            put("expected_dtype", "float32-little-endian")
            put("actual_dtype", "Kotlin Float / IEEE-754 binary32")
            put("shape_mismatch", false)
            put("dtype_mismatch", false)
            put("maximum_absolute_difference", maximumAbsoluteDifference)
            put("mean_absolute_difference", sumAbsoluteDifference / valueCount)
            put("rmse", sqrt(sumSquaredDifference / valueCount))
            put(
                "relative_l2_error",
                sqrt(sumSquaredDifference) / max(sqrt(sumSquaredReference), 1e-30),
            )
            put("expected_nonfinite_count", expectedNonFinite)
            put("actual_nonfinite_count", actualNonFinite)
            put("channels", channelReports)
        }
    }

    private fun loadManifest(context: Context, sampleDirectory: String): JSONObject =
        JSONObject(
            context.assets
                .open("golden_validation/$sampleDirectory/sample_manifest.json")
                .bufferedReader()
                .use { it.readText() }
        )

    private fun loadMatrix(
        context: Context,
        asset: String,
        channels: Int,
        samples: Int,
    ): Array<FloatArray> {
        val bytes = context.assets.open(asset).use { it.readBytes() }
        val expectedBytes = channels * samples * Float.SIZE_BYTES
        require(bytes.size == expectedBytes) {
            "$asset contains ${bytes.size} bytes; expected $expectedBytes"
        }
        val flat = FloatArray(channels * samples)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(flat)
        return Array(channels) { channel ->
            flat.copyOfRange(channel * samples, (channel + 1) * samples)
        }
    }
}
