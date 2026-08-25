package org.dgist.bladderbenchmark

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.dgist.bladderbenchmark.preprocessing.PreprocessingMetadata
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
import java.security.MessageDigest
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class HeldOutRawRfConcordanceTest {
    @Test
    fun evaluateFrozenRawRfPipelineOnceOnHeldOutTest() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val testContext = instrumentation.context
        val targetContext = instrumentation.targetContext
        val manifest = JSONObject(
            testContext.assets.open("heldout_raw_manifest.json").bufferedReader().use { it.readText() }
        )
        assertEquals("held-out wearable test", manifest.getString("partition"))
        assertTrue(manifest.getBoolean("one_time_evaluation_after_validation_freeze"))
        assertEquals(60, manifest.getInt("sample_count"))

        val samples = manifest.getJSONArray("samples")
        val expectedInputs = loadFlatFloats(
            testContext,
            manifest.getString("expected_inputs_asset"),
            60 * PreprocessingMetadata.LOGICAL_CHANNELS * PreprocessingMetadata.MODEL_SAMPLES,
        )
        val seenIds = mutableSetOf<String>()
        val sampleReports = JSONArray()
        val csv = StringBuilder(
            "index,sample_id,target_ml,android_raw_rf_u8u8_ml,desktop_u8u8_ml," +
                "prediction_abs_difference_ml,tensor_max_abs_difference,tensor_mean_abs_difference," +
                "tensor_rmse,alarm_target,alarm_predicted\n"
        )
        val targets = DoubleArray(samples.length())
        val predictions = DoubleArray(samples.length())
        var totalTensorAbsoluteDifference = 0.0
        var totalTensorSquaredDifference = 0.0
        var totalTensorValues = 0L
        var maximumTensorDifference = 0.0
        var predictionDifferenceSum = 0.0
        var maximumPredictionDifference = 0.0
        var nonFiniteTensorCount = 0
        var nonFinitePredictionCount = 0
        val preprocessor = WearableRfPreprocessor()

        OrtModelRunner(targetContext).use { runner ->
            for (position in 0 until samples.length()) {
                val row = samples.getJSONObject(position)
                val index = row.getInt("index")
                assertEquals(position, index)
                val sampleId = row.getString("sample_id")
                assertTrue("Duplicate sample ID: $sampleId", seenIds.add(sampleId))
                val rawBytes = testContext.assets.open(row.getString("asset")).use { it.readBytes() }
                assertEquals(row.getString("raw_sha256"), sha256(rawBytes))
                val raw = bytesToMatrix(
                    rawBytes,
                    PreprocessingMetadata.PHYSICAL_CHANNELS,
                    PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL,
                )
                val actualInput = preprocessor.process(raw)
                val expectedOffset = index * actualInput.size
                var sampleMaximum = 0.0
                var sampleAbsoluteSum = 0.0
                var sampleSquaredSum = 0.0
                for (valueIndex in actualInput.indices) {
                    val actual = actualInput[valueIndex].toDouble()
                    val expected = expectedInputs[expectedOffset + valueIndex].toDouble()
                    if (!actual.isFinite()) nonFiniteTensorCount += 1
                    if (actual.isFinite() && expected.isFinite()) {
                        val difference = abs(actual - expected)
                        sampleMaximum = max(sampleMaximum, difference)
                        sampleAbsoluteSum += difference
                        sampleSquaredSum += difference * difference
                        maximumTensorDifference = max(maximumTensorDifference, difference)
                        totalTensorAbsoluteDifference += difference
                        totalTensorSquaredDifference += difference * difference
                        totalTensorValues += 1
                    }
                }

                val prediction = runner.runInput(actualInput).toDouble()
                if (!prediction.isFinite()) nonFinitePredictionCount += 1
                val desktopPrediction = row.getDouble("expected_desktop_u8u8_ml")
                val predictionDifference = abs(prediction - desktopPrediction)
                predictionDifferenceSum += predictionDifference
                maximumPredictionDifference = max(maximumPredictionDifference, predictionDifference)
                val target = row.getDouble("target_ml")
                targets[index] = target
                predictions[index] = prediction
                val alarmTarget = target >= ALARM_THRESHOLD_ML
                val alarmPrediction = prediction >= ALARM_THRESHOLD_ML
                val sampleMeanDifference = sampleAbsoluteSum / actualInput.size
                val sampleRmse = sqrt(sampleSquaredSum / actualInput.size)

                sampleReports.put(
                    JSONObject().apply {
                        put("index", index)
                        put("sample_id", sampleId)
                        put("target_ml", target)
                        put("android_raw_rf_u8u8_ml", prediction)
                        put("desktop_u8u8_ml", desktopPrediction)
                        put("prediction_absolute_difference_ml", predictionDifference)
                        put("tensor_maximum_absolute_difference", sampleMaximum)
                        put("tensor_mean_absolute_difference", sampleMeanDifference)
                        put("tensor_rmse", sampleRmse)
                        put("tensor_nonfinite_count", actualInput.count { !it.isFinite() })
                        put("alarm_target", alarmTarget)
                        put("alarm_predicted", alarmPrediction)
                    }
                )
                csv.append(
                    String.format(
                        Locale.US,
                        "%d,%s,%.9f,%.9f,%.9f,%.9g,%.9g,%.9g,%.9g,%s,%s\n",
                        index,
                        sampleId,
                        target,
                        prediction,
                        desktopPrediction,
                        predictionDifference,
                        sampleMaximum,
                        sampleMeanDifference,
                        sampleRmse,
                        alarmTarget,
                        alarmPrediction,
                    )
                )
            }
        }

        assertEquals(60, seenIds.size)
        assertEquals(0, nonFiniteTensorCount)
        assertEquals(0, nonFinitePredictionCount)
        val performance = performanceMetrics(targets, predictions)
        val outputDirectory = targetContext.getExternalFilesDir(null) ?: targetContext.filesDir
        val report = JSONObject().apply {
            put("scope", "Frozen Android six-channel raw RF preprocessing and U8U8 inference")
            put("partition", "held-out wearable test")
            put("sample_count", samples.length())
            put("implementation_freeze_manifest_sha256", manifest.getString("implementation_freeze_manifest_sha256"))
            put("model_sha256", manifest.getString("model_sha256"))
            put("missing_sample_count", 0)
            put("duplicate_sample_count", samples.length() - seenIds.size)
            put("nonfinite_tensor_count", nonFiniteTensorCount)
            put("nonfinite_prediction_count", nonFinitePredictionCount)
            put("tensor_maximum_absolute_difference", maximumTensorDifference)
            put("tensor_mean_absolute_difference", totalTensorAbsoluteDifference / totalTensorValues)
            put("tensor_rmse", sqrt(totalTensorSquaredDifference / totalTensorValues))
            put("prediction_maximum_absolute_difference_ml", maximumPredictionDifference)
            put("prediction_mean_absolute_difference_ml", predictionDifferenceSum / samples.length())
            put("performance", performance)
            put("samples", sampleReports)
        }
        File(outputDirectory, "heldout_raw_rf_concordance_android.json")
            .writeText(report.toString(2) + "\n")
        File(outputDirectory, "heldout_raw_rf_predictions_android.csv")
            .writeText(csv.toString())
    }

    private fun performanceMetrics(targets: DoubleArray, predictions: DoubleArray): JSONObject {
        val count = targets.size
        val targetMean = targets.average()
        var absoluteSum = 0.0
        var squaredSum = 0.0
        var totalTargetSquares = 0.0
        var withinFifty = 0
        var truePositive = 0
        var trueNegative = 0
        var falsePositive = 0
        var falseNegative = 0
        for (index in targets.indices) {
            val error = predictions[index] - targets[index]
            absoluteSum += abs(error)
            squaredSum += error * error
            val centeredTarget = targets[index] - targetMean
            totalTargetSquares += centeredTarget * centeredTarget
            if (abs(error) <= 50.0) withinFifty += 1
            val targetAlarm = targets[index] >= ALARM_THRESHOLD_ML
            val predictedAlarm = predictions[index] >= ALARM_THRESHOLD_ML
            when {
                targetAlarm && predictedAlarm -> truePositive += 1
                !targetAlarm && !predictedAlarm -> trueNegative += 1
                !targetAlarm && predictedAlarm -> falsePositive += 1
                else -> falseNegative += 1
            }
        }
        val precision = truePositive.toDouble() / (truePositive + falsePositive)
        val recall = truePositive.toDouble() / (truePositive + falseNegative)
        val f1 = 2.0 * precision * recall / (precision + recall)
        return JSONObject().apply {
            put("mae_ml", absoluteSum / count)
            put("rmse_ml", sqrt(squaredSum / count))
            put("r2", 1.0 - squaredSum / totalTargetSquares)
            put("within_50_ml_accuracy", withinFifty.toDouble() / count)
            put("alarm_threshold_ml", ALARM_THRESHOLD_ML)
            put("alarm_accuracy", (truePositive + trueNegative).toDouble() / count)
            put("alarm_precision", precision)
            put("alarm_recall", recall)
            put("alarm_f1", f1)
            put("true_positive", truePositive)
            put("true_negative", trueNegative)
            put("false_positive", falsePositive)
            put("false_negative", falseNegative)
        }
    }

    private fun loadFlatFloats(context: Context, asset: String, count: Int): FloatArray {
        val bytes = context.assets.open(asset).use { it.readBytes() }
        assertEquals(count * Float.SIZE_BYTES, bytes.size)
        val values = FloatArray(count)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values)
        return values
    }

    private fun bytesToMatrix(bytes: ByteArray, channels: Int, samples: Int): Array<FloatArray> {
        assertEquals(channels * samples * Float.SIZE_BYTES, bytes.size)
        val flat = FloatArray(channels * samples)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(flat)
        return Array(channels) { channel ->
            flat.copyOfRange(channel * samples, (channel + 1) * samples)
        }
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { byte -> "%02x".format(byte) }

    companion object {
        private const val ALARM_THRESHOLD_ML = 200.0
    }
}
