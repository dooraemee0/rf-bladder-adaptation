package org.dgist.bladderbenchmark

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import ai.onnxruntime.OnnxTensor
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.ceil
import kotlin.math.sqrt

data class LatencyStats(
    val meanMs: Double,
    val medianMs: Double,
    val standardDeviationMs: Double,
    val percentile90Ms: Double,
    val minimumMs: Double,
    val maximumMs: Double,
)

data class BenchmarkOutput(
    val reportFile: File,
    val predictionsFile: File,
    val stats: LatencyStats,
    val maxDesktopDifferenceMl: Double,
)

object BenchmarkHarness {
    fun run(
        context: Context,
        warmupRuns: Int = 30,
        measuredRuns: Int = 500,
    ): BenchmarkOutput {
        val samples = TestAssetLoader.load(context)
        val runner = OrtModelRunner(context)
        val tensors = samples.map { runner.createTensor(it.input) }
        val activityManager = context.getSystemService(ActivityManager::class.java)
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val powerManager = context.getSystemService(PowerManager::class.java)
        try {
            val thermalStatusBefore = thermalStatus(powerManager)
            repeat(warmupRuns) { index -> runner.runTensor(tensors[index % tensors.size]) }
            val pssBeforeKb = Debug.getPss()
            val latencyNs = LongArray(measuredRuns)
            repeat(measuredRuns) { index ->
                val start = System.nanoTime()
                runner.runTensor(tensors[index % tensors.size])
                latencyNs[index] = System.nanoTime() - start
            }
            val pssAfterKb = Debug.getPss()
            val thermalStatusAfter = thermalStatus(powerManager)
            val stats = calculateStats(latencyNs)
            val predictions = samples.mapIndexed { index, sample ->
                sample to runner.runTensor(tensors[index])
            }
            val maxDifference = predictions.maxOf {
                kotlin.math.abs(it.second.toDouble() - it.first.desktop8BitMl.toDouble())
            }

            val outputDir = context.getExternalFilesDir(null) ?: context.filesDir
            val predictionsFile = File(outputDir, "android_predictions.csv")
            predictionsFile.bufferedWriter().use { writer ->
                writer.appendLine("sample_id,target_ml,desktop_8bit_ml,prediction_ml,difference_ml")
                predictions.forEach { (sample, prediction) ->
                    writer.appendLine(
                        "${sample.sampleId},${sample.targetMl},${sample.desktop8BitMl},$prediction," +
                            "${prediction - sample.desktop8BitMl}"
                    )
                }
            }

            val report = JSONObject().apply {
                put("benchmark_scope", "OrtSession.run model-only")
                put("batch_size", 1)
                put("input_shape", JSONArray(listOf(1, 7, 400)))
                put("execution_provider", "CPUExecutionProvider")
                put("intra_op_threads", 1)
                put("inter_op_threads", 1)
                put("warmup_runs", warmupRuns)
                put("measured_runs", measuredRuns)
                put("model_asset_read_ms", runner.assetReadTimeNs / 1e6)
                put("session_create_ms", runner.sessionCreateTimeNs / 1e6)
                put("latency_mean_ms", stats.meanMs)
                put("latency_median_ms", stats.medianMs)
                put("latency_standard_deviation_ms", stats.standardDeviationMs)
                put("latency_p90_ms", stats.percentile90Ms)
                put("latency_minimum_ms", stats.minimumMs)
                put("latency_maximum_ms", stats.maximumMs)
                put("pss_before_kb", pssBeforeKb)
                put("pss_after_kb", pssAfterKb)
                put("total_ram_bytes", memoryInfo.totalMem)
                put("thermal_status_before", thermalStatusBefore)
                put("thermal_status_after", thermalStatusAfter)
                put("android_desktop_max_abs_difference_ml", maxDifference)
                put("manufacturer", Build.MANUFACTURER)
                put("device_model", Build.MODEL)
                put("soc_manufacturer", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else "unavailable")
                put("soc_model", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "unavailable")
                put("android_version", Build.VERSION.RELEASE)
                put("android_sdk", Build.VERSION.SDK_INT)
                put("available_processors", Runtime.getRuntime().availableProcessors())
                put("onnx_runtime_version", "1.23.2")
                put("excluded", JSONArray(listOf("model loading", "RF preprocessing", "BLE", "UI update")))
            }
            val reportFile = File(outputDir, "benchmark_report.json")
            reportFile.writeText(report.toString(2) + "\n")
            return BenchmarkOutput(reportFile, predictionsFile, stats, maxDifference)
        } finally {
            tensors.forEach(OnnxTensor::close)
            runner.close()
        }
    }

    private fun calculateStats(valuesNs: LongArray): LatencyStats {
        val sortedMs = valuesNs.map { it / 1e6 }.sorted()
        val mean = sortedMs.average()
        val variance = sortedMs.sumOf { (it - mean) * (it - mean) } / sortedMs.size
        fun percentile(fraction: Double): Double {
            val index = ceil(fraction * sortedMs.size).toInt().coerceIn(1, sortedMs.size) - 1
            return sortedMs[index]
        }
        val middle = sortedMs.size / 2
        val median = if (sortedMs.size % 2 == 0) {
            (sortedMs[middle - 1] + sortedMs[middle]) / 2.0
        } else {
            sortedMs[middle]
        }
        return LatencyStats(
            meanMs = mean,
            medianMs = median,
            standardDeviationMs = sqrt(variance),
            percentile90Ms = percentile(0.90),
            minimumMs = sortedMs.first(),
            maximumMs = sortedMs.last(),
        )
    }

    private fun thermalStatus(powerManager: PowerManager): Int? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) powerManager.currentThermalStatus else null
}
