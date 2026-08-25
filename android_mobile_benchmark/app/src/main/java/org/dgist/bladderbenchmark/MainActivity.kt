package org.dgist.bladderbenchmark

import android.app.Activity
import android.os.Bundle
import android.os.SystemClock
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

class MainActivity : Activity() {
    @Volatile
    private var modelRunner: OrtModelRunner? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val density = resources.displayMetrics.density
        val padding = (20 * density).toInt()
        val title = TextView(this).apply {
            text = "Local smartphone inference"
            textSize = 24f
        }
        val configuration = TextView(this).apply {
            text = "Execution mode: Local / server-free\n" +
                "Model: DER++ seed 3\n" +
                "Quantization: static U8U8 QDQ\n" +
                "Input: six-channel raw RF"
            textSize = 15f
            setPadding(0, padding / 2, 0, padding)
        }
        val status = TextView(this).apply {
            text = "Ready for smartphone-resident RF preprocessing and inference."
            textSize = 18f
            setPadding(0, padding, 0, padding)
        }
        val localButton = Button(this).apply {
            text = "Run local RF-to-volume inference"
            setOnClickListener {
                isEnabled = false
                status.text = "Processing raw RF locally..."
                thread {
                    val message = try {
                        val sample = LocalRawRfAssetLoader.load(applicationContext)[1]
                        val result = LocalInferencePipeline(getOrCreateRunner()).run(sample)
                        val uiRequestNs = SystemClock.elapsedRealtimeNanos()
                        writeLocalReport(sample, result, uiRequestNs)
                        val alarmText = if (result.alarmActive) "ALARM: 200 mL threshold reached" else "Normal: below 200 mL"
                        "Predicted bladder volume\n${"%.1f".format(result.predictedVolumeMl)} mL\n\n" +
                            "$alarmText\n\n" +
                            "Source: ${result.sampleId} (${sample.partition})\n" +
                            "Processing: entirely on this smartphone"
                    } catch (error: Throwable) {
                        "Local inference failed:\n${error.stackTraceToString()}"
                    }
                    runOnUiThread {
                        status.text = message
                        isEnabled = true
                    }
                }
            }
        }
        val legacyButton = Button(this).apply {
            text = "Legacy model-only validation (debug)"
            setOnClickListener {
                isEnabled = false
                status.text = "Running legacy model-only validation..."
                thread {
                    val message = try {
                        val output = BenchmarkHarness.run(applicationContext)
                        "Legacy model-only debug report saved.\n" +
                            "Report: ${output.reportFile.absolutePath}\n" +
                            "Predictions: ${output.predictionsFile.absolutePath}"
                    } catch (error: Throwable) {
                        "Legacy validation failed:\n${error.stackTraceToString()}"
                    }
                    runOnUiThread {
                        status.text = message
                        isEnabled = true
                    }
                }
            }
        }

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(padding, padding, padding, padding)
            addView(title, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(configuration, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(localButton, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(status, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(legacyButton, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        }
        setContentView(ScrollView(this).apply { addView(content) })
    }

    @Synchronized
    private fun getOrCreateRunner(): OrtModelRunner {
        val existing = modelRunner
        if (existing != null) return existing
        return OrtModelRunner(applicationContext).also { modelRunner = it }
    }

    private fun writeLocalReport(
        sample: LocalRawRfSample,
        result: LocalInferenceResult,
        uiRequestNs: Long,
    ): File {
        val outputDir = getExternalFilesDir(null) ?: filesDir
        val report = JSONObject().apply {
            put("scope", "local raw-RF file to UI scheduling")
            put("server_used", false)
            put("sample_id", sample.sampleId)
            put("partition", sample.partition)
            put("predicted_volume_ml", result.predictedVolumeMl)
            put("alarm_threshold_ml", 200)
            put("alarm_active", result.alarmActive)
            put("t_input_ready_ns", result.inputReadyNs)
            put("t_preprocessing_done_ns", result.preprocessingDoneNs)
            put("t_inference_done_ns", result.inferenceDoneNs)
            put("t_ui_request_ns", uiRequestNs)
            put("preprocessing_ms", result.preprocessingMs)
            put("inference_ms", result.inferenceMs)
            put("raw_rf_to_volume_ms", result.rawRfToVolumeMs)
            put("ui_scheduling_ms", (uiRequestNs - result.inferenceDoneNs) / 1e6)
            put("excluded", "BLE acquisition and transfer")
        }
        return File(outputDir, "local_inference_report.json").also {
            it.writeText(report.toString(2) + "\n")
        }
    }

    override fun onDestroy() {
        modelRunner?.close()
        modelRunner = null
        super.onDestroy()
    }
}
