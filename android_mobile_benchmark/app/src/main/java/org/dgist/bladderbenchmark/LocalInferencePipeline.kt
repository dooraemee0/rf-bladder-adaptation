package org.dgist.bladderbenchmark

import android.os.SystemClock
import org.dgist.bladderbenchmark.preprocessing.WearableRfPreprocessor

data class LocalInferenceResult(
    val sampleId: String,
    val predictedVolumeMl: Float,
    val alarmActive: Boolean,
    val inputReadyNs: Long,
    val preprocessingDoneNs: Long,
    val inferenceDoneNs: Long,
) {
    val preprocessingMs: Double get() = (preprocessingDoneNs - inputReadyNs) / 1e6
    val inferenceMs: Double get() = (inferenceDoneNs - preprocessingDoneNs) / 1e6
    val rawRfToVolumeMs: Double get() = (inferenceDoneNs - inputReadyNs) / 1e6
}

class LocalInferencePipeline(
    private val modelRunner: OrtModelRunner,
    private val preprocessor: WearableRfPreprocessor = WearableRfPreprocessor(),
) {
    fun run(sample: LocalRawRfSample): LocalInferenceResult {
        val inputReadyNs = SystemClock.elapsedRealtimeNanos()
        val modelInput = preprocessor.process(sample.physicalTraces)
        val preprocessingDoneNs = SystemClock.elapsedRealtimeNanos()
        val prediction = modelRunner.runInput(modelInput)
        val inferenceDoneNs = SystemClock.elapsedRealtimeNanos()
        return LocalInferenceResult(
            sampleId = sample.sampleId,
            predictedVolumeMl = prediction,
            alarmActive = prediction >= 200.0f,
            inputReadyNs = inputReadyNs,
            preprocessingDoneNs = preprocessingDoneNs,
            inferenceDoneNs = inferenceDoneNs,
        )
    }
}
