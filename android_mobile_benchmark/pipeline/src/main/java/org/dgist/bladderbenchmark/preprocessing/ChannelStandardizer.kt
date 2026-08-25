package org.dgist.bladderbenchmark.preprocessing

import kotlin.math.sqrt

object ChannelStandardizer {
    fun standardize(values: Array<FloatArray>): Array<FloatArray> {
        require(values.size == PreprocessingMetadata.LOGICAL_CHANNELS)
        return Array(values.size) { channelIndex ->
            val channel = values[channelIndex]
            require(channel.size == PreprocessingMetadata.RETAINED_SAMPLES)
            require(channel.all(Float::isFinite)) {
                "Standardization input channel $channelIndex contains NaN or infinity"
            }
            val mean = channel.fold(0.0) { sum, value -> sum + value.toDouble() } / channel.size
            var squaredDifference = 0.0
            for (value in channel) {
                val difference = value.toDouble() - mean
                squaredDifference += difference * difference
            }
            val standardDeviation = sqrt(squaredDifference / channel.size)
            val denominator = standardDeviation + PreprocessingMetadata.STANDARDIZATION_EPSILON
            FloatArray(channel.size) { sample ->
                ((channel[sample].toDouble() - mean) / denominator).toFloat()
            }
        }
    }

    fun padTailMean(standardized: Array<FloatArray>): Array<FloatArray> {
        require(standardized.size == PreprocessingMetadata.LOGICAL_CHANNELS)
        return Array(standardized.size) { channelIndex ->
            val channel = standardized[channelIndex]
            require(channel.size == PreprocessingMetadata.RETAINED_SAMPLES)
            val output = FloatArray(PreprocessingMetadata.MODEL_SAMPLES)
            channel.copyInto(output)
            var sum = 0.0
            for (sample in channel.size - PreprocessingMetadata.TAIL_MEAN_SAMPLES until channel.size) {
                sum += channel[sample].toDouble()
            }
            val fill = (sum / PreprocessingMetadata.TAIL_MEAN_SAMPLES).toFloat()
            for (sample in channel.size until output.size) {
                output[sample] = fill
            }
            output
        }
    }
}
