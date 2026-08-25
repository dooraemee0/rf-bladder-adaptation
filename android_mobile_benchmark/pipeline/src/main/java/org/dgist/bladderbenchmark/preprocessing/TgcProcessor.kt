package org.dgist.bladderbenchmark.preprocessing

import kotlin.math.exp

object TgcProcessor {
    fun apply(amplitudeScaled: Array<FloatArray>): Array<FloatArray> {
        requireShape(amplitudeScaled, PreprocessingMetadata.PHYSICAL_CHANNELS, PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL)
        val midpoint = PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL / 2
        val gains = FloatArray(PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL) { sampleIndex ->
            val distanceMm = sampleIndex.toFloat() /
                PreprocessingMetadata.SAMPLING_FREQUENCY_HZ *
                PreprocessingMetadata.SOUND_SPEED_M_PER_S /
                2.0f *
                1000.0f
            val coefficient = if (sampleIndex < midpoint) 0.15f else 0.4f
            exp((coefficient * distanceMm / 10.0f).toDouble()).toFloat()
        }
        return Array(PreprocessingMetadata.PHYSICAL_CHANNELS) { channel ->
            FloatArray(PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL) { sample ->
                amplitudeScaled[channel][sample] * gains[sample]
            }
        }
    }

    private fun requireShape(values: Array<FloatArray>, channels: Int, samples: Int) {
        require(values.size == channels) { "Expected $channels channels, found ${values.size}" }
        values.forEachIndexed { index, channel ->
            require(channel.size == samples) {
                "Expected $samples samples in channel $index, found ${channel.size}"
            }
        }
    }
}
