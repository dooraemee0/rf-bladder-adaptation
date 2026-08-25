package org.dgist.bladderbenchmark.preprocessing

object GaussianChannelSmoother {
    private val weights = doubleArrayOf(
        0.00013383062461474175,
        0.0044318616200312655,
        0.05399112742070441,
        0.24197144565660073,
        0.39894346935609776,
        0.24197144565660073,
        0.05399112742070441,
        0.0044318616200312655,
        0.00013383062461474175,
    )
    private const val RADIUS = 4

    fun apply(logicalChannels: Array<FloatArray>): Array<FloatArray> {
        require(logicalChannels.size == PreprocessingMetadata.LOGICAL_CHANNELS)
        logicalChannels.forEachIndexed { index, channel ->
            require(channel.size == PreprocessingMetadata.CROPPED_SAMPLES) {
                "Unexpected logical channel length at $index: ${channel.size}"
            }
        }
        val output = Array(PreprocessingMetadata.LOGICAL_CHANNELS) {
            FloatArray(PreprocessingMetadata.CROPPED_SAMPLES)
        }
        smoothGroup(logicalChannels, output, startChannel = 0, channelCount = 3)
        smoothGroup(logicalChannels, output, startChannel = 3, channelCount = 4)
        return output
    }

    private fun smoothGroup(
        input: Array<FloatArray>,
        output: Array<FloatArray>,
        startChannel: Int,
        channelCount: Int,
    ) {
        for (outputOffset in 0 until channelCount) {
            for (sample in 0 until PreprocessingMetadata.CROPPED_SAMPLES) {
                var sum = 0.0
                for (kernelIndex in weights.indices) {
                    val sourceOffset = reflectIndex(
                        outputOffset + kernelIndex - RADIUS,
                        channelCount,
                    )
                    sum += weights[kernelIndex] * input[startChannel + sourceOffset][sample].toDouble()
                }
                output[startChannel + outputOffset][sample] = sum.toFloat()
            }
        }
    }

    private fun reflectIndex(index: Int, length: Int): Int {
        var reflected = index
        while (reflected < 0 || reflected >= length) {
            reflected = if (reflected < 0) -reflected - 1 else 2 * length - reflected - 1
        }
        return reflected
    }
}
