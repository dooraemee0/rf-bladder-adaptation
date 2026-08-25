package org.dgist.bladderbenchmark.preprocessing

object ChannelConstructor {
    fun construct(clippedCropped: Array<FloatArray>): Array<FloatArray> {
        require(clippedCropped.size == PreprocessingMetadata.PHYSICAL_CHANNELS) {
            "Expected six physical channels, found ${clippedCropped.size}"
        }
        clippedCropped.forEachIndexed { index, channel ->
            require(channel.size == PreprocessingMetadata.CROPPED_SAMPLES) {
                "Unexpected cropped length in physical channel $index: ${channel.size}"
            }
        }

        val result = Array(PreprocessingMetadata.LOGICAL_CHANNELS) {
            FloatArray(PreprocessingMetadata.CROPPED_SAMPLES)
        }
        clippedCropped[4].copyInto(result[0])
        for (sample in 0 until PreprocessingMetadata.CROPPED_SAMPLES) {
            result[1][sample] = (clippedCropped[1][sample] + clippedCropped[2][sample]) / 2.0f
        }
        clippedCropped[5].copyInto(result[2])
        for (channel in 0 until 4) {
            clippedCropped[channel].copyInto(result[channel + 3])
        }
        return result
    }
}
