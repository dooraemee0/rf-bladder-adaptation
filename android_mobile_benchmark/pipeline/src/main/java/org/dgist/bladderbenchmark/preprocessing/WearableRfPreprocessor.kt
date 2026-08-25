package org.dgist.bladderbenchmark.preprocessing

data class PreprocessingStages(
    val parsedSixChannelRaw: Array<FloatArray>,
    val amplitudeScaled: Array<FloatArray>,
    val tgcOutput: Array<FloatArray>,
    val bandpassFiltered: Array<FloatArray>,
    val clippedCropped: Array<FloatArray>,
    val logicalSevenChannel: Array<FloatArray>,
    val gaussianSmoothed: Array<FloatArray>,
    val downsampled220: Array<FloatArray>,
    val standardized: Array<FloatArray>,
    val padded7x400: Array<FloatArray>,
) {
    fun modelInputFlat(): FloatArray = flatten(padded7x400)

    fun namedStages(): LinkedHashMap<String, Array<FloatArray>> = linkedMapOf(
        "01_parsed_six_channel_raw" to parsedSixChannelRaw,
        "02_amplitude_scaled" to amplitudeScaled,
        "03_tgc_output" to tgcOutput,
        "04_bandpass_filtered" to bandpassFiltered,
        "05_clipped_cropped" to clippedCropped,
        "06_logical_seven_channel" to logicalSevenChannel,
        "07_gaussian_smoothed" to gaussianSmoothed,
        "08_downsampled_220" to downsampled220,
        "09_standardized" to standardized,
        "10_padded_7x400" to padded7x400,
    )

    companion object {
        fun flatten(channels: Array<FloatArray>): FloatArray {
            val values = FloatArray(channels.sumOf { it.size })
            var destination = 0
            for (channel in channels) {
                channel.copyInto(values, destination)
                destination += channel.size
            }
            return values
        }
    }
}

class WearableRfPreprocessor {
    fun process(physicalRaw: Array<FloatArray>): FloatArray =
        processWithStages(physicalRaw).modelInputFlat()

    fun processWithStages(physicalRaw: Array<FloatArray>): PreprocessingStages {
        val parsed = validateAndCopy(physicalRaw)
        val scaled = Array(PreprocessingMetadata.PHYSICAL_CHANNELS) { channel ->
            FloatArray(PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL) { sample ->
                parsed[channel][sample] / PreprocessingMetadata.AMPLITUDE_DIVISOR
            }
        }
        val tgc = TgcProcessor.apply(scaled)
        val filtered = ButterworthFilter.apply(tgc)
        val clippedCropped = Array(PreprocessingMetadata.PHYSICAL_CHANNELS) { channel ->
            FloatArray(PreprocessingMetadata.CROPPED_SAMPLES) { croppedSample ->
                val value = filtered[channel][PreprocessingMetadata.CROP_START + croppedSample]
                value.coerceIn(PreprocessingMetadata.CLIP_MIN, PreprocessingMetadata.CLIP_MAX)
            }
        }
        val logical = ChannelConstructor.construct(clippedCropped)
        val smoothed = GaussianChannelSmoother.apply(logical)
        val downsampled = Array(PreprocessingMetadata.LOGICAL_CHANNELS) { channel ->
            FloatArray(PreprocessingMetadata.RETAINED_SAMPLES) { sample ->
                smoothed[channel][sample * PreprocessingMetadata.DOWNSAMPLE_RATE]
            }
        }
        val standardized = ChannelStandardizer.standardize(downsampled)
        val padded = ChannelStandardizer.padTailMean(standardized)
        require(padded.all { channel -> channel.all(Float::isFinite) }) {
            "Preprocessing produced NaN or infinity"
        }
        return PreprocessingStages(
            parsedSixChannelRaw = parsed,
            amplitudeScaled = scaled,
            tgcOutput = tgc,
            bandpassFiltered = filtered,
            clippedCropped = clippedCropped,
            logicalSevenChannel = logical,
            gaussianSmoothed = smoothed,
            downsampled220 = downsampled,
            standardized = standardized,
            padded7x400 = padded,
        )
    }

    private fun validateAndCopy(physicalRaw: Array<FloatArray>): Array<FloatArray> {
        require(physicalRaw.size == PreprocessingMetadata.PHYSICAL_CHANNELS) {
            "Expected ${PreprocessingMetadata.PHYSICAL_CHANNELS} physical RF traces, found ${physicalRaw.size}"
        }
        return Array(physicalRaw.size) { channelIndex ->
            val channel = physicalRaw[channelIndex]
            require(channel.size == PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL) {
                "Expected ${PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL} samples in physical channel " +
                    "$channelIndex, found ${channel.size}"
            }
            require(channel.all(Float::isFinite)) {
                "Physical RF channel $channelIndex contains NaN or infinity"
            }
            channel.copyOf()
        }
    }
}
