package org.dgist.bladderbenchmark.preprocessing

object PreprocessingMetadata {
    const val PHYSICAL_CHANNELS = 6
    const val RAW_SAMPLES_PER_CHANNEL = 5120
    const val LOGICAL_CHANNELS = 7
    const val CROP_START = 300
    const val CROP_END_EXCLUSIVE = 3000
    const val CROPPED_SAMPLES = CROP_END_EXCLUSIVE - CROP_START
    const val DOWNSAMPLE_RATE = 10
    const val RETAINED_SAMPLES = 220
    const val MODEL_SAMPLES = 400
    const val TAIL_MEAN_SAMPLES = 30
    const val AMPLITUDE_DIVISOR = 1700.0f
    const val CLIP_MIN = -0.1f
    const val CLIP_MAX = 0.1f
    const val STANDARDIZATION_EPSILON = 1e-8
    const val SAMPLING_FREQUENCY_HZ = 20_000_000.0f
    const val SOUND_SPEED_M_PER_S = 1500.0f

    val physicalChannelOrder = listOf("S1", "S2", "S3", "S4", "H_left", "H_right")
    val logicalChannelOrder = listOf(
        "H_left",
        "H_virtual_mean_S2_S3",
        "H_right",
        "S1",
        "S2",
        "S3",
        "S4",
    )
}
