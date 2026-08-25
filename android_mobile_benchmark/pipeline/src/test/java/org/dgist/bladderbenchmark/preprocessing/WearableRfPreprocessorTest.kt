package org.dgist.bladderbenchmark.preprocessing

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class WearableRfPreprocessorTest {
    @Test
    fun virtualHorizontalChannelIsMeanOfS2AndS3() {
        val physical = Array(6) { channel ->
            FloatArray(PreprocessingMetadata.CROPPED_SAMPLES) { sample -> channel * 10.0f + sample }
        }
        val logical = ChannelConstructor.construct(physical)
        for (sample in 0 until PreprocessingMetadata.CROPPED_SAMPLES) {
            assertEquals((physical[1][sample] + physical[2][sample]) / 2.0f, logical[1][sample], 0.0f)
        }
        assertArrayEquals(physical[4], logical[0], 0.0f)
        assertArrayEquals(physical[5], logical[2], 0.0f)
    }

    @Test
    fun paddingRepeatsPerChannelTailMeanRatherThanZero() {
        val standardized = Array(7) { channel ->
            FloatArray(220) { sample -> channel + sample / 100.0f }
        }
        val padded = ChannelStandardizer.padTailMean(standardized)
        assertEquals(7, padded.size)
        padded.forEachIndexed { channelIndex, channel ->
            assertEquals(400, channel.size)
            val fill = channel[220]
            assertNotEquals(0.0f, fill)
            assertTrue((220 until 400).all { channel[it] == fill })
            assertArrayEquals(standardized[channelIndex], channel.copyOfRange(0, 220), 0.0f)
        }
    }

    @Test
    fun malformedPhysicalInputFailsExplicitly() {
        val wrongChannelCount = Array(5) { FloatArray(5120) }
        assertThrows(IllegalArgumentException::class.java) {
            WearableRfPreprocessor().process(wrongChannelCount)
        }

        val wrongLength = Array(6) { FloatArray(5119) }
        assertThrows(IllegalArgumentException::class.java) {
            WearableRfPreprocessor().process(wrongLength)
        }
    }

    @Test
    fun constantChannelsRemainFiniteAndStandardizeToZero() {
        val constant = Array(7) { FloatArray(220) { 2.0f } }
        val standardized = ChannelStandardizer.standardize(constant)
        assertTrue(standardized.all { channel -> channel.all { it == 0.0f && it.isFinite() } })
    }
}
