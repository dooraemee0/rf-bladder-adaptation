package org.dgist.bladderbenchmark

import android.content.Context
import org.dgist.bladderbenchmark.preprocessing.PreprocessingMetadata
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

data class LocalRawRfSample(
    val sampleId: String,
    val partition: String,
    val targetMl: Float,
    val expectedU8U8Ml: Float,
    val physicalTraces: Array<FloatArray>,
)

object LocalRawRfAssetLoader {
    fun load(context: Context): List<LocalRawRfSample> {
        val manifest = JSONObject(
            context.assets.open("local_raw_manifest.json").bufferedReader().use { it.readText() }
        )
        require(manifest.getString("partition") == "wearable validation only")
        require(!manifest.getBoolean("held_out_test_used"))
        val samples = manifest.getJSONArray("samples")
        return List(samples.length()) { index ->
            val row = samples.getJSONObject(index)
            val asset = row.getString("asset")
            val bytes = context.assets.open(asset).use { it.readBytes() }
            val expectedBytes = PreprocessingMetadata.PHYSICAL_CHANNELS *
                PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL * Float.SIZE_BYTES
            require(bytes.size == expectedBytes) {
                "Raw RF asset $asset has ${bytes.size} bytes; expected $expectedBytes"
            }
            require(sha256(bytes) == row.getString("sha256")) { "Raw RF asset SHA mismatch: $asset" }
            val flat = FloatArray(
                PreprocessingMetadata.PHYSICAL_CHANNELS * PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL
            )
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(flat)
            val traces = Array(PreprocessingMetadata.PHYSICAL_CHANNELS) { channel ->
                val start = channel * PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL
                flat.copyOfRange(start, start + PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL)
            }
            LocalRawRfSample(
                sampleId = row.getString("sample_id"),
                partition = row.getString("partition"),
                targetMl = row.getDouble("target_ml").toFloat(),
                expectedU8U8Ml = row.getDouble("expected_u8u8_ml").toFloat(),
                physicalTraces = traces,
            )
        }
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { byte -> "%02x".format(byte) }
}
