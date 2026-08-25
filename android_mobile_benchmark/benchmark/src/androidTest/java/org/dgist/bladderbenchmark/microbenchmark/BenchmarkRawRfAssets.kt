package org.dgist.bladderbenchmark.microbenchmark

import android.content.Context
import org.dgist.bladderbenchmark.preprocessing.PreprocessingMetadata
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

object BenchmarkRawRfAssets {
    fun loadPredeclaredValidationSample(context: Context): Array<FloatArray> {
        val manifest = JSONObject(
            context.assets.open("local_raw_manifest.json").bufferedReader().use { it.readText() }
        )
        require(manifest.getString("partition") == "wearable validation only")
        require(!manifest.getBoolean("held_out_test_used"))
        val sample = manifest.getJSONArray("samples").getJSONObject(0)
        val asset = sample.getString("asset")
        val bytes = context.assets.open(asset).use { it.readBytes() }
        require(sha256(bytes) == sample.getString("sha256"))
        val expectedValues = PreprocessingMetadata.PHYSICAL_CHANNELS *
            PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL
        require(bytes.size == expectedValues * Float.SIZE_BYTES)
        val flat = FloatArray(expectedValues)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(flat)
        return Array(PreprocessingMetadata.PHYSICAL_CHANNELS) { channel ->
            val start = channel * PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL
            flat.copyOfRange(start, start + PreprocessingMetadata.RAW_SAMPLES_PER_CHANNEL)
        }
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { byte -> "%02x".format(byte) }
}
