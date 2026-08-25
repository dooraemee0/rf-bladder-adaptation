package org.dgist.bladderbenchmark

import android.content.Context
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class TestSample(
    val index: Int,
    val sampleId: String,
    val targetMl: Float,
    val desktop8BitMl: Float,
    val input: FloatArray,
)

object TestAssetLoader {
    const val CHANNELS = 7
    const val SAMPLES_PER_CHANNEL = 400
    const val VALUES_PER_SAMPLE = CHANNELS * SAMPLES_PER_CHANNEL

    fun load(context: Context): List<TestSample> {
        val manifestText = context.assets.open("test_manifest.json").bufferedReader().use { it.readText() }
        val manifest = JSONObject(manifestText)
        val shape = manifest.getJSONArray("shape")
        require(shape.getInt(1) == CHANNELS && shape.getInt(2) == SAMPLES_PER_CHANNEL) {
            "Expected input shape (N, 7, 400), found $shape"
        }
        val sampleCount = shape.getInt(0)
        val bytes = context.assets.open("test_inputs.bin").use { it.readBytes() }
        require(bytes.size == sampleCount * VALUES_PER_SAMPLE * Float.SIZE_BYTES) {
            "Input binary has ${bytes.size} bytes; expected ${sampleCount * VALUES_PER_SAMPLE * Float.SIZE_BYTES}"
        }
        val allValues = FloatArray(sampleCount * VALUES_PER_SAMPLE)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(allValues)

        val rows = manifest.getJSONArray("samples")
        require(rows.length() == sampleCount)
        return List(sampleCount) { index ->
            val row = rows.getJSONObject(index)
            val start = index * VALUES_PER_SAMPLE
            TestSample(
                index = index,
                sampleId = row.getString("sample_id"),
                targetMl = row.getDouble("target_ml").toFloat(),
                desktop8BitMl = when {
                    row.has("desktop_8bit_ml") -> row.getDouble("desktop_8bit_ml").toFloat()
                    else -> row.getDouble("desktop_int8_ml").toFloat()
                },
                input = allValues.copyOfRange(start, start + VALUES_PER_SAMPLE),
            )
        }
    }
}
