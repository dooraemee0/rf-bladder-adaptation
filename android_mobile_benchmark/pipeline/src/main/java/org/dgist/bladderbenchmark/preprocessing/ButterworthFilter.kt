package org.dgist.bladderbenchmark.preprocessing

object ButterworthFilter {
    private const val EDGE_SAMPLES = 45

    // scipy.signal.butter(N=7, Wn=[1.5e6, 4.5e6], btype="band", fs=20e6)
    private val b = doubleArrayOf(
        0.0009628940476967194,
        0.0,
        -0.006740258333877035,
        0.0,
        0.020220775001631105,
        0.0,
        -0.03370129166938518,
        0.0,
        0.03370129166938518,
        0.0,
        -0.020220775001631105,
        0.0,
        0.006740258333877035,
        0.0,
        -0.0009628940476967194,
    )
    private val a = doubleArrayOf(
        1.0,
        -6.453395330000912,
        20.913214038490448,
        -44.949286859709844,
        71.45464796016648,
        -88.59231588623601,
        88.05843254742703,
        -71.10446812961118,
        46.806571375594096,
        -24.98830436639225,
        10.658709904313934,
        -3.5303740116686915,
        0.8623915685376176,
        -0.14017787154703928,
        0.011662728049189596,
    )

    // scipy.signal.lfilter_zi(b, a), frozen with SciPy 1.15.1.
    private val steadyState = doubleArrayOf(
        -0.0009628940476932927,
        -0.0009628940477154043,
        0.005777364286233293,
        0.00577736428607927,
        -0.014443410715307003,
        -0.01444341071561057,
        0.01925788095407634,
        0.0192578809538327,
        -0.0144434107153921,
        -0.014443410715477722,
        0.005777364286189899,
        0.005777364286177802,
        -0.000962894047696279,
        -0.0009628940476967592,
    )

    fun apply(channels: Array<FloatArray>): Array<FloatArray> =
        Array(channels.size) { channel -> filtfilt(channels[channel]) }

    fun filtfilt(input: FloatArray): FloatArray {
        require(input.size > EDGE_SAMPLES) {
            "Input length ${input.size} must exceed filtfilt edge length $EDGE_SAMPLES"
        }
        require(input.all { it.isFinite() }) { "Band-pass input contains NaN or infinity" }

        val extended = oddExtension(input)
        val forward = lfilter(extended)
        forward.reverse()
        val backward = lfilter(forward)
        backward.reverse()
        return FloatArray(input.size) { index -> backward[index + EDGE_SAMPLES].toFloat() }
    }

    private fun oddExtension(input: FloatArray): DoubleArray {
        val result = DoubleArray(input.size + 2 * EDGE_SAMPLES)
        val leftEndpoint = input.first().toDouble()
        val rightEndpoint = input.last().toDouble()
        for (offset in 0 until EDGE_SAMPLES) {
            result[offset] = 2.0 * leftEndpoint - input[EDGE_SAMPLES - offset].toDouble()
            result[EDGE_SAMPLES + input.size + offset] =
                2.0 * rightEndpoint - input[input.lastIndex - 1 - offset].toDouble()
        }
        for (index in input.indices) {
            result[EDGE_SAMPLES + index] = input[index].toDouble()
        }
        return result
    }

    private fun lfilter(input: DoubleArray): DoubleArray {
        val state = DoubleArray(steadyState.size) { index -> steadyState[index] * input.first() }
        val output = DoubleArray(input.size)
        for (sample in input.indices) {
            val x = input[sample]
            val y = b[0] * x + state[0]
            output[sample] = y
            for (index in 0 until state.lastIndex) {
                state[index] = state[index + 1] + b[index + 1] * x - a[index + 1] * y
            }
            state[state.lastIndex] = b.last() * x - a.last() * y
        }
        return output
    }
}
