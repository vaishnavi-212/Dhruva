package com.dhruva.nav

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/** How "level" is defined for the speed model. Must match the frame the model was trained in. */
enum class LevelFrame {
    /** The training transform with trailing windows: "down" rebuilt from 8 s of linear acceleration. */
    REPLICA,
    /** Levelled by the phone's own gravity sensor, trailing 8 s. */
    GRAVITY
}

/**
 * The speed model's input, built on the phone exactly as the laptop builds it
 * (scripts/train_eval_phone_features.py, golden-tested in ImuFeaturesGoldenTest):
 *
 *   sensor events -> 10 Hz grid by linear interpolation (numpy.interp semantics)
 *   -> level frame from TRAILING 8 s averages (a phone cannot look ahead)
 *   -> 6 channels: gyroscope then linear acceleration, rotated into that frame
 *   -> ring of the latest [window] rows, handed to the model channel-first.
 *
 * Pure Kotlin with no Android types, so the golden test runs on the host. Preparing inputs
 * differently from training has broken our results four times; change nothing here without
 * regenerating the golden files and re-running the test.
 */
class ImuFeatures(
    private val frame: LevelFrame,
    val window: Int = 300,
    private val onRow: ((Double, FloatArray) -> Unit)? = null
) {
    companion object {
        const val HZ = 10.0
        const val STEP_S = 0.1
        const val TRAIL = 81                 // max(int(8 s * 10 Hz) | 1, 3), as the laptop
        const val G = 9.80665
        const val CHANNELS = 6
        const val ACC = 0                    // TYPE_LINEAR_ACCELERATION, m/s^2, gravity removed
        const val GYRO = 1                   // TYPE_GYROSCOPE, rad/s
        const val GRAV = 2                   // TYPE_GRAVITY, m/s^2
        private const val KEEP_S = 1.0       // raw events kept behind the next grid time
    }

    private class Stream {
        val t = ArrayDeque<Double>()
        val v = ArrayDeque<DoubleArray>()
    }

    private val streams = Array(3) { Stream() }
    private var t0 = Double.NaN
    private var k = 0L

    private val rows = Array(window) { FloatArray(CHANNELS) }
    private var head = 0
    var filled = 0
        private set
    var lastRowTimeS = Double.NaN
        private set

    private val accRing = Array(TRAIL) { DoubleArray(3) }
    private val accSum = DoubleArray(3)
    private val rawRing = Array(TRAIL) { DoubleArray(3) }
    private val rawSum = DoubleArray(3)
    private val gravRing = Array(TRAIL) { DoubleArray(3) }
    private val gravSum = DoubleArray(3)
    private var nTrail = 0L

    /**
     * One sensor event. [tS] is seconds on ONE clock shared by all three sensors
     * (SensorEvent.timestamp / 1e9). Returns how many 10 Hz rows this event completed.
     */
    fun push(kind: Int, tS: Double, x: Double, y: Double, z: Double): Int {
        val s = streams[kind]
        s.t.addLast(tS)
        s.v.addLast(doubleArrayOf(x, y, z))
        if (t0.isNaN()) {
            if (streams.any { it.t.isEmpty() }) return 0
            t0 = streams.maxOf { it.t.first() }
        }
        var n = 0
        while (true) {
            val g = t0 + k * STEP_S
            if (streams.any { it.t.last() < g }) break
            emit(g, interp(streams[ACC], g), interp(streams[GYRO], g), interp(streams[GRAV], g))
            k++
            n++
        }
        val cutoff = t0 + k * STEP_S - KEEP_S
        for (st in streams) {
            while (st.t.size > 2 && st.t[1] <= cutoff) { st.t.removeFirst(); st.v.removeFirst() }
        }
        return n
    }

    /** Forget everything, e.g. after the sensor stream stalled or the screen was off. */
    fun reset() {
        for (st in streams) { st.t.clear(); st.v.clear() }
        t0 = Double.NaN; k = 0; head = 0; filled = 0; nTrail = 0; lastRowTimeS = Double.NaN
        for (sum in arrayOf(accSum, rawSum, gravSum)) sum.fill(0.0)
    }

    /**
     * The latest full window, normalised, channel-first and flattened for an ONNX input of
     * shape (1, 6, window). Null until [window] rows exist.
     */
    fun window(mean: DoubleArray, std: DoubleArray, out: FloatArray = FloatArray(CHANNELS * window)): FloatArray? {
        if (filled < window) return null
        for (j in 0 until window) {
            val row = rows[(head + j) % window]          // oldest first
            for (c in 0 until CHANNELS) out[c * window + j] = ((row[c] - mean[c]) / std[c]).toFloat()
        }
        return out
    }

    // numpy.interp: linear between the events bracketing g, clamped at the ends
    private fun interp(s: Stream, g: Double): DoubleArray {
        val t = s.t
        if (g <= t.first()) return s.v.first()
        if (g >= t.last()) return s.v.last()
        var i = 0
        while (t[i + 1] < g) i++
        val a = s.v[i]; val b = s.v[i + 1]
        val span = t[i + 1] - t[i]
        if (span <= 0.0) return b
        val u = (g - t[i]) / span
        return DoubleArray(3) { a[it] + u * (b[it] - a[it]) }
    }

    private fun emit(tS: Double, a: DoubleArray, w: DoubleArray, gr: DoubleArray) {
        val idx = (nTrail % TRAIL).toInt()
        val count = minOf(nTrail + 1, TRAIL.toLong()).toDouble()
        val r = when (frame) {
            LevelFrame.REPLICA -> {
                slide(accRing, accSum, idx, a)
                val sm = DoubleArray(3) { accSum[it] / count }
                val n = max(norm(sm), 1e-6)
                slide(rawRing, rawSum, idx, DoubleArray(3) { a[it] + sm[it] / n * G })
                basis(DoubleArray(3) { rawSum[it] / count })
            }
            LevelFrame.GRAVITY -> {
                slide(gravRing, gravSum, idx, gr)
                basis(DoubleArray(3) { gravSum[it] / count })
            }
        }
        nTrail++
        val row = rows[head]
        for (i in 0..2) {
            row[i] = (r[i][0] * w[0] + r[i][1] * w[1] + r[i][2] * w[2]).toFloat()
            row[3 + i] = (r[i][0] * a[0] + r[i][1] * a[1] + r[i][2] * a[2]).toFloat()
        }
        head = (head + 1) % window
        filled = minOf(filled + 1, window)
        lastRowTimeS = tS
        onRow?.invoke(tS, row)
    }

    private fun slide(ring: Array<DoubleArray>, sum: DoubleArray, idx: Int, v: DoubleArray) {
        if (nTrail >= TRAIL) for (i in 0..2) sum[i] -= ring[idx][i]
        for (i in 0..2) { ring[idx][i] = v[i]; sum[i] += v[i] }
    }

    // rows [east, north, down], exactly dhruva.align.estimate_tilt's convention
    private fun basis(d: DoubleArray): Array<DoubleArray> {
        val dn = max(norm(d), 1e-9)
        val down = DoubleArray(3) { d[it] / dn }
        val ref = if (abs(down[0]) > 0.9) doubleArrayOf(0.0, 1.0, 0.0) else doubleArrayOf(1.0, 0.0, 0.0)
        val east = cross(ref, down)
        val en = max(norm(east), 1e-9)
        for (i in 0..2) east[i] /= en
        return arrayOf(east, cross(down, east), down)
    }

    private fun norm(v: DoubleArray) = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
}
