package com.dhruva.nav

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * GNSS+INS fusion filter: a line-by-line port of engine/dhruva/fusion.py
 * (GnssInsFusion), checked
 * against it by FusionEngineGoldenTest. Runs 10 times a second.
 *
 * State (5 numbers): x, y (m, east/north from the map origin), v (m/s), psi
 * (heading, rad, 0 = east,
 * counter-clockwise), b (gyro bias, rad/s). P is its 5 x 5 uncertainty,
 * stored row by row.
 *
 * Every epoch: predict with the gyro turn rate -> then each measurement
 * that is available:
 * GNSS position (+ speed) when GPS is usable
 * road heading + road position (cross-track only) when bound to a road
 * AI speed from the model during a blackout
 * During a blackout there is no branch and no reset: the GNSS update is
 * simply skipped and the
 * uncertainty grows on its own. That is the "seamless" hand-over, in both
 * directions.
 */
class FusionEngine(private val cfg: Config = Config()) {

    data class Config(
        val qAccel: Double = 1.2, // m/s^2, unmodelled acceleration
        val qYawRate: Double = 0.35, // rad/s, gyro heading noise
        val qBias: Double = 1e-5, // rad/s, bias random walk
        val rGnssPos: Double = 4.0, // m
        val rGnssSpeed: Double = 1.0, // m/s
        val rModelSpeed: Double = 3.5, // m/s, the AI speed's uncertainty
        val roadHeadingSigma: Double = 0.05, // rad
        val roadCrossSigma: Double = 2.5 // m, half a lane
    )

    enum class Mode { INIT, GNSS_AIDED, DEAD_RECKONING }

    val x = DoubleArray(5)
    val p = DoubleArray(25)

    var mode = Mode.INIT
        private set

    var epochsSinceGnss = 0
        private set

    val posX get() = x[0]
    val posY get() = x[1]
    val speed get() = x[2]
    val heading get() = x[3]
    val bias get() = x[4]

    /** 1-sigma position uncertainty, metres. */
    val posSigma
        get() = sqrt(max(p[0] + p[6], 0.0) / 2.0)

    fun initialise(
        px: Double,
        py: Double,
        v: Double,
        psi: Double
    ) {
        x[0] = px
        x[1] = py
        x[2] = v
        x[3] = psi
        x[4] = 0.0

        p.fill(0.0)

        p[0] = 16.0
        p[6] = 16.0
        p[12] = 1.0
        p[18] = 0.05
        p[24] = 1e-4

        mode = Mode.GNSS_AIDED
        epochsSinceGnss = 0
    }

    /** Move the state forward by [dt] seconds using the gyro turn rate
     * (rad/s about vertical). */
    fun predict(
        gyroZ: Double,
        accelFwd: Double,
        dt: Double
    ) {
        val px = x[0]
        val py = x[1]
        val v = x[2]
        val psi = x[3]
        val b = x[4]

        val w = gyroZ - b
        val c = cos(psi)
        val s = sin(psi)

        x[0] = px + v * c * dt
        x[1] = py + v * s * dt
        x[2] = max(v + accelFwd * dt, 0.0)
        x[3] = psi + w * dt

        val f = identity()

        f[0 * 5 + 2] = c * dt
        f[0 * 5 + 3] = -v * s * dt
        f[1 * 5 + 2] = s * dt
        f[1 * 5 + 3] = v * c * dt
        f[3 * 5 + 4] = -dt

        val q = DoubleArray(25)

        q[2 * 5 + 2] = sq(cfg.qAccel * dt)
        q[3 * 5 + 3] = sq(cfg.qYawRate * dt)
        q[4 * 5 + 4] = sq(cfg.qBias * dt)

        q[0] = sq(0.5 * cfg.qAccel * dt * dt)
        q[6] = q[0]

        val fp = mul(f, 5, 5, p, 5)
        val fpft = mul(
            fp,
            5,
            5,
            transpose(f, 5, 5),
            5
        )

        for (i in 0 until 25)
            p[i] = fpft[i] + q[i]
    }

    fun updateGnssPosition(
        gx: Double,
        gy: Double,
        sigma: Double = cfg.rGnssPos
    ) {
        val h = DoubleArray(10)
        h[0] = 1.0
        h[5 + 1] = 1.0

        val r = doubleArrayOf(
            gx - x[0],
            gy - x[1]
        )

        val rr = doubleArrayOf(
            sq(sigma),
            0.0,
            0.0,
            sq(sigma)
        )

        update(h, 2, r, rr)

        mode = Mode.GNSS_AIDED
        epochsSinceGnss = 0
    }

    fun updateGnssSpeed(v: Double) =
        updateSpeed(v, cfg.rGnssSpeed)

    /** The AI element: the model's speed, with its own uncertainty. */
    fun updateModelSpeed(
        v: Double,
        sigma: Double = cfg.rModelSpeed
    ) =
        updateSpeed(v, sigma)

    fun updateRoadHeading(roadHeading: Double) {
        val h = DoubleArray(5)
        h[3] = 1.0

        val d = roadHeading - x[3]

        update(
            h,
            1,
            doubleArrayOf(
                atan2(
                    sin(d),
                    cos(d)
                )
            ),
            doubleArrayOf(
                sq(cfg.roadHeadingSigma)
            )
        )
    }

    /** Bind to the road sideways only: along-track position is exactly what
     * we don't know in a blackout. */
    fun updateRoadPosition(
        roadX: Double,
        roadY: Double,
        roadHeading: Double
    ) {
        val nx = -sin(roadHeading)
        val ny = cos(roadHeading)

        val h = DoubleArray(5)
        h[0] = nx
        h[1] = ny

        val resid =
            nx * (roadX - x[0]) +
                    ny * (roadY - x[1])

        update(
            h,
            1,
            doubleArrayOf(resid),
            doubleArrayOf(
                sq(cfg.roadCrossSigma)
            )
        )
    }

    /**
     * One epoch, exactly fusion.py's step(). Pass null for anything not
     * available this epoch.
     * Order matters and matches Python: predict, GNSS (or not), road
     * heading, road position, AI speed.
     */
    fun step(
        dt: Double,
        gyroZ: Double,
        accelFwd: Double = 0.0,
        gnssX: Double? = null,
        gnssY: Double? = null,
        gnssSpeed: Double? = null,
        gnssSigma: Double? = null,
        modelSpeed: Double? = null,
        modelSigma: Double? = null,
        roadHeading: Double? = null,
        roadX: Double? = null,
        roadY: Double? = null
    ) {
        predict(
            gyroZ,
            accelFwd,
            dt
        )

        if (gnssX != null && gnssY != null) {
            updateGnssPosition(
                gnssX,
                gnssY,
                gnssSigma ?: cfg.rGnssPos
            )

            if (gnssSpeed != null)
                updateGnssSpeed(gnssSpeed)
        } else {
            epochsSinceGnss++
            mode = Mode.DEAD_RECKONING
        }

        if (roadHeading != null)
            updateRoadHeading(roadHeading)

        if (roadX != null && roadY != null)
            updateRoadPosition(
                roadX,
                roadY,
                roadHeading ?: x[3]
            )

        if (modelSpeed != null)
            updateModelSpeed(
                modelSpeed,
                modelSigma ?: cfg.rModelSpeed
            )
    }

    // ---------- the Kalman update, for 1 or 2 measurements (Joseph form, as
    // in Python) ----------

    private fun updateSpeed(
        v: Double,
        sigma: Double
    ) {
        val h = DoubleArray(5)
        h[2] = 1.0

        update(
            h,
            1,
            doubleArrayOf(v - x[2]),
            doubleArrayOf(sq(sigma))
        )
    }

    /** h: m x 5, resid: m, rr: m x m, all row by row. m is 1 or 2. */
    private fun update(
        h: DoubleArray,
        m: Int,
        resid: DoubleArray,
        rr: DoubleArray
    ) {
        val ht = transpose(h, m, 5)
        val pht = mul(p, 5, 5, ht, m)
        val s = mul(h, m, 5, pht, m)

        for (i in 0 until m * m)
            s[i] += rr[i]

        val sInv = invert(s, m)
        val k = mul(pht, 5, m, sInv, m)

        val kr = mul(
            k,
            5,
            m,
            resid,
            1
        )

        for (i in 0 until 5)
            x[i] += kr[i]

        val a = identity()

        val kh = mul(
            k,
            5,
            m,
            h,
            5
        )

        for (i in 0 until 25)
            a[i] -= kh[i]

        val apat = mul(
            mul(
                a,
                5,
                5,
                p,
                5
            ),
            5,
            5,
            transpose(a, 5, 5),
            5
        )

        val krkt = mul(
            mul(
                k,
                5,
                m,
                rr,
                m
            ),
            5,
            m,
            transpose(k, 5, m),
            5
        )

        for (i in 0 until 25)
            p[i] = apat[i] + krkt[i]

        x[3] = atan2(
            sin(x[3]),
            cos(x[3])
        ) // keep heading in -pi..pi
    }

    companion object {

        private fun sq(v: Double) = v * v

        private fun identity() =
            DoubleArray(25).also {
                for (i in 0 until 5)
                    it[i * 5 + i] = 1.0
            }

        /** (r x n) times (n x c). */
        private fun mul(
            a: DoubleArray,
            r: Int,
            n: Int,
            b: DoubleArray,
            c: Int
        ): DoubleArray {
            val out = DoubleArray(r * c)

            for (i in 0 until r)
                for (j in 0 until c) {
                    var acc = 0.0

                    for (k in 0 until n)
                        acc +=
                            a[i * n + k] *
                                    b[k * c + j]

                    out[i * c + j] = acc
                }

            return out
        }

        private fun transpose(
            a: DoubleArray,
            r: Int,
            c: Int
        ): DoubleArray {
            val out = DoubleArray(r * c)

            for (i in 0 until r)
                for (j in 0 until c)
                    out[j * r + i] =
                        a[i * c + j]

            return out
        }

        private fun invert(
            s: DoubleArray,
            m: Int
        ): DoubleArray =
            if (m == 1) {
                doubleArrayOf(
                    1.0 / s[0]
                )
            } else {
                val det =
                    s[0] * s[3] -
                            s[1] * s[2]

                doubleArrayOf(
                    s[3] / det,
                    -s[1] / det,
                    -s[2] / det,
                    s[0] / det
                )
            }
    }
}

/**
 * What the user sees: a dot that glides to the filter's estimate instead of
 * teleporting when GPS
 * returns after a long blackout. Port of fusion.py's SeamlessOutput. The
 * filter stays optimal inside;
 * only the displayed position is smoothed, at most [maxRateMps] and with
 * time constant [tauS].
 */
class SeamlessDot(
    private val tauS: Double = 1.8,
    private val maxRateMps: Double = 25.0,
    private val snapBelowM: Double = 0.5
) {
    var outX = Double.NaN
        private set

    var outY = Double.NaN
        private set

    var slewing = false
        private set

    /** Largest gap between the shown dot and the filter when a glide
     * started, metres. */
    var lastGlideStartGapM = 0.0
        private set

    fun update(
        fx: Double,
        fy: Double,
        dt: Double
    ) {
        if (outX.isNaN()) {
            outX = fx
            outY = fy
            return
        }

        val gx = fx - outX
        val gy = fy - outY
        val d = hypot(gx, gy)

        if (d <= snapBelowM) {
            outX = fx
            outY = fy
            slewing = false
            return
        }

        val alpha = 1.0 - exp(-dt / tauS)

        var sx = gx * alpha
        var sy = gy * alpha

        val maxStep = maxRateMps * dt
        val n = hypot(sx, sy)

        if (n > maxStep) {
            sx = sx / n * maxStep
            sy = sy / n * maxStep
        }

        if (!slewing && d > 2.0) {
            slewing = true
            lastGlideStartGapM = d
        }

        outX += sx
        outY += sy
    }
}