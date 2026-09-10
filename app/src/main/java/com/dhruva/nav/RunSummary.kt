package com.dhruva.nav

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sqrt

/**
 * The post-run summary the phone can produce ON ITS OWN, with no laptop and no
 * server.
 *
 * The key realisation: the phone already holds BOTH tracks. It logged every GPS
 * fix, and its own DeadReckoner produced a position at every step. Drift is just
 * the distance between the two at the end, divided by how far the vehicle
 * actually travelled. No pipeline needed.
 *
 * The formula matches harness/metrics.py exactly:
 *
 *     drift %  =  final position error  /  cumulative TRUE path length  x 100
 *
 * Denominator is path length walked along the truth track, NOT straight-line
 * distance from the start. The team already fixed this once in the dashboard;
 * it is repeated here so the two never disagree.
 */
object RunSummary {

    private const val R = 6378137.0
    private const val ISRO_LIMIT_PCT = 10.0
    /** measured on 11 rides: 1-sigma is ~19% of distance since the last fix */
    private const val SIGMA_PER_METRE = 0.19

    data class Result(
        val distanceM: Double,
        val durationS: Double,
        val finalErrorM: Double,
        val driftPct: Double,
        val passes: Boolean,
        val meanSpeedMps: Double,
        val confidence90M: Double,
        val gpsFixes: Int,
        val imuHz: Double
    )

    private fun metres(aLat: Double, aLon: Double, bLat: Double, bLon: Double): Double {
        val x = Math.toRadians(bLon - aLon) * cos(Math.toRadians(aLat)) * R
        val y = Math.toRadians(bLat - aLat) * R
        return sqrt(x * x + y * y)
    }

    /**
     * @param truth    GPS fixes in order, (lat, lon)
     * @param pred     dead-reckoned positions in order, (lat, lon)
     * @param durationS wall-clock length of the ride
     * @param imuSamples how many IMU rows were written (for the Hz readout)
     * @param blackoutFromIndex index into `truth` at which GNSS was cut. Distance
     *        is counted from HERE, not from the start of the ride.
     *
     * Why that index matters: ISRO's metric is drift over the GNSS-DENIED
     * distance. Counting the whole ride puts the GPS-healthy leg into the
     * denominator, which silently divides the drift down -- on the 9 Sept run
     * that leg was twice as long as the blackout, so the figure shown would have
     * been about a third of the truth. A number that flatters us by accident is
     * the one that gets found in Q&A.
     */
    fun compute(
        truth: List<Pair<Double, Double>>,
        pred: List<Pair<Double, Double>>,
        durationS: Double,
        imuSamples: Int,
        blackoutFromIndex: Int = 0,
        blackoutDurationS: Double = durationS
    ): Result {
        if (truth.size < 2 || pred.isEmpty()) {
            return Result(0.0, durationS, 0.0, 0.0, false, 0.0, 0.0, truth.size, 0.0)
        }
        // cumulative TRUE path length SINCE GNSS WAS CUT -- the denominator that matters
        var dist = 0.0
        val from = blackoutFromIndex.coerceIn(0, truth.size - 1).coerceAtLeast(1)
        for (i in from until truth.size) {
            val step = metres(truth[i - 1].first, truth[i - 1].second,
                              truth[i].first, truth[i].second)
            // ignore fixes implying an impossible speed (matches dhruva/gpsclean.py)
            if (step < 15.0 * (durationS / truth.size).coerceAtLeast(0.2)) dist += step
        }
        val t = truth.last(); val p = pred.last()
        val err = metres(t.first, t.second, p.first, p.second)
        val drift = if (dist > 1.0) err / dist * 100.0 else 0.0
        return Result(
            distanceM = dist,
            durationS = durationS,
            finalErrorM = err,
            driftPct = drift,
            passes = dist > 1.0 && drift < ISRO_LIMIT_PCT,
            // Distance and time must describe the SAME stretch. Denied distance over
            // whole-session time printed 7.9 km/h on 10 Sept for a blackout ridden at
            // about 22 km/h.
            meanSpeedMps = if (blackoutDurationS > 0) dist / blackoutDurationS else 0.0,
            // the honest 90% circle after travelling this far without a fix
            confidence90M = 2.146 * (SIGMA_PER_METRE * dist).coerceAtLeast(2.0),
            gpsFixes = truth.size,
            imuHz = if (durationS > 0) imuSamples / durationS else 0.0
        )
    }

    /** One-line verdict for the top of the summary card. */
    fun verdict(r: Result): String =
        if (r.passes) "PASS — %.1f%% drift, inside ISRO's 10%% limit".format(r.driftPct)
        else "FAIL — %.1f%% drift, over ISRO's 10%% limit".format(r.driftPct)
}
