package com.dhruva.nav

import android.content.Context
import org.json.JSONObject
import kotlin.math.*

/**
 * Keeps the predicted dot ON THE ROAD.
 *
 * Without this the app tracks a free 2-D position: it counts distance and turns,
 * and every small heading error compounds. Measured on a real 1.7 km ride, the
 * dot ends up 514 m away -- in the middle of a field.
 *
 * With this it is a train on rails. We tell it which road it is on, so it cannot
 * drift sideways at all. The only thing left to get wrong is how far along it has
 * gone. Same ride, same dumb constant speed: 166 m, and always on the road.
 *
 * This is plain geometry -- no model, no training. Verified against the Python
 * pipeline: both give 166 m / 9.8% on the 5 Sept ride.
 */
class RoadBinder(routeJson: String) {

    private val xs: DoubleArray          // metres east of the route's first point
    private val ys: DoubleArray          // metres north
    private val cum: DoubleArray         // distance along the route at each point
    private val lat0: Double
    private val lon0: Double
    private val mPerDegLon: Double

    /** How far along the road we currently are, in metres. */
    var arcM: Double = 0.0
        private set

    val lengthM: Double get() = cum[cum.size - 1]

    init {
        val pts = JSONObject(routeJson).getJSONArray("points")
        val n = pts.length()
        require(n >= 2) { "route needs at least 2 points" }

        lat0 = pts.getJSONArray(0).getDouble(0)
        lon0 = pts.getJSONArray(0).getDouble(1)
        mPerDegLon = M_PER_DEG * cos(Math.toRadians(lat0))

        xs = DoubleArray(n); ys = DoubleArray(n); cum = DoubleArray(n)
        for (i in 0 until n) {
            val p = pts.getJSONArray(i)
            xs[i] = (p.getDouble(1) - lon0) * mPerDegLon
            ys[i] = (p.getDouble(0) - lat0) * M_PER_DEG
            if (i > 0) cum[i] = cum[i - 1] + hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        }
    }

    /**
     * Call ONCE, with the last good GPS fix before the signal died.
     * Finds where on the road that is. Everything after is just walking along it.
     */
    fun start(lat: Double, lon: Double) {
        val px = (lon - lon0) * mPerDegLon
        val py = (lat - lat0) * M_PER_DEG
        var bestD = Double.MAX_VALUE
        var bestS = 0.0
        for (i in 0 until xs.size - 1) {
            val ax = xs[i]; val ay = ys[i]
            val bx = xs[i + 1] - ax; val by = ys[i + 1] - ay
            val len2 = bx * bx + by * by
            val u = if (len2 == 0.0) 0.0
                    else ((px - ax) * bx + (py - ay) * by).div(len2).coerceIn(0.0, 1.0)
            val qx = ax + u * bx; val qy = ay + u * by
            val d = hypot(px - qx, py - qy)
            if (d < bestD) { bestD = d; bestS = cum[i] + u * sqrt(len2) }
        }
        arcM = bestS
    }

    /**
     * Call on every step while GNSS is gone. Advances along the road and returns
     * the new position as (lat, lon).
     *
     * @param speedMps  current speed estimate (the last GPS speed is fine)
     * @param dtS       seconds since the previous call
     */
    fun advance(speedMps: Double, dtS: Double): Pair<Double, Double> {
        arcM = (arcM + max(speedMps, 0.0) * dtS).coerceIn(0.0, lengthM)
        return at(arcM)
    }

    /** Position at a given distance along the road, as (lat, lon). */
    fun at(s: Double): Pair<Double, Double> {
        val d = s.coerceIn(0.0, lengthM)
        var lo = 0; var hi = cum.size - 1
        while (lo < hi - 1) {                       // binary search, cheap at 100 Hz
            val mid = (lo + hi) / 2
            if (cum[mid] <= d) lo = mid else hi = mid
        }
        val segLen = cum[lo + 1] - cum[lo]
        val u = if (segLen < 1e-9) 0.0 else (d - cum[lo]) / segLen
        val x = xs[lo] + u * (xs[lo + 1] - xs[lo])
        val y = ys[lo] + u * (ys[lo + 1] - ys[lo])
        return Pair(lat0 + y / M_PER_DEG, lon0 + x / mPerDegLon)
    }

    companion object {
        private const val M_PER_DEG = 111320.0

        /** Load route.json from app/src/main/assets/ */
        fun fromAssets(ctx: Context, name: String = "route.json") =
            RoadBinder(ctx.assets.open(name).bufferedReader().use { it.readText() })
    }
}
