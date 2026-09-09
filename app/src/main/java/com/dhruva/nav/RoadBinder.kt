package com.dhruva.nav

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.*

/**
 * Keeps the predicted dot ON THE ROAD.
 *
 * Without this the app tracks a free 2-D position: it counts distance and turns,
 * and every small heading error compounds. Measured on a real 1.7 km ride with
 * GNSS off, the dot ends up 514 m away, in the middle of a field.
 *
 * With this it is a train on rails. We tell it which road it is on, so it cannot
 * drift sideways -- sideways stops existing. The only thing left to get wrong is
 * how far ALONG it has gone. Same ride, same constant speed: 166 m.
 *
 * Plain geometry, no model. Verified against the Python pipeline: both give
 * 166 m / 9.8% on the 5 Sept ride.
 *
 * WHAT A ROUTE IS -- and is not. A route is ONE PATH: the sequence of road
 * points for a journey we have driven before. It is NOT a map of every road.
 * That is why the asset carries SEVERAL routes and start() picks whichever one
 * we are actually on; ride somewhere none of them covers and it refuses rather
 * than snapping to a road on the other side of town. Choosing between roads at a
 * junction with no GNSS is map matching -- requirement 3, the HMM, and it is not
 * what this class does.
 */
class RoadBinder(routeJson: String) {

    private var xs = DoubleArray(0)      // metres east of the route's first point
    private var ys = DoubleArray(0)      // metres north
    private var cum = DoubleArray(0)     // distance along the route at each point
    private var lat0 = 0.0
    private var lon0 = 0.0
    private var mPerDegLon = 1.0

    private val alternates: List<JSONObject>

    /** Which route is currently loaded, for the status line. */
    var routeName: String = "route"
        private set

    /** How far along the road we currently are, in metres. */
    var arcM: Double = 0.0
        private set

    /** How far the start point was from the chosen route. */
    var snapDistanceM: Double = Double.NaN
        private set

    /**
     * False when no route in the asset matches where we are.
     *
     * On 7 Sept a ride was recorded 3.2 km from the route the app carried.
     * start() obediently snapped to the nearest point on it and slid the dot
     * along a road on the other side of town; the screen read "2000 m apart" the
     * whole way. A wrong road is far worse than no road.
     *
     * When this is false, do NOT call advance(). Fall back to the free position.
     */
    var bound: Boolean = false
        private set

    /** Freeze the dot -- e.g. while stopped at the start line. */
    var paused: Boolean = false

    val lengthM: Double get() = if (cum.isEmpty()) 0.0 else cum[cum.size - 1]

    init {
        val root = JSONObject(routeJson)
        // Two shapes accepted:  {"points":[...]}  or  {"routes":[{name,points},...]}
        alternates = if (root.has("routes")) {
            val arr = root.getJSONArray("routes")
            (0 until arr.length()).map { arr.getJSONObject(it) }
        } else {
            listOf(root)
        }
        load(alternates.first())
    }

    private fun load(route: JSONObject) {
        val pts: JSONArray = route.getJSONArray("points")
        val n = pts.length()
        require(n >= 2) { "route needs at least 2 points" }
        routeName = route.optString("name", "route")

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

    /** Nearest arc-length on the currently loaded route, and how far off it we are. */
    private fun project(lat: Double, lon: Double): Pair<Double, Double> {
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
            val d = hypot(px - (ax + u * bx), py - (ay + u * by))
            if (d < bestD) { bestD = d; bestS = cum[i] + u * sqrt(len2) }
        }
        return Pair(bestD, bestS)
    }

    /**
     * Call ONCE, with the last good GPS fix before the signal dies.
     *
     * Tries every route in the asset and keeps the nearest. Returns false, and
     * leaves `bound` false, if none of them is within MAX_SNAP_M.
     */
    fun start(lat: Double, lon: Double): Boolean {
        var bestRoute = 0
        var bestD = Double.MAX_VALUE
        var bestS = 0.0
        for ((idx, r) in alternates.withIndex()) {
            load(r)
            val (d, s) = project(lat, lon)
            if (d < bestD) { bestD = d; bestS = s; bestRoute = idx }
        }
        load(alternates[bestRoute])
        snapDistanceM = bestD
        arcM = bestS
        bound = bestD <= MAX_SNAP_M
        return bound
    }

    /**
     * Call on every step. Advances along the road and returns (lat, lon).
     *
     * WHAT TO PASS as speedMps:
     *   while GNSS is alive : the LIVE GPS speed, every fix -- stops then
     *                         register on their own and the dot holds still.
     *   after GNSS dies     : the last speed seen before it died, held.
     *
     * Anything below MIN_SPEED_MPS counts as stopped. Without that, a phone on a
     * desk creeps forward forever on GPS speed noise.
     *
     * Known limit: during a blackout, with the speed frozen, a genuine stop
     * cannot be detected. On the 7 Sept ride that injected 11 m over 31 s of
     * stops on a 1467 m ride -- under 1%. Do NOT try to fix it with accelerometer
     * variance: measured, a stopped engine idles at std 1.751 and a moving one
     * at 1.757.
     */
    fun advance(speedMps: Double, dtS: Double): Pair<Double, Double> {
        if (speedMps >= MIN_SPEED_MPS && !paused) {
            arcM = (arcM + speedMps * dtS).coerceIn(0.0, lengthM)
        }
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

        /** Below this, treat the vehicle as stopped. GPS speed noise at a
         *  standstill measured 0.10-0.25 m/s on our own runs. */
        const val MIN_SPEED_MPS = 0.5

        /** No route within this distance means we are somewhere we have never
         *  ridden. Our good rides sit within 50 m of their own route. */
        const val MAX_SNAP_M = 150.0

        /** Prefers routes.json (several routes); falls back to route.json. */
        fun fromAssets(ctx: Context): RoadBinder {
            val name = ctx.assets.list("")?.firstOrNull { it == "routes.json" } ?: "route.json"
            return RoadBinder(ctx.assets.open(name).bufferedReader().use { it.readText() })
        }
    }
}
