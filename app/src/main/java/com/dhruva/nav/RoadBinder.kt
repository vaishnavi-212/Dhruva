package com.dhruva.nav

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.*

/**
 * Keeps the predicted dot ON THE ROAD.
 *
 * Without this the app tracks a free 2-D position and every small heading error
 * compounds: measured on a real 1.7 km ride with GNSS off, the dot ends 514 m
 * away in a field. With it the dot is a train on rails -- it cannot drift
 * sideways, and only how far ALONG the road it has gone can be wrong.
 *
 * WHAT A ROUTE IS. One stored path we have driven before -- not a map of every
 * road. The asset carries several. This class follows the chosen one and knows
 * nothing about side streets, so:
 *
 *  - If the rider leaves the route, the dot does NOT follow them. That is the
 *    10 Sept failure: the rider turned into a housing grid and the dot carried
 *    on down the stored road. Pick the route you will actually ride, and ride it.
 *  - At the END of the route the dot STOPS and `atRouteEnd` becomes true. Earlier
 *    route files were padded with 500 m of perfectly straight synthetic line at
 *    each end; past the real road the dot drove down that line through houses.
 *    The files no longer contain any padding.
 *
 * Choosing which road at a junction with no GNSS is map matching -- requirement
 * 3, the HMM -- and is not what this class does.
 */
class RoadBinder(routeJson: String) {

    private var xs = DoubleArray(0)      // metres east of the route's first point
    private var ys = DoubleArray(0)      // metres north
    private var cum = DoubleArray(0)     // distance along the route at each point
    private var lat0 = 0.0
    private var lon0 = 0.0
    private var mPerDegLon = 1.0

    private val routes: List<JSONObject>

    /** Names of every route in the asset, in file order. */
    val routeNames: List<String>

    /**
     * null = AUTO: bind to whichever route is nearest. Where stored routes share
     * a road and split later (every campus route does), auto is a guess. For a
     * demo, select the route you are about to ride.
     */
    var selectedIndex: Int? = null
        private set

    var routeName: String = "route"
        private set

    /** Distance along the route, metres from its first point. */
    var arcM: Double = 0.0
        private set

    /** +1 travelling in the route's stored order, -1 against it. */
    var direction: Int = 1
        private set

    /** Distance from the start fix to the route that was chosen (or the nearest one, if refused). */
    var snapDistanceM: Double = Double.NaN
        private set

    /**
     * False when no candidate route is within MAX_SNAP_M. A wrong road is far
     * worse than no road: when this is false, do NOT call advance().
     */
    var bound: Boolean = false
        private set

    /** Freeze the dot -- e.g. while stopped at the start line. */
    var paused: Boolean = false

    val lengthM: Double get() = if (cum.isEmpty()) 0.0 else cum[cum.size - 1]

    /** True once the dot has reached the end of the known road and is holding there. */
    val atRouteEnd: Boolean
        get() = bound && ((direction > 0 && arcM >= lengthM - END_EPS_M) ||
                          (direction < 0 && arcM <= END_EPS_M))

    init {
        val root = JSONObject(routeJson)
        routes = if (root.has("routes")) {
            val arr = root.getJSONArray("routes")
            (0 until arr.length()).map { arr.getJSONObject(it) }
        } else {
            listOf(root)
        }
        require(routes.isNotEmpty()) { "no routes in asset" }
        routeNames = routes.mapIndexed { i, r -> r.optString("name", "route ${i + 1}") }
        load(routes.first())
    }

    /** Select a route by index, or null for AUTO. Clears any existing binding. */
    fun select(index: Int?) {
        selectedIndex = index?.takeIf { it in routes.indices }
        bound = false
    }

    /** Cycle AUTO -> route 1 -> route 2 -> ... -> AUTO. Returns a label for the button. */
    fun cycleSelection(): String {
        val cur = selectedIndex
        selectedIndex = when {
            cur == null -> 0
            cur + 1 < routes.size -> cur + 1
            else -> null
        }
        bound = false
        return selectionLabel()
    }

    fun selectionLabel(): String = selectedIndex?.let { "Route: ${routeNames[it]}" } ?: "Route: auto"

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

    /** Nearest point on each segment: distance off the road, arc-length, unit tangent. */
    private class Cand(val d: Double, val s: Double, val tx: Double, val ty: Double)

    private fun candidates(lat: Double, lon: Double): List<Cand> {
        val px = (lon - lon0) * mPerDegLon
        val py = (lat - lat0) * M_PER_DEG
        val out = ArrayList<Cand>(xs.size)
        for (i in 0 until xs.size - 1) {
            val ax = xs[i]; val ay = ys[i]
            val bx = xs[i + 1] - ax; val by = ys[i + 1] - ay
            val len2 = bx * bx + by * by
            if (len2 < 1e-9) continue
            val u = (((px - ax) * bx + (py - ay) * by) / len2).coerceIn(0.0, 1.0)
            val d = hypot(px - (ax + u * bx), py - (ay + u * by))
            val len = sqrt(len2)
            out.add(Cand(d, cum[i] + u * len, bx / len, by / len))
        }
        return out
    }

    /**
     * Call ONCE, with the last good GPS fix before the signal dies.
     *
     * @param bearingDeg the fix's bearing (clockwise from north), or null when
     *        the phone has none or is barely moving. It decides which WAY along
     *        the route we are travelling -- without it a route ridden in reverse
     *        sends the dot backwards.
     *
     * Where the route passes the same spot twice -- a closed circuit starts and
     * ends at the same gate -- several points on it are equally near. The one
     * chosen is the one travelling our way with the most road still ahead, so
     * a circuit does not bind to its own finish line and stop dead.
     */
    fun start(lat: Double, lon: Double, bearingDeg: Float? = null): Boolean {
        val bE = bearingDeg?.let { sin(Math.toRadians(it.toDouble())) }
        val bN = bearingDeg?.let { cos(Math.toRadians(it.toDouble())) }

        var bestIdx = -1; var bestD = Double.MAX_VALUE; var bestS = 0.0
        var bestDir = 1; var bestAhead = -1.0
        var nearestAnyD = Double.MAX_VALUE

        val indices = selectedIndex?.let { listOf(it) } ?: routes.indices.toList()
        for (idx in indices) {
            load(routes[idx])
            val cands = candidates(lat, lon)
            if (cands.isEmpty()) continue
            val dMin = cands.minOf { it.d }
            nearestAnyD = min(nearestAnyD, dMin)
            if (dMin > MAX_SNAP_M) continue
            for (c in cands) {
                if (c.d > dMin + TIE_M) continue
                val dir = if (bE != null && bN != null && c.tx * bE + c.ty * bN < 0) -1 else 1
                val ahead = if (dir > 0) lengthM - c.s else c.s
                // across routes: nearer wins; within TIE_M of each other, more road ahead wins
                val better = when {
                    bestIdx < 0 -> true
                    c.d < bestD - TIE_M -> true
                    c.d <= bestD + TIE_M && ahead > bestAhead -> true
                    else -> false
                }
                if (better) {
                    bestIdx = idx; bestD = c.d; bestS = c.s; bestDir = dir; bestAhead = ahead
                }
            }
        }

        if (bestIdx < 0) {
            bound = false
            snapDistanceM = nearestAnyD
            return false
        }
        load(routes[bestIdx])
        arcM = bestS
        direction = bestDir
        snapDistanceM = bestD
        bound = true
        return true
    }

    /**
     * Call on every step while bound. Returns (lat, lon).
     *
     * speedMps: the speed held from the last GPS fix before the outage. Below
     * MIN_SPEED_MPS nothing moves, so a phone on a desk does not creep. At the
     * end of the route the dot holds and `atRouteEnd` goes true.
     *
     * Known limit: with speed held, a genuine stop during the outage is not
     * detected -- 11 m over 31 s of stops on the 7 Sept ride. Accelerometer
     * variance cannot fix it: a stopped engine idles at std 1.751, a moving one
     * at 1.757.
     */
    fun advance(speedMps: Double, dtS: Double): Pair<Double, Double> {
        if (bound && !paused && speedMps >= MIN_SPEED_MPS) {
            arcM = (arcM + direction * speedMps * dtS).coerceIn(0.0, lengthM)
        }
        return at(arcM)
    }

    /** Position at a given distance along the route, as (lat, lon). */
    fun at(s: Double): Pair<Double, Double> {
        val d = s.coerceIn(0.0, lengthM)
        var lo = 0; var hi = cum.size - 1
        while (lo < hi - 1) {
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
        private const val END_EPS_M = 2.0

        /** Candidates within this of the nearest are treated as equally near. */
        private const val TIE_M = 15.0

        /** Below this, treat the vehicle as stopped (GPS noise at a standstill: 0.10-0.25 m/s). */
        const val MIN_SPEED_MPS = 0.5

        /** No route within this means we are somewhere never ridden. Good rides sit within 50 m. */
        const val MAX_SNAP_M = 150.0

        fun fromAssets(ctx: Context): RoadBinder =
            RoadBinder(ctx.assets.open("routes.json").bufferedReader().use { it.readText() })
    }
}
