package com.dhruva.nav

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot

/**
 * Turn-by-turn guidance along a planned route: where we are on it, the next turn, arrival, and
 * whether we have left it.
 *
 * Progress along the route comes from GPS while it works, and from the dot riding the route
 * (RoadBinder.arcM) during a blackout -- the same numbers either way, so the voice keeps guiding
 * with GPS off. Instructions use VoiceGuide.phrase, so the wording loosens as the confidence
 * circle grows ("turn left in about 100 metres").
 */

/** A route as a polyline in local metres, with distance along it. */
class RouteGuide(points: List<Pair<Double, Double>>) {
    private val lat0 = points.first().first
    private val lon0 = points.first().second
    private val kx = M_PER_DEG * cos(Math.toRadians(lat0))
    val xs = DoubleArray(points.size) { (points[it].second - lon0) * kx }
    val ys = DoubleArray(points.size) { (points[it].first - lat0) * M_PER_DEG }
    val cum = DoubleArray(points.size).also { c -> for (i in 1 until points.size) c[i] = c[i - 1] + hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) }
    val lengthM: Double get() = cum.last()

    class Proj(val s: Double, val offM: Double)

    fun toLocal(lat: Double, lon: Double): Pair<Double, Double> = ((lon - lon0) * kx) to ((lat - lat0) * M_PER_DEG)
    fun toLatLon(x: Double, y: Double): Pair<Double, Double> = (lat0 + y / M_PER_DEG) to (lon0 + x / kx)

    /** The route from arc [from] to its end, in local metres: what the route guard watches after a cut. */
    fun ahead(from: Double): Pair<DoubleArray, DoubleArray> {
        val xs2 = ArrayList<Double>(); val ys2 = ArrayList<Double>()
        val (x0, y0) = xy(from); xs2.add(x0); ys2.add(y0)
        for (i in cum.indices) if (cum[i] > from) { xs2.add(xs[i]); ys2.add(ys[i]) }
        return xs2.toDoubleArray() to ys2.toDoubleArray()
    }

    /**
     * Nearest point on the route to (lat, lon). With [nearS], only the stretch from 60 m behind to
     * 250 m ahead of it is searched, so a route that passes the same road twice does not jump.
     */
    fun project(lat: Double, lon: Double, nearS: Double? = null): Proj {
        val px = (lon - lon0) * kx; val py = (lat - lat0) * M_PER_DEG
        var best = Proj(0.0, Double.MAX_VALUE)
        for (i in 0 until xs.size - 1) {
            if (nearS != null && (cum[i + 1] < nearS - BEHIND_M || cum[i] > nearS + AHEAD_M)) continue
            val ax = xs[i]; val ay = ys[i]; val bx = xs[i + 1] - ax; val by = ys[i + 1] - ay
            val l2 = bx * bx + by * by
            val u = if (l2 < 1e-9) 0.0 else (((px - ax) * bx + (py - ay) * by) / l2).coerceIn(0.0, 1.0)
            val d = hypot(px - (ax + u * bx), py - (ay + u * by))
            if (d < best.offM) best = Proj(cum[i] + u * (cum[i + 1] - cum[i]), d)
        }
        return best
    }

    /** Direction of travel (radians, counter-clockwise from east) between two distances along the route. */
    fun heading(s1: Double, s2: Double): Double {
        val (x1, y1) = xy(s1); val (x2, y2) = xy(s2)
        return atan2(y2 - y1, x2 - x1)
    }

    private fun xy(s: Double): Pair<Double, Double> {
        val d = s.coerceIn(0.0, lengthM)
        var i = cum.indexOfLast { it <= d }.coerceIn(0, cum.size - 2)
        while (i < cum.size - 2 && cum[i + 1] - cum[i] < 1e-9) i++
        val seg = cum[i + 1] - cum[i]
        val u = if (seg < 1e-9) 0.0 else (d - cum[i]) / seg
        return (xs[i] + u * (xs[i + 1] - xs[i])) to (ys[i] + u * (ys[i + 1] - ys[i]))
    }

    companion object {
        const val M_PER_DEG = 111320.0
        const val BEHIND_M = 60.0
        const val AHEAD_M = 250.0
    }
}

/** A turn at a junction, [s] metres along the route. [angleDeg] > 0 is to the left. */
class Maneuver(val s: Double, val angleDeg: Double) {
    val instruction: String get() {
        val side = if (angleDeg > 0) "left" else "right"
        val a = abs(angleDeg)
        return when {
            a >= 165 -> "Make a U-turn"
            a >= 140 -> "Turn sharp $side"
            a >= 50 -> "Turn $side"
            else -> "Keep $side"
        }
    }
    val arrow: String get() = when {
        abs(angleDeg) >= 165 -> "⤺"
        angleDeg > 0 -> if (abs(angleDeg) >= 50) "↰" else "↖"
        else -> if (abs(angleDeg) >= 50) "↱" else "↗"
    }

    companion object {
        const val MIN_TURN_DEG = 30.0     // less than this at a junction is "carry on straight"
        const val LOOK_M = 20.0           // compare the road 20 m before and after the junction
        const val MERGE_M = 25.0          // two junctions this close (a divided road) are one turn

        /**
         * Turns along a route: only at JUNCTIONS (a road point shared by 3+ road directions), so a
         * bend in a road with no side road is never announced.
         * [nodes] are the route's road nodes, points[1..] (points[0] is where we started).
         */
        fun fromRoute(guide: RouteGuide, nodes: IntArray, degree: (Int) -> Int): List<Maneuver> {
            val raw = ArrayList<Maneuver>()
            for (k in 0 until nodes.size - 1) {
                val i = k + 1                                        // index in the polyline
                if (i >= guide.cum.size - 1 || degree(nodes[k]) < 3) continue
                val s = guide.cum[i]
                if (s < LOOK_M / 2 || s > guide.lengthM - LOOK_M / 2) continue
                val hin = guide.heading(s - LOOK_M, s)
                val hout = guide.heading(s, s + LOOK_M)
                val d = Math.toDegrees(wrap(hout - hin))
                if (abs(d) >= MIN_TURN_DEG) raw.add(Maneuver(s, d))
            }
            val out = ArrayList<Maneuver>()
            for (m in raw) {
                val last = out.lastOrNull()
                if (last != null && m.s - last.s < MERGE_M) out[out.size - 1] = Maneuver(last.s, Math.toDegrees(wrap(Math.toRadians(last.angleDeg + m.angleDeg))))
                else out.add(m)
            }
            return out.filter { abs(it.angleDeg) >= MIN_TURN_DEG }
        }

        fun wrap(a: Double): Double {
            var x = a
            while (x > Math.PI) x -= 2 * Math.PI
            while (x < -Math.PI) x += 2 * Math.PI
            return x
        }
    }
}

/**
 * Says the right thing at the right distance. Call [update] about once a second with the distance
 * along the route; it returns what to show and, at most once per event, what to say.
 */
class Navigator(val placeName: String, val guide: RouteGuide, val maneuvers: List<Maneuver>) {

    class Update(val status: String, val speak: String?, val arrived: Boolean)

    private val saidFar = HashSet<Int>()
    private val saidNow = HashSet<Int>()
    var arrived = false
        private set

    /** [radius90M] = 0 with GPS; the confidence circle during a blackout. */
    fun update(s: Double, radius90M: Double, phrase: (String, Double, Double) -> String): Update {
        val toGo = (guide.lengthM - s).coerceAtLeast(0.0)
        if (arrived || toGo <= ARRIVE_M) {
            val first = !arrived
            arrived = true
            return Update("Arrived · $placeName", if (first) "You have arrived at $placeName." else null, true)
        }
        val k = maneuvers.indexOfFirst { it.s > s + PASSED_M }
        var speak: String? = null
        val status: String
        if (k >= 0) {
            val m = maneuvers[k]; val d = m.s - s
            if (d <= NOW_M && saidNow.add(k)) { saidFar.add(k); speak = m.instruction + " now." }
            else if (d <= FAR_M && saidFar.add(k)) speak = phrase(m.instruction, d, radius90M)
            status = "%s %s in %s · %s to go".format(m.arrow, m.instruction, metres(d), metres(toGo))
        } else {
            if (toGo <= FAR_M && saidFar.add(-1)) speak = phrase("Your destination is ahead", toGo, radius90M)
            status = "Straight on · %s to go".format(metres(toGo))
        }
        return Update(status, speak, false)
    }

    companion object {
        const val ARRIVE_M = 30.0
        const val FAR_M = 150.0
        const val NOW_M = 30.0
        const val PASSED_M = 5.0
        fun metres(m: Double) = if (m < 1000) "%.0f m".format(Math.round(m / 10.0) * 10.0) else "%.1f km".format(m / 1000)
    }
}

/**
 * Off the route with GPS: 2 fixes in a row more than 40 m off (and further off than the fix's own
 * accuracy) -- but only once the rider has JOINED the route (been within 25 m of it). A route starts
 * at the road nearest the rider, who may be standing in a building 60 m from it.
 */
class OffRouteDetector {
    private var count = 0
    var joined = false
        private set

    fun onFix(offM: Double, accuracyM: Float): Boolean {
        if (offM <= JOIN_M) joined = true
        if (!joined) return false
        count = if (offM > OFF_M && offM > accuracyM) count + 1 else 0
        return count >= FIXES
    }
    fun reset() { count = 0; joined = false }

    companion object {
        const val OFF_M = 40.0
        const val JOIN_M = 25.0
        const val FIXES = 2
    }
}
