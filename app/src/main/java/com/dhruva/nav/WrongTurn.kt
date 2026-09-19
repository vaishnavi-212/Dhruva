package com.dhruva.nav

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * A wrong turn with GPS OFF: notice it from the gyro, work out which road was taken, re-route.
 *
 * Ports of engine/scripts/eval_route_guard.py (v2) and eval_reroute.py, measured on recorded rides
 * (RESOURCES 8ap, 8aq), mounted phone: 14/14 wrong turns caught, median 19 m after leaving the route,
 * 9/10 correct rides quiet; the road actually taken chosen 10/12 (every junction with one other road
 * right). RouteGuardGoldenTest checks this port alarms at the same sample as Python on a real ride.
 */

private fun wrapRad(x: Double) = atan2(sin(x), cos(x))
private fun wrapDeg(x: Double) = Math.toDegrees(wrapRad(Math.toRadians(x)))

/** A polyline resampled every metre with unwrapped heading: mapmatch.RoadPolyline(xy, 1.0). */
class Poly1m(xIn: DoubleArray, yIn: DoubleArray) {
    val s: DoubleArray; val x: DoubleArray; val y: DoubleArray; val heading: DoubleArray
    val length: Double

    init {
        val sIn = DoubleArray(xIn.size)
        for (i in 1 until xIn.size) sIn[i] = sIn[i - 1] + hypot(xIn[i] - xIn[i - 1], yIn[i] - yIn[i - 1])
        val keep = (xIn.indices).filter { it == 0 || sIn[it] - sIn[it - 1] > 1e-6 }
        val ks = DoubleArray(keep.size) { sIn[keep[it]] }
        val kx = DoubleArray(keep.size) { xIn[keep[it]] }
        val ky = DoubleArray(keep.size) { yIn[keep[it]] }
        val n = maxOf(1, ceil(ks.last()).toInt())
        s = DoubleArray(n) { it.toDouble() }
        x = DoubleArray(n) { interp(s[it], ks, kx) }
        y = DoubleArray(n) { interp(s[it], ks, ky) }
        val raw = DoubleArray(n) { i ->
            if (n == 1) 0.0 else {
                val (dx, dy) = when (i) {
                    0 -> (x[1] - x[0]) to (y[1] - y[0])
                    n - 1 -> (x[n - 1] - x[n - 2]) to (y[n - 1] - y[n - 2])
                    else -> ((x[i + 1] - x[i - 1]) / 2) to ((y[i + 1] - y[i - 1]) / 2)
                }
                atan2(dy, dx)
            }
        }
        heading = unwrap(raw)
        length = s.last()
    }

    fun xyAt(a: Double): Pair<Double, Double> { val c = a.coerceIn(0.0, length); return interp(c, s, x) to interp(c, s, y) }

    /** eval_reroute.heading_at: degrees, from points span/2 either side. */
    fun headingAtDeg(a: Double, span: Double = 10.0): Double {
        val (ax, ay) = xyAt(a - span / 2); val (bx, by) = xyAt(a + span / 2)
        return Math.toDegrees(atan2(by - ay, bx - ax))
    }

    /** eval_reroute.unwrapped_change: total heading change (deg) between two arcs, following every bend. */
    fun changeDeg(s0: Double, s1: Double, step: Double = 5.0): Double {
        val end = maxOf(s1, s0 + step)
        val hs = ArrayList<Double>()
        var a = s0
        while (a < end - 1e-9) { hs.add(headingAtDeg(a)); a += step }
        var sum = 0.0
        for (i in 1 until hs.size) sum += wrapDeg(hs[i] - hs[i - 1])
        return sum
    }

    companion object {
        /** numpy.interp: linear, clamped at both ends. */
        fun interp(v: Double, xp: DoubleArray, fp: DoubleArray): Double {
            if (v <= xp[0]) return fp[0]
            if (v >= xp[xp.size - 1]) return fp[fp.size - 1]
            var lo = 0; var hi = xp.size - 1
            while (hi - lo > 1) { val m = (lo + hi) ushr 1; if (xp[m] <= v) lo = m else hi = m }
            val span = xp[hi] - xp[lo]
            return if (span == 0.0) fp[hi] else fp[lo] + (fp[hi] - fp[lo]) * (v - xp[lo]) / span
        }

        /** numpy.unwrap with the default discontinuity of pi. */
        fun unwrap(p: DoubleArray): DoubleArray {
            val out = p.copyOf(); var corr = 0.0
            for (i in 1 until p.size) {
                val dd = p[i] - p[i - 1]
                var ddmod = ((dd + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI
                if (ddmod == -Math.PI && dd > 0) ddmod = Math.PI
                if (abs(dd) >= Math.PI) corr += ddmod - dd
                out[i] = p[i] + corr
            }
            return out
        }
    }
}

/**
 * OFF ROUTE detector (eval_route_guard.evaluate, v2). Built at the cut from the planned route AHEAD
 * of the dot; fed ~10 times a second with the distance travelled since the cut (AI speed) and the
 * gyro heading since the cut. Compares the heading CHANGE over a sliding window of travel with the
 * route's, allowing for the along-track uncertainty; 60 degrees unexplained for 2 s = off route.
 */
class RouteGuard(routeX: DoubleArray, routeY: DoubleArray) {
    val road = Poly1m(routeX, routeY)
    private val psiR = road.heading.let { h -> DoubleArray(h.size) { h[it] - h[0] } }
    private val ts = ArrayList<Double>(); private val arcs = ArrayList<Double>(); private val psis = ArrayList<Double>()
    private var runStart = Double.NaN
    var alarmIndex = -1; private set
    var lastMismatchDeg = 0.0; private set
    val size: Int get() = arcs.size

    fun arcAt(i: Int) = arcs[i]
    fun psiAt(i: Int) = psis[i]

    /** Distance travelled since the cut -> first sample index at or beyond it (numpy searchsorted, left). */
    fun indexAtArc(a: Double): Int {
        var lo = 0; var hi = arcs.size
        while (lo < hi) { val m = (lo + hi) ushr 1; if (arcs[m] < a) lo = m + 1 else hi = m }
        return lo
    }

    /** Returns true exactly once: at the sample the alarm is raised. */
    fun add(t: Double, arcSinceCut: Double, psiSinceCut: Double): Boolean {
        ts.add(t); arcs.add(arcSinceCut); psis.add(psiSinceCut)
        val k = arcs.size - 1
        val s = arcSinceCut
        val tol = TOL_M + TOL_FRAC * s
        val win = 2.0 * tol + WIN_EXTRA_M
        var mismatch = 0.0
        if (s >= win) {
            val kb = indexAtArc(s - win)
            val dm = psis[k] - psis[kb]
            mismatch = Double.MAX_VALUE
            for (i in 0..20) {
                val off = -1.0 + i * 0.1
                val so = s + off * tol
                val dr = Poly1m.interp(so.coerceIn(0.0, road.length), road.s, psiR) -
                         Poly1m.interp((so - win).coerceIn(0.0, road.length), road.s, psiR)
                mismatch = minOf(mismatch, abs(wrapRad(dm - dr)))
            }
        }
        lastMismatchDeg = Math.toDegrees(mismatch)
        if (alarmIndex >= 0) return false
        if (lastMismatchDeg > ALARM_DEG) {
            if (runStart.isNaN()) runStart = t
            if (t - runStart >= ALARM_HOLD_S) { alarmIndex = k; return true }
        } else runStart = Double.NaN
        return false
    }

    fun tolAt(s: Double) = TOL_M + TOL_FRAC * s

    companion object {
        const val ALARM_DEG = 60.0
        const val ALARM_HOLD_S = 2.0
        const val TOL_M = 15.0
        const val TOL_FRAC = 0.08
        const val WIN_EXTRA_M = 20.0
    }
}

/**
 * Which road did the rider take? (eval_reroute.py) Candidate junctions: route junctions from the
 * detector's window before the alarm up to tol past it. Candidate roads: every branch there the
 * route did not take, followed straight on for 300 m. Winner: the one whose heading change over the
 * same stretch best matches what the gyro measured. Distances here are ROUTE arcs (the full route).
 */
class Rebinder(private val city: CityPack, private val guide: RouteGuide, private val routeNodes: IntArray,
               private val cutArc: Double, private val guard: RouteGuard) {

    class Choice(val junction: Int, val branch: IntArray, val road: Poly1m, val costDeg: Double,
                 val gyroDeg: Double, val pathDeg: Double, val candidates: Int)

    private class Cand(val jn: Int, val sJ: Double, val sStart: Double, val decideArc: Double, val branch: IntArray, val road: Poly1m)

    private val cands = ArrayList<Cand>()

    /** Route arc (since the cut) the phone must reach before every candidate can be judged. */
    val readyAtSinceCut: Double

    init {
        val k = guard.alarmIndex
        val aAlarm = guard.arcAt(k)                      // since the cut
        val sAlarm = cutArc + aAlarm                     // on the route
        val tol = guard.tolAt(aAlarm)
        val lo = sAlarm - (2 * tol + RouteGuard.WIN_EXTRA_M) - tol
        val hi = sAlarm + tol
        var ready = aAlarm
        for (j in 0 until routeNodes.size - 1) {
            val jn = routeNodes[j]
            val sJ = guide.cum[j + 1]
            if (city.degree(jn) < 3 || sJ < lo || sJ > hi || sJ < cutArc) continue
            val prefix = prefixUpTo(sJ)
            for (nb in city.neighbours(jn)) {
                if (nb == routeNodes[j + 1]) continue                        // the road the route takes
                if (j > 0 && nb == routeNodes[j - 1]) continue               // the road we came in on
                val br = straightestPath(city, jn, nb, 300.0)
                if (br.size < 2) continue
                val xs = DoubleArray(prefix.size + br.size); val ys = DoubleArray(prefix.size + br.size)
                prefix.forEachIndexed { i, p -> xs[i] = p.first; ys[i] = p.second }
                br.forEachIndexed { i, n -> val (x, y) = guide.toLocal(city.lat[n], city.lon[n]); xs[prefix.size + i] = x; ys[prefix.size + i] = y }
                val road = Poly1m(xs, ys)
                if (abs(wrapDeg(road.headingAtDeg(sJ + 15) - road.headingAtDeg(maxOf(sJ - 15, 0.0)))) > 160) continue   // the way in
                val sStart = maxOf(sJ - tol - LEAD_M, cutArc)
                val decide = sJ + tol + 30.0
                ready = maxOf(ready, decide - cutArc)
                cands.add(Cand(jn, sJ, sStart, decide, br, road))
            }
        }
        readyAtSinceCut = ready
    }

    val hasCandidates: Boolean get() = cands.isNotEmpty()

    /**
     * Call once the phone has travelled readyAtSinceCut since the cut (or it has waited long enough).
     * Best shape match wins. ADDED to eval_reroute (20 Sept): when several roads match within
     * TIE_DEG -- two left turns 28 m apart look the same to a gyro -- the one whose junction is
     * nearest to where the gyro says the turn happened wins.
     */
    fun choose(): Choice? {
        if (cands.isEmpty()) return null
        val kLast = guard.size - 1
        val scored = cands.map { c ->
            val kb = minOf(guard.indexAtArc(c.sStart - cutArc), kLast)
            val ka = maxOf(guard.alarmIndex, minOf(guard.indexAtArc(c.decideArc - cutArc), kLast))
            val gyro = Math.toDegrees(guard.psiAt(ka) - guard.psiAt(kb))
            val path = c.road.changeDeg(c.sStart, cutArc + guard.arcAt(ka))
            val cost = abs(wrapDeg(gyro - path))
            // where the turn happened: the gyro heading has done half of its change
            var kt = kb
            while (kt < ka && abs(guard.psiAt(kt) - guard.psiAt(kb)) < abs(guard.psiAt(ka) - guard.psiAt(kb)) / 2) kt++
            val turnAt = cutArc + guard.arcAt(kt)
            if (DEBUG) System.out.println("  candidate junction ${c.jn} at ${"%.0f".format(c.sJ)} m -> node ${c.branch[1]}: gyro ${"%.0f".format(gyro)} " +
                               "path ${"%.0f".format(path)} cost ${"%.0f".format(cost)}, gyro turn at ${"%.0f".format(turnAt)} m")
            Triple(c, Choice(c.jn, c.branch, c.road, cost, gyro, path, cands.size), abs(turnAt - c.sJ))
        }
        val bestCost = scored.minOf { it.second.costDeg }
        return scored.filter { it.second.costDeg <= bestCost + TIE_DEG }.minByOrNull { it.third }!!.second
    }

    /** The planned route (local metres) up to arc sJ. */
    private fun prefixUpTo(sJ: Double): List<Pair<Double, Double>> {
        val out = ArrayList<Pair<Double, Double>>()
        for (i in guide.cum.indices) { if (guide.cum[i] >= sJ) break; out.add(guide.xs[i] to guide.ys[i]) }
        return out
    }

    companion object {
        const val LEAD_M = 20.0
        /** Stop waiting for more evidence after this long past the alarm and decide with what we have. */
        const val MAX_WAIT_S = 6.0
        const val TIE_DEG = 15.0
        var DEBUG = false
    }
}

/** From junction [jn] into [first], then keep taking the straightest road onward, for [length] m. Node list. */
internal fun straightestPath(city: CityPack, jn: Int, first: Int, length: Double): IntArray {
    val path = arrayListOf(jn, first)
    var total = CityPack.metres(city.lat[jn], city.lon[jn], city.lat[first], city.lon[first])
    while (total < length) {
        val a = path[path.size - 2]; val b = path[path.size - 1]
        val h = atan2(city.lat[b] - city.lat[a], (city.lon[b] - city.lon[a]) * cos(Math.toRadians(city.lat[b])))
        var bestN = -1; var bestTurn = Double.MAX_VALUE
        for (n in city.neighbours(b)) {
            if (n == a) continue
            val h2 = atan2(city.lat[n] - city.lat[b], (city.lon[n] - city.lon[b]) * cos(Math.toRadians(city.lat[b])))
            val turn = abs(wrapRad(h2 - h))
            if (turn < bestTurn) { bestTurn = turn; bestN = n }
        }
        if (bestN < 0 || bestN in path) break
        total += CityPack.metres(city.lat[b], city.lon[b], city.lat[bestN], city.lon[bestN])
        path.add(bestN)
    }
    return path.toIntArray()
}
