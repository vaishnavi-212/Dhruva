package com.dhruva.nav

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The demo, simulated on the real map: planned KLE Tech -> Unkal Lake, GPS cut 60 m in, and at the
 * campus gate the rider goes the other way. The gyro heading is what the rider's real path implies.
 */
class WrongTurnTest {
    private val city by lazy { CityPack.parse(File("src/main/assets/city_pack_hubli.json").readText()) }
    private val kle = 15.3693 to 75.1219
    private val cut = 60.0

    private fun planned(): Pair<CityPack.Route, RouteGuide> {
        val lake = city.search("unkal lake", kle.first, kle.second).first().first
        val r = city.route(kle.first, kle.second, lake.arrive)!!
        return r to RouteGuide(r.points)
    }

    /** Feed the guard 1 m at a time at 8 m/s along [path] (local metres, same origin as the route). */
    private fun ride(guard: RouteGuard, path: Poly1m, untilSinceCut: Double): Int {
        var alarm = -1; var a = 0.0
        val h0 = Poly1m.interp(cut, path.s, path.heading)
        while (a <= untilSinceCut) {
            val psi = Poly1m.interp(cut + a, path.s, path.heading) - h0
            if (guard.add(a / 8.0, a, psi)) alarm = guard.size - 1
            a += 1.0
        }
        return alarm
    }

    @Test
    fun followingTheRouteStaysQuiet() {
        val (_, g) = planned()
        val (ax, ay) = g.ahead(cut)
        val guard = RouteGuard(ax, ay)
        assertEquals(-1, ride(guard, Poly1m(g.xs, g.ys), g.lengthM - cut - 5))
    }

    @Test
    fun wrongWayAtTheGateIsCaughtRoadFoundAndReRouted() {
        val (r, g) = planned()
        // the gate: the route's first turn
        val gate = Maneuver.fromRoute(g, r.nodes) { city.degree(it) }.first()
        val j = (0 until r.nodes.size - 1).first { g.cum[it + 1] >= gate.s - 0.5 }
        val jn = r.nodes[j]; val next = r.nodes[j + 1]; val prev = r.nodes[j - 1]
        val wrong = city.neighbours(jn).filter { it != next && it != prev }
        assertTrue("a junction has another road", wrong.isNotEmpty())
        for (nb in wrong) {
            // the rider's real path: route to the gate, then the other road
            val br = straightestPath(city, jn, nb, 300.0)
            val xs = (0..j).map { g.xs[it] } + br.map { g.toLocal(city.lat[it], city.lon[it]).first }
            val ys = (0..j).map { g.ys[it] } + br.map { g.toLocal(city.lat[it], city.lon[it]).second }
            val truePath = Poly1m(xs.toDoubleArray(), ys.toDoubleArray())
            val (ax, ay) = g.ahead(cut)
            val guard = RouteGuard(ax, ay)
            val sJ = g.cum[j + 1]
            val alarm = ride(guard, truePath, sJ - cut + 150)
            if (alarm < 0) { println("road to node $nb: no alarm (it runs close to the route's own heading)"); continue }
            val delay = guard.arcAt(alarm) - (sJ - cut)
            val rb = Rebinder(city, g, r.nodes, cut, guard)
            ride(guard, truePath, maxOf(rb.readyAtSinceCut, sJ - cut + 150))     // keep riding until it can decide
            Rebinder.DEBUG = true
            val choice = rb.choose()
            assertNotNull(choice); choice!!
            println("wrong road via node $nb: alarm ${"%.0f".format(delay)} m after the gate, ${choice.candidates} candidate road(s), " +
                    "chose node ${choice.branch[1]} (gyro ${"%.0f".format(choice.gyroDeg)}° vs path ${"%.0f".format(choice.pathDeg)}°)")
            assertTrue("caught within 80 m of the gate (was $delay)", delay < 80)
            assertEquals("the road actually taken", nb, choice.branch[1])
            // re-route from where the dot now is on that road, facing the way the rider goes
            val lake = city.search("unkal lake", kle.first, kle.second).first().first
            val d = routeFromBranch(choice, g, lake.arrive, sJ + 40.0)
            assertNotNull(d)
            println("  re-route: ${"%.0f".format(d!!.route.lengthM)} m, U-turn first: ${d.uTurn}")
        }
    }

    /** Where the dot is on the chosen road, and the directed re-route from there. */
    private fun routeFromBranch(c: Rebinder.Choice, g: RouteGuide, targets: IntArray, sNow: Double): CityPack.Directed? {
        val (x, y) = c.road.xyAt(sNow)
        val (la, lo) = g.toLatLon(x, y)
        var best = 0; var bd = Double.MAX_VALUE
        for (k in 0 until c.branch.size - 1) {
            val d = CityPack.metres(la, lo, city.lat[c.branch[k + 1]], city.lon[c.branch[k + 1]])
            if (d < bd) { bd = d; best = k }
        }
        return city.routeFrom(c.branch[best], c.branch[best + 1], la, lo, targets)
    }
}
