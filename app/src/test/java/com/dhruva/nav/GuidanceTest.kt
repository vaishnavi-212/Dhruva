package com.dhruva.nav

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class GuidanceTest {

    private val city by lazy { CityPack.parse(File("src/main/assets/city_pack_hubli.json").readText()) }
    private val phrase = { ins: String, d: Double, _: Double -> "$ins in ${Math.round(d / 10) * 10} metres." }

    /** An L: 200 m north, then 200 m west (a left turn) at a junction. */
    private fun lRoute(): RouteGuide {
        val d = 1.0 / 111320.0
        val pts = (0..20).map { (15.37 + it * 10 * d) to 75.12 } +
                  (1..20).map { (15.37 + 200 * d) to (75.12 - it * 10 * d / Math.cos(Math.toRadians(15.37))) }
        return RouteGuide(pts)
    }

    @Test
    fun leftTurnAtAJunctionIsFound_bendsWithoutJunctionAreNot() {
        val g = lRoute()
        val nodes = IntArray(40) { it }                       // points[1..40]
        val atJunction = Maneuver.fromRoute(g, nodes) { if (it == 19) 3 else 2 }   // node 19 = points[20], the corner
        assertEquals(1, atJunction.size)
        assertEquals("Turn left", atJunction[0].instruction)
        assertEquals(200.0, atJunction[0].s, 1.0)
        assertTrue(Maneuver.fromRoute(g, nodes) { 2 }.isEmpty())   // same corner, no side road: not announced
    }

    @Test
    fun navigatorSpeaksFarThenNowThenArrives_eachOnce() {
        val g = lRoute()
        val nav = Navigator("Test Lake", g, Maneuver.fromRoute(g, IntArray(40) { it }) { if (it == 19) 3 else 2 })
        val said = ArrayList<String>()
        var s = 0.0
        while (s <= g.lengthM) { nav.update(s, 0.0, phrase).speak?.let { said.add(it) }; s += 5.0 }
        nav.update(g.lengthM, 0.0, phrase).speak?.let { said.add(it) }
        println(said)
        assertEquals(listOf("Turn left in 150 metres.", "Turn left now.", "Your destination is ahead in 150 metres.",
            "You have arrived at Test Lake."), said)
        assertTrue(nav.arrived)
    }

    @Test
    fun projectionGivesProgressAndDistanceOff() {
        val g = lRoute()
        val d = 1.0 / 111320.0
        val p = g.project(15.37 + 100 * d, 75.12 + 30 * d / Math.cos(Math.toRadians(15.37)))
        assertEquals(100.0, p.s, 0.5)
        assertEquals(30.0, p.offM, 0.5)
    }

    @Test
    fun offRouteWaitsUntilTheRiderHasJoinedTheRoute() {
        val o = OffRouteDetector()
        repeat(5) { assertFalse(o.onFix(58.0, 5f)) }   // standing in a building 58 m from the route start
        assertFalse(o.onFix(10.0, 5f))                 // on the road now
        assertFalse(o.onFix(60.0, 5f))
        assertTrue(o.onFix(60.0, 5f))
    }

    @Test
    fun offRouteNeedsTwoFixesInARow() {
        val o = OffRouteDetector()
        o.onFix(0.0, 5f)                               // joined
        assertFalse(o.onFix(60.0, 5f))
        assertFalse(o.onFix(10.0, 5f))        // back on: count resets
        assertFalse(o.onFix(60.0, 5f))
        assertTrue(o.onFix(55.0, 5f))
        o.reset(); o.onFix(0.0, 5f)
        assertFalse(o.onFix(60.0, 80f))       // a 80 m-accuracy fix cannot say we are 60 m off
        assertFalse(o.onFix(60.0, 80f))
    }

    @Test
    fun demoRouteKleTechToUnkalLake() {
        val lake = city.search("unkal lake", 15.3693, 75.1219).first().first
        val r = city.route(15.3693, 75.1219, lake.arrive)
        assertNotNull(r); r!!
        val g = RouteGuide(r.points)
        val ms = Maneuver.fromRoute(g, r.nodes) { city.degree(it) }
        println("KLE Tech -> ${lake.name}: ${"%.0f".format(g.lengthM)} m, turns: " +
                ms.joinToString { "%s at %.0f m (%.0f°)".format(it.instruction, it.s, it.angleDeg) })
        assertEquals(r.lengthM, g.lengthM, r.lengthM * 0.005)
        assertTrue("a route this long has a few turns, not dozens", ms.size in 1..8)
    }
}
