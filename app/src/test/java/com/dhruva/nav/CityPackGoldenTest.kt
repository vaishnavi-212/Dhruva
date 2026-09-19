package com.dhruva.nav

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The phone must route and search exactly like the laptop reference (engine/dhruva/citypack.py).
 * Golden file: engine/scripts/make_golden_city.py (41 routes, 12 searches, from the shipped pack).
 */
class CityPackGoldenTest {

    private val city by lazy { CityPack.parse(File("src/main/assets/city_pack_hubli.json").readText()) }
    private val golden by lazy {
        JSONObject(File(javaClass.classLoader!!.getResource("golden/city_golden.json")!!.toURI()).readText())
    }

    @Test
    fun routesMatchPython() {
        val routes = golden.getJSONArray("routes")
        var worstLen = 0.0; var worstSec = 0.0; var worstEnd = 0.0
        val t0 = System.nanoTime()
        for (i in 0 until routes.length()) {
            val g = routes.getJSONObject(i)
            val place = city.places[g.getInt("place")]
            assertEquals("route $i place", g.getString("name"), place.name)
            val r = city.route(g.getDouble("lat"), g.getDouble("lon"), place.arrive)
            if (!g.getBoolean("reachable")) { assertNull("route $i should be unreachable", r); continue }
            r!!
            assertEquals("route $i points", g.getInt("n_points"), r.points.size)
            worstLen = maxOf(worstLen, kotlin.math.abs(r.lengthM - g.getDouble("length_m")))
            worstSec = maxOf(worstSec, kotlin.math.abs(r.seconds - g.getDouble("seconds")))
            val end = g.getJSONArray("end")
            worstEnd = maxOf(worstEnd, CityPack.metres(r.points.last().first, r.points.last().second, end.getDouble(0), end.getDouble(1)))
        }
        val ms = (System.nanoTime() - t0) / 1e6 / routes.length()
        val demo = city.route(golden.getDouble("from_lat"), golden.getDouble("from_lon"), city.places[routes.getJSONObject(0).getInt("place")].arrive)!!
        println("routes: ${routes.length()}, worst |length| ${worstLen} m, |time| ${worstSec} s, end ${worstEnd} m; " +
                "${"%.1f".format(ms)} ms per route here; demo KLE Tech -> Unkal Lake ${"%.0f".format(demo.lengthM)} m")
        assertTrue("length differs by $worstLen m", worstLen < 1e-6)
        assertTrue("time differs by $worstSec s", worstSec < 1e-6)
        assertTrue("end differs by $worstEnd m", worstEnd < 1e-6)
    }

    @Test
    fun searchMatchesPython() {
        val searches = golden.getJSONArray("searches")
        val la = golden.getDouble("from_lat"); val lo = golden.getDouble("from_lon")
        for (i in 0 until searches.length()) {
            val g = searches.getJSONObject(i)
            val want = g.getJSONArray("top").let { a -> List(a.length()) { a.getString(it) } }
            val got = city.search(g.getString("query"), la, lo, limit = 5).map { it.first.name }
            assertEquals("search '${g.getString("query")}'", want, got)
        }
    }

    @Test
    fun typoAndAliasFindTheLake() {
        val top = city.search("unkal lake", 15.3693, 75.1219).first().first
        assertEquals("Unakal Kere", top.name)
        assertEquals("Unakal Kere", city.search("unakal kere", 15.3693, 75.1219).first().first.name)
    }
}
