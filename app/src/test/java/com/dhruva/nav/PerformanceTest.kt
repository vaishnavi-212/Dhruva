package com.dhruva.nav

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** The work the screen does must be small enough that a phone never stalls on it. */
class PerformanceTest {

    private val json by lazy { File("src/main/assets/city_pack_hubli.json").readText() }

    @Test
    fun loadingTheCityAndSearchingIsCheap() {
        val t0 = System.nanoTime()
        val city = CityPack.parse(json)                       // background thread in the app
        val parseMs = (System.nanoTime() - t0) / 1e6

        val queries = listOf("u", "un", "unk", "unka", "unkal", "unkal l", "unkal la", "unkal lake",
            "kle", "lhc", "hosp", "railway", "park", "bank", "bvb", "vidyanagar")
        val t1 = System.nanoTime()
        for (q in queries) city.search(q, 15.3693, 75.1219, limit = 8)
        val perKeystrokeMs = (System.nanoTime() - t1) / 1e6 / queries.size

        val lake = city.search("unkal lake", 15.3693, 75.1219).first().first
        val t2 = System.nanoTime()
        repeat(5) { city.route(15.3693, 75.1219, lake.arrive) }
        val routeMs = (System.nanoTime() - t2) / 1e6 / 5

        println("city pack: parse ${"%.0f".format(parseMs)} ms (background), " +
                "search ${"%.2f".format(perKeystrokeMs)} ms per keystroke (main thread), " +
                "route ${"%.0f".format(routeMs)} ms (background)")
        assertTrue("a keystroke must not block the screen: $perKeystrokeMs ms", perKeystrokeMs < 50.0)
        assertTrue("parsing must stay well under a screen timeout: $parseMs ms", parseMs < 5000.0)
    }
}
