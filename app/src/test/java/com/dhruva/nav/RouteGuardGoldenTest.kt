package com.dhruva.nav

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * The phone's route guard must raise OFF ROUTE at the same sample as the Python evaluation.
 * Golden: engine/scripts/make_golden_route_guard.py (94CC_6RQ 30 Aug 16:39, a wrong and a true destination).
 */
class RouteGuardGoldenTest {
    private fun arr(a: JSONArray) = DoubleArray(a.length()) { a.getDouble(it) }

    @Test
    fun alarmsAtTheSameSampleAsPython() {
        val g = JSONObject(File(javaClass.classLoader!!.getResource("golden/golden_route_guard.json")!!.toURI()).readText())
        val cases = g.getJSONArray("cases")
        for (c in 0 until cases.length()) {
            val o = cases.getJSONObject(c)
            val guard = RouteGuard(arr(o.getJSONArray("route_x")), arr(o.getJSONArray("route_y")))
            val t = arr(o.getJSONArray("t")); val arc = arr(o.getJSONArray("arc")); val psi = arr(o.getJSONArray("psi"))
            val mis = arr(o.getJSONArray("mismatch_deg"))
            var worst = 0.0; var alarm = -1
            for (i in t.indices) {
                if (guard.add(t[i], arc[i], psi[i])) alarm = i
                worst = maxOf(worst, abs(guard.lastMismatchDeg - mis[i]))
            }
            val want = if (o.isNull("alarm_index")) -1 else o.getInt("alarm_index")
            println("${o.getString("name")}: ${t.size} samples, alarm Kotlin $alarm vs Python $want, worst |mismatch| $worst deg")
            assertTrue("mismatch differs by $worst deg", worst < 1e-3)     // 2.4e-4 deg seen: float order, no decision changes
            assertEquals("${o.getString("name")} alarm sample", want, alarm)
        }
    }
}
