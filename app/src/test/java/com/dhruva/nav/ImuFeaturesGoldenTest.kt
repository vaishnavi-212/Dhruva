package com.dhruva.nav

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * The phone must build the speed model's inputs exactly as the laptop does.
 * Golden files: engine/scripts/make_golden_features.py (60 s of the 11 Sept 19-41-28 ride).
 */
class ImuFeaturesGoldenTest {

    private fun golden(name: String) = File(javaClass.classLoader!!.getResource("golden/$name")!!.toURI())

    private fun events(name: String, kind: Int) = golden(name).readLines().drop(1).map { line ->
        val p = line.split(",")
        Triple(p[0].toLong(), kind, doubleArrayOf(p[1].toDouble(), p[2].toDouble(), p[3].toDouble()))
    }

    private fun check(frame: LevelFrame, expectedFile: String) {
        val all = (events("raw_acc.csv", ImuFeatures.ACC) + events("raw_gyro.csv", ImuFeatures.GYRO) +
                events("raw_grav.csv", ImuFeatures.GRAV)).sortedWith(compareBy({ it.first }, { it.second }))
        val got = ArrayList<Pair<Double, FloatArray>>()
        val features = ImuFeatures(frame) { t, row -> got.add(t to row.copyOf()) }
        for ((tNs, kind, v) in all) features.push(kind, tNs / 1e9, v[0], v[1], v[2])

        val expected = golden(expectedFile).readLines().drop(1).map { l -> l.split(",").map { it.toDouble() } }
        assertEquals("rows", expected.size, got.size)
        var worst = 0.0
        for (i in expected.indices) {
            assertEquals("grid time, row $i", expected[i][0], got[i].first, 1e-6)
            for (c in 0 until ImuFeatures.CHANNELS) worst = maxOf(worst, abs(expected[i][1 + c] - got[i].second[c]))
        }
        println("$frame: ${got.size} rows, worst |Kotlin - Python| = $worst")
        assertTrue("worst feature difference $worst", worst < 1e-4)
    }

    @Test
    fun replicaFrameMatchesLaptop() = check(LevelFrame.REPLICA, "expected_replica.csv")

    @Test
    fun gravityFrameMatchesLaptop() = check(LevelFrame.GRAVITY, "expected_gravity.csv")

    @Test
    fun windowIsChannelFirstOldestFirst() {
        val f = ImuFeatures(LevelFrame.GRAVITY, window = 3)
        assertEquals(null, f.window(DoubleArray(6), DoubleArray(6) { 1.0 }))
        val rows = ArrayList<FloatArray>()
        val g = ImuFeatures(LevelFrame.GRAVITY, window = 3) { _, r -> rows.add(r.copyOf()) }
        var t = 0.0
        while (rows.size < 4) {
            for (kind in 0..2) g.push(kind, t, if (kind == ImuFeatures.GRAV) 0.0 else t, 0.0, if (kind == ImuFeatures.GRAV) 9.8 else 0.0)
            t += 0.05
        }
        val w = g.window(DoubleArray(6), DoubleArray(6) { 1.0 })!!
        for (c in 0 until 6) for (j in 0 until 3) assertEquals(rows[rows.size - 3 + j][c], w[c * 3 + j], 1e-6f)
    }
}
