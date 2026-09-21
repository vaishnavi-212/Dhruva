package com.dhruva.nav

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs
import kotlin.math.hypot

/**
 * The Kotlin fusion filter must give the same answer as the Python one,
 * epoch by epoch.
 *
 * Golden file: engine/scripts/make_golden_fusion.py (one benchmark ride,
 * phone conditions, 10 Hz).
 */
class FusionEngineGoldenTest {

    private fun golden(): Pair<Map<String, Double>, List<Map<String, Double>>> {
        val lines =
            File(
                javaClass.classLoader!!
                    .getResource("golden/fusion_golden.csv")!!
                    .toURI()
            ).readLines()

        val init =
            Regex("""(\w+)=(-?[\d.eE+-]+)""")
                .findAll(lines[0])
                .associate {
                    it.groupValues[1] to it.groupValues[2].toDouble()
                }

        val cols = lines[1].split(",")

        val rows =
            lines.drop(2).map { l ->
                cols.zip(
                    l.split(",").map {
                        if (it == "nan")
                            Double.NaN
                        else
                            it.toDouble()
                    }
                ).toMap()
            } // Python writes NaN as "nan"

        return init to rows
    }

    private fun opt(v: Double): Double? =
        if (v.isNaN()) null else v

    @Test
    fun matchesPythonEveryEpoch() {
        val (init, rows) = golden()

        val f = FusionEngine()

        f.initialise(
            init.getValue("x"),
            init.getValue("y"),
            init.getValue("v"),
            init.getValue("psi")
        )

        val dot = SeamlessDot()

        var worstPos = 0.0
        var worstPsi = 0.0
        var worstP = 0.0
        var worstDot = 0.0

        val t0 = System.nanoTime()

        for (r in rows) {

            f.step(
                dt = r.getValue("dt"),
                gyroZ = r.getValue("gyro_z"),
                accelFwd = r.getValue("accel_fwd"),

                gnssX = opt(r.getValue("gnss_x")),
                gnssY = opt(r.getValue("gnss_y")),
                gnssSpeed = opt(r.getValue("gnss_speed")),

                modelSpeed = opt(r.getValue("model_speed")),

                roadHeading = r.getValue("road_heading"),
                roadX = r.getValue("road_x"),
                roadY = r.getValue("road_y")
            )

            dot.update(
                f.posX,
                f.posY,
                r.getValue("dt")
            )

            worstPos = maxOf(
                worstPos,
                hypot(
                    f.posX - r.getValue("x"),
                    f.posY - r.getValue("y")
                )
            )

            worstPsi = maxOf(
                worstPsi,
                abs(
                    f.heading - r.getValue("psi")
                )
            )

            worstP = maxOf(
                worstP,
                abs(
                    f.p[0] - r.getValue("p00")
                ) / maxOf(
                    1.0,
                    r.getValue("p00")
                )
            )

            worstDot = maxOf(
                worstDot,
                hypot(
                    dot.outX - r.getValue("dot_x"),
                    dot.outY - r.getValue("dot_y")
                )
            )

            val pyMode =
                when (r.getValue("mode").toInt()) {
                    1 -> FusionEngine.Mode.GNSS_AIDED
                    2 -> FusionEngine.Mode.DEAD_RECKONING
                    else -> FusionEngine.Mode.INIT
                }

            assertEquals(pyMode, f.mode)
        }

        val epochsPerS =
            rows.size /
                    ((System.nanoTime() - t0) / 1e9)

        println(
            "fusion: ${rows.size} epochs, " +
                    "worst |pos| ${worstPos} m, " +
                    "|psi| ${worstPsi} rad, " +
                    "P rel ${worstP}, " +
                    "dot ${worstDot} m; " +
                    "${epochsPerS.toInt()} epochs/s on this machine"
        )

        assertTrue(
            "position differs by $worstPos m",
            worstPos < 1e-6
        )

        assertTrue(
            "heading differs by $worstPsi rad",
            worstPsi < 1e-9
        )

        assertTrue(
            "covariance differs by $worstP",
            worstP < 1e-9
        )

        assertTrue(
            "shown dot differs by $worstDot m",
            worstDot < 1e-6
        )
    }
}