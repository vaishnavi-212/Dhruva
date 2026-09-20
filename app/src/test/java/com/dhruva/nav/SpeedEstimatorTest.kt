package com.dhruva.nav

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.Executor

/**
 * The model must run ONLY during a blackout, and answer the moment one
 * starts.
 *
 * A fake model (always 5 m/s) replaces the ONNX file, and runs on the test's
 * own thread.
 */
class SpeedEstimatorTest {

    private var calls = 0

    private fun estimator() = SpeedEstimator(
        frame = LevelFrame.REPLICA,
        window = 300,
        mean = DoubleArray(6),
        std = DoubleArray(6) { 1.0 },
        predict = { calls++; 5.0 },
        closeModel = {},
        everyS = 0.5,
        worker = Executor { it.run() }
    )

    /** Feeds all three sensors at 50 Hz from [fromS] to [toS] seconds. */
    private fun feed(e: SpeedEstimator, fromS: Double, toS: Double) {
        var t = fromS

        while (t < toS) {
            val ns = (t * 1e9).toLong()

            e.onSensor(
                ImuFeatures.ACC,
                ns,
                floatArrayOf(0.1f, 0f, 0f)
            )

            e.onSensor(
                ImuFeatures.GYRO,
                ns,
                floatArrayOf(0f, 0f, 0.01f)
            )

            e.onSensor(
                ImuFeatures.GRAV,
                ns,
                floatArrayOf(0f, 0f, 9.8f)
            )

            t += 0.02
        }
    }

    @Test
    fun modelNeverRunsWhileGpsIsFine() {
        val e = estimator()

        feed(e, 0.0, 60.0)

        assertEquals(0, calls)
        assertTrue("window should be full after 60 s", e.ready)
        assertFalse(e.freshAt(60.0))
    }

    @Test
    fun blackoutGetsAnAnswerAtTheNextRow() {
        val e = estimator()

        feed(e, 0.0, 40.0)

        e.setActive(true)

        feed(e, 40.0, 40.2) // 0.2 s later

        assertEquals(5.0, e.speedMps, 1e-9)
        assertTrue(e.freshAt(40.2))
    }

    @Test
    fun runsTwiceASecondDuringABlackout() {
        val e = estimator()

        feed(e, 0.0, 40.0)

        e.setActive(true)

        feed(e, 40.0, 50.0) // 10 s of blackout

        assertTrue(
            "expected about 20 runs, got $calls",
            calls in 19..21
        )
    }

    @Test
    fun gpsBackForgetsTheAnswerAndStopsTheModel() {
        val e = estimator()

        feed(e, 0.0, 40.0)

        e.setActive(true)

        feed(e, 40.0, 45.0)

        e.setActive(false)

        val before = calls

        feed(e, 45.0, 60.0)

        assertEquals(before, calls)
        assertFalse(e.freshAt(60.0))
    }

    @Test
    fun noAnswerBeforeThirtySecondsAreBuffered() {
        val e = estimator()

        e.setActive(true)

        feed(e, 0.0, 20.0) // blackout from the start, only 20 s of motion

        assertEquals(0, calls)

        feed(e, 20.0, 31.0)

        assertTrue(calls > 0)
    }
}