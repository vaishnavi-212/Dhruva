package com.dhruva.nav

import org.junit.Assert.assertTrue
import org.junit.Test

/** The card's one-liner must never contradict itself. */
class RunSummaryTest {

    private fun result(distanceM: Double, driftPct: Double, passes: Boolean) = RunSummary.Result(
        distanceM = distanceM, durationS = 60.0, finalErrorM = distanceM * driftPct / 100.0,
        driftPct = driftPct, passes = passes, meanSpeedMps = 5.0, confidence90M = 10.0,
        gpsFixes = 60, imuHz = 50.0
    )

    @Test
    fun aGoodBlackoutPasses() {
        assertTrue(RunSummary.verdict(result(485.0, 4.5, true)).startsWith("PASS"))
    }

    @Test
    fun aDriftyBlackoutFails() {
        assertTrue(RunSummary.verdict(result(485.0, 22.0, false)).startsWith("FAIL"))
    }

    @Test
    fun tooShortToScoreSaysSo_notFail() {
        val v = RunSummary.verdict(result(0.0, 0.0, false))
        assertTrue("said: $v", v.startsWith("NOT SCORABLE"))
        assertTrue("said: $v", !v.contains("0.0% drift"))
    }
}
