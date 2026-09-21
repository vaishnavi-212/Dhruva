package com.dhruva.nav

import org.junit.Assert.assertEquals
import org.junit.Test

/** The GPS switch must flip on real loss, flip back on real recovery, and
 * never flap. */
class GnssSwitchTest {

    private val changes = mutableListOf<Pair<GnssMode, Long>>()

    private val sw = GnssSwitch(onChange = { m, _, t ->
        changes.add(m to t)
    })

    /** Good fixes (±5 m, 9 satellites) once a second, ticking 10 times a
     * second, from [fromS] to [toS]. */
    private fun goodFixes(
        fromS: Int,
        toS: Int,
        acc: Float = 5f,
        sats: Int = 9
    ) {
        for (s in fromS until toS) {
            sw.onSatellites(s * 1000L, sats)
            sw.onFix(s * 1000L, acc)

            for (k in 1..9)
                sw.tick(s * 1000L + k * 100L)
        }
    }

    /** No fixes at all (a tunnel), ticking 10 times a second. */
    private fun silence(fromS: Int, toS: Int) {
        for (ms in fromS * 1000L until toS * 1000L step 100L)
            sw.tick(ms)
    }

    @Test
    fun staysOnGnssWhileFixesAreGood() {
        goodFixes(0, 30)

        assertEquals(GnssMode.GNSS, sw.mode)
        assertEquals(0, changes.size)
    }

    @Test
    fun tunnelIsDetectedTwoAndAHalfSecondsAfterTheLastFix() {
        goodFixes(0, 10) // last fix at 9.0
        silence(10, 20)

        assertEquals(GnssMode.DEAD_RECKONING, sw.mode)
        assertEquals(11_600L, changes[0].second) // first tick
        // more than 2.5 s after 9.0 s
    }

    @Test
    fun comesBackAfterTwoGoodFixesNotOne() {
        goodFixes(0, 10)
        silence(10, 20)

        sw.onFix(20_000L, 5f)

        assertEquals(GnssMode.DEAD_RECKONING, sw.mode) // one fix is not
        // enough

        sw.onFix(21_000L, 5f)

        assertEquals(GnssMode.GNSS, sw.mode)
        assertEquals(21_000L, changes[1].second)
    }

    @Test
    fun tooFewSatellitesForASecondMeansLost() {
        goodFixes(0, 10)
        goodFixes(10, 15, sats = 3)

        assertEquals(GnssMode.DEAD_RECKONING, sw.mode)
        assertEquals(11_000L, changes[0].second)
    }

    @Test
    fun aCoarseNetworkFixMeansLostButAMediumOneDoesNot() {
        goodFixes(0, 10)

        sw.onFix(10_000L, 45f) // worse than 30 m, better than 60 m
        assertEquals(GnssMode.GNSS, sw.mode)

        sw.onFix(11_000L, 150f) // tunnel: coarse
        // Wi-Fi/cell position

        assertEquals(GnssMode.DEAD_RECKONING, sw.mode)
    }

    @Test
    fun simulationForcesDeadReckoningAndOnlyRealGpsEndsIt() {
        goodFixes(0, 10)

        sw.setSimulated(true, 10_000L)

        assertEquals(GnssMode.DEAD_RECKONING, sw.mode)
        assertEquals(true, sw.realGnssOk) // the sky is
        // still fine

        silence(10, 20) // real GPS also
        // dies during the simulation

        sw.setSimulated(false, 20_000L)

        assertEquals(GnssMode.DEAD_RECKONING, sw.mode) // still no real
        // GPS: stay on dead reckoning

        goodFixes(20, 23)

        assertEquals(GnssMode.GNSS, sw.mode)
    }

    @Test
    fun aFlickeringTunnelMouthDoesNotFlap() {
        goodFixes(0, 10)
        silence(10, 15) // lost

        for (s in 15 until 30) { // good, missing,
            // good, missing ...
            if (s % 2 == 1)
                sw.onFix(s * 1000L, 8f)

            for (k in 0..9)
                sw.tick(s * 1000L + k * 100L)
        }

        assertEquals(1, changes.size) // only the first
        // loss, never back
    }
}