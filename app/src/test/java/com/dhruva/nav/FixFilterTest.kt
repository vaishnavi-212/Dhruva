package com.dhruva.nav

import com.dhruva.nav.FixFilter.Verdict.ACCEPT
import com.dhruva.nav.FixFilter.Verdict.GLITCH
import com.dhruva.nav.FixFilter.Verdict.NEW_TRACK
import com.dhruva.nav.FixFilter.Verdict.STALE
import org.junit.Assert.assertEquals
import org.junit.Test

class FixFilterTest {
    private val m = 1.0 / 111320.0                 // one metre of latitude
    private val lake = 15.3776 to 75.1133
    private val kle = 15.3693 to 75.1219

    @Test
    fun cachedFixFromTheLastRideIsNeverUsed() {
        val f = FixFilter()
        assertEquals(STALE, f.check(lake.first, lake.second, 0, ageS = 240.0))
        assertEquals(ACCEPT, f.check(kle.first, kle.second, 1000, ageS = 0.5))      // first real fix: no line from the lake
        assertEquals(ACCEPT, f.check(kle.first + 8 * m, kle.second, 2000, ageS = 0.5))
    }

    @Test
    fun oneWildFixIsDroppedAndTheTrackCarriesOn() {
        val f = FixFilter()
        f.check(kle.first, kle.second, 0, 0.2)
        assertEquals(GLITCH, f.check(kle.first + 900 * m, kle.second, 1000, 0.2))
        assertEquals(ACCEPT, f.check(kle.first + 16 * m, kle.second, 2000, 0.2))
    }

    @Test
    fun aWrongFirstFixIsReplacedByANewTrack_notJoinedByAStraightLine() {
        val f = FixFilter()
        f.check(lake.first, lake.second, 0, 0.2)                                  // wrong, but fresh
        assertEquals(GLITCH, f.check(kle.first, kle.second, 1000, 0.2))
        assertEquals(GLITCH, f.check(kle.first + 8 * m, kle.second, 2000, 0.2))
        assertEquals(NEW_TRACK, f.check(kle.first + 16 * m, kle.second, 3000, 0.2))
        assertEquals(ACCEPT, f.check(kle.first + 24 * m, kle.second, 4000, 0.2))
    }

    @Test
    fun afterAPauseTheLineIsNotJoinedAcrossTheGap() {
        val f = FixFilter()
        f.check(kle.first, kle.second, 0, 0.2)
        assertEquals(NEW_TRACK, f.check(kle.first + 300 * m, kle.second, 60_000, 0.2))
        assertEquals(ACCEPT, f.check(kle.first + 308 * m, kle.second, 61_000, 0.2))
    }
}
