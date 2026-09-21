package com.dhruva.nav

/**
 * Decides whether a GPS fix is used, BEFORE it reaches the map, the route guidance or the score.
 *
 * 20 Sept (emulator): the navigation screen drew a straight blue line from where the previous ride
 * ended to where this one started. The first fix Android delivered was its CACHED location from
 * minutes earlier, and every later fix was joined to it. Real phones do the same when the screen
 * is opened far from where the phone last had a fix (the hostel, then the parking lot).
 *
 *   STALE      older than 10 s when it arrives: a cached fix. Never used.
 *   GLITCH     implies more than 216 km/h from the last good fix: dropped. If the next fixes
 *              agree with each other at the new place, the jump was real (the anchor was wrong),
 *              and the next one is accepted as NEW_TRACK.
 *   NEW_TRACK  use it, but start a new line: do not join it to the previous fix with a straight
 *              line. Also after a 20 s+ gap (app paused) with 50 m+ of movement.
 */
class FixFilter {

    enum class Verdict { ACCEPT, NEW_TRACK, STALE, GLITCH }

    private var lastLat = 0.0; private var lastLon = 0.0; private var lastMs = 0L
    private var hasLast = false
    private var pendLat = 0.0; private var pendLon = 0.0; private var pendMs = 0L
    private var pending = 0

    var rejected = 0
        private set

    /** [ageS] = how old the fix already is when it arrives; [tMs] = when it was taken. */
    fun check(lat: Double, lon: Double, tMs: Long, ageS: Double): Verdict {
        if (ageS > STALE_S) { rejected++; return Verdict.STALE }
        if (!hasLast) return accept(lat, lon, tMs, Verdict.ACCEPT)

        val d = CityPack.metres(lastLat, lastLon, lat, lon)
        val dt = (tMs - lastMs) / 1000.0
        if (dt > 0 && d / dt <= MAX_MPS) {
            pending = 0
            val v = if (dt >= GAP_S && d >= GAP_M) Verdict.NEW_TRACK else Verdict.ACCEPT
            return accept(lat, lon, tMs, v)
        }
        // impossible from the last good fix: a glitch, unless the fixes after it keep agreeing
        val agrees = pending > 0 && (tMs - pendMs) > 0 &&
            CityPack.metres(pendLat, pendLon, lat, lon) / ((tMs - pendMs) / 1000.0) <= MAX_MPS
        pending = if (agrees) pending + 1 else 1
        pendLat = lat; pendLon = lon; pendMs = tMs
        if (pending >= CONFIRM) { pending = 0; return accept(lat, lon, tMs, Verdict.NEW_TRACK) }
        rejected++
        return Verdict.GLITCH
    }

    private fun accept(lat: Double, lon: Double, tMs: Long, v: Verdict): Verdict {
        lastLat = lat; lastLon = lon; lastMs = tMs; hasLast = true
        return v
    }

    companion object {
        const val STALE_S = 10.0
        const val MAX_MPS = 60.0        // 216 km/h: nothing we navigate moves this fast
        const val CONFIRM = 3           // fixes in a row agreeing on the new place
        const val GAP_S = 20.0
        const val GAP_M = 50.0
    }
}
