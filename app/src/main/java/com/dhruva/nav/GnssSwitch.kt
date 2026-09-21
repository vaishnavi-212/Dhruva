package com.dhruva.nav

/** What the navigation is running on right now. */
enum class GnssMode { GNSS, DEAD_RECKONING }

/**
 * The Seamless GNSS Deficit Handler's decision: is GNSS usable right now?
 *
 * Pure Kotlin, no Android types, so it is unit-tested on the laptop.
 * [GnssWatcher] feeds it on the phone. Times are milliseconds on ONE monotonic clock
 * (SystemClock.elapsedRealtime()).
 *
 * GNSS is declared LOST when any of these happens:
 * - no fix for [lostAfterMs] (fixes normally arrive every 1 s, so 2.5 s =
 *   two missed fixes)
 * - fewer than [minSatellites] satellites used in the fix, for
 *   [weakSatHoldMs]
 * - a fix worse than 2 x [maxAccuracyM] (a tunnel often gives a coarse Wi-Fi/cell
 *   position instead)
 *
 * GNSS is declared BACK after [regainFixes] good fixes in a row (accuracy <=
 * [maxAccuracyM], enough satellites, each within [streakGapMs] of the one before), so a
 * flickering tunnel mouth doesn't flip the mode back and forth.
 *
 * [setSimulated] is the demo switch: it forces DEAD_RECKONING while real GPS
 * keeps running, so the real track stays available for scoring. [realGnssOk] always tells
 * the truth about the sky.
 */
class GnssSwitch(
    private val lostAfterMs: Long = 2_500,
    private val minSatellites: Int = 4,
    private val weakSatHoldMs: Long = 1_000,
    private val maxAccuracyM: Float = 30f,
    private val regainFixes: Int = 2,
    private val streakGapMs: Long = 1_500,
    private val onChange: (GnssMode, String, Long) -> Unit = { _, _, _ -> }
) {
    var mode = GnssMode.GNSS
        private set

    var reason = "waiting for first fix"
        private set

    var realGnssOk = true
        private set

    var simulated = false
        private set

    var lastFixMs = -1L
        private set

    var lastAccuracyM = Float.NaN
        private set

    /** Satellites used in the last fix; -1 until the phone reports satellite status. */
    var satellitesUsed = -1
        private set

    var switches = 0
        private set

    private var weakSinceMs = -1L
    private var goodStreak = 0

    val hasFix: Boolean
        get() = lastFixMs >= 0

    /** A position fix arrived (any provider). */
    fun onFix(
        tMs: Long,
        accuracyM: Float
    ) {
        val continues =
            lastFixMs >= 0 &&
                    tMs - lastFixMs <= streakGapMs

        lastFixMs = tMs
        lastAccuracyM = accuracyM

        val enoughSats =
            satellitesUsed < 0 ||
                    satellitesUsed >= minSatellites

        val good =
            accuracyM <= maxAccuracyM &&
                    enoughSats

        goodStreak = when {
            !good -> 0

            continues ->
                goodStreak + 1

            else ->
                1 // a gap before
            // this fix: the streak starts again
        }

        when {
            !realGnssOk &&
                    goodStreak >= regainFixes ->
                setReal(
                    true,
                    tMs,
                    "GPS back: $goodStreak good fixes, ±${accuracyM.toInt()} m"
                )

            realGnssOk &&
                    accuracyM > 2 * maxAccuracyM ->
                setReal(
                    false,
                    tMs,
                    "fix too coarse: ±${accuracyM.toInt()} m"
                )
        }
    }

    /** Satellite status arrived: how many satellites the receiver used in its fix. */
    fun onSatellites(
        tMs: Long,
        used: Int
    ) {
        satellitesUsed = used

        if (used < minSatellites) {
            if (weakSinceMs < 0) {
                weakSinceMs = tMs
            }
        } else {
            weakSinceMs = -1
        }

        tick(tMs)
    }

    /** Call about 10 times a second: this is what notices that fixes have STOPPED arriving. */
    fun tick(tMs: Long) {
        if (!realGnssOk) return

        if (
            lastFixMs >= 0 &&
            tMs - lastFixMs > lostAfterMs
        ) {
            setReal(
                false,
                tMs,
                "no fix for %.1f s".format(
                    java.util.Locale.US,
                    (tMs - lastFixMs) / 1000.0
                )
            )
        } else if (
            weakSinceMs >= 0 &&
            tMs - weakSinceMs >= weakSatHoldMs
        ) {
            setReal(
                false,
                tMs,
                "only $satellitesUsed satellites"
            )
        }
    }

    /** The demo's "Simulate GPS loss" switch. */
    fun setSimulated(
        on: Boolean,
        tMs: Long
    ) {
        simulated = on

        update(
            tMs,
            if (on) "GPS loss simulated"
            else "simulation ended"
        )
    }

    private fun setReal(
        ok: Boolean,
        tMs: Long,
        why: String
    ) {
        realGnssOk = ok
        goodStreak = 0

        update(
            tMs,
            why
        )
    }

    private fun update(
        tMs: Long,
        why: String
    ) {
        val m =
            if (
                simulated ||
                !realGnssOk
            ) {
                GnssMode.DEAD_RECKONING
            } else {
                GnssMode.GNSS
            }

        if (m != mode) {
            mode = m
            reason = why
            switches++

            onChange(
                m,
                why,
                tMs
            )
        }
    }
}