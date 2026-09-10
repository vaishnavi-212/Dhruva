package com.dhruva.nav

/**
 * Estimates and removes the gyroscope's zero-rate offset.
 *
 * A MEMS gyro reads a small non-zero rate when perfectly still, and the size of
 * that offset belongs to the individual chip: 0.010 deg/s on one of our phones,
 * 0.707 deg/s on the other -- 0.3 versus 21 degrees of heading error over 30 s.
 * It has to be measured on the device, every session.
 *
 * HOW: average the gyro while GNSS says we are stopped; lock the average when a
 * blackout starts (nothing can tell us we are stopped once GNSS is gone); unlock
 * when GNSS returns so it keeps refining.
 *
 * 10 Sept lesson: the first blackout came before 3 s of standing still, the old
 * version locked an EMPTY estimate, never unlocked, and showed "locked: 0.000"
 * for the rest of the session -- which reads as a measurement and was not one.
 * Now an unmeasured bias says so.
 */
class GyroBias {

    private var sumX = 0.0; private var sumY = 0.0; private var sumZ = 0.0
    private var n = 0
    private var bx = 0.0; private var by = 0.0; private var bz = 0.0

    /** Enough stationary samples to trust the estimate. */
    var ready = false
        private set

    /** Locked for the duration of a blackout. */
    var frozen = false
        private set

    /** Pass stationary = true ONLY when GNSS speed says so, and never during a blackout. */
    fun observe(x: Float, y: Float, z: Float, stationary: Boolean) {
        if (frozen || !stationary) return
        sumX += x; sumY += y; sumZ += z; n++
        if (n >= MIN_SAMPLES) {
            bx = sumX / n; by = sumY / n; bz = sumZ / n
            ready = true
        }
    }

    /** Blackout starts. If not yet measured, correction stays zero -- and the status says so. */
    fun freeze() { frozen = true }

    /** GNSS is back: keep learning. */
    fun unfreeze() { frozen = false }

    fun correctX(v: Float) = if (ready) v - bx.toFloat() else v
    fun correctY(v: Float) = if (ready) v - by.toFloat() else v
    fun correctZ(v: Float) = if (ready) v - bz.toFloat() else v

    fun magnitudeDegPerSec(): Double = Math.toDegrees(Math.sqrt(bx * bx + by * by + bz * bz))

    fun sampleCount() = n

    fun statusText(): String = when {
        ready && frozen -> "gyro bias locked: %.3f deg/s".format(magnitudeDegPerSec())
        ready           -> "gyro bias: %.3f deg/s".format(magnitudeDegPerSec())
        frozen          -> "gyro bias NOT measured — stop 3 s before cutting GPS"
        else            -> "gyro bias: measuring %d/%d — stop the vehicle".format(n, MIN_SAMPLES)
    }

    companion object {
        /** ~3 s of gyro at SENSOR_DELAY_GAME. */
        const val MIN_SAMPLES = 150
    }
}
