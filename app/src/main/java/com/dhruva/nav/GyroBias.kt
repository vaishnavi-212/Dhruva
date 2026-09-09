package com.dhruva.nav

/**
 * Estimates and removes the gyroscope's zero-rate offset.
 *
 * A MEMS gyro reads a small non-zero rate when perfectly still, and the size of
 * that offset is a property of the individual chip. Measured on our own two
 * phones: 0.010 deg/s on one, 0.707 deg/s on the other. Integrated for 30
 * seconds that is 0.3 degrees of heading error on the first phone and 21 on the
 * second -- the difference between a clean demo and a visibly crooked one.
 *
 * You cannot read it off a datasheet and you cannot assume it. It has to be
 * measured, on the device, every session.
 *
 * HOW: average the gyro while GNSS says we are stopped, then freeze that average
 * when the blackout starts. It needs no cooperation from the rider -- every ride
 * begins stationary -- and it uses only what is available BEFORE the outage,
 * which is the same principle the speed and heading now follow.
 */
class GyroBias {

    private var sumX = 0.0; private var sumY = 0.0; private var sumZ = 0.0
    private var n = 0

    private var bx = 0.0; private var by = 0.0; private var bz = 0.0

    /** True once enough stationary samples have been seen to trust the estimate. */
    var ready = false
        private set

    /** Frozen once the blackout starts -- no GNSS then, so no way to know we are still. */
    var frozen = false
        private set

    /**
     * Feed a gyro sample. Only pass `stationary = true` when GNSS speed says so;
     * a moving vehicle turning would otherwise be averaged in as "bias".
     */
    fun observe(x: Float, y: Float, z: Float, stationary: Boolean) {
        if (frozen || !stationary) return
        sumX += x; sumY += y; sumZ += z; n++
        if (n >= MIN_SAMPLES) {
            bx = sumX / n; by = sumY / n; bz = sumZ / n
            ready = true
        }
    }

    fun freeze() { frozen = true }

    fun reset() {
        sumX = 0.0; sumY = 0.0; sumZ = 0.0; n = 0
        bx = 0.0; by = 0.0; bz = 0.0
        ready = false; frozen = false
    }

    fun correctX(v: Float) = v - bx.toFloat()
    fun correctY(v: Float) = v - by.toFloat()
    fun correctZ(v: Float) = v - bz.toFloat()

    /** Magnitude in deg/s, for the status line. Above ~0.5 is a phone worth avoiding. */
    fun magnitudeDegPerSec(): Double =
        Math.toDegrees(Math.sqrt(bx * bx + by * by + bz * bz))

    fun sampleCount() = n

    companion object {
        /** ~3 s at 100 Hz. Enough to average out noise, short enough to get at a red light. */
        const val MIN_SAMPLES = 300
    }
}
