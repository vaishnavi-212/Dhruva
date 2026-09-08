package com.dhruva.nav

import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Same contract as the Python side: predict(acc, gyro, dt, init) -> (xy, sigma)
 * Deliberately crude -- it integrates yaw and assumes constant speed, so it
 * drifts. That is fine: if the UI looks right with this, it looks better with
 * the real model, and NOTHING in the UI code changes when it is swapped in.
 */
class DeadReckoner(
    private var heading: Double,   // radians, 0 = east, counter-clockwise positive
    private var speed: Double,     // m/s at the moment GNSS dropped
    private var x: Double, private var y: Double
) {
    private var distanceSinceFix = 0.0

    /** Current heading, radians, 0 = east. */
    fun heading(): Double = heading

    /**
     * Feed one IMU sample.
     *
     * @param yawRate rotation about the TRUE VERTICAL in rad/s -- not raw gyro z.
     *   Raw z is only the vertical axis when the phone lies flat. Project the
     *   gyro vector onto the gravity direction first (NavigateActivity does).
     */
    fun step(yawRate: Double, dt: Double): Triple<Double, Double, Double> {
        heading += yawRate * dt
        val stepLen = speed * dt
        x += stepLen * cos(heading)
        y += stepLen * sin(heading)
        distanceSinceFix += stepLen

        // measured on 11 real rides: 1-sigma is about 19% of distance travelled,
        // and the 90% circle is 2.146x that (Rayleigh)
        val sigma = max(0.19 * distanceSinceFix, 2.0)
        return Triple(x, y, 2.146 * sigma)
    }

    fun onGnssFix(newX: Double, newY: Double, newSpeed: Double) {
        x = newX; y = newY; speed = newSpeed
        distanceSinceFix = 0.0   // confidence collapses back to the floor
    }

    /**
     * Set the absolute heading from a GNSS bearing.
     *
     * WITHOUT THIS THE DOT RUNS OFF THE MAP. Heading used to start at 0.0 -- due
     * east -- and was only ever changed by the gyro, which measures *turns*, not
     * direction. So the instant a blackout began the estimate set off due east no
     * matter which way the vehicle was actually pointing, drawing one long
     * straight line across the map. That was the 7 Sept screenshot.
     *
     * @param bearingDeg Android's Location.bearing: degrees CLOCKWISE FROM NORTH.
     *   Our heading is counter-clockwise from east, hence 90 - bearing.
     */
    fun setHeadingFromBearing(bearingDeg: Float) {
        heading = Math.toRadians(90.0 - bearingDeg.toDouble())
    }
}
