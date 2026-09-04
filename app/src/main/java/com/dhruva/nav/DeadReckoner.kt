package com.dhruva.nav

import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Same contract as the Python side: predict(acc, gyro, dt, init) -> (xy, sigma)
 * Deliberately crude — it integrates gyro heading and assumes constant speed,
 * so it drifts. That is fine: if the UI looks right with this, it looks better
 * with the real model, and NOTHING in the UI code changes when it's swapped in.
 */
class DeadReckoner(
    private var heading: Double,   // radians, 0 = east
    private var speed: Double,     // m/s at the moment GNSS dropped
    private var x: Double, private var y: Double
) {
    private var distanceSinceFix = 0.0

    /** Feed one IMU sample. Returns position and an HONEST confidence radius. */
    fun step(gyroZ: Double, dt: Double): Triple<Double, Double, Double> {
        heading += gyroZ * dt
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
}