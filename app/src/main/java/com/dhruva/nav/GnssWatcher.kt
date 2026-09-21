package com.dhruva.nav

import android.annotation.SuppressLint
import android.content.Context
import android.location.GnssStatus
import android.location.LocationManager
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log

/**
 * The Android side of the GNSS deficit handler: feeds [GnssSwitch] with
 * satellite status from the GNSS chip, and a tick 10 times a second so it
 * notices when fixes stop arriving.
 *
 * Position fixes are fed by NavigateActivity's existing location callback.
 */
class GnssWatcher(
    ctx: Context,
    private val sw: GnssSwitch
) {
    private val lm =
        ctx.getSystemService(LocationManager::class.java)

    private val handler =
        Handler(Looper.getMainLooper())

    private var running = false

    private val status =
        object : GnssStatus.Callback() {
            override fun onSatelliteStatusChanged(
                s: GnssStatus
            ) {
                var used = 0

                for (
                i in 0 until s.satelliteCount
                ) {
                    if (s.usedInFix(i)) used++
                }

                sw.onSatellites(
                    SystemClock.elapsedRealtime(),
                    used
                )
            }
        }

    private val tick =
        object : Runnable {
            override fun run() {
                sw.tick(
                    SystemClock.elapsedRealtime()
                )

                handler.postDelayed(
                    this,
                    100
                )
            }
        }

    @SuppressLint("MissingPermission")
    // MainActivity asks for location
    // before opening Navigate
    fun start() {
        if (running) return

        running = true

        try {
            lm?.registerGnssStatusCallback(
                status,
                handler
            )
        } catch (e: SecurityException) {
            Log.w(
                "Dhruva",
                "no location permission: satellite status unavailable"
            )
        }

        handler.post(tick)
    }
    fun stop() {
        if (!running) return
        running = false
        lm?.unregisterGnssStatusCallback(status)
        handler.removeCallbacks(tick)
    }
}