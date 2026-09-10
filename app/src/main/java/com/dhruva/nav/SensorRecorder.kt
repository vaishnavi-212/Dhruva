package com.dhruva.nav

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Looper
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SensorRecorder(private val ctx: Context) : SensorEventListener {

    private val sm = ctx.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private var t0Nanos = 0L
    private val writers = mutableMapOf<String, BufferedWriter>()
    private val counts = mutableMapOf<String, Long>()
    private var lastFlushNanos = 0L
    private lateinit var dir: File

    /** True once stop() has run. stop() is safe to call more than once. */
    @Volatile var stopped = false
        private set

    var lastFix: Location? = null           // the UI reads this for the blue dot

    private val fused = LocationServices.getFusedLocationProviderClient(ctx)
    private val locCb = object : LocationCallback() {
        override fun onLocationResult(r: LocationResult) {
            val l = r.lastLocation ?: return
            lastFix = l
            if (stopped || t0Nanos == 0L) return   // wait for the sensor clock
            val elapsed = (l.elapsedRealtimeNanos - t0Nanos) / 1_000_000_000.0
            writers["Location"]?.apply {
                write("${l.elapsedRealtimeNanos},${"%.6f".format(elapsed)}," +
                      "${l.accuracy},${l.speed},${l.bearing},${l.altitude}," +
                      "${l.longitude},${l.latitude}\n")
                // One line a second never fills a 4 KB buffer, so without this the
                // GPS file stays EMPTY on disk until close -- exactly the 10 Sept zip.
                flush()
            }
        }
    }

    /** Sensor events received so far, for the live Hz readout. */
    fun eventCount(name: String): Long = counts[name] ?: 0L

    fun start(): File {
        val stamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US).format(Date())
        dir = File(ctx.getExternalFilesDir(null), "DhruvaRun_$stamp").apply { mkdirs() }

        // one header per file -- note the z,y,x order
        listOf("Accelerometer", "Gyroscope", "Gravity", "Magnetometer").forEach { name ->
            val w = File(dir, "$name.csv").bufferedWriter()
            w.write("time,seconds_elapsed,z,y,x\n")
            writers[name] = w
        }
        val loc = File(dir, "Location.csv").bufferedWriter()
        loc.write("time,seconds_elapsed,horizontalAccuracy,speed,bearing,altitude,longitude,latitude\n")
        writers["Location"] = loc

        t0Nanos = 0L   // set by the first sensor event, shared by ALL files

        // Requested at 100 Hz. Android treats this as a hint -- phones deliver
        // 125-150 Hz -- which is fine: the training rides were 100-125 Hz. What
        // must NOT come back is SENSOR_DELAY_FASTEST (200/400 Hz on 7 Sept), whose
        // 40x decimation aliased vibration into the model: 19.5% vs 16.0%.
        mapOf(
            Sensor.TYPE_LINEAR_ACCELERATION to "Accelerometer",   // gravity already removed
            Sensor.TYPE_GYROSCOPE to "Gyroscope",
            Sensor.TYPE_GRAVITY to "Gravity",
            Sensor.TYPE_MAGNETIC_FIELD to "Magnetometer"
        ).forEach { (type, _) ->
            sm.getDefaultSensor(type)?.let { sm.registerListener(this, it, SAMPLING_PERIOD_US) }
        }
        return dir
    }

    fun startGps() {
        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            return   // caller must request permission before calling this
        }
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
            .setMinUpdateIntervalMillis(1000L)
            .build()
        fused.requestLocationUpdates(req, locCb, Looper.getMainLooper())
    }

    override fun onSensorChanged(e: SensorEvent) {
        if (stopped) return
        if (t0Nanos == 0L) { t0Nanos = e.timestamp; lastFlushNanos = e.timestamp }
        val elapsed = (e.timestamp - t0Nanos) / 1_000_000_000.0
        val name = when (e.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> "Accelerometer"
            Sensor.TYPE_GYROSCOPE -> "Gyroscope"
            Sensor.TYPE_GRAVITY -> "Gravity"
            Sensor.TYPE_MAGNETIC_FIELD -> "Magnetometer"
            else -> return
        }
        // z,y,x -- deliberately in this order
        writers[name]?.write(
            "${e.timestamp},${"%.6f".format(elapsed)}," +
                    "${e.values[2]},${e.values[1]},${e.values[0]}\n")
        counts[name] = (counts[name] ?: 0L) + 1

        // Flush every 2 s, so a killed app or a crash loses at most 2 s -- not the
        // whole ride sitting in memory.
        if (e.timestamp - lastFlushNanos > FLUSH_EVERY_NANOS) {
            writers.values.forEach { it.flush() }
            lastFlushNanos = e.timestamp
        }
    }

    override fun onAccuracyChanged(s: Sensor?, a: Int) {}

    /**
     * Stop everything and close the files. Idempotent: MainActivity calls it
     * synchronously on Stop, and RecordingService.onDestroy() calls it again.
     */
    fun stop() {
        if (stopped) return
        stopped = true
        sm.unregisterListener(this)
        fused.removeLocationUpdates(locCb)      // previously never removed
        writers.values.forEach { it.flush(); it.close() }
        writers.clear()
    }

    companion object {
        /** 10 000 us = 100 Hz requested. */
        const val SAMPLING_PERIOD_US = 10_000
        private const val FLUSH_EVERY_NANOS = 2_000_000_000L
    }
}
