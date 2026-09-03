package com.dhruva.nav

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SensorRecorder(private val ctx: Context) : SensorEventListener {

    private val sm = ctx.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private var t0Nanos = 0L
    private val writers = mutableMapOf<String, BufferedWriter>()
    private lateinit var dir: File

    var lastFix: Location? = null           // the UI reads this for the blue dot
    val hz = mutableMapOf<String, Int>()    // events per second, per sensor, for the display

    fun start(): File {
        val stamp = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.US).format(Date())
        dir = File(ctx.getExternalFilesDir(null), "DhruvaRun_$stamp").apply { mkdirs() }

        // one header per file — note the z,y,x order
        listOf("Accelerometer", "Gyroscope", "Gravity", "Magnetometer").forEach { name ->
            val w = File(dir, "$name.csv").bufferedWriter()
            w.write("time,seconds_elapsed,z,y,x\n")
            writers[name] = w
        }

        val loc = File(dir, "Location.csv").bufferedWriter()
        loc.write("time,seconds_elapsed,horizontalAccuracy,speed,bearing,altitude,longitude,latitude\n")
        writers["Location"] = loc

        t0Nanos = 0L   // set by the first sensor event, shared by ALL files

        // SENSOR_DELAY_FASTEST asks for the highest rate the hardware offers
        mapOf(
            Sensor.TYPE_LINEAR_ACCELERATION to "Accelerometer",   // gravity already removed
            Sensor.TYPE_GYROSCOPE to "Gyroscope",
            Sensor.TYPE_GRAVITY to "Gravity",
            Sensor.TYPE_MAGNETIC_FIELD to "Magnetometer"
        ).forEach { (type, _) ->
            sm.getDefaultSensor(type)?.let {
                sm.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
            }
        }

        return dir
    }

    override fun onSensorChanged(e: SensorEvent) {
        if (t0Nanos == 0L) t0Nanos = e.timestamp   // the shared clock origin
        val elapsed = (e.timestamp - t0Nanos) / 1_000_000_000.0
        val name = when (e.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> "Accelerometer"
            Sensor.TYPE_GYROSCOPE -> "Gyroscope"
            Sensor.TYPE_GRAVITY -> "Gravity"
            Sensor.TYPE_MAGNETIC_FIELD -> "Magnetometer"
            else -> return
        }
        // z,y,x — deliberately in this order
        writers[name]?.write(
            "${e.timestamp},${"%.6f".format(elapsed)}," +
                    "${e.values[2]},${e.values[1]},${e.values[0]}\n")
    }

    override fun onAccuracyChanged(s: Sensor?, a: Int) {}

    fun stop() {
        sm.unregisterListener(this)
        writers.values.forEach { it.flush(); it.close() }
        writers.clear()
    }
}