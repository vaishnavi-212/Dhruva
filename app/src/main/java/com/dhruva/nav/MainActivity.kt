package com.dhruva.nav

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationServices
import java.io.File

class MainActivity : AppCompatActivity() {

    private val askPerms = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { granted ->
        if (granted[Manifest.permission.ACCESS_FINE_LOCATION] != true) {
            Toast.makeText(this, "Location permission is required to record", Toast.LENGTH_LONG).show()
        }
    }

    private lateinit var btnStartStop: Button
    private lateinit var btnShare: Button
    private lateinit var tvGpsStatus: TextView
    private lateinit var tvHz: TextView
    private lateinit var tvElapsed: TextView
    private lateinit var tvDistance: TextView

    private lateinit var btnNavigate: Button

    private var isRecording = false
    private var recordingStartMs = 0L
    private var finishedRunDir: File? = null

    private val prevLineCounts = mutableMapOf<String, Long>()
    private var prevLocation: android.location.Location? = null
    private var totalDistanceM = 0.0

    private val handler = Handler(Looper.getMainLooper())
    private val statusUpdater = object : Runnable {
        override fun run() {
            updateLiveStatus()
            handler.postDelayed(this, 1000L)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        askPerms.launch(arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        ))

        btnStartStop = findViewById(R.id.btnStartStop)
        btnShare = findViewById(R.id.btnShare)
        btnNavigate = findViewById(R.id.btnNavigate)
        tvGpsStatus = findViewById(R.id.tvGpsStatus)
        tvHz = findViewById(R.id.tvHz)
        tvElapsed = findViewById(R.id.tvElapsed)
        tvDistance = findViewById(R.id.tvDistance)

        btnShare.isEnabled = latestRunDir() != null

        btnStartStop.setOnClickListener {
            if (!isRecording) attemptStart() else stopRecording()
        }

        btnShare.setOnClickListener {
            if (isRecording) {
                // Zipping a folder that is still being written produces truncated
                // files -- the 10 Sept zip. Stop first.
                Toast.makeText(this, "Stop recording first — the files are still being written",
                    Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            // finishedRunDir is in-memory only, so it is null on every fresh
            // launch -- and `?.let {}` then did NOTHING, silently. That is the
            // "Share Last Run does nothing" bug: the button worked, there was
            // just no run in memory to share.
            val dir = finishedRunDir ?: latestRunDir()
            if (dir == null) {
                Toast.makeText(this, "No recorded runs found", Toast.LENGTH_SHORT).show()
            } else {
                try {
                    shareRun(this, dir)
                } catch (e: Exception) {
                    Toast.makeText(this, "Share failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
        btnNavigate.setOnClickListener {
            startActivity(Intent(this, NavigateActivity::class.java))
        }
    }

    override fun onResume() {
        super.onResume()
        // The activity can be rebuilt while the service keeps recording (rotation,
        // memory pressure, returning from Navigate). Trust the service, not our field.
        val rec = RecordingService.activeRecorder
        if (!isRecording && rec != null && !rec.stopped) {
            isRecording = true
            recordingStartMs = RecordingService.startedAtMs
            btnStartStop.text = "Stop Recording"
            handler.removeCallbacks(statusUpdater)
            handler.post(statusUpdater)
        } else if (isRecording && rec != null && rec.stopped) {
            isRecording = false
            btnStartStop.text = "Start Recording"
            handler.removeCallbacks(statusUpdater)
        }
        btnShare.isEnabled = !isRecording && (finishedRunDir ?: latestRunDir()) != null
    }

    /** Newest DhruvaRun_* folder on disk. Survives app restarts. */
    private fun latestRunDir(): File? =
        getExternalFilesDir(null)
            ?.listFiles { f -> f.isDirectory && f.name.startsWith("DhruvaRun_") }
            ?.maxByOrNull { it.name }

    private fun attemptStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Location permission not granted yet", Toast.LENGTH_SHORT).show()
            return
        }

        val fused = LocationServices.getFusedLocationProviderClient(this)
        fused.lastLocation.addOnSuccessListener { loc ->
            if (loc == null) {
                Toast.makeText(this, "No GPS fix yet — step outside and try again", Toast.LENGTH_LONG).show()
            } else {
                startRecording()
            }
        }
    }

    private fun startRecording() {
        startForegroundService(Intent(this, RecordingService::class.java))
        isRecording = true
        recordingStartMs = System.currentTimeMillis()
        prevLineCounts.clear()
        prevLocation = null
        totalDistanceM = 0.0
        finishedRunDir = null
        btnStartStop.text = "Stop Recording"
        btnShare.isEnabled = false
        handler.post(statusUpdater)
    }

    private fun stopRecording() {
        finishedRunDir = RecordingService.activeRunDir
        // Flush and close NOW. stopService() only schedules the service's
        // onDestroy(); a Share pressed in that gap zipped half-written files --
        // on 10 Sept, 41 s of IMU cut mid-line and an empty Location.csv.
        // stop() is idempotent, so onDestroy() calling it again is harmless.
        RecordingService.activeRecorder?.stop()
        stopService(Intent(this, RecordingService::class.java))
        isRecording = false
        btnStartStop.text = "Start Recording"
        handler.removeCallbacks(statusUpdater)
        btnShare.isEnabled = (finishedRunDir ?: latestRunDir()) != null
    }

    private fun updateLiveStatus() {
        val rec = RecordingService.activeRecorder ?: return

        // Hz from the recorder's own event counters. This used to re-read every CSV
        // in full once a second -- megabytes per tick on the main thread by the end
        // of a long ride, the same thread that receives the sensor events.
        val sensorFiles = listOf("Accelerometer", "Gyroscope", "Gravity", "Magnetometer")
        val hzText = StringBuilder()
        sensorFiles.forEach { name ->
            val count = rec.eventCount(name)
            val prev = prevLineCounts[name] ?: count
            prevLineCounts[name] = count
            hzText.append("$name: ${(count - prev).coerceAtLeast(0L)} Hz   ")
        }
        tvHz.text = hzText.toString().ifBlank { "Accel: -- Hz" }

        // GPS status + rolling distance
        val fix = rec.lastFix
        if (fix != null) {
            tvGpsStatus.text = "GPS: locked, accuracy ${"%.1f".format(fix.accuracy)} m"

            val prev = prevLocation
            if (prev == null) {
                prevLocation = fix
            } else {
                val moved = prev.distanceTo(fix)
                // cap accuracy's influence: never block movement above 8m,
                // never allow jitter under 3m through — this was the bug:
                // using raw 15m accuracy as the floor blocked all real movement
                val noiseFloor = fix.accuracy.coerceIn(3f, 8f)
                if (moved > noiseFloor) {
                    totalDistanceM += moved
                    prevLocation = fix
                }
            }
        } else {
            tvGpsStatus.text = "GPS: waiting for fix..."
        }

        // elapsed time
        val elapsedSec = (System.currentTimeMillis() - recordingStartMs) / 1000
        tvElapsed.text = "Elapsed: %02d:%02d".format(elapsedSec / 60, elapsedSec % 60)

        // distance
        tvDistance.text = "Distance: %.1f m".format(totalDistanceM)
    }
}