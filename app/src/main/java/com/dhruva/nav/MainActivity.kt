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
import kotlin.jvm.java

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

    private val prevLineCounts = mutableMapOf<String, Int>()
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

        btnStartStop.setOnClickListener {
            if (!isRecording) attemptStart() else stopRecording()
        }

        btnShare.setOnClickListener {
            finishedRunDir?.let { shareRun(this, it) }
        }
        btnNavigate.setOnClickListener {
            startActivity(Intent(this, NavigateActivity::class.java))
        }
    }

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
        stopService(Intent(this, RecordingService::class.java))
        isRecording = false
        btnStartStop.text = "Start Recording"
        handler.removeCallbacks(statusUpdater)
        btnShare.isEnabled = finishedRunDir != null
    }

    private fun updateLiveStatus() {
        val rec = RecordingService.activeRecorder ?: return
        val dir = RecordingService.activeRunDir

        // Hz — count new CSV lines written since the last poll, one second ago
        val sensorFiles = listOf("Accelerometer", "Gyroscope", "Gravity", "Magnetometer")
        val hzText = StringBuilder()
        if (dir != null) {
            sensorFiles.forEach { name ->
                val f = File(dir, "$name.csv")
                val lines = if (f.exists()) f.readLines().size else 0
                val prev = prevLineCounts[name] ?: 0
                val hz = (lines - prev).coerceAtLeast(0)
                prevLineCounts[name] = lines
                hzText.append("$name: $hz Hz   ")
            }
        }
        tvHz.text = hzText.toString().ifBlank { "Accel: -- Hz" }

        // GPS status + rolling distance
        val fix = rec.lastFix
        if (fix != null) {
            tvGpsStatus.text = "GPS: locked, accuracy ${"%.1f".format(fix.accuracy)} m"
            prevLocation?.let { totalDistanceM += it.distanceTo(fix) }
            prevLocation = fix
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