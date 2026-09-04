package com.dhruva.nav

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import java.io.File

class RecordingService : Service() {

    private lateinit var recorder: SensorRecorder

    companion object {
        // lets MainActivity read live status without a full bind/unbind setup —
        // simplest thing that works for a two-person hackathon build
        var activeRecorder: SensorRecorder? = null
        var activeRunDir: File? = null
    }

    override fun onCreate() {
        super.onCreate()
        // API 26+ REQUIRES a notification channel or startForeground() throws
        val ch = NotificationChannel("dhruva", "Recording",
            NotificationManager.IMPORTANCE_LOW)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
            .createNotificationChannel(ch)

        val note = Notification.Builder(this, "dhruva")
            .setContentTitle("Dhruva is recording")
            .setContentText("Sensors and GPS are being logged")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .build()
        startForeground(1, note)

        recorder = SensorRecorder(this)
        val dir = recorder.start()
        recorder.startGps()

        activeRecorder = recorder
        activeRunDir = dir
    }

    override fun onDestroy() {
        recorder.stop()
        activeRecorder = null
        super.onDestroy()
    }

    override fun onBind(i: Intent?) = null
}