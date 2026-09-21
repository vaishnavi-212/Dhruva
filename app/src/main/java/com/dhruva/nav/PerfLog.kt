package com.dhruva.nav

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Debug
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Measures what the AI speed model costs on a real phone: time per run,
 * memory, battery.
 *
 * Every [everyMs] it writes one row to perf_<date>_<time>.csv in the app's
 * files folder
 * (Android/data/com.dhruva.nav/files/) and the same row to logcat with the
 * tag "Dhruva".
 * Pull the file with: adb pull
 * /sdcard/Android/data/com.dhruva.nav/files/<name>.csv
 */
class PerfLog(
    private val ctx: Context,
    private val estimator: () -> SpeedEstimator?,
    private val blackout: () -> Boolean,
    private val everyMs: Long = 5_000L
) {

    private val handler = Handler(Looper.getMainLooper())
    private var out: BufferedWriter? = null
    private var startMs = 0L

    var file: File? = null
        private set

    private val tick = object : Runnable {
        override fun run() {
            writeRow()
            handler.postDelayed(this, everyMs)
        }
    }

    fun start() {
        if (out != null) return

        val stamp = SimpleDateFormat(
            "yyyy-MM-dd_HH-mm-ss",
            Locale.US
        ).format(Date())

        val f = File(
            ctx.getExternalFilesDir(null),
            "perf_$stamp.csv"
        )

        out = f.bufferedWriter().also {
            it.write(
                "wall_ms,elapsed_s,blackout,ai_active,inferences," +
                        "infer_ms_last,infer_ms_mean,infer_ms_max," +
                        "battery_pct,current_ua,voltage_mv,temp_c," +
                        "java_heap_mb,native_heap_mb,pss_mb\n"
            )
        }

        file = f
        startMs = System.currentTimeMillis()

        handler.post(tick)

        Log.i(
            "Dhruva",
            "perf log -> ${f.absolutePath}"
        )
    }

    fun stop() {
        handler.removeCallbacks(tick)

        out?.let {
            it.flush()
            it.close()
        }

        out = null
    }

    private fun writeRow() {
        val w = out ?: return

        val now = System.currentTimeMillis()
        val e = estimator()

        val n = e?.inferences ?: 0L

        val mean =
            if (e != null && n > 0) {
                e.totalInferMs / n
            } else {
                0.0
            }

        val bm = ctx.getSystemService(BatteryManager::class.java)

        val pct =
            bm?.getIntProperty(
                BatteryManager.BATTERY_PROPERTY_CAPACITY
            ) ?: -1

        val currentUa =
            bm?.getIntProperty(
                BatteryManager.BATTERY_PROPERTY_CURRENT_NOW
            ) ?: 0 // sign differs by vendor

        val sticky = ctx.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )

        val voltage =
            sticky?.getIntExtra(
                BatteryManager.EXTRA_VOLTAGE,
                -1
            ) ?: -1

        val tempC =
            (
                    sticky?.getIntExtra(
                        BatteryManager.EXTRA_TEMPERATURE,
                        -10
                    ) ?: -10
                    ) / 10.0

        val rt = Runtime.getRuntime()

        val javaMb =
            (rt.totalMemory() - rt.freeMemory()) /
                    1_048_576.0

        val nativeMb =
            Debug.getNativeHeapAllocatedSize() /
                    1_048_576.0

        // ONNX Runtime lives here
        val info = Debug.MemoryInfo().also {
            Debug.getMemoryInfo(it)
        }

        val pssMb =
            info.totalPss / 1024.0

        // the app's real share of RAM
        val row =
            "%d,%.1f,%d,%d,%d,%.2f,%.2f,%.2f,%d,%d,%d,%.1f,%.1f,%.1f,%.1f".format(
                Locale.US,
                now,
                (now - startMs) / 1000.0,
                if (blackout()) 1 else 0,
                if (e?.active == true) 1 else 0,
                n,
                e?.lastInferMs ?: 0.0,
                mean,
                e?.maxInferMs ?: 0.0,
                pct,
                currentUa,
                voltage,
                tempC,
                javaMb,
                nativeMb,
                pssMb
            )

        w.write(row)
        w.write("\n")
        w.flush()

        Log.i("Dhruva", "perf $row")
    }
}