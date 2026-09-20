package com.dhruva.nav

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * trip_summary.json: everything the Finish Run card shows, plus the paths,
 * saved next to the ride's sensor files so a shared zip can be scored on the laptop with no
 * screenshots.
 *
 * Built by hand (no org.json) so the unit test runs on the laptop. Fields
 * other people fill later:
 * reacquire_jump_m J2, Part 7: how far the dot jumped when GPS came back
 * destination, planned_route, wrong_turn_alarms Nakul's destination flow
 */
object TripSummary {
    const val FORMAT = "dhruva-trip/1"

    data class Trip(
        val createdMs: Long,
        val appVersion: String,
        val phone: String,
        val android: String,
        val blackoutStartMs: Long,
        val blackoutEndMs: Long?, // null = blackout still running
        // when Finish Run was pressed
        val blackoutDurationS: Double,
        val result: RunSummary.Result,
        val verdict: String,
        val aiShare: Double?, // fraction of blackout steps that
        // used the AI speed, 0..1
        val modelRuns: Long,
        val inferMsMean: Double,
        val inferMsMax: Double,
        val roadBinding: Boolean,
        val routeName: String?,
        val truthPath: List<Pair<Double, Double>>, // GPS during the
        // blackout, (lat, lon)
        val predictedPath: List<Pair<Double, Double>>, // Dhruva's dot
        val reacquireJumpM: Double? = null
    )

    /** Where the file goes: the ride being recorded if there is one, else
     * files/trips/. */
    fun folderFor(ctx: Context): File {
        val rec = RecordingService.activeRecorder
        val run = RecordingService.activeRunDir

        if (rec != null && !rec.stopped && run != null && run.isDirectory) {
            return run
        }

        return File(
            ctx.getExternalFilesDir(null),
            "trips"
        ).apply {
            mkdirs()
        }
    }

    fun save(t: Trip, dir: File): File {
        val name =
            if (dir.name.startsWith("DhruvaRun_")) {
                "trip_summary.json"
            } else {
                "trip_" +
                        SimpleDateFormat(
                            "yyyy-MM-dd_HH-mm-ss",
                            Locale.US
                        ).format(Date(t.createdMs)) +
                        ".json"
            }

        val f = File(dir, name)
        f.writeText(toJson(t))
        return f
    }

    fun toJson(t: Trip): String {
        val r = t.result

        val iso = SimpleDateFormat(
            "yyyy-MM-dd'T'HH:mm:ss'Z'",
            Locale.US
        ).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }

        val sb = StringBuilder()

        sb.append("{\n")
        sb.append(" \"format\": ").append(str(FORMAT)).append(",\n")
        sb.append(" \"created_utc\": ")
            .append(str(iso.format(Date(t.createdMs))))
            .append(",\n")
        sb.append(" \"app_version\": ")
            .append(str(t.appVersion))
            .append(",\n")

        sb.append(" \"phone\": {\"model\": ")
            .append(str(t.phone))
            .append(", \"android\": ")
            .append(str(t.android))
            .append("},\n")

        sb.append(" \"blackout\": {\"start_ms\": ")
            .append(t.blackoutStartMs)
            .append(", \"end_ms\": ")
            .append(t.blackoutEndMs?.toString() ?: "null")
            .append(", \"duration_s\": ")
            .append(num(t.blackoutDurationS))
            .append(", \"session_s\": ")
            .append(num(r.durationS))
            .append("},\n")

        sb.append(" \"result\": {\"verdict\": ")
            .append(str(t.verdict))
            .append(", \"passes\": ")
            .append(r.passes)
            .append(", \"distance_m\": ")
            .append(num(r.distanceM))
            .append(", \"final_error_m\": ")
            .append(num(r.finalErrorM))
            .append(", \"drift_pct\": ")
            .append(num(r.driftPct))
            .append(", \"mean_speed_kmh\": ")
            .append(num(r.meanSpeedMps * 3.6))
            .append(", \"confidence90_m\": ")
            .append(num(r.confidence90M))
            .append(", \"gps_fixes\": ")
            .append(r.gpsFixes)
            .append(", \"imu_hz\": ")
            .append(num(r.imuHz))
            .append("},\n")

        sb.append(" \"speed\": {\"ai_share\": ")
            .append(
                t.aiShare?.let {
                    num(it)
                } ?: "null"
            )
            .append(", \"model_runs\": ")
            .append(t.modelRuns)
            .append(", \"infer_ms_mean\": ")
            .append(num(t.inferMsMean))
            .append(", \"infer_ms_max\": ")
            .append(num(t.inferMsMax))
            .append(", \"model_file\": \"speed_app.onnx\"},\n")

        sb.append(" \"road\": {\"binding_on\": ")
            .append(t.roadBinding)
            .append(", \"route\": ")
            .append(
                t.routeName?.let {
                    str(it)
                } ?: "null"
            )
            .append("},\n")

        sb.append(" \"reacquire_jump_m\": ")
            .append(
                t.reacquireJumpM?.let {
                    num(it)
                } ?: "null"
            )
            .append(",\n")

        sb.append(" \"destination\": null,\n")
        sb.append(" \"planned_route\": null,\n")
        sb.append(" \"wrong_turn_alarms\": [],\n")

        sb.append(" \"truth_path\": ")
            .append(path(t.truthPath))
            .append(",\n")

        sb.append(" \"predicted_path\": ")
            .append(path(t.predictedPath))
            .append("\n")

        sb.append("}\n")

        return sb.toString()
    }

    private fun num(v: Double) =
        if (v.isNaN() || v.isInfinite()) {
            "null"
        } else {
            String.format(
                Locale.US,
                "%.3f",
                v
            )
        }

    private fun path(p: List<Pair<Double, Double>>) =
        p.joinToString(
            ", ",
            "[",
            "]"
        ) {
            String.format(
                Locale.US,
                "[%.7f, %.7f]",
                it.first,
                it.second
            )
        }

    private fun str(s: String): String {
        val b = StringBuilder("\"")

        for (c in s) {
            when (c) {
                '"' -> b.append("\\\"")
                '\\' -> b.append("\\\\")
                '\n' -> b.append("\\n")
                '\r' -> b.append("\\r")
                '\t' -> b.append("\\t")
                else ->
                    if (c < ' ') {
                        b.append(
                            String.format(
                                Locale.US,
                                "\\u%04x",
                                c.code
                            )
                        )
                    } else {
                        b.append(c)
                    }
            }
        }

        return b.append('"').toString()
    }
}