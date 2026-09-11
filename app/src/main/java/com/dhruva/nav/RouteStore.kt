package com.dhruva.nav

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.cos
import kotlin.math.hypot

/**
 * Routes the phone LEARNS by driving them once with GPS.
 *
 * Road binding needs the road you are on. Until now that meant riding a route,
 * sending the file to a laptop, snapping it to OpenStreetMap, rebuilding the app
 * and reinstalling -- for every new road. Now: ride the circuit once with GPS on,
 * tap "Save track as route", and the next ride can bind to it with GNSS off.
 *
 * The saved polyline is the phone's own GPS track, cleaned. At 2-5 m accuracy on
 * a moving vehicle that is as good a road centre-line as OSM's, and it is the
 * road actually ridden -- including campus lanes OSM draws coarsely or not at all.
 *
 * Learned routes live in the app's private storage and are listed before the
 * stored ones in the route picker.
 */
object RouteStore {

    private const val FILE = "learned_routes.json"
    private const val M_PER_DEG = 111320.0

    /** A 1 Hz fix further than this from the last kept one is a GPS spike (216 km/h). */
    private const val SPIKE_M = 60.0
    /** Closer than this adds nothing but points. */
    private const val MIN_SPACING_M = 2.0
    const val MIN_ROUTE_M = 200.0

    data class Saved(val name: String, val lengthM: Double, val points: Int, val closedLoop: Boolean)

    /**
     * Clean a GPS track and save it as a route. Returns null, with nothing saved,
     * if the track is too short to be worth binding to.
     */
    fun saveTrack(ctx: Context, track: List<Pair<Double, Double>>): Saved? {
        if (track.size < 20) return null
        val lat0 = track.first().first
        val k = M_PER_DEG * cos(Math.toRadians(lat0))
        fun d(a: Pair<Double, Double>, b: Pair<Double, Double>) =
            hypot((b.second - a.second) * k, (b.first - a.first) * M_PER_DEG)

        val kept = ArrayList<Pair<Double, Double>>(track.size)
        for (p in track) {
            val last = kept.lastOrNull()
            if (last == null) { kept.add(p); continue }
            val step = d(last, p)
            if (step > SPIKE_M) continue          // spike: skip it, keep the previous anchor
            if (step < MIN_SPACING_M) continue
            kept.add(p)
        }
        if (kept.size < 10) return null
        var length = 0.0
        for (i in 1 until kept.size) length += d(kept[i - 1], kept[i])
        if (length < MIN_ROUTE_M) return null

        val name = "learned " + SimpleDateFormat("dd MMM HH:mm", Locale.US).format(Date())
        val pts = JSONArray()
        kept.forEach { pts.put(JSONArray().put(round7(it.first)).put(round7(it.second))) }
        val route = JSONObject()
            .put("name", name).put("length_m", Math.round(length * 10) / 10.0)
            .put("n_points", kept.size).put("source", "phone GPS track").put("points", pts)

        val doc = readLearned(ctx)
        val arr = doc.getJSONArray("routes")
        val updated = JSONArray().put(route)                 // newest first
        for (i in 0 until arr.length()) updated.put(arr.getJSONObject(i))
        doc.put("routes", updated)
        File(ctx.filesDir, FILE).writeText(doc.toString())

        return Saved(name, length, kept.size, d(kept.first(), kept.last()) < 50.0)
    }

    /** Learned routes first, then the ones shipped in assets/routes.json. */
    fun combinedJson(ctx: Context): String {
        val out = JSONArray()
        val learned = readLearned(ctx).getJSONArray("routes")
        for (i in 0 until learned.length()) out.put(learned.getJSONObject(i))
        try {
            val shipped = JSONObject(ctx.assets.open("routes.json").bufferedReader().use { it.readText() })
                .getJSONArray("routes")
            for (i in 0 until shipped.length()) out.put(shipped.getJSONObject(i))
        } catch (e: Exception) { /* no shipped routes: learned ones still work */ }
        return JSONObject().put("routes", out).toString()
    }

    fun learnedCount(ctx: Context): Int = readLearned(ctx).getJSONArray("routes").length()

    private fun readLearned(ctx: Context): JSONObject {
        val f = File(ctx.filesDir, FILE)
        return try {
            if (f.exists()) JSONObject(f.readText()) else JSONObject().put("routes", JSONArray())
        } catch (e: Exception) {
            JSONObject().put("routes", JSONArray())      // corrupt file: start clean rather than crash
        }
    }

    private fun round7(v: Double) = Math.round(v * 1e7) / 1e7
}
