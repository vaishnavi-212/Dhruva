package com.dhruva.nav

import android.graphics.Color
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Polyline

/**
 * Draws the two paths that ARE the demo: where GPS says we went (solid) and
 * where dead reckoning thinks we went (dotted).
 *
 * While GNSS is healthy the two sit on top of each other. The instant the
 * blackout toggle is flipped, the dotted line starts its own life and the gap
 * between them grows. That divergence, watched live, explains the whole project
 * faster than any slide.
 *
 * Usage in NavigateActivity:
 *
 *     private val paths = LivePathOverlay(map)
 *     // every GPS fix, whether or not we are in blackout:
 *     paths.addTruth(lat, lon)
 *     // every DeadReckoner step:
 *     paths.addPredicted(lat, lon)
 *     // when the toggle changes:
 *     paths.setBlackout(on)
 */
class LivePathOverlay(private val map: MapView) {

    private val truth = Polyline(map).apply {
        outlinePaint.color = Color.parseColor("#2471A3")   // solid blue
        outlinePaint.strokeWidth = 9f
    }
    private val predicted = Polyline(map).apply {
        outlinePaint.color = Color.parseColor("#E67E22")   // amber
        outlinePaint.strokeWidth = 7f
        // dashes: 18 px on, 14 px off. Colour alone is not enough on a projector.
        outlinePaint.pathEffect = android.graphics.DashPathEffect(floatArrayOf(18f, 14f), 0f)
    }
    /** straight line between the two current heads — makes the error legible */
    private val gap = Polyline(map).apply {
        outlinePaint.color = Color.parseColor("#C0392B")
        outlinePaint.strokeWidth = 4f
    }

    private var lastTruth: GeoPoint? = null
    private var lastPred: GeoPoint? = null
    private var blackout = false

    init {
        map.overlays.add(truth)
        map.overlays.add(predicted)
        map.overlays.add(gap)
    }

    fun addTruth(lat: Double, lon: Double) {
        val p = GeoPoint(lat, lon)
        lastTruth = p
        truth.addPoint(p)
        refreshGap()
    }

    fun addPredicted(lat: Double, lon: Double) {
        val p = GeoPoint(lat, lon)
        lastPred = p
        predicted.addPoint(p)
        refreshGap()
    }

    fun setBlackout(on: Boolean) {
        blackout = on
        // amber while dead reckoning, green while it is merely shadowing GNSS
        predicted.outlinePaint.color =
            Color.parseColor(if (on) "#E67E22" else "#1E8449")
        map.invalidate()
    }

    /** Current separation in metres — show it as a live number next to the map. */
    fun currentErrorMetres(): Double {
        val t = lastTruth ?: return 0.0
        val p = lastPred ?: return 0.0
        return t.distanceToAsDouble(p)
    }

    private fun refreshGap() {
        val t = lastTruth; val p = lastPred
        gap.setPoints(if (blackout && t != null && p != null) listOf(t, p) else emptyList())
        map.invalidate()
    }

    fun clear() {
        truth.setPoints(emptyList()); predicted.setPoints(emptyList()); gap.setPoints(emptyList())
        lastTruth = null; lastPred = null
        map.invalidate()
    }
}
