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

    /**
     * The predicted line stays EMPTY until this is called.
     *
     * Contract A, rule 2: "xy[0] must equal init['xy']". If the reckoner emits a
     * position before it has been initialised from a real GNSS fix, that first
     * point sits at whatever the default was -- and the polyline draws a single
     * straight line from there to the real track, right across the map. It looks
     * like the estimate ran away. It did not; it was never started.
     *
     * Call this once, on the first GPS fix with usable accuracy, with the same
     * position you initialise the DeadReckoner from.
     */
    fun startPredicted(lat: Double, lon: Double) {
        armed = true
        val p = GeoPoint(lat, lon)
        lastPred = p
        predicted.addPoint(p)
        map.invalidate()
    }

    private var armed = false

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
        if (!armed) return                       // not initialised yet -- drop it
        val p = GeoPoint(lat, lon)

        // Safety net, not the fix: at 10 Hz nothing on a road moves 100 m between
        // two samples. A jump that large is a bad frame, never a real position.
        // If this ever fires, the real bug is upstream -- log it, do not live with it.
        lastPred?.let {
            if (it.distanceToAsDouble(p) > MAX_STEP_M) { dropped++; return }
        }

        lastPred = p
        predicted.addPoint(p)
        refreshGap()
    }

    /** How many impossible jumps were rejected. Should be 0. Show it while testing. */
    var dropped = 0
        private set

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

    companion object {
        /** Largest believable move between two predicted samples. */
        const val MAX_STEP_M = 100.0
    }

    private fun refreshGap() {
        val t = lastTruth; val p = lastPred
        gap.setPoints(if (blackout && t != null && p != null) listOf(t, p) else emptyList())
        map.invalidate()
    }

    fun clear() {
        truth.setPoints(emptyList()); predicted.setPoints(emptyList()); gap.setPoints(emptyList())
        lastTruth = null; lastPred = null; armed = false; dropped = 0
        map.invalidate()
    }
}
