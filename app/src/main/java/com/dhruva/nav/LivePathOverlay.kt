package com.dhruva.nav

import android.graphics.Color
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Polyline

/**
 * Draws the two paths that ARE the demo: where GPS says we went (solid blue) and
 * where dead reckoning thinks we went (dashed amber).
 *
 * EVERY BLACKOUT GETS ITS OWN LINE. On 10 Sept three blackouts in one session were
 * drawn as one polyline, so the end of each was joined to the start of the next
 * by a long straight dashed segment that looked like the estimate had jumped.
 * Earlier blackouts stay on the map, dimmed, so the live one is unambiguous.
 *
 *     paths.addTruth(lat, lon)          // every GPS fix, blackout or not
 *     paths.startPredicted(lat, lon)    // at the start of EACH blackout
 *     paths.addPredicted(lat, lon)      // every dead-reckoning step
 *     paths.setBlackout(on)             // when the toggle changes
 */
class LivePathOverlay(private val map: MapView) {

    private val truth = Polyline(map).apply {
        outlinePaint.color = Color.parseColor("#2471A3")
        outlinePaint.strokeWidth = 9f
    }
    /** straight line between the two current heads -- makes the error legible */
    private val gap = Polyline(map).apply {
        outlinePaint.color = Color.parseColor("#C0392B")
        outlinePaint.strokeWidth = 4f
    }

    private val segments = mutableListOf<Polyline>()
    private var current: Polyline? = null

    private var lastTruth: GeoPoint? = null
    private var lastPred: GeoPoint? = null
    private var blackout = false
    private var armed = false

    /** Impossible jumps rejected. Must read 0; shown on screen as BAD FRAMES. */
    var dropped = 0
        private set

    init {
        map.overlays.add(truth)
        map.overlays.add(gap)
    }

    private fun newSegment() = Polyline(map).apply {
        outlinePaint.color = Color.parseColor("#E67E22")
        outlinePaint.strokeWidth = 7f
        // dashes: colour alone is not enough on a projector
        outlinePaint.pathEffect = android.graphics.DashPathEffect(floatArrayOf(18f, 14f), 0f)
    }

    /**
     * Start a NEW predicted line at the last good fix. Nothing is drawn before
     * this -- Contract A: xy[0] must equal init['xy'].
     */
    fun startPredicted(lat: Double, lon: Double) {
        current?.outlinePaint?.color = Color.parseColor("#95A5A6")   // dim the finished one
        val seg = newSegment()
        val under = map.overlays.indexOf(gap).coerceAtLeast(0)
        map.overlays.add(under, seg)
        segments.add(seg)
        current = seg

        armed = true
        val p = GeoPoint(lat, lon)
        lastPred = p
        seg.addPoint(p)
        refreshGap()
    }

    fun addTruth(lat: Double, lon: Double) {
        val p = GeoPoint(lat, lon)
        lastTruth = p
        truth.addPoint(p)
        refreshGap()
    }

    fun addPredicted(lat: Double, lon: Double) {
        val seg = current
        if (!armed || seg == null) return
        val p = GeoPoint(lat, lon)
        // Safety net, not a fix: nothing on a road moves 100 m between two samples.
        lastPred?.let {
            if (it.distanceToAsDouble(p) > MAX_STEP_M) { dropped++; return }
        }
        lastPred = p
        seg.addPoint(p)
        refreshGap()
    }

    fun setBlackout(on: Boolean) {
        blackout = on
        // amber while dead reckoning; green once GNSS is back
        current?.outlinePaint?.color = Color.parseColor(if (on) "#E67E22" else "#1E8449")
        refreshGap()
    }

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
        segments.forEach { map.overlays.remove(it) }
        segments.clear(); current = null
        truth.setPoints(emptyList()); gap.setPoints(emptyList())
        lastTruth = null; lastPred = null; armed = false; dropped = 0
        map.invalidate()
    }

    companion object {
        const val MAX_STEP_M = 100.0
    }
}
