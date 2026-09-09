package com.dhruva.nav

import android.content.Context
import android.widget.Toast
import org.osmdroid.tileprovider.cachemanager.CacheManager
import org.osmdroid.util.BoundingBox
import org.osmdroid.views.MapView

/**
 * Pre-downloads map tiles so the map still DRAWS with no network.
 *
 * Nothing in the estimation needs the internet -- sensors, dead reckoning, road
 * binding, the run summary and the road data are all local, and a whole ride has
 * already been recorded in airplane mode. The one thing that does need it is the
 * map picture, and that is precisely what a judge sees during the two demos
 * where the phone has no signal.
 *
 * Same model as offline Google Maps: download the area once, in advance.
 *
 * ZOOM RANGE matters. Each level up quadruples the tile count, so a whole city
 * at z18 is tens of thousands of tiles and OSM's servers will rate-limit you.
 * 15-18 over a campus-sized box is a few thousand and takes a couple of minutes
 * on wifi.
 */
object OfflineMap {

    const val MIN_ZOOM = 15
    const val MAX_ZOOM = 18

    /**
     * Download every tile for the currently visible area. Run it on wifi, at a
     * desk, the day BEFORE the demo -- not in the field.
     */
    fun downloadCurrentView(ctx: Context, map: MapView, onDone: (String) -> Unit) {
        val box: BoundingBox = map.boundingBox
        val mgr = CacheManager(map)
        val tiles = mgr.possibleTilesInArea(box, MIN_ZOOM, MAX_ZOOM)
        mgr.downloadAreaAsync(ctx, box, MIN_ZOOM, MAX_ZOOM,
            object : CacheManager.CacheManagerCallback {
                override fun onTaskComplete() {
                    onDone("Map saved for offline use ($tiles tiles)")
                }
                override fun onTaskFailed(errors: Int) {
                    onDone("Map download finished with $errors errors — re-run it")
                }
                override fun updateProgress(progress: Int, currentZoomLevel: Int,
                                            zoomMin: Int, zoomMax: Int) {}
                override fun downloadStarted() {}
                override fun setPossibleTilesInArea(total: Int) {}
            })
        Toast.makeText(ctx, "Downloading $tiles tiles, z$MIN_ZOOM-$MAX_ZOOM…",
            Toast.LENGTH_LONG).show()
    }

    /**
     * Stop osmdroid asking the network for tiles.
     *
     * Call this with `false` for the offline demos. Without it the map spends
     * the whole demo timing out on requests that cannot succeed, and draws grey
     * where a cached tile was available all along.
     */
    fun setOnline(map: MapView, online: Boolean) {
        map.setUseDataConnection(online)
        map.invalidate()
    }
}
