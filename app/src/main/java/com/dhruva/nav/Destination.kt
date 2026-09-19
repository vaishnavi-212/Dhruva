package com.dhruva.nav

import android.app.Activity
import android.graphics.Color
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ListView
import androidx.appcompat.app.AlertDialog
import org.json.JSONArray
import org.json.JSONObject
import org.osmdroid.util.BoundingBox
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Polyline

/**
 * A trip to a searched destination: the place, and the road line to it from where GPS last put us.
 *
 * The route is handed to RoadBinder in the same format as a stored route, so everything that
 * already works in a blackout -- AI speed, the dot held on the road, the run summary -- works on
 * the planned route unchanged.
 */
class PlannedRoute(val place: CityPack.Place, val route: CityPack.Route) {

    val lengthKm: Double get() = route.lengthM / 1000.0
    val minutes: Double get() = route.seconds / 60.0

    /** "Unakal Kere · 1.5 km · 3 min" */
    val label: String get() = "%s · %.1f km · %.0f min".format(place.name, lengthKm, maxOf(1.0, minutes))

    /** The route in RoadBinder's format: {"name", "points": [[lat, lon], ...]}. */
    fun toRouteJson(): String {
        val pts = JSONArray()
        route.points.forEach { (la, lo) -> pts.put(JSONArray().put(la).put(lo)) }
        return JSONObject().put("name", place.name).put("points", pts).toString()
    }

    /** The planned route drawn under the dot: a wide teal line, like any navigation app. */
    fun draw(map: MapView): Polyline {
        val line = Polyline(map).apply {
            setPoints(route.points.map { GeoPoint(it.first, it.second) })
            outlinePaint.color = Color.parseColor("#B34CD3C2")      // dhruva_teal, 70% opaque
            outlinePaint.strokeWidth = 16f
            outlinePaint.strokeCap = android.graphics.Paint.Cap.ROUND
        }
        map.overlays.add(0, line)                                     // under the dot and the paths
        map.zoomToBoundingBox(BoundingBox.fromGeoPoints(line.actualPoints).increaseByScale(1.3f), true)
        map.invalidate()
        return line
    }
}

/**
 * "Where to?": a search box whose results update as you type, nearest first, with distance.
 * Works with no network: the places are in the city pack.
 */
object DestinationPicker {

    fun show(activity: Activity, city: CityPack, lat: Double, lon: Double, onPick: (CityPack.Place) -> Unit) {
        val box = EditText(activity).apply {
            hint = "Search a place — e.g. Unkal Lake"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
            isSingleLine = true
        }
        val hits = ArrayList<CityPack.Place>()
        val adapter = ArrayAdapter<String>(activity, android.R.layout.simple_list_item_1, ArrayList())
        val list = ListView(activity).apply { this.adapter = adapter }
        val body = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 0)
            addView(box, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(list, ViewGroup.LayoutParams.MATCH_PARENT, (activity.resources.displayMetrics.heightPixels * 0.45).toInt())
        }
        val dialog = AlertDialog.Builder(activity)
            .setTitle("Where to?")
            .setView(body)
            .setNegativeButton("Cancel", null)
            .create()

        box.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {
                val found = city.search(s?.toString() ?: "", lat, lon, limit = 8)
                hits.clear(); hits.addAll(found.map { it.first })
                adapter.clear()
                adapter.addAll(found.map { (p, d) -> "%s\n%s · %s".format(p.name, distanceText(d), kindText(p.kind)) })
                adapter.notifyDataSetChanged()
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
        })
        list.setOnItemClickListener { _, _, position, _ ->
            dialog.dismiss()
            onPick(hits[position])
        }
        dialog.show()
        box.requestFocus()
    }

    private fun distanceText(m: Double) = if (m < 1000) "%.0f m away".format(m) else "%.1f km away".format(m / 1000)

    /** "amenity=bus_station" -> "bus station" */
    private fun kindText(kind: String) = kind.substringAfter('=').replace('_', ' ').let { if (it == "yes") "building" else it }
}
