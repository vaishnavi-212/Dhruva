package com.dhruva.nav

import kotlin.math.cos

const val EARTH_RADIUS_M = 6378137.0   // earth radius, metres

// (lat, lon) -> local metres, east and north of the origin fix
fun toXY(lat: Double, lon: Double, lat0: Double, lon0: Double): Pair<Double, Double> {
    val x = Math.toRadians(lon - lon0) * cos(Math.toRadians(lat0)) * EARTH_RADIUS_M
    val y = Math.toRadians(lat - lat0) * EARTH_RADIUS_M
    return Pair(x, y)
}

// local metres -> (lat, lon), to place the dot back on the map
fun toLatLon(x: Double, y: Double, lat0: Double, lon0: Double): Pair<Double, Double> {
    val lat = lat0 + Math.toDegrees(y / EARTH_RADIUS_M)
    val lon = lon0 + Math.toDegrees(x / (EARTH_RADIUS_M * cos(Math.toRadians(lat0))))
    return Pair(lat, lon)
}