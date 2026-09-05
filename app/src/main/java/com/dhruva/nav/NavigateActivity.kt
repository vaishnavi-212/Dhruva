package com.dhruva.nav

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Looper
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polygon
import android.widget.Button

class NavigateActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var mapView: MapView
    private lateinit var tvMode: TextView
    private lateinit var switchBlackout: Switch

    private lateinit var btnMapStandard: Button
    private lateinit var btnMapTerrain: Button
    private lateinit var sm: SensorManager
    private var gyro: Sensor? = null
    private var lastGyroTimeNanos = 0L

    private var lat0: Double? = null
    private var lon0: Double? = null
    private var deadReckoner: DeadReckoner? = null
    private var blackoutOn = false

    private lateinit var dotMarker: Marker
    private var confCircle: Polygon? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        // osmdroid refuses to fetch tiles without this — must be set before setContentView
        Configuration.getInstance().userAgentValue = packageName
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_navigate)

        mapView = findViewById(R.id.mapView)
        tvMode = findViewById(R.id.tvMode)
        switchBlackout = findViewById(R.id.switchBlackout)

        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(18.0)

        dotMarker = Marker(mapView).apply {
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        }
        mapView.overlays.add(dotMarker)

        sm = getSystemService(SENSOR_SERVICE) as SensorManager
        gyro = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        switchBlackout.setOnCheckedChangeListener { _, isChecked ->
            blackoutOn = isChecked
            tvMode.text = if (isChecked) "Mode: DEAD RECKONING (simulated)" else "Mode: GNSS"
        }
        btnMapStandard = findViewById(R.id.btnMapStandard)
        btnMapTerrain = findViewById(R.id.btnMapTerrain)

        btnMapStandard.setOnClickListener {
            mapView.setTileSource(TileSourceFactory.MAPNIK)
            mapView.invalidate()
        }
        btnMapTerrain.setOnClickListener {
            mapView.setTileSource(TileSourceFactory.OpenTopo)
            mapView.invalidate()
        }

        startGpsUpdates()
    }

    override fun onResume() {
        super.onResume()
        mapView.onResume()
        gyro?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
        sm.unregisterListener(this)
    }

    private fun startGpsUpdates() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            return   // MainActivity already asks for this before launching us
        }
        val fused = LocationServices.getFusedLocationProviderClient(this)
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
            .setMinUpdateIntervalMillis(1000L)
            .build()
        fused.requestLocationUpdates(req, object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return
                if (lat0 == null) {
                    // first fix ever — this becomes the map's local origin
                    lat0 = loc.latitude
                    lon0 = loc.longitude
                    mapView.controller.setCenter(GeoPoint(loc.latitude, loc.longitude))
                }
                val (x, y) = toXY(loc.latitude, loc.longitude, lat0!!, lon0!!)

                if (deadReckoner == null) {
                    deadReckoner = DeadReckoner(heading = 0.0, speed = loc.speed.toDouble(), x = x, y = y)
                } else {
                    deadReckoner!!.onGnssFix(x, y, loc.speed.toDouble())
                }

                // only let a real fix move the dot when we are NOT simulating a blackout
                if (!blackoutOn) {
                    placeDot(loc.latitude, loc.longitude, isBlue = true, radiusM = 0.0)
                }
            }
        }, Looper.getMainLooper())
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_GYROSCOPE) return
        val dr = deadReckoner ?: return
        val o0 = lat0 ?: return
        val o1 = lon0 ?: return

        if (lastGyroTimeNanos == 0L) { lastGyroTimeNanos = event.timestamp; return }
        val dt = (event.timestamp - lastGyroTimeNanos) / 1_000_000_000.0
        lastGyroTimeNanos = event.timestamp

        if (!blackoutOn) return   // only drive the dot with dead reckoning during a simulated blackout

        val gyroZ = event.values[2].toDouble()
        val (x, y, radius) = dr.step(gyroZ, dt)
        val (lat, lon) = toLatLon(x, y, o0, o1)
        placeDot(lat, lon, isBlue = false, radiusM = radius)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun placeDot(lat: Double, lon: Double, isBlue: Boolean, radiusM: Double) {
        val point = GeoPoint(lat, lon)
        dotMarker.position = point
        dotMarker.icon = ContextCompat.getDrawable(
            this,
            if (isBlue) android.R.drawable.presence_online else android.R.drawable.presence_away
        )
        // blue system icon while GNSS is healthy, amber while dead reckoning

        confCircle?.let { mapView.overlays.remove(it) }
        if (radiusM > 0.0) {
            val circle = Polygon(mapView)
            circle.points = Polygon.pointsAsCircle(point, radiusM)
            circle.fillPaint.color = 0x334CD3C2
            circle.outlinePaint.color = 0xFF4CD3C2.toInt()
            circle.outlinePaint.strokeWidth = 2f
            mapView.overlays.add(circle)
            confCircle = circle
        }

        mapView.controller.animateTo(point)
        mapView.invalidate()
    }
}