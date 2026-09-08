package com.dhruva.nav

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Bundle
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
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

class NavigateActivity : AppCompatActivity(), SensorEventListener {

    // GPS jitter fix: a fix must move further than this, capped between
    // 3m and 8m regardless of the phone's reported accuracy, before we
    // treat it as real movement. Using raw accuracy (can be 15m+) as the
    // floor was the earlier bug — it blocked all short real movement too.
    private val MIN_MOVEMENT_M = 3f
    private val MAX_NOISE_FLOOR_M = 8f
    private var lastAcceptedGps: Location? = null

    private lateinit var mapView: MapView
    private lateinit var tvMode: TextView
    private lateinit var switchBlackout: Switch
    private lateinit var btnMapStandard: Button
    private lateinit var btnMapTerrain: Button
    private lateinit var tvErrorLabel: TextView

    private lateinit var summaryCard: LinearLayout
    private lateinit var summaryVerdict: TextView
    private lateinit var summaryDistance: TextView
    private lateinit var summaryError: TextView
    private lateinit var summaryDrift: TextView
    private lateinit var summarySpeed: TextView
    private lateinit var summaryConf: TextView
    private lateinit var summaryHz: TextView
    private lateinit var btnFinishRun: Button

    private lateinit var sm: SensorManager
    private var gyro: Sensor? = null
    private var lastGyroTimeNanos = 0L

    private var lat0: Double? = null
    private var lon0: Double? = null
    private var deadReckoner: DeadReckoner? = null
    private var blackoutOn = false

    private lateinit var dotMarker: Marker
    private var confCircle: Polygon? = null

    // Guide 3 add-on: draws the live truth/predicted divergence
    private lateinit var paths: LivePathOverlay

    // Guide 3 add-on: tracked for the on-phone RunSummary verdict
    private val gpsPoints = mutableListOf<Pair<Double, Double>>()
    private val predPoints = mutableListOf<Pair<Double, Double>>()
    private var sessionStartMs = 0L
    private var gyroSampleCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        // osmdroid refuses to fetch tiles without this — must be set before setContentView
        Configuration.getInstance().userAgentValue = packageName
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_navigate)

        mapView = findViewById(R.id.mapView)
        tvMode = findViewById(R.id.tvMode)
        switchBlackout = findViewById(R.id.switchBlackout)
        btnMapStandard = findViewById(R.id.btnMapStandard)
        btnMapTerrain = findViewById(R.id.btnMapTerrain)
        tvErrorLabel = findViewById(R.id.tvErrorLabel)

        summaryCard = findViewById(R.id.summaryCard)
        summaryVerdict = findViewById(R.id.summaryVerdict)
        summaryDistance = findViewById(R.id.summaryDistance)
        summaryError = findViewById(R.id.summaryError)
        summaryDrift = findViewById(R.id.summaryDrift)
        summarySpeed = findViewById(R.id.summarySpeed)
        summaryConf = findViewById(R.id.summaryConf)
        summaryHz = findViewById(R.id.summaryHz)
        btnFinishRun = findViewById(R.id.btnFinishRun)

        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(18.0)

        dotMarker = Marker(mapView).apply {
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        }
        mapView.overlays.add(dotMarker)

        // Guide 3 add-on: solid blue truth line + dashed amber predicted line
        paths = LivePathOverlay(mapView)
        sessionStartMs = System.currentTimeMillis()

        sm = getSystemService(SENSOR_SERVICE) as SensorManager
        gyro = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        btnMapStandard.setOnClickListener {
            mapView.setTileSource(TileSourceFactory.MAPNIK)
            mapView.invalidate()
        }
        btnMapTerrain.setOnClickListener {
            mapView.setTileSource(TileSourceFactory.OpenTopo)
            mapView.invalidate()
        }

        switchBlackout.setOnCheckedChangeListener { _, isChecked ->
            blackoutOn = isChecked
            tvMode.text = if (isChecked) "Mode: DEAD RECKONING (simulated)" else "Mode: GNSS"
            paths.setBlackout(isChecked)
        }

        btnFinishRun.setOnClickListener {
            val durationS = (System.currentTimeMillis() - sessionStartMs) / 1000.0
            val r = RunSummary.compute(
                truth = gpsPoints,
                pred = predPoints,
                durationS = durationS,
                imuSamples = gyroSampleCount
            )
            summaryVerdict.text = RunSummary.verdict(r)
            summaryDistance.text = "Distance: %.0f m".format(r.distanceM)
            summaryError.text = "Final error: %.1f m".format(r.finalErrorM)
            summaryDrift.text = "Drift: %.1f%%".format(r.driftPct)
            summarySpeed.text = "Mean speed: %.1f km/h".format(r.meanSpeedMps * 3.6)
            summaryConf.text = "Confidence: ±%.0f m (90%%)".format(r.confidence90M)
            summaryHz.text = "IMU rate: %.0f Hz".format(r.imuHz)
            summaryCard.visibility = View.VISIBLE
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

                // GPS jitter fix: only treat this fix as real movement if it moved
                // further than a capped noise floor. Standing still with 15m
                // accuracy no longer wanders, but real short walks still register.
                val anchor = lastAcceptedGps
                val movedM = anchor?.distanceTo(loc) ?: Float.MAX_VALUE
                val noiseFloor = loc.accuracy.coerceIn(MIN_MOVEMENT_M, MAX_NOISE_FLOOR_M)
                val isRealMovement = anchor == null || movedM > noiseFloor

                if (isRealMovement) {
                    lastAcceptedGps = loc
                    // Guide 3 add-on: keep feeding the truth line even during a
                    // simulated blackout — the comparison is the whole point
                    paths.addTruth(loc.latitude, loc.longitude)
                    gpsPoints.add(loc.latitude to loc.longitude)
                }

                if (lat0 == null) {
                    // first fix ever — this becomes the map's local origin
                    lat0 = loc.latitude
                    lon0 = loc.longitude
                    mapView.controller.setCenter(GeoPoint(loc.latitude, loc.longitude))
                }
                val (x, y) = toXY(loc.latitude, loc.longitude, lat0!!, lon0!!)

                if (deadReckoner == null) {
                    deadReckoner = DeadReckoner(heading = 0.0, speed = loc.speed.toDouble(), x = x, y = y)
                } else if (isRealMovement) {
                    deadReckoner!!.onGnssFix(x, y, loc.speed.toDouble())
                }

                // only let a real fix move the dot when we are NOT simulating a blackout
                if (!blackoutOn && isRealMovement) {
                    placeDot(loc.latitude, loc.longitude, isBlue = true, radiusM = 0.0)
                }
            }
        }, Looper.getMainLooper())
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_GYROSCOPE) return
        gyroSampleCount++

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

        // Guide 3 add-on: predicted track + live divergence label
        paths.addPredicted(lat, lon)
        predPoints.add(lat to lon)
        tvErrorLabel.text = "%.0f m apart".format(paths.currentErrorMetres())
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
            circle.fillColor = 0x334CD3C2   // translucent teal
            circle.strokeColor = 0xFF4CD3C2.toInt()
            circle.strokeWidth = 2f
            mapView.overlays.add(circle)
            confCircle = circle
        }

        mapView.controller.animateTo(point)
        mapView.invalidate()
    }
}