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
    // Longest gap between gyro samples we will integrate. At SENSOR_DELAY_GAME
    // real gaps are ~20 ms; anything past this means the stream stalled.
    private val MAX_STEP_S = 0.5

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
    private var gravitySensor: Sensor? = null
    private var lastGyroTimeNanos = 0L

    // Unit gravity vector in DEVICE coordinates. Needed to work out which way is
    // actually "up" -- raw gyro z is only the vertical axis when the phone lies
    // flat on a table, which it never is on a handlebar.
    private val gravityUnit = floatArrayOf(0f, 0f, 1f)

    // Road binding (Guide 3, Section 10)
    private var road: RoadBinder? = null
    private var roadBindingOn = false
    private var lastGoodSpeedMps = 0.0
    private lateinit var switchRoadBinding: Switch

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
        switchRoadBinding = findViewById(R.id.switchRoadBinding)
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
        gravitySensor = sm.getDefaultSensor(Sensor.TYPE_GRAVITY)

        // route.json ships in assets/. It is route-specific: a file for the wrong
        // area is worse than no file, which is why start() below is checked.
        road = try { RoadBinder.fromAssets(this) } catch (e: Exception) { null }

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

            if (isChecked) {
                // Anchor the estimate to where we actually are, facing the way we
                // are actually facing. Skipping this is what sent the dot east.
                lastAcceptedGps?.let { loc ->
                    if (loc.hasBearing()) deadReckoner?.setHeadingFromBearing(loc.bearing)
                    paths.startPredicted(loc.latitude, loc.longitude)

                    if (roadBindingOn) {
                        val ok = road?.start(loc.latitude, loc.longitude) ?: false
                        if (!ok) {
                            roadBindingOn = false
                            switchRoadBinding.isChecked = false
                            tvErrorLabel.text = "route.json is for another area (%.0f m away)"
                                .format(road?.snapDistanceM ?: 0.0)
                        }
                    }
                }
            }
        }

        switchRoadBinding.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked && road == null) {
                switchRoadBinding.isChecked = false
                tvErrorLabel.text = "no route.json in assets"
                return@setOnCheckedChangeListener
            }
            roadBindingOn = isChecked
            if (isChecked && blackoutOn) {
                lastAcceptedGps?.let { road?.start(it.latitude, it.longitude) }
            }
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
        gravitySensor?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
        sm.unregisterListener(this)
        // MUST reset. Otherwise the next gyro event after a resume reports a dt
        // of however long the screen was off, and speed * dt teleports the dot
        // across the map in a single straight segment -- the runaway dashed line
        // in the 7 Sept screenshot.
        lastGyroTimeNanos = 0L
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

                // Freeze the speed the moment the blackout starts -- during one we
                // are pretending these fixes do not exist.
                if (!blackoutOn && loc.speed > 0.5f) lastGoodSpeedMps = loc.speed.toDouble()

                if (deadReckoner == null) {
                    deadReckoner = DeadReckoner(
                        heading = if (loc.hasBearing()) Math.toRadians(90.0 - loc.bearing.toDouble()) else 0.0,
                        speed = loc.speed.toDouble(), x = x, y = y
                    )
                } else if (isRealMovement && !blackoutOn) {
                    // Only while GNSS is genuinely in use. Correcting the estimate
                    // from GPS during a "simulated blackout" makes the whole demo
                    // meaningless -- it was doing exactly that.
                    deadReckoner!!.onGnssFix(x, y, loc.speed.toDouble())
                    if (loc.hasBearing() && loc.speed > 1.0f) {
                        deadReckoner!!.setHeadingFromBearing(loc.bearing)
                    }
                }

                // only let a real fix move the dot when we are NOT simulating a blackout
                if (!blackoutOn && isRealMovement) {
                    placeDot(loc.latitude, loc.longitude, isBlue = true, radiusM = 0.0)
                }
            }
        }, Looper.getMainLooper())
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_GRAVITY) {
            val n = Math.sqrt(
                (event.values[0] * event.values[0] + event.values[1] * event.values[1] +
                        event.values[2] * event.values[2]).toDouble()
            ).toFloat()
            if (n > 1f) for (i in 0..2) gravityUnit[i] = event.values[i] / n
            return
        }
        if (event.sensor.type != Sensor.TYPE_GYROSCOPE) return
        gyroSampleCount++

        val dr = deadReckoner ?: return
        val o0 = lat0 ?: return
        val o1 = lon0 ?: return

        if (lastGyroTimeNanos == 0L) { lastGyroTimeNanos = event.timestamp; return }
        val dt = (event.timestamp - lastGyroTimeNanos) / 1_000_000_000.0
        lastGyroTimeNanos = event.timestamp
        // A stalled sensor stream must never be integrated as real elapsed time.
        if (dt <= 0.0 || dt > MAX_STEP_S) return

        if (!blackoutOn) return   // only drive the dot with dead reckoning during a simulated blackout

        // Rotation about the TRUE vertical, not the phone's z axis. Reduces to
        // gyro z when the phone happens to be flat, and stays correct when it is
        // not -- feeding raw z to a yaw-dependent estimator cost us 12.2% vs 8.3%
        // drift once already (RESOURCES 8g).
        val yawRate = (event.values[0] * gravityUnit[0] +
                event.values[1] * gravityUnit[1] +
                event.values[2] * gravityUnit[2]).toDouble()

        val (x, y, radius) = dr.step(yawRate, dt)
        var lat: Double; var lon: Double
        val free = toLatLon(x, y, o0, o1)

        val rb = road
        if (roadBindingOn && rb != null && rb.bound) {
            // Constrained to the road: the dot cannot drift sideways at all, and
            // only how far ALONG the road can be wrong.
            val p = rb.advance(lastGoodSpeedMps, dt)
            lat = p.first; lon = p.second
        } else {
            lat = free.first; lon = free.second
        }

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