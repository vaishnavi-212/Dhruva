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
import android.os.SystemClock
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
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
import org.osmdroid.views.overlay.Polyline

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
    private lateinit var tvSensorStatus: TextView
    private lateinit var btnSaveMap: Button

    private lateinit var summaryCard: LinearLayout
    private lateinit var summaryVerdict: TextView
    private lateinit var summaryDistance: TextView
    private lateinit var summaryError: TextView
    private lateinit var summaryDrift: TextView
    private lateinit var summarySpeed: TextView
    private lateinit var summaryConf: TextView
    private lateinit var summaryHz: TextView
    private lateinit var btnFinishRun: Button
    private lateinit var btnCloseSummary: Button

    private lateinit var sm: SensorManager
    private var gyro: Sensor? = null
    private var gravitySensor: Sensor? = null
    private var linAccSensor: Sensor? = null
    private var lastGyroTimeNanos = 0L

    // Speed from the trained model. Null if its assets are missing -- the blackout then carries
    // the GPS speed held before it, exactly as before.
    private var speedAi: SpeedEstimator? = null
    private var aiSteps = 0L
    private var heldSteps = 0L
    private var perf: PerfLog? = null
// Benchmark mode: run the model while standing still, for the battery test (long-press the status line).
    private var benchmark = false

    // Unit gravity vector in DEVICE coordinates. Needed to work out which way is
    // actually "up" -- raw gyro z is only the vertical axis when the phone lies
    // flat on a table, which it never is on a handlebar.
    private val gravityUnit = floatArrayOf(0f, 0f, 1f)

    // Road binding (Guide 3, Section 10)
    private var road: RoadBinder? = null
    private var roadBindingOn = false
    private var lastGoodSpeedMps = 0.0
    // Recent MOVING GPS speeds (time ms, m/s). The speed carried into a blackout is their median.
    private val recentSpeeds = ArrayDeque<Pair<Long, Double>>()
    private val MOVING_MPS = 1.5
    private val SPEED_WINDOW_MS = 30_000L
    private var lastFixMs = 0L
    private val TRUTH_STALE_S = 10.0
    private lateinit var switchRoadBinding: Switch

    private val gyroBias = GyroBias()
    private var voice: VoiceGuide? = null
    private var stationaryNow = false
    private var lastTruthPoint: GeoPoint? = null
    private var blackoutFromIndex = 0
    private var blackoutToIndex = -1
    private var blackoutStartMs = 0L
    private var blackoutEndMs = 0L
    private lateinit var btnRoute: Button
    private lateinit var btnSaveRoute: Button

    // Destination: search a place, route to it offline, bind the dot to that route.
    private lateinit var btnWhereTo: Button
    private var city: CityPack? = null
    private var planned: PlannedRoute? = null
    private var routeLine: Polyline? = null
    // Turn-by-turn on the planned route: progress from GPS, or from the dot during a blackout.
    private var guide: RouteGuide? = null
    private var navigator: Navigator? = null
    private val offRoute = OffRouteDetector()
    private var progressS = 0.0
    private var lastRerouteMs = 0L
    private var lastGuideMs = 0L
    private val recentFixes = ArrayDeque<Location>()      // for the rider's direction at a re-route
    private val fixFilter = FixFilter()                   // drops cached and impossible fixes before anything uses them
    // Wrong turn with GPS off: the gyro watches the planned route ahead of the dot.
    private var routeNodes: IntArray? = null
    private var routeGuard: RouteGuard? = null
    private var rebinder: Rebinder? = null
    private var guardCutArc = 0.0
    private var guardT0 = 0.0
    private var guardLastFeed = 0.0
    private var guardAlarmT = 0.0
    private var psiSinceCut = 0.0
    private val wrongTurns = mutableListOf<TripSummary.WrongTurn>()

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

    // Seamless GNSS deficit handler: decides GNSS vs dead reckoning by itself (Part 5).
    private lateinit var gnss: GnssSwitch
    private var gnssWatcher: GnssWatcher? = null

    // GNSS+INS fusion filter, 10 times a second (Part 7).
    private val fusion = FusionEngine()
    private val seamless = SeamlessDot()
    private val fusionHandler = android.os.Handler(Looper.getMainLooper())
    private var fusionReady = false
    private var lastFusionMs = 0L
    private var yawSum = 0.0 // gyro turn accumulated since the last fusion step, rad
    private var yawTime = 0.0 // and over how long, s
    private var pendingFix: DoubleArray? = null // x, y, sigma, speed of a fix not yet given to the filter
    private var roadArcNear = Double.NaN // where on the route the filter last was, m
    private var dotFromFusion = false // long-press the mode text to switch
    private var fusionEpochs = 0L
    private var fusionRateStartMs = 0L
    private var fusionHz = 0.0
    private var blackoutDistM = 0.0
    private var reacquireJumpM: Double? = null

    private val fusionTick = object : Runnable {
        override fun run() {
            fusionStep()
            fusionHandler.postDelayed(this, 100)
        }
    }

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
        tvSensorStatus = findViewById(R.id.tvSensorStatus)
        btnSaveMap = findViewById(R.id.btnSaveMap)

        summaryCard = findViewById(R.id.summaryCard)
        summaryVerdict = findViewById(R.id.summaryVerdict)
        summaryDistance = findViewById(R.id.summaryDistance)
        summaryError = findViewById(R.id.summaryError)
        summaryDrift = findViewById(R.id.summaryDrift)
        summarySpeed = findViewById(R.id.summarySpeed)
        summaryConf = findViewById(R.id.summaryConf)
        summaryHz = findViewById(R.id.summaryHz)
        btnFinishRun = findViewById(R.id.btnFinishRun)
        btnCloseSummary = findViewById(R.id.btnCloseSummary)
        btnRoute = findViewById(R.id.btnRoute)
        btnSaveRoute = findViewById(R.id.btnSaveRoute)
        btnWhereTo = findViewById(R.id.btnWhereTo)

        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(18.0)

        dotMarker = Marker(mapView).apply {
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        }
        mapView.overlays.add(dotMarker)

        // Guide 3 add-on: solid blue truth line + dashed amber predicted line
        paths = LivePathOverlay(mapView)
        voice = VoiceGuide(this)
        sessionStartMs = System.currentTimeMillis()

        sm = getSystemService(SENSOR_SERVICE) as SensorManager
        gyro = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        gravitySensor = sm.getDefaultSensor(Sensor.TYPE_GRAVITY)
        linAccSensor = sm.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        speedAi = try { SpeedEstimator(this) } catch (e: Exception) { null }
        // Warm from the start. The model needs 30 s of sensors before its first answer; starting it
        // only at the cut meant the first 30 s of every blackout ran on a guessed held speed
        // (24 Sept: held 32 km/h against a real 24 km/h). It costs ~15 ms twice a second.
        speedAi?.setActive(true)
        gnss = GnssSwitch(onChange = { mode, why, tMs ->
            onGnssModeChanged(mode, why, tMs) })
        gnssWatcher = GnssWatcher(this, gnss)
        perf = PerfLog(this, { speedAi }, { blackoutOn }).also { it.start() }
        tvMode.setOnLongClickListener {
            dotFromFusion = !dotFromFusion
            Toast.makeText(this, if (dotFromFusion) "Dot: fusion filter (10 Hz)" else "Dot: road tracker",
                Toast.LENGTH_SHORT).show()
            true
        }

        tvSensorStatus.setOnLongClickListener {
            benchmark = !benchmark
            speedAi?.setActive(true)

            Toast.makeText(
                this,
                if (benchmark) "Benchmark ON: model runs twice a second"
                else "Benchmark OFF",
                Toast.LENGTH_SHORT
            ).show()

            true
        }

        // routes.json ships in assets/: several stored routes, no synthetic padding.
        // A route for the wrong area is worse than none, which is why start() is checked.
        // Routes learned on this phone first, then the ones shipped with the app.
        road = try { RoadBinder(RouteStore.combinedJson(this)) } catch (e: Exception) { null }

        // AUTO binds to the nearest stored route, which is a guess wherever routes
        // share a road and split later. For a demo, tap to pick the route you will ride.
        btnRoute.text = road?.selectionLabel() ?: "Route: none"
        btnRoute.setOnClickListener {
            val rb = road ?: return@setOnClickListener
            if (blackoutOn) {
                Toast.makeText(this, "Choose the route before cutting GPS", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            btnRoute.text = rb.cycleSelection()
            if (roadBindingOn) lastAcceptedGps?.let { bindRoad(it) }
        }

        // The city pack is 1.15 MB of roads and places: read it off the main thread.
        Thread {
            val c = try { CityPack.fromAssets(this) } catch (e: Exception) { null }
            runOnUiThread {
                city = c
                btnWhereTo.isEnabled = c != null
                btnWhereTo.text = if (c != null) "Where to?" else "No city map"
            }
        }.start()

        btnWhereTo.setOnClickListener {
            val c = city ?: return@setOnClickListener
            if (blackoutOn) {
                Toast.makeText(this, "Choose the destination before cutting GPS", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val fix = lastAcceptedGps ?: run {
                Toast.makeText(this, "Waiting for a GPS fix — the route starts from where you are", Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            DestinationPicker.show(this, c, fix.latitude, fix.longitude) { place -> planTo(place) }
        }

        btnSaveRoute.setOnClickListener {
            if (blackoutOn) {
                Toast.makeText(this, "Save the route after GPS is back on", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val saved = RouteStore.saveTrack(this, gpsPoints.toList())
            if (saved == null) {
                Toast.makeText(this, "Track too short to save — ride at least %.0f m with GPS on"
                    .format(RouteStore.MIN_ROUTE_M), Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            clearPlan()                              // a learned route replaces a planned one
            road = try { RoadBinder(RouteStore.combinedJson(this)) } catch (e: Exception) { null }
            road?.select(0)                          // the route just learned is listed first
            btnRoute.text = road?.selectionLabel() ?: "Route: none"
            if (roadBindingOn) {                     // the old binding belonged to the old route set
                roadBindingOn = false
                switchRoadBinding.isChecked = false
            }
            Toast.makeText(this, "Saved '%s' — %.0f m%s. Selected for road binding."
                .format(saved.name, saved.lengthM, if (saved.closedLoop) ", closed loop" else ""),
                Toast.LENGTH_LONG).show()
        }

        btnSaveMap.setOnClickListener {
            // Do this at a desk on wifi, the day BEFORE the demo. Pan and zoom to
            // the demo area first -- it downloads exactly what is on screen.
            OfflineMap.downloadCurrentView(this, mapView) { msg ->
                runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_LONG).show() }
            }
        }

        btnMapStandard.setOnClickListener {
            mapView.setTileSource(TileSourceFactory.MAPNIK)
            mapView.invalidate()
        }
        btnMapTerrain.setOnClickListener {
            mapView.setTileSource(TileSourceFactory.OpenTopo)
            mapView.invalidate()
        }

        switchBlackout.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked && lastGoodSpeedMps < MOVING_MPS) {
                // No measured moving speed yet. A blackout now would carry a speed of zero
                // and the dot would sit still for the whole outage (11 Sept, run 3).
                switchBlackout.isChecked = false
                Toast.makeText(this, "Ride with GPS on until you are moving — no speed measured yet",
                    Toast.LENGTH_LONG).show()
                return@setOnCheckedChangeListener
            }
            // The switch no longer runs the blackout itself. It tells the GNSS switch "simulated", and the
            // GNSS switch calls enterBlackout()/exitBlackout() -- exactly what a real tunnel does.
            gnss.setSimulated(isChecked, SystemClock.elapsedRealtime())
        }

        switchRoadBinding.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked && road == null) {
                switchRoadBinding.isChecked = false
                tvErrorLabel.text = "no routes.json in assets"
                return@setOnCheckedChangeListener
            }
            if (blackoutOn) {
                // Binding needs the last GOOD fix. Mid-blackout the only fixes are
                // ones we are pretending do not exist -- binding to them is cheating.
                if (isChecked != roadBindingOn) {
                    switchRoadBinding.isChecked = roadBindingOn
                    Toast.makeText(this, "Set road binding before cutting GPS",
                        Toast.LENGTH_SHORT).show()
                }
                return@setOnCheckedChangeListener
            }
            roadBindingOn = isChecked
            if (isChecked) lastAcceptedGps?.let { bindRoad(it) }
        }

        btnFinishRun.setOnClickListener {
            val now = System.currentTimeMillis()
            val durationS = (now - sessionStartMs) / 1000.0
            if (blackoutStartMs == 0L || predPoints.size < 2) {
                Toast.makeText(this, "No GPS blackout in this session — nothing to score",
                    Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            val truthAgeS = if (lastFixMs == 0L) 9999.0 else (now - lastFixMs) / 1000.0
            if (truthAgeS > TRUTH_STALE_S) {
                // 11 Sept run 3: phone location was switched off mid-run, and the card scored
                // the dot against a GPS point three minutes old. Refuse instead.
                summaryVerdict.text = "NOT SCORABLE — real GPS stopped %.0f s before Finish".format(truthAgeS)
                summaryDistance.text = "Phone location was switched off, so there is nothing to compare against."
                summaryError.text = "Use only 'Simulate GPS loss'. Never turn phone location off in a scored run."
                summaryDrift.text = ""; summarySpeed.text = ""; summaryConf.text = ""; summaryHz.text = ""
                summaryCard.visibility = View.VISIBLE
                btnFinishRun.visibility = View.GONE
                return@setOnClickListener
            }
            // Score ONLY the most recent blackout: truth up to where it ended, time
            // from its start to its end (or to now, if it is still running).
            val truthEnd = if (blackoutToIndex >= 0) blackoutToIndex else gpsPoints.size
            val blackoutDurationS =
                ((if (blackoutEndMs > 0L) blackoutEndMs else now) - blackoutStartMs) / 1000.0
            val r = RunSummary.compute(
                truth = gpsPoints.subList(0, truthEnd).toList(),
                pred = predPoints.toList(),
                durationS = durationS,
                imuSamples = gyroSampleCount,
                blackoutFromIndex = blackoutFromIndex,
                blackoutDurationS = blackoutDurationS
            )
            summaryVerdict.text = RunSummary.verdict(r)
            summaryDistance.text = "Distance without GPS: %.0f m".format(r.distanceM)
            summaryError.text = "Final error: %.1f m".format(r.finalErrorM)
            summaryDrift.text = "Drift: %.1f%%".format(r.driftPct)
            summarySpeed.text = "Mean speed: %.1f km/h".format(r.meanSpeedMps * 3.6)
            summaryConf.text = "Confidence: ±%.0f m (90%%)".format(r.confidence90M)
            summaryHz.text = "IMU rate: %.0f Hz".format(r.imuHz) + " · speed: " +
                (if (aiSteps + heldSteps > 0) "%.0f%% AI model".format(100.0 * aiSteps / (aiSteps + heldSteps)) else "n/a")
            summaryCard.visibility = View.VISIBLE
            btnFinishRun.visibility = View.GONE
            saveTrip(r, now, truthEnd, blackoutDurationS)

        }

        btnCloseSummary.setOnClickListener {
            summaryCard.visibility = View.VISIBLE
            btnFinishRun.visibility = View.GONE


        }

        startGpsUpdates()
    }

    /**
     * Route from the last good fix to [place], draw it, and bind the dot to it.
     * From here on the planned route IS the road the blackout rides on.
     */
    private fun planTo(place: CityPack.Place, reroute: Boolean = false) {
        val c = city ?: return
        val fix = lastAcceptedGps ?: return
        if (!reroute) btnWhereTo.text = "Routing to ${place.name}…"
        // Routing searches 34,000 road points: off the main thread, so the screen never freezes.
        Thread {
            val r = c.route(fix.latitude, fix.longitude, place.arrive)
            runOnUiThread { applyRoute(place, fix, r, reroute) }
        }.start()
    }

    private fun applyRoute(place: CityPack.Place, fix: Location, r: CityPack.Route?, reroute: Boolean = false) {
        if (blackoutOn) {                                   // GPS was cut while routing: keep what we had
            btnWhereTo.text = planned?.let { "→ " + it.label } ?: "Where to?"
            return
        }
        if (r == null || r.points.size < 2) {
            btnWhereTo.text = planned?.let { "→ " + it.label } ?: "Where to?"
            Toast.makeText(this, "No road route to ${place.name} from here", Toast.LENGTH_LONG).show()
            return
        }
        clearPlan()
        val p = PlannedRoute(place, r)
        planned = p
        routeLine = p.draw(mapView, zoomToFit = !reroute)
        road = try { RoadBinder(p.toRouteJson()).also { it.select(0) } } catch (e: Exception) { null }
        btnRoute.text = road?.selectionLabel() ?: "Route: none"
        btnWhereTo.text = "→ " + p.label
        roadBindingOn = true
        switchRoadBinding.isChecked = true
        bindRoad(fix)

        val g = RouteGuide(r.points)
        guide = g
        routeNodes = r.nodes
        navigator = Navigator(place.name, g, Maneuver.fromRoute(g, r.nodes) { city?.degree(it) ?: 2 })
        progressS = 0.0
        offRoute.reset()
        if (!reroute) {
            voice?.say("Route to %s. %.1f kilometres.".format(place.name, p.lengthKm))
            return
        }
        // Re-routed: if the new route starts back the way the rider is going, say so first.
        val riding = riderHeading()
        val uTurn = riding != null && g.lengthM > 30 &&
            Math.abs(Math.toDegrees(Maneuver.wrap(g.heading(0.0, 30.0) - riding))) > U_TURN_DEG
        voice?.say(if (uTurn) "Re-routing. Make a U-turn." else "Re-routing.")
        android.util.Log.i("DhruvaNav", "re-routed to ${place.name}: %.0f m, u-turn=$uTurn".format(g.lengthM))
        tvErrorLabel.text = if (uTurn) "Re-routed · make a U-turn" else "Re-routed"
    }

    /** At the cut: watch the planned route ahead of the dot for a turn the rider did not follow. */
    private fun armRouteGuard() {
        val g = guide; val rb = road
        routeGuard = null; rebinder = null
        if (planned == null || g == null || rb == null || !rb.bound || !roadBindingOn) return
        // An unmeasured gyro bias drifts the heading and the guard fires on its own drift: three
        // alarms in 1.2 km on 24 Sept, each one moving the dot onto a road nobody took.
        if (!gyroBias.ready) {
            tvErrorLabel.text = "Wrong-turn alarm off — gyro bias was never measured (stand still 5 s with GPS on)"
            android.util.Log.i("DhruvaNav", "route guard NOT armed: gyro bias never measured")
            return
        }
        guardCutArc = rb.arcM
        val (ax, ay) = g.ahead(guardCutArc)
        if (ax.size < 2) return
        routeGuard = RouteGuard(ax, ay)
        psiSinceCut = 0.0; guardT0 = Double.NaN; guardLastFeed = 0.0
        android.util.Log.i("DhruvaNav", "route guard armed at %.0f m along the route, %.0f m of route ahead"
            .format(guardCutArc, (guide?.lengthM ?: 0.0) - guardCutArc))
    }

    private fun watchForWrongTurn(nowS: Double, rb: RoadBinder) {
        val guard = routeGuard ?: return
        if (guardT0.isNaN()) guardT0 = nowS
        val t = nowS - guardT0
        if (t - guardLastFeed < 0.1 && guard.size > 0) return
        guardLastFeed = t
        val since = rb.arcM - guardCutArc
        if (guard.add(t, since, psiSinceCut)) {
            voice?.say("Wrong turn.")
            wrongTurns.add(TripSummary.WrongTurn(since, 0.0, 0.0, 0, null, false))   // filled in when the road is chosen
            android.util.Log.i("DhruvaNav", "wrong turn: %.0f m after the cut".format(since))
            tvErrorLabel.text = "Wrong turn — finding the road you took…"
            val c = city; val g = guide; val nodes = routeNodes
            rebinder = if (c != null && g != null && nodes != null) Rebinder(c, g, nodes, guardCutArc, guard).takeIf { it.hasCandidates } else null
            guardAlarmT = t
            if (rebinder == null) { tvErrorLabel.text = "Off route — no side road found here"; routeGuard = null }
            return
        }
        val rbd = rebinder ?: return
        if (since < rbd.readyAtSinceCut && t - guardAlarmT < Rebinder.MAX_WAIT_S) return
        rebinder = null; routeGuard = null
        val choice = rbd.choose() ?: return
        rerouteFromBranch(choice, rb.arcM)
    }

    /** Put the dot on the road the rider actually took and route from there, still with GPS off. */
    private fun rerouteFromBranch(choice: Rebinder.Choice, routeArc: Double) {
        val c = city ?: return; val g = guide ?: return; val p = planned ?: return
        val (x, y) = choice.road.xyAt(routeArc)
        val (la, lo) = g.toLatLon(x, y)
        var k = 0; var bd = Double.MAX_VALUE
        for (i in 0 until choice.branch.size - 1) {
            val d = CityPack.metres(la, lo, c.lat[choice.branch[i + 1]], c.lon[choice.branch[i + 1]])
            if (d < bd) { bd = d; k = i }
        }
        val a = choice.branch[k]; val b = choice.branch[k + 1]
        Thread {
            val d = c.routeFrom(a, b, la, lo, p.place.arrive)
            runOnUiThread {
                if (d == null || !blackoutOn) return@runOnUiThread
                routeLine?.let { mapView.overlays.remove(it) }
                val np = PlannedRoute(p.place, d.route)
                planned = np
                routeLine = np.draw(mapView, zoomToFit = false)
                road = try { RoadBinder(np.toRouteJson()).also { it.select(0) } } catch (e: Exception) { null }
                road?.start(la, lo, null)
                val ng = RouteGuide(d.route.points)
                guide = ng; routeNodes = d.route.nodes
                navigator = Navigator(p.place.name, ng, Maneuver.fromRoute(ng, d.route.nodes) { c.degree(it) })
                progressS = 0.0
                armRouteGuard()                            // a second wrong turn is caught too
                voice?.say(if (d.uTurn) "Re-routing. Make a U-turn." else "Re-routing.")
                tvErrorLabel.text = "Re-routed from the road you took (gyro %+.0f°)%s".format(choice.gyroDeg, if (d.uTurn) " · make a U-turn" else "")
                if (wrongTurns.isNotEmpty()) {
                    val w = wrongTurns.removeAt(wrongTurns.size - 1)
                    wrongTurns.add(TripSummary.WrongTurn(w.atM, choice.gyroDeg, choice.pathDeg, choice.candidates,
                        d.route.lengthM, d.uTurn))
                }
                android.util.Log.i("DhruvaNav", "wrong-turn re-route: %.0f m, u-turn=%s, %d candidate road(s), gyro %.0f vs road %.0f"
                    .format(d.route.lengthM, d.uTurn, choice.candidates, choice.gyroDeg, choice.pathDeg))
            }
        }.start()
    }

    /** Direction the rider is moving (radians, counter-clockwise from east), from the last ~10 m of real fixes. */
    private fun riderHeading(): Double? {
        val now = recentFixes.lastOrNull() ?: return null
        val back = recentFixes.lastOrNull { it.distanceTo(now) >= 10f } ?: return null
        val (x, y) = toXY(now.latitude, now.longitude, back.latitude, back.longitude)
        return Math.atan2(y, x)
    }

    /** With GPS: move along the route, speak the next turn, re-route when we have left it. */
    private fun guideWithGps(loc: Location) {
        val g = guide ?: return
        val nav = navigator ?: return
        val p = planned ?: return
        if (nav.arrived) return
        val pr = g.project(loc.latitude, loc.longitude, progressS)
        val now = System.currentTimeMillis()
        if (offRoute.onFix(pr.offM, loc.accuracy) && now - lastRerouteMs > REROUTE_GAP_MS) {
            lastRerouteMs = now
            offRoute.reset()
            tvErrorLabel.text = "Off route (%.0f m) — re-routing".format(pr.offM)
            planTo(p.place, reroute = true)
            return
        }
        if (pr.offM <= OffRouteDetector.OFF_M) progressS = maxOf(progressS, pr.s)
        showGuidance(nav.update(progressS, 0.0) { a, d, r -> voice?.phrase(a, d, r) ?: "$a." })
    }

    private fun showGuidance(u: Navigator.Update) {
        btnWhereTo.text = u.status
        u.speak?.let { voice?.say(it); android.util.Log.i("DhruvaNav", "say: $it") }
    }

    private fun clearPlan() {
        routeLine?.let { mapView.overlays.remove(it); mapView.invalidate() }
        routeLine = null
        planned = null
        guide = null
        navigator = null
        btnWhereTo.text = if (city != null) "Where to?" else btnWhereTo.text
    }


    /** Called by GnssSwitch the moment the mode changes. Part 7 makes this
     * drive the whole blackout. */
    /** Called by GnssSwitch the moment the mode changes. */
    private fun onGnssModeChanged(mode: GnssMode, why: String, tMs: Long) {
        android.util.Log.i("Dhruva", "GNSS mode -> $mode ($why) at $tMs ms")
        Toast.makeText(this, if (mode == GnssMode.GNSS) "GPS back: $why" else "GPS lost: $why", Toast.LENGTH_SHORT).show()
        if (mode == GnssMode.DEAD_RECKONING) enterBlackout(why) else exitBlackout()
    }

    /** GPS lost (real or simulated): start dead reckoning from the last good fix. */
    private fun enterBlackout(why: String) {
        if (blackoutOn) return
        blackoutOn = true
        speedAi?.setActive(true)
        tvMode.text = "Mode: DEAD RECKONING (" + (if (gnss.simulated) "simulated" else why) + ")\n" +
                (if (speedAi != null) "AI speed · fallback %.0f km/h" else "holding %.0f km/h").format(lastGoodSpeedMps * 3.6)
        paths.setBlackout(true)
        // The last fix may have said "stationary". If that flag stayed true
        // through the outage, the bias estimator would average a MOVING gyro.
        stationaryNow = false
        gyroBias.freeze()
        aiSteps = 0; heldSteps = 0
        blackoutDistM = 0.0
        reacquireJumpM = null
        voice?.reset()
        voice?.say("Satellite signal lost. Switching to inertial navigation.")
        // Each blackout is scored on its own: denied distance from HERE,
        // denied time from NOW, and a prediction list that starts empty.
        // On 10 Sept a third blackout inherited the first two's points.
        blackoutFromIndex = gpsPoints.size
        blackoutToIndex = -1
        blackoutStartMs = System.currentTimeMillis()
        blackoutEndMs = 0L
        predPoints.clear()
        // Anchor to where we are, facing the way we are facing.
        lastAcceptedGps?.let { loc ->
            if (loc.hasBearing()) deadReckoner?.setHeadingFromBearing(loc.bearing)
            if (roadBindingOn) {
                bindRoad(loc)
                roadArcNear = road?.arcM ?: Double.NaN
            }
            // Start the predicted line WHERE THE DOT STARTS. Bound to a road, that is the point on the
            // road, which can be 100 m from the fix; anchoring at the fix made every later point look
            // like a >100 m jump and the overlay rejected them all ("BAD FRAMES", 22 Sept).
            val rb0 = road
            val start = if (roadBindingOn && rb0 != null && rb0.bound) rb0.at(rb0.arcM)
                        else loc.latitude to loc.longitude
            paths.startPredicted(start.first, start.second) // a NEW line
            predPoints.add(start)
            // Watch the planned route from here: with GPS off only the gyro can tell us
            // the rider has taken a different road.
            armRouteGuard()
        }
    }

    /** GPS usable again: stop dead reckoning and record how far off the dot was. */
    private fun exitBlackout() {
        if (!blackoutOn) return
        blackoutOn = false
        speedAi?.setActive(true)                 // stays warm for the next blackout
        tvMode.text = "Mode: GNSS"
        paths.setBlackout(false)
        // "0 m apart" while GNSS is healthy is not a result -- there is no
        // prediction to compare against yet, and it reads like a perfect
        // score to anyone watching. Say what is actually happening.
        tvErrorLabel.text = "GNSS locked — tracking"
        blackoutEndMs = System.currentTimeMillis()
        blackoutToIndex = gpsPoints.size
        gyroBias.unfreeze() // GNSS is back: keep learning the bias
        routeGuard = null; rebinder = null // GPS itself checks the route again (the off-route detector)
        // The dot's error at the moment GPS returns: a real-world accuracy number, even in a real tunnel.
        val p = predPoints.lastOrNull(); val t = lastTruthPoint
        if (p != null && t != null) {
            val d = FloatArray(1)
            android.location.Location.distanceBetween(p.first, p.second, t.latitude, t.longitude, d)
            reacquireJumpM = d[0].toDouble()
            android.util.Log.i("Dhruva", "GPS back: dot was %.1f m off".format(reacquireJumpM))
        }
        voice?.say("Satellite signal back.")
    }

    /** One step of the GNSS+INS fusion filter. Runs 10 times a second from fusionTick. */
    private fun fusionStep() {
        val now = SystemClock.elapsedRealtime()
        val dt = if (lastFusionMs == 0L) 0.1 else ((now - lastFusionMs) / 1000.0).coerceIn(0.05, 0.5)
        lastFusionMs = now
        val o0 = lat0; val o1 = lon0
        if (!fusionReady || o0 == null || o1 == null) return

        val wz = if (yawTime > 0.0) yawSum / yawTime else 0.0 // average turn rate since the last step
        yawSum = 0.0; yawTime = 0.0

        val fix = pendingFix; pendingFix = null

        // Speed measurement: the AI model during a blackout; held GPS speed if the model has no answer yet.
        val ai = speedAi
        val aiFresh = ai != null && ai.freshAt(SystemClock.elapsedRealtimeNanos() / 1e9)
        val modelSpeed = if (!blackoutOn) null else if (aiFresh) ai!!.speedMps else lastGoodSpeedMps

        // Road: nearest point within 40 m of where we last were on the route.
        var roadPt: RoadBinder.RoadPoint? = null
        val rb = road
        if (roadBindingOn && rb != null && rb.bound) {
            val (la, lo) = toLatLon(fusion.posX, fusion.posY, o0, o1)
            roadPt = rb.nearestNear(la, lo, if (roadArcNear.isNaN()) rb.arcM else roadArcNear)
            roadPt?.let { roadArcNear = it.arcM }
        }
        val roadXY = roadPt?.let { toXY(it.lat, it.lon, o0, o1) }

        fusion.step(dt, wz, 0.0,
            gnssX = fix?.get(0), gnssY = fix?.get(1), gnssSigma = fix?.get(2),
            gnssSpeed = fix?.get(3)?.takeIf { !it.isNaN() },
            modelSpeed = modelSpeed,
            roadHeading = roadPt?.headingRad, roadX = roadXY?.first, roadY = roadXY?.second)

        seamless.update(fusion.posX, fusion.posY, dt)
        if (blackoutOn) blackoutDistM += fusion.speed * dt
        fusionEpochs++
        if (fusionRateStartMs == 0L) fusionRateStartMs = now
        if (fusionEpochs % 50 == 0L && now > fusionRateStartMs) fusionHz = fusionEpochs * 1000.0 / (now - fusionRateStartMs)

        if (!dotFromFusion) return // the road tracker draws the dot (default)

        val (la, lo) = toLatLon(seamless.outX, seamless.outY, o0, o1)
        // The honest 90% circle from distance without GPS (as DeadReckoner), not the filter's own covariance.
        val radius = if (blackoutOn) 2.146 * maxOf(0.19 * blackoutDistM, 2.0) else 0.0
        placeDot(la, lo, isBlue = !blackoutOn, radiusM = radius)
        if (blackoutOn) {
            paths.addPredicted(la, lo)
            predPoints.add(la to lo)
            tvErrorLabel.text = "%.0f m apart · fusion".format(paths.currentErrorMetres())
        }
    }

    /** Writes trip_summary.json next to the ride's sensor files (or
     * files/trips/ if not recording). */
    private fun saveTrip(
        r: RunSummary.Result,
        nowMs: Long,
        truthEnd: Int,
        blackoutDurationS: Double
    ) {
        try {
            val ai = speedAi

            val trip = TripSummary.Trip(
                createdMs = nowMs,
                appVersion = packageManager.getPackageInfo(
                    packageName,
                    0
                ).versionName ?: "?",
                phone = "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}",
                android = android.os.Build.VERSION.RELEASE,
                blackoutStartMs = blackoutStartMs,
                blackoutEndMs = if (blackoutEndMs > 0L) blackoutEndMs else null,
                blackoutDurationS = blackoutDurationS,
                result = r,
                verdict = RunSummary.verdict(r),
                aiShare = if (aiSteps + heldSteps > 0)
                    aiSteps.toDouble() / (aiSteps + heldSteps)
                else null,
                modelRuns = ai?.inferences ?: 0L,
                inferMsMean = if (ai != null && ai.inferences > 0)
                    ai.totalInferMs / ai.inferences
                else 0.0,
                inferMsMax = ai?.maxInferMs ?: 0.0,
                roadBinding = roadBindingOn,
                routeName = if (roadBindingOn) road?.routeName else null,
                truthPath = gpsPoints.subList(
                    blackoutFromIndex.coerceIn(0, truthEnd),
                    truthEnd
                ).toList(),
                predictedPath = predPoints.toList(),
                reacquireJumpM = reacquireJumpM,
                dotSource = if (dotFromFusion) "fusion" else "road_tracker",
                destination = planned?.let { p ->
                    TripSummary.Dest(p.place.name, p.place.lat, p.place.lon, navigator?.arrived == true,
                        guide?.let { (it.lengthM - progressS).coerceAtLeast(0.0) })
                },
                plannedRoute = planned?.let { p ->
                    TripSummary.PlannedRouteInfo(p.route.lengthM, p.route.seconds, p.route.points)
                },
                wrongTurns = wrongTurns.toList()
            )

            val f = TripSummary.save(
                trip,
                TripSummary.folderFor(this)
            )

            android.util.Log.i(
                "Dhruva",
                "trip summary -> ${f.absolutePath}"
            )

            Toast.makeText(
                this,
                "Saved ${f.name}",
                Toast.LENGTH_SHORT
            ).show()
        } catch (e: Exception) {
            android.util.Log.e(
                "Dhruva",
                "trip summary failed",
                e
            )

            Toast.makeText(
                this,
                "Could not save the trip summary: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }
    /** Bind to a route from a GOOD fix, and say exactly what happened. */
    private fun bindRoad(loc: Location) {
        val rb = road ?: return
        // Bearing decides which way along the route we travel; below ~1 m/s it is noise.
        // A planned route is already in travel order, so it is always ridden forwards.
        val bearing = if (planned == null && loc.hasBearing() && loc.speed > 1f) loc.bearing else null
        if (rb.start(loc.latitude, loc.longitude, bearing)) {
            tvErrorLabel.text = "bound to %s (%.0f m off, %s)".format(
                rb.routeName, rb.snapDistanceM, if (rb.direction > 0) "forward" else "reverse")
        } else {
            roadBindingOn = false                 // set BEFORE the switch, so its listener is a no-op
            switchRoadBinding.isChecked = false
            tvErrorLabel.text = "No route for this area (nearest %.0f m away) — riding free"
                .format(rb.snapDistanceM)
        }
    }

    override fun onResume() {
        super.onResume()
        mapView.onResume()
        gyro?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        // gravity at GAME too: the speed model needs all three streams well above 10 Hz
        gravitySensor?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        linAccSensor?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gnssWatcher?.start()
        fusionHandler.removeCallbacks(fusionTick)
        fusionHandler.post(fusionTick)
    }

    override fun onDestroy() {
        super.onDestroy()
        voice?.shutdown()
        perf?.stop()
        speedAi?.close()
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
        sm.unregisterListener(this)
        gnssWatcher?.stop()
        fusionHandler.removeCallbacks(fusionTick)
        lastFusionMs = 0L
        speedAi?.pause()
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
                // A cached fix from before this screen opened, or a jump no vehicle can make, must
                // never reach the map, the route or the score.
                val ageS = (android.os.SystemClock.elapsedRealtimeNanos() - loc.elapsedRealtimeNanos) / 1e9
                when (fixFilter.check(loc.latitude, loc.longitude, loc.time, ageS)) {
                    FixFilter.Verdict.STALE, FixFilter.Verdict.GLITCH -> return
                    FixFilter.Verdict.NEW_TRACK -> { paths.breakTruth(); recentFixes.clear(); lastAcceptedGps = null }
                    FixFilter.Verdict.ACCEPT -> {}
                }

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
                    lastTruthPoint = GeoPoint(loc.latitude, loc.longitude)
                    gpsPoints.add(loc.latitude to loc.longitude)
                }
                gnss.onFix(SystemClock.elapsedRealtime(), loc.accuracy)

                if (lat0 == null) {
                    // first fix ever — this becomes the map's local origin
                    lat0 = loc.latitude
                    lon0 = loc.longitude
                    mapView.controller.setCenter(GeoPoint(loc.latitude, loc.longitude))
                }
                val (x, y) = toXY(loc.latitude, loc.longitude, lat0!!, lon0!!)
                // Fusion (Part 7): start at the first fix, then hand it every fix while GNSS is usable.
                if (!fusionReady) {
                    fusion.initialise(x, y, if (loc.hasSpeed()) loc.speed.toDouble() else 0.0,
                        if (loc.hasBearing()) Math.toRadians(90.0 - loc.bearing) else 0.0)
                    fusionReady = true
                } else if (gnss.mode == GnssMode.GNSS) {
                    pendingFix = doubleArrayOf(x, y, loc.accuracy.toDouble().coerceAtLeast(3.0),
                        if (loc.hasSpeed()) loc.speed.toDouble() else Double.NaN)
                }

                // Freeze the speed the moment the blackout starts -- during one we
                // are pretending these fixes do not exist.
                lastFixMs = System.currentTimeMillis()
                if (!blackoutOn) {
                    // The held speed is the MEDIAN of the last 30 s of moving fixes, not the
                    // last single fix. On 11 Sept the switch was flipped 12 m into the ride
                    // while every fix still read 0.0-0.35 m/s: the held speed was zero and
                    // the dot never moved.
                    if (loc.hasSpeed() && loc.speed > MOVING_MPS) {
                        recentSpeeds.addLast(lastFixMs to loc.speed.toDouble())
                    }
                    while (recentSpeeds.isNotEmpty() &&
                           lastFixMs - recentSpeeds.first().first > SPEED_WINDOW_MS) {
                        recentSpeeds.removeFirst()
                    }
                    if (recentSpeeds.isNotEmpty()) {
                        val sorted = recentSpeeds.map { it.second }.sorted()
                        lastGoodSpeedMps = sorted[sorted.size / 2]
                    }
                }

                // Gyro bias can only be measured while genuinely still, and only
                // GNSS can tell us that -- accelerometer variance cannot: measured
                // on the 7 Sept ride, a stopped engine idles at std 1.751 and a
                // moving one at 1.757.
                stationaryNow = !blackoutOn && loc.speed < 0.5f

                if (deadReckoner == null) {
                    deadReckoner = DeadReckoner(
                        heading = if (loc.hasBearing()) Math.toRadians(90.0 - loc.bearing.toDouble()) else 0.0,
                        speed = loc.speed.toDouble(), x = x, y = y
                    )
                } else if (isRealMovement && !blackoutOn) {
                    // ONLY while GNSS is genuinely in use.
                    //
                    // The branches were the wrong way round: this arm rebuilt the
                    // reckoner with heading = 0.0 on every healthy fix, and the arm
                    // below -- which is reached only when blackoutOn is true --
                    // called onGnssFix(), so the "blackout" estimate was still being
                    // corrected from live GPS on every fix. The dot would have sat
                    // exactly on the truth line, read 0 m apart, and the confidence
                    // circle would never have grown. It would have looked perfect
                    // and measured nothing.
                    deadReckoner!!.onGnssFix(x, y, loc.speed.toDouble())
                    if (loc.hasBearing() && loc.speed > 1.0f) {
                        deadReckoner!!.setHeadingFromBearing(loc.bearing)
                    }
                }
                // During a blackout: nothing. No position, no speed, no heading.
                // That is what makes it a blackout.

                // only let a real fix move the dot when we are NOT simulating a blackout
                if (!blackoutOn && isRealMovement && !dotFromFusion) {
                    placeDot(loc.latitude, loc.longitude, isBlue = true, radiusM = 0.0)
                    recentFixes.addLast(loc)
                    while (recentFixes.size > 10) recentFixes.removeFirst()
                    guideWithGps(loc)
                }
            }
        }, Looper.getMainLooper())
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LINEAR_ACCELERATION -> speedAi?.onSensor(ImuFeatures.ACC, event.timestamp, event.values)
            Sensor.TYPE_GYROSCOPE -> speedAi?.onSensor(ImuFeatures.GYRO, event.timestamp, event.values)
            Sensor.TYPE_GRAVITY -> speedAi?.onSensor(ImuFeatures.GRAV, event.timestamp, event.values)
        }
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

        gyroBias.observe(event.values[0], event.values[1], event.values[2], stationaryNow)
        if (gyroSampleCount % 100 == 0) {
            val bias = gyroBias.statusText()
            // A rejected jump means something upstream produced an impossible
            // position. It must read 0. If it climbs, say so -- do not ride on.
            // If the PHONE's location goes off, there is no truth left to score against.
            val gpsAgeS = if (lastFixMs > 0) (System.currentTimeMillis() - lastFixMs) / 1000 else 0
            val truthNote = if (gpsAgeS > 5) " · REAL GPS LOST ${gpsAgeS}s — score invalid" else ""
            val aiNote = speedAi?.let {
                when {
                    it.freshAt(event.timestamp / 1e9) ->
                        " · AI speed %.0f km/h (%.1f ms)"
                            .format(it.speedMps * 3.6, it.lastInferMs)

                    it.ready ->
                        " · AI speed ready (warm)"

                    else ->
                        " · AI speed warming up %.0f/30 s"
                            .format(it.bufferedS)
                }
            } ?: " · AI speed OFF (model missing)"
            val gnssNote = " · " + (if (gnss.mode == GnssMode.GNSS) "GNSS"
            else "DR: ${gnss.reason}") +
                    (if (gnss.satellitesUsed >= 0) " (${gnss.satellitesUsed} sats)" else "")
            val fusionNote = " · fusion %.1f Hz%s".format(fusionHz, if (dotFromFusion) " (drawing)" else "")
            tvSensorStatus.text =
                (if (paths.dropped > 0) "$bias · ${paths.dropped} BAD FRAMES" else bias) + aiNote + gnssNote + fusionNote + truthNote

        }


        // Rotation about the TRUE vertical, not the phone's z axis. Reduces to
        // gyro z when the phone happens to be flat, and stays correct when it is
        // not -- feeding raw z to a yaw-dependent estimator cost us 12.2% vs 8.3%
        // drift once already (RESOURCES 8g).
        val gx = gyroBias.correctX(event.values[0])
        val gy = gyroBias.correctY(event.values[1])
        val gz = gyroBias.correctZ(event.values[2])

        val yawRate = (gx * gravityUnit[0] +
                gy * gravityUnit[1] +
                gz * gravityUnit[2]).toDouble()
        // Fusion (Part 7) needs the turn rate all the time, not only in a blackout.
        yawSum += yawRate * dt; yawTime += dt
        if (!blackoutOn) return // only drive the dot with dead reckoning during a blackout

        // Speed: the trained model when it has a fresh answer, else the GPS speed held at the cut.
        val nowS = event.timestamp / 1e9
        val ai = speedAi
        val speedNow = if (ai != null && ai.freshAt(nowS)) { aiSteps++; ai.speedMps } else { heldSteps++; lastGoodSpeedMps }
        dr.setSpeed(speedNow)
        val (x, y, radius) = dr.step(yawRate, dt)
        var lat: Double; var lon: Double
        val free = toLatLon(x, y, o0, o1)

        val rb = road
        if (roadBindingOn && rb != null && rb.bound) {
            // Constrained to the road: the dot cannot drift sideways at all, and
            // only how far ALONG the road can be wrong.
            val p = rb.advance(speedNow, dt)
            lat = p.first; lon = p.second
        } else {
            lat = free.first; lon = free.second
        }


        placeDot(lat, lon, isBlue = false, radiusM = radius)

        // Wrong turn with GPS off: integrate the gyro, feed the route guard 10 times a second.
        psiSinceCut += yawRate * dt
        if (routeGuard != null && rb != null && rb.bound) watchForWrongTurn(nowS, rb)

        // Keep guiding with GPS off: the dot's distance along the planned route drives the voice.
        val nav = navigator
        if (nav != null && planned != null && roadBindingOn && rb != null && rb.bound) {
            val nowMs = System.currentTimeMillis()
            if (nowMs - lastGuideMs >= 1000L) {
                lastGuideMs = nowMs
                progressS = rb.arcM
                showGuidance(nav.update(progressS, radius) { a, d, r -> voice?.phrase(a, d, r) ?: "$a." })
            }
        }

        // Guide 3 add-on: predicted track + live divergence label
        paths.addPredicted(lat, lon)
        predPoints.add(lat to lon)
        val apart = "%.0f m apart".format(paths.currentErrorMetres())
        tvErrorLabel.text =
            if (roadBindingOn && rb != null && rb.atRouteEnd) "$apart · end of known route, holding"
            else apart

        // Contribution 9: the wording loosens as the circle grows.
        voice?.announceConfidence(radius, radius / 2.146 / 0.19)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    companion object {
        /** No second re-route within this: a fresh route needs a few fixes to settle. */
        const val REROUTE_GAP_MS = 8_000L
        /** The new route starts this far from the way the rider is going: a U-turn. */
        const val U_TURN_DEG = 120.0
    }

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

        // During a blackout, centring on the predicted dot pushes the truth line
        // off the screen -- and the comparison between them IS the demo. In the
        // 9 Sept recording the blue line left the right edge entirely. Centre on
        // the midpoint of the two heads instead.
        val t = lastTruthPoint
        if (blackoutOn && t != null) {
            mapView.controller.animateTo(
                GeoPoint((t.latitude + lat) / 2.0, (t.longitude + lon) / 2.0)
            )
        } else {
            mapView.controller.animateTo(point)
        }
        mapView.invalidate()
    }
}