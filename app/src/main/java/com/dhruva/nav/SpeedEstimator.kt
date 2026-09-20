package com.dhruva.nav

import android.content.Context

import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Speed from the trained model, live on the phone.
 *
 * Sensor events ALWAYS go into [ImuFeatures], so the last 30 s are ready the
 * instant GPS drops.
 * The model itself runs only while [active] (a GNSS blackout): every
 * [everyS] seconds the latest
 * window is copied and the model runs on a background thread, so the map
 * never stutters. The
 * answer is the speed over the 30 s that just ENDED -- a phone cannot look
 * ahead.
 *
 * Callers must check [freshAt] and fall back to the GPS speed held before
 * the blackout when the
 * model has no answer yet (the first 30 s of a session) or its answer is
 * stale.
 */
class SpeedEstimator internal constructor(
    frame: LevelFrame,
    window: Int,
    private val mean: DoubleArray,
    private val std: DoubleArray,
    private val predict: (FloatArray) -> Double, // normalised window
    private val closeModel: () -> Unit,
    private val everyS: Double = 0.5,
    private val worker: Executor = Executors.newSingleThreadExecutor()
) : AutoCloseable {

    /** The app's constructor: loads speed_app.onnx and its normalisation
     * from assets. */
    constructor(ctx: Context, everyS: Double = 0.5) : this(SpeedModel(ctx), everyS)

    private constructor(m: SpeedModel, everyS: Double) :
            this(
                m.frame,
                m.window,
                m.mean,
                m.std,
                { m.speedMps(it) },
                { m.close() },
                everyS
            )

    companion object {
        /** A gap longer than this in any sensor stream means the stream
         * stalled (screen off). */
        private const val STALL_S = 0.5
        private const val FRESH_S = 3.0
    }

    private val window = window
    private val features = ImuFeatures(frame, window) // the frame comes
    // from the model itself

    private val busy = AtomicBoolean(false)
    private val lastEventS = DoubleArray(3) { Double.NaN }
    private var lastInferS = Double.NEGATIVE_INFINITY

    /** True during a GNSS blackout: only then does the model run. */
    @Volatile var active = false
        private set

    @Volatile var speedMps = Double.NaN
        private set

    /** Sensor-clock time (s) of the end of the window behind [speedMps]. */
    @Volatile var speedTimeS = Double.NaN
        private set

    @Volatile var lastInferMs = 0.0
        private set

    @Volatile var inferences = 0L
        private set

    @Volatile var resets = 0L
        private set

    /** Seconds of motion buffered, up to the 30 s the model needs. */
    val bufferedS: Double
        get() = features.filled / ImuFeatures.HZ

    /** True when a full 30 s window is buffered, so the model can answer the
     * moment it is needed. */
    val ready: Boolean
        get() = features.filled >= window

    /** True when the model has an answer less than [FRESH_S] old at sensor
     * time [nowS]. */
    fun freshAt(nowS: Double) =
        !speedMps.isNaN() && nowS - speedTimeS <= FRESH_S

    /**
     * Start (GPS lost) or stop (GPS back) running the model. Starting asks
     * for an answer at the
     * very next 10 Hz row; stopping forgets the old answer so a later
     * blackout never uses it.
     */
    fun setActive(on: Boolean) {
        if (on == active) return

        active = on

        if (on) {
            lastInferS = Double.NEGATIVE_INFINITY
        } else {
            speedMps = Double.NaN
            speedTimeS = Double.NaN
        }
    }

    /**
     * One event from TYPE_LINEAR_ACCELERATION ([ImuFeatures.ACC]),
     * TYPE_GYROSCOPE ([ImuFeatures.GYRO])
     * or TYPE_GRAVITY ([ImuFeatures.GRAV]); [timestampNs] is
     * SensorEvent.timestamp.
     */
    fun onSensor(kind: Int, timestampNs: Long, v: FloatArray) {
        val tS = timestampNs / 1e9
        val prev = lastEventS[kind]

        if (!prev.isNaN() && (tS - prev > STALL_S || tS < prev)) {
            features.reset()
            lastEventS.fill(Double.NaN)
            lastInferS = Double.NEGATIVE_INFINITY
            resets++
        }

        lastEventS[kind] = tS

        if (
            features.push(
                kind,
                tS,
                v[0].toDouble(),
                v[1].toDouble(),
                v[2].toDouble()
            ) == 0
        ) return

        if (!active) return // GPS is fine: keep buffering, don't run the model

        val rowS = features.lastRowTimeS

        if (rowS - lastInferS < everyS || busy.get()) return

        val input = features.window(mean, std)
            ?: return // a new array each time: the worker owns it

        lastInferS = rowS
        busy.set(true)

        worker.execute {
            try {
                val t0 = System.nanoTime()
                val v = predict(input)

                lastInferMs = (System.nanoTime() - t0) / 1e6

                if (active) { // GPS may have come back while we ran
                    speedMps = v
                    speedTimeS = rowS
                }

                inferences++
            } catch (_: Exception) {
                // keep the previous answer; freshAt() will expire it
            } finally {
                busy.set(false)
            }
        }
    }

    /** Call from onPause: the next events must not be joined to the old
     * ones. */
    fun pause() {
        features.reset()
        lastEventS.fill(Double.NaN)
        lastInferS = Double.NEGATIVE_INFINITY
    }

    override fun close() {
        (worker as? ExecutorService)?.shutdown()
        closeModel()
    }
}