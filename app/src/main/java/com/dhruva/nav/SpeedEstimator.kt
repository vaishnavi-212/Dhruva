package com.dhruva.nav

import android.content.Context
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Speed from the trained model, live on the phone.
 *
 * Sensor events arrive on the main thread and go into [ImuFeatures]. Every [everyS] seconds the
 * latest 30 s window is copied and the model runs on a background thread, so the map never
 * stutters. The answer is the speed over the 30 s that just ENDED -- a phone cannot look ahead.
 *
 * Callers must check [freshAt] and fall back to the GPS speed held before the blackout when the
 * model has no answer yet (the first 30 s of a session) or its answer is stale.
 */
class SpeedEstimator(
    ctx: Context,
    private val everyS: Double = 0.5
) : AutoCloseable {

    companion object {
        /** A gap longer than this in any sensor stream means the stream stalled (screen off). */
        private const val STALL_S = 0.5
        private const val FRESH_S = 3.0
    }

    private val model = SpeedModel(ctx)
    private val features = ImuFeatures(model.frame, model.window)   // the frame comes from the model itself
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private val lastEventS = DoubleArray(3) { Double.NaN }
    private var lastInferS = Double.NEGATIVE_INFINITY

    @Volatile var speedMps = Double.NaN; private set
    /** Sensor-clock time (s) of the end of the window behind [speedMps]. */
    @Volatile var speedTimeS = Double.NaN; private set
    @Volatile var lastInferMs = 0.0; private set
    @Volatile var inferences = 0L; private set
    @Volatile var resets = 0L; private set

    /** Seconds of motion buffered, up to the 30 s the model needs. */
    val bufferedS: Double get() = features.filled / ImuFeatures.HZ

    /** True when the model has an answer less than [FRESH_S] old at sensor time [nowS]. */
    fun freshAt(nowS: Double) = !speedMps.isNaN() && nowS - speedTimeS <= FRESH_S

    /**
     * One event from TYPE_LINEAR_ACCELERATION ([ImuFeatures.ACC]), TYPE_GYROSCOPE ([ImuFeatures.GYRO])
     * or TYPE_GRAVITY ([ImuFeatures.GRAV]); [timestampNs] is SensorEvent.timestamp.
     */
    fun onSensor(kind: Int, timestampNs: Long, v: FloatArray) {
        val tS = timestampNs / 1e9
        val prev = lastEventS[kind]
        if (!prev.isNaN() && (tS - prev > STALL_S || tS < prev)) {
            features.reset(); lastEventS.fill(Double.NaN); lastInferS = Double.NEGATIVE_INFINITY; resets++
        }
        lastEventS[kind] = tS
        if (features.push(kind, tS, v[0].toDouble(), v[1].toDouble(), v[2].toDouble()) == 0) return
        val rowS = features.lastRowTimeS
        if (rowS - lastInferS < everyS || busy.get()) return
        val input = features.window(model.mean, model.std) ?: return   // a new array each time: the worker owns it
        lastInferS = rowS
        busy.set(true)
        worker.execute {
            try {
                val t0 = System.nanoTime()
                val v = model.speedMps(input)
                lastInferMs = (System.nanoTime() - t0) / 1e6
                speedMps = v
                speedTimeS = rowS
                inferences++
            } catch (_: Exception) {
                // keep the previous answer; freshAt() will expire it
            } finally {
                busy.set(false)
            }
        }
    }

    /** Call from onPause: the next events must not be joined to the old ones. */
    fun pause() {
        features.reset(); lastEventS.fill(Double.NaN); lastInferS = Double.NEGATIVE_INFINITY
    }

    override fun close() {
        worker.shutdown()
        model.close()
    }
}
