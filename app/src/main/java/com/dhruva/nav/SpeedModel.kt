package com.dhruva.nav

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.FloatBuffer

/**
 * The trained speed model (ResNet1D, 997,633 parameters), running on the phone with ONNX Runtime.
 *
 * Input: the last 30 s of motion as 6 levelled channels at 10 Hz, normalised, shape (1, 6, 300),
 * built by [ImuFeatures]. Output: metres travelled over that window, so speed = metres / 30 s.
 * The normalisation file ships next to the model and comes from the same checkpoint.
 */
class SpeedModel(
    ctx: Context,
    modelAsset: String = "speed_app.onnx",
    normAsset: String = "speed_app_norm.json"
) : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val window: Int
    val modelHz: Double
    val mean: DoubleArray
    val std: DoubleArray
    /** The level frame the model was trained in. The phone builds its inputs the same way. */
    val frame: LevelFrame

    init {
        val norm = JSONObject(ctx.assets.open(normAsset).bufferedReader().use { it.readText() })
        // A model trained on look-ahead features would run here without complaint and be quietly
        // wrong (RESOURCES 8ao). Refuse it; the app then falls back to the held GPS speed.
        frame = when (norm.optString("frame")) {
            "replica" -> LevelFrame.REPLICA
            "gravity" -> LevelFrame.GRAVITY
            else -> throw IllegalStateException(
                "speed model trained in frame '${norm.optString("frame")}', which a phone cannot compute")
        }
        window = norm.getInt("window")
        modelHz = norm.getDouble("model_hz")
        mean = DoubleArray(ImuFeatures.CHANNELS) { norm.getJSONArray("mean").getDouble(it) }
        std = DoubleArray(ImuFeatures.CHANNELS) { norm.getJSONArray("std").getDouble(it) }
        session = env.createSession(ctx.assets.open(modelAsset).use { it.readBytes() }, OrtSession.SessionOptions())
    }

    /** Metres travelled over the window. [input] is (1, 6, window) flattened channel-first. */
    fun metres(input: FloatArray): Float {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, ImuFeatures.CHANNELS.toLong(), window.toLong())).use { t ->
            session.run(mapOf("imu" to t)).use { r ->
                @Suppress("UNCHECKED_CAST")
                return (r.get(0).value as Array<FloatArray>)[0][0]
            }
        }
    }

    /** Speed in m/s, never negative. */
    fun speedMps(input: FloatArray): Double = maxOf(metres(input).toDouble() / (window / modelHz), 0.0)

    override fun close() = session.close()
}
