package com.dhruva.nav

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

/**
 * CONTRIBUTION 9 — confidence-aware guidance.
 *
 * The idea: a navigation system that has lost GNSS should not keep speaking with
 * the same certainty it had with twelve satellites. As the confidence circle
 * grows the wording should loosen with it, so the rider is told how much to
 * trust what they are hearing.
 *
 *   tight   (< 25 m)  "Turn left in 100 metres."
 *   medium  (< 75 m)  "Turn left in about 100 metres."
 *   loose   (>= 75 m) "Turn left in roughly 100 metres — position uncertain."
 *
 * This only became honest once contribution 5 gave us a radius worth quoting.
 * The filter's own covariance was overconfident by three to four orders of
 * magnitude (RESOURCES 8e); the radius used here is the empirical one,
 * sigma = 0.19 x distance since the last fix, which measured 86% coverage at a
 * nominal 90%.
 *
 * Deliberately NOT a language model. It is a lookup on a measured number, it
 * runs with the network off, and it cannot invent a road that is not there.
 */
class VoiceGuide(ctx: Context) {

    private var tts: TextToSpeech? = null
    private var ready = false
    private var lastSpokenBand = ""
    private var lastSpeakMs = 0L

    init {
        tts = TextToSpeech(ctx) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.UK
                ready = true
            }
        }
    }

    enum class Band { TIGHT, MEDIUM, LOOSE }

    fun band(radius90M: Double): Band = when {
        radius90M < 25.0 -> Band.TIGHT
        radius90M < 75.0 -> Band.MEDIUM
        else -> Band.LOOSE
    }

    /** How an instruction should be phrased at this confidence. */
    fun phrase(instruction: String, distanceM: Double, radius90M: Double): String =
        when (band(radius90M)) {
            Band.TIGHT  -> "$instruction in ${round10(distanceM)} metres."
            Band.MEDIUM -> "$instruction in about ${round10(distanceM)} metres."
            Band.LOOSE  -> "$instruction in roughly ${round10(distanceM)} metres. " +
                           "Position uncertain to ${radius90M.toInt()} metres."
        }

    /**
     * Announce that confidence has changed band. Speaks at most once per band
     * change and never more often than MIN_GAP_MS -- a system that narrates its
     * own uncertainty every second is worse than one that says nothing.
     */
    fun announceConfidence(radius90M: Double, distanceSinceFixM: Double) {
        val b = band(radius90M).name
        val now = System.currentTimeMillis()
        if (b == lastSpokenBand || now - lastSpeakMs < MIN_GAP_MS) return
        lastSpokenBand = b
        lastSpeakMs = now
        say(when (band(radius90M)) {
            Band.TIGHT  -> "Satellite fix lost. Still tracking accurately."
            Band.MEDIUM -> "Position accurate to about ${radius90M.toInt()} metres."
            Band.LOOSE  -> "${distanceSinceFixM.toInt()} metres without a fix. " +
                           "Position uncertain to ${radius90M.toInt()} metres."
        })
    }

    fun say(text: String) {
        if (!ready) return
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "dhruva")
    }

    fun reset() { lastSpokenBand = ""; lastSpeakMs = 0L }

    fun shutdown() {
        tts?.stop(); tts?.shutdown(); tts = null; ready = false
    }

    private fun round10(m: Double) = (Math.round(m / 10.0) * 10).toInt()

    companion object {
        const val MIN_GAP_MS = 8_000L
    }
}
