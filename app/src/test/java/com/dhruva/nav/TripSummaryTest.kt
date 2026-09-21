package com.dhruva.nav

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** trip_summary.json must be valid JSON with the fields the laptop reads. */
class TripSummaryTest {

    private fun sample() = TripSummary.Trip(
        createdMs = 1_758_000_000_000L,
        appVersion = "1.0",
        phone = "Pixel \"7\"",
        android = "14",
        blackoutStartMs = 1_758_000_000_000L - 90_000L,
        blackoutEndMs = null,
        blackoutDurationS = 89.0,
        result = RunSummary.Result(
            485.4,
            89.0,
            21.8,
            4.49,
            true,
            5.45,
            197.9,
            170,
            49.8
        ),
        verdict = "PASS — 4.5% drift, inside ISRO's 10% limit",
        aiShare = 0.97,
        modelRuns = 178,
        inferMsMean = 3.2,
        inferMsMax = 11.0,
        roadBinding = true,
        routeName = "longloop_appride",
        truthPath = listOf(
            15.3704 to 75.1219,
            15.3705 to 75.1220
        ),
        predictedPath = listOf(
            15.3704 to 75.1219,
            15.3706 to 75.1221
        )
    )

    @Test
    fun destinationAndWrongTurnsAreRecorded() {
        val j = TripSummary.toJson(sample().copy(
            destination = TripSummary.Dest("Unakal Kere", 15.3782, 75.1048, arrived = true, remainingM = 0.0),
            plannedRoute = TripSummary.PlannedRouteInfo(1483.0, 174.0, listOf(15.3693 to 75.1219, 15.3700 to 75.1210)),
            wrongTurns = listOf(TripSummary.WrongTurn(212.0, 92.0, 92.5, 5, 1460.0, uTurn = false))
        ))
        for (key in listOf(
            "\"destination\": {\"name\": \"Unakal Kere\"",
            "\"arrived\": true",
            "\"planned_route\": {\"length_m\": 1483.000",
            "\"at_m\": 212.000",
            "\"candidate_roads\": 5",
            "\"u_turn\": false"
        )) {
            assertTrue("missing $key in\n$j", j.contains(key))
        }
    }

    @Test
    fun containsTheFieldsTheLaptopReads() {
        val j = TripSummary.toJson(sample())

        for (key in listOf(
            "\"format\": \"dhruva-trip/1\"",
            "\"drift_pct\": 4.490",
            "\"ai_share\": 0.970",
            "\"end_ms\": null",
            "\"duration_s\": 89.000",
            "\"reacquire_jump_m\": null",
            "\"wrong_turn_alarms\": []",
            "\"phone\": {\"model\": \"Pixel \\\"7\\\"\""
        )) {
            assertTrue("missing $key", j.contains(key))
        }
    }

    @Test
    fun writesAFileTheLaptopCanOpen() {
        val dir = File("build/test-trips").apply { mkdirs() }

        val f = TripSummary.save(sample(), dir)

        assertTrue(f.exists() && f.length() > 100)

        println("wrote ${f.absolutePath}")
    }
}