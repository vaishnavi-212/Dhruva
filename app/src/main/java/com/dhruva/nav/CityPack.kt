package com.dhruva.nav

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.PriorityQueue
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * The offline city: roads to route on and places to search for, with no network.
 *
 * Built on the laptop from OpenStreetMap (engine/scripts/build_city_pack.py) and shipped as
 * assets/city_pack_hubli.json: 835 places, 1,474 km of road around KLE Tech, 1.15 MB.
 * This is a line-for-line port of engine/dhruva/citypack.py; CityPackGoldenTest checks that both
 * give the same routes and the same search results.
 *
 * Route cost is travel TIME at a typical two-wheeler speed per road class, so routes prefer main
 * roads. One-way streets are respected. A place's "arrive" nodes are the road points within 40 m of
 * its outline (60 m of a point), so a route to a lake ends at the nearest reachable shore road.
 */
class CityPack(
    val lat: DoubleArray,
    val lon: DoubleArray,
    val ways: List<Way>,
    val places: List<Place>,
) {
    class Way(val nodes: IntArray, val cls: Int, val oneway: Int, val private: Boolean)
    class Place(val name: String, val alt: String, val kind: String, val lat: Double, val lon: Double, val arrive: IntArray)
    /** [nodes] are the road nodes of points[1..]; points[0] is the start, snapped onto the road. */
    class Route(val points: List<Pair<Double, Double>>, val nodes: IntArray, val lengthM: Double, val seconds: Double)

    // adjacency in flat arrays, in the same order Python builds its lists
    private val adjStart: IntArray
    private val adjTo: IntArray
    private val adjCost: DoubleArray
    // segments for snapping: node a, node b, way index
    private val segA: IntArray
    private val segB: IntArray
    private val segWay: IntArray
    // how many different road points each node connects to, ignoring one-ways: 3+ is a junction
    private val degreeOf: IntArray
    private val neighbourOf: Array<IntArray>

    fun degree(node: Int): Int = degreeOf[node]

    /** Road points joined to [node] by a road, either direction (a rider can physically go either way). */
    fun neighbours(node: Int): IntArray = neighbourOf[node]

    init {
        val n = lat.size
        val count = IntArray(n)
        var nSeg = 0
        for (w in ways) {
            for (k in 0 until w.nodes.size - 1) {
                if (w.oneway != -1) count[w.nodes[k]]++
                if (w.oneway != 1) count[w.nodes[k + 1]]++
                nSeg++
            }
        }
        adjStart = IntArray(n + 1)
        for (i in 0 until n) adjStart[i + 1] = adjStart[i] + count[i]
        adjTo = IntArray(adjStart[n]); adjCost = DoubleArray(adjStart[n])
        val fill = adjStart.copyOf(n)
        segA = IntArray(nSeg); segB = IntArray(nSeg); segWay = IntArray(nSeg)
        var s = 0
        for ((wi, w) in ways.withIndex()) {
            val f = secondsPerMetre(w)
            for (k in 0 until w.nodes.size - 1) {
                val a = w.nodes[k]; val b = w.nodes[k + 1]
                val c = metres(lat[a], lon[a], lat[b], lon[b]) * f
                if (w.oneway != -1) { adjTo[fill[a]] = b; adjCost[fill[a]] = c; fill[a]++ }
                if (w.oneway != 1) { adjTo[fill[b]] = a; adjCost[fill[b]] = c; fill[b]++ }
                segA[s] = a; segB[s] = b; segWay[s] = wi; s++
            }
        }
        val nb = Array(n) { HashSet<Int>(4) }
        for (i in 0 until nSeg) { nb[segA[i]].add(segB[i]); nb[segB[i]].add(segA[i]) }
        degreeOf = IntArray(n) { nb[it].size }
        neighbourOf = Array(n) { nb[it].toIntArray().also { a -> a.sort() } }
    }

    // search words for every place, worked out once at load (off the main thread) so typing costs
    // only comparisons
    private val placeWords: List<List<String>> = places.map { tokens(it.name + " " + it.alt) }
    private val nameLengths: List<List<Int>> =
        places.map { p -> listOf(p.name, p.alt).filter { it.isNotEmpty() }.map { tokens(it).size } }

    class Snap(val a: Int, val b: Int, val t: Double, val distM: Double, val way: Int)

    /** Nearest point on any road. */
    fun snap(la: Double, lo: Double): Snap {
        val k = cos(Math.toRadians(la))
        var best = Double.MAX_VALUE; var bi = 0; var bt = 0.0
        for (i in segA.indices) {
            val a = segA[i]; val b = segB[i]
            val ax = (lon[a] - lo) * k; val ay = lat[a] - la
            val bx = (lon[b] - lo) * k; val by = lat[b] - la
            val dx = bx - ax; val dy = by - ay
            val l2 = dx * dx + dy * dy
            val t = if (l2 == 0.0) 0.0 else minOf(1.0, maxOf(0.0, -(ax * dx + ay * dy) / l2))
            val ex = ax + t * dx; val ey = ay + t * dy
            val d2 = ex * ex + ey * ey
            if (d2 < best) { best = d2; bi = i; bt = t }
        }
        return Snap(segA[bi], segB[bi], bt, Math.toRadians(sqrt(best)) * R_EARTH, segWay[bi])
    }

    /** Fastest path from a position to ANY of the target nodes; null if none can be reached. */
    fun route(la: Double, lo: Double, targets: IntArray): Route? {
        val isTarget = HashSet<Int>(targets.size * 2).apply { targets.forEach { add(it) } }
        val sn = snap(la, lo)
        val a = sn.a; val b = sn.b; val t = sn.t
        val w = ways[sn.way]
        val f = secondsPerMetre(w)
        val len = metres(lat[a], lon[a], lat[b], lon[b])
        val plat = lat[a] + t * (lat[b] - lat[a]); val plon = lon[a] + t * (lon[b] - lon[a])

        val dist = HashMap<Int, Double>(); val prev = HashMap<Int, Int>()
        // (cost, node), ties broken by node index exactly like Python's heapq on tuples
        val pq = PriorityQueue<Pair<Double, Int>>(compareBy<Pair<Double, Int>> { it.first }.thenBy { it.second })
        if (w.oneway != 1) { dist[a] = t * len * f; pq.add(dist[a]!! to a) }
        if (w.oneway != -1 && (1 - t) * len * f < (dist[b] ?: Double.POSITIVE_INFINITY)) {
            dist[b] = (1 - t) * len * f; pq.add(dist[b]!! to b)
        }
        val done = HashSet<Int>()
        while (pq.isNotEmpty()) {
            val (d, u) = pq.poll()!!
            if (!done.add(u)) continue
            if (u in isTarget) {
                val path = arrayListOf(u)
                while (prev.containsKey(path.last())) path.add(prev[path.last()]!!)
                path.reverse()
                val pts = ArrayList<Pair<Double, Double>>(path.size + 1)
                pts.add(plat to plon)
                for (i in path) pts.add(lat[i] to lon[i])
                var length = 0.0
                for (i in 1 until pts.size) length += metres(pts[i - 1].first, pts[i - 1].second, pts[i].first, pts[i].second)
                return Route(pts, path.toIntArray(), length, d)
            }
            for (e in adjStart[u] until adjStart[u + 1]) {
                val v = adjTo[e]; val nd = d + adjCost[e]
                if (nd < (dist[v] ?: Double.POSITIVE_INFINITY)) { dist[v] = nd; prev[v] = u; pq.add(nd to v) }
            }
        }
        return null
    }

    /**
     * Fastest path from a point on the road a->b, travelling TOWARDS b (the way the rider is going),
     * to any target. Turning back towards a is allowed but costs [uTurnPenaltyS] extra, so it is chosen
     * only when going on is much longer. [uTurn] is true when the answer starts by turning back.
     */
    class Directed(val route: Route, val uTurn: Boolean)

    fun routeFrom(a: Int, b: Int, pLat: Double, pLon: Double, targets: IntArray, uTurnPenaltyS: Double = 30.0): Directed? {
        val isTarget = HashSet<Int>(targets.size * 2).apply { targets.forEach { add(it) } }
        val w = ways.firstOrNull { wy -> (0 until wy.nodes.size - 1).any { k ->
            (wy.nodes[k] == a && wy.nodes[k + 1] == b) || (wy.nodes[k] == b && wy.nodes[k + 1] == a) } }
        val f = if (w != null) secondsPerMetre(w) else 3.6 / KMH[5]
        val dist = HashMap<Int, Double>(); val prev = HashMap<Int, Int>()
        val pq = PriorityQueue<Pair<Double, Int>>(compareBy<Pair<Double, Int>> { it.first }.thenBy { it.second })
        dist[b] = metres(pLat, pLon, lat[b], lon[b]) * f; pq.add(dist[b]!! to b)
        val back = metres(pLat, pLon, lat[a], lon[a]) * f + uTurnPenaltyS
        if (back < (dist[a] ?: Double.POSITIVE_INFINITY)) { dist[a] = back; pq.add(back to a) }
        val done = HashSet<Int>()
        while (pq.isNotEmpty()) {
            val (d, u) = pq.poll()!!
            if (!done.add(u)) continue
            if (u in isTarget) {
                val path = arrayListOf(u)
                while (prev.containsKey(path.last())) path.add(prev[path.last()]!!)
                path.reverse()
                val pts = ArrayList<Pair<Double, Double>>(path.size + 1)
                pts.add(pLat to pLon)
                for (i in path) pts.add(lat[i] to lon[i])
                var length = 0.0
                for (i in 1 until pts.size) length += metres(pts[i - 1].first, pts[i - 1].second, pts[i].first, pts[i].second)
                return Directed(Route(pts, path.toIntArray(), length, d), uTurn = path.first() == a)
            }
            for (e in adjStart[u] until adjStart[u + 1]) {
                val v = adjTo[e]; val nd = d + adjCost[e]
                if (nd < (dist[v] ?: Double.POSITIVE_INFINITY)) { dist[v] = nd; prev[v] = u; pq.add(nd to v) }
            }
        }
        return null
    }

    /** Every typed word must match a word of the name; best matches first, then nearest first. */
    fun search(query: String, la: Double?, lo: Double?, limit: Int = 8): List<Pair<Place, Double>> {
        val qt = tokens(query)
        if (qt.isEmpty()) return emptyList()
        class Hit(val score: Int, val d: Double, val i: Int)
        val out = ArrayList<Hit>()
        for ((i, p) in places.withIndex()) {
            val words = placeWords[i]
            var score = 0; var ok = true
            for ((k, q) in qt.withIndex()) {
                val m = words.maxOfOrNull { tokenMatch(q, it, k == qt.size - 1) } ?: 0
                if (m == 0) { ok = false; break }
                score += m
            }
            if (!ok) continue
            if (nameLengths[i].any { it == qt.size }) score += 1
            val d = if (la != null && lo != null) metres(la, lo, p.lat, p.lon) else 0.0
            out.add(Hit(score, d, i))
        }
        out.sortWith(compareBy<Hit>({ -it.score }, { it.d }, { it.i }))
        return out.take(limit).map { places[it.i] to it.d }
    }

    companion object {
        const val R_EARTH = 6371008.8
        val KMH = doubleArrayOf(40.0, 35.0, 30.0, 25.0, 20.0, 20.0, 15.0, 10.0, 20.0)   // same classes as citypack.py
        const val PRIVATE_FACTOR = 5.0
        private val ALIASES = mapOf("kere" to "lake", "lake" to "kere", "stn" to "station", "rly" to "railway",
            "univ" to "university", "hosp" to "hospital")
        private val NON_WORD = Regex("[^a-z0-9 ]")

        fun secondsPerMetre(w: Way) = 3.6 / KMH[w.cls] * (if (w.private) PRIVATE_FACTOR else 1.0)

        /** Equirectangular distance, identical to citypack.metres. */
        fun metres(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
            val k = cos(Math.toRadians((lat1 + lat2) / 2))
            return R_EARTH * hypot(Math.toRadians(lon2 - lon1) * k, Math.toRadians(lat2 - lat1))
        }

        fun tokens(s: String): List<String> = NON_WORD.replace(s.lowercase(), " ").split(" ").filter { it.isNotEmpty() }

        /** True if a and b differ by at most one insert, delete or substitution. */
        fun edit1(x: String, y: String): Boolean {
            var a = x; var b = y
            if (kotlin.math.abs(a.length - b.length) > 1) return false
            if (a.length > b.length) { val tmp = a; a = b; b = tmp }
            var i = 0; var j = 0; var diff = 0
            while (i < a.length && j < b.length) {
                if (a[i] == b[j]) { i++; j++; continue }
                diff++
                if (diff > 1) return false
                if (a.length == b.length) i++
                j++
            }
            return diff + (b.length - j) + (a.length - i) <= 1
        }

        /** 2 exact (or alias), 1 prefix of the word being typed or one typo, 0 no match. */
        fun tokenMatch(q: String, w: String, last: Boolean): Int = when {
            q == w || ALIASES[q] == w -> 2
            last && q.length >= 2 && w.startsWith(q) -> 1
            q.length >= 4 && edit1(q, w) -> 1
            else -> 0
        }

        fun parse(json: String): CityPack {
            val o = JSONObject(json)
            require(o.getString("format") == "dhruva-city/1") { "unknown city pack format" }
            val la = o.getJSONArray("lat"); val lo = o.getJSONArray("lon")
            val lat = DoubleArray(la.length()) { la.getLong(it) / 1e7 }
            val lon = DoubleArray(lo.length()) { lo.getLong(it) / 1e7 }
            val wa = o.getJSONArray("ways")
            val ways = List(wa.length()) { i ->
                val w = wa.getJSONArray(i)
                Way(ints(w.getJSONArray(0)), w.getInt(1), w.getInt(2), w.getInt(3) != 0)
            }
            val pa = o.getJSONArray("places")
            val places = List(pa.length()) { i ->
                val p = pa.getJSONObject(i)
                Place(p.getString("name"), p.optString("alt", ""), p.getString("kind"),
                    p.getDouble("lat"), p.getDouble("lon"), ints(p.getJSONArray("arrive")))
            }
            return CityPack(lat, lon, ways, places)
        }

        fun fromAssets(ctx: Context, name: String = "city_pack_hubli.json"): CityPack =
            parse(ctx.assets.open(name).bufferedReader().use { it.readText() })

        private fun ints(a: JSONArray) = IntArray(a.length()) { a.getInt(it) }
    }
}
