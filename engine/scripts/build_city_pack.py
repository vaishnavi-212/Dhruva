#!/usr/bin/env python3
"""Build the offline city pack the app routes and searches on (dhruva/citypack.py has the format).

Inputs, fetched once from OpenStreetMap (Overpass) for the box 15.33-15.41 N, 75.07-75.17 E
(about 9 x 11 km: KLE Tech, Vidyanagar, Unkal, Hubli centre and station):
  data/city/hubli_roads.json            car/two-wheeler roads with node ids (out body geom)
  data/city/hubli_places.json           named places (out tags center)
  data/city/hubli_areas.json            outlines of named areas: lakes, parks, campuses (ways)
  data/city/hubli_area_relations.json   the same for multipolygons (Unakal Kere is one)

Only the largest connected road network is kept, so every search result is reachable.

    python scripts/build_city_pack.py      ->  data/city/city_pack_hubli.json
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhruva.citypack import CLASSES, LINK, ARRIVE_POINT_M, ARRIVE_AREA_M, metres

D = "data/city"
KINDS = ("amenity", "tourism", "leisure", "natural", "shop", "office", "healthcare", "historic", "railway",
         "public_transport", "place", "landuse", "building")
SKIP_SERVICE = {"parking_aisle", "drive-through"}
# what people in Hubballi call a place when OSM uses another name (local knowledge, kept short)
LOCAL_ALIASES = {"Unakal Kere": "Unkal Lake", "KLE Technological University": "KLE Tech"}
SAME_PLACE_M = 300.0     # one name mapped as a point, an outline and a building: keep one


def main():
    roads = json.load(open(f"{D}/hubli_roads.json"))["elements"]
    ways, coord = [], {}
    for w in roads:
        tg = w["tags"]; hw = LINK.get(tg["highway"], tg["highway"])
        if hw not in CLASSES or tg.get("service") in SKIP_SERVICE or tg.get("area") == "yes":
            continue
        ow = tg.get("oneway", "no")
        oneway = 1 if ow in ("yes", "true", "1") or tg.get("junction") in ("roundabout", "circular") else -1 if ow == "-1" else 0
        private = int(tg.get("access") in ("private", "no"))
        for nid, g in zip(w["nodes"], w["geometry"]):
            coord[nid] = (g["lat"], g["lon"])
        ways.append((w["nodes"], CLASSES.index(hw), oneway, private))

    # largest connected network (ignoring direction)
    parent = {n: n for n in coord}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for nodes, *_ in ways:
        for a, b in zip(nodes[:-1], nodes[1:]):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb
    big = Counter(find(n) for n in coord).most_common(1)[0][0]
    keep = [w for w in ways if find(w[0][0]) == big]
    ids = sorted({n for w in keep for n in w[0]})
    ix = {n: i for i, n in enumerate(ids)}
    lat = [round(coord[n][0] * 1e7) for n in ids]; lon = [round(coord[n][1] * 1e7) for n in ids]
    pack_ways = [[[ix[n] for n in nodes], c, o, p] for nodes, c, o, p in keep]
    print(f"roads: {len(ways)} ways -> kept {len(keep)} in the main network, {len(ids)} nodes "
          f"({len(coord) - len(ids)} nodes on disconnected bits dropped)")

    # outlines for areas
    outline = {}
    for e in json.load(open(f"{D}/hubli_areas.json"))["elements"]:
        if e.get("geometry"): outline[(e["type"], e["id"])] = [(g["lat"], g["lon"]) for g in e["geometry"]]
    for e in json.load(open(f"{D}/hubli_area_relations.json"))["elements"]:
        pts = [(g["lat"], g["lon"]) for m in e.get("members", []) if m.get("role", "outer") in ("outer", "") for g in m.get("geometry", [])]
        if pts: outline[("relation", e["id"])] = pts

    # road nodes on a coarse grid for the arrival search
    cell = 0.001
    grid = {}
    for i, n in enumerate(ids):
        grid.setdefault((int(coord[n][0] / cell), int(coord[n][1] / cell)), []).append(i)
    def near(la, lo, r):
        k = int(r / 111.0 / 1000 / cell) + 1; ci, cj = int(la / cell), int(lo / cell); out = []
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                for i in grid.get((ci + di, cj + dj), []):
                    if metres(la, lo, lat[i] / 1e7, lon[i] / 1e7) <= r: out.append(i)
        return out

    places, seen = [], set()
    for e in json.load(open(f"{D}/hubli_places.json"))["elements"]:
        tg = e["tags"]; name = tg["name"].strip()
        c = (e["lat"], e["lon"]) if "lat" in e else (e["center"]["lat"], e["center"]["lon"])
        kind = next((f"{k}={tg[k]}" for k in KINDS if k in tg), "place")
        key = (name.lower(), round(c[0], 3), round(c[1], 3))
        if key in seen: continue
        seen.add(key)
        pts = outline.get((e["type"], e["id"]))
        if pts:                                               # every road node within 40 m of the outline
            arrive = sorted({i for la, lo in pts[::max(1, len(pts) // 200)] for i in near(la, lo, ARRIVE_AREA_M)})
        else:
            arrive = near(*c, ARRIVE_POINT_M)
        if not arrive:                                        # nothing close: the nearest road node
            arrive = [min(range(len(ids)), key=lambda i: metres(*c, lat[i] / 1e7, lon[i] / 1e7))]
        alt = " ".join(v for k, v in tg.items() if k in ("alt_name", "name:en", "old_name", "short_name") and v != name)
        if name in LOCAL_ALIASES: alt = (alt + " " + LOCAL_ALIASES[name]).strip()
        places.append(dict(name=name, alt=alt, kind=kind, lat=round(c[0], 7), lon=round(c[1], 7), arrive=arrive))

    # the same name within 300 m is one place: keep the entry with an outline (most arrival points), merge arrivals
    places.sort(key=lambda p: -len(p["arrive"]))
    merged = []
    for p in places:
        twin = next((q for q in merged if q["name"].lower() == p["name"].lower()
                     and metres(p["lat"], p["lon"], q["lat"], q["lon"]) < SAME_PLACE_M), None)
        if twin: twin["arrive"] = sorted(set(twin["arrive"]) | set(p["arrive"]))
        else: merged.append(p)
    print(f"places: {len(places)} entries -> {len(merged)} after merging the same name within {SAME_PLACE_M:.0f} m")
    places = sorted(merged, key=lambda p: p["name"].lower())

    pack = dict(format="dhruva-city/1", area="Hubballi (KLE Tech, Vidyanagar, Unkal, centre, station)",
                bbox=[15.33, 75.07, 15.41, 75.17], source="OpenStreetMap contributors, ODbL",
                lat=lat, lon=lon, ways=pack_ways, places=places)
    out = f"{D}/city_pack_hubli.json"
    json.dump(pack, open(out, "w"), separators=(",", ":"))
    km = sum(metres(lat[a] / 1e7, lon[a] / 1e7, lat[b] / 1e7, lon[b] / 1e7) for w in pack_ways for a, b in zip(w[0][:-1], w[0][1:])) / 1000
    print(f"wrote {out}: {os.path.getsize(out) / 1e6:.2f} MB, {km:.0f} km of road")


if __name__ == "__main__":
    main()
