#!/usr/bin/env python3
"""Golden answers for the phone's router and search (the Kotlin port must reproduce them).

40 routes from random road-side positions (seeded) to random places, plus the demo route KLE Tech ->
Unkal Lake, and 12 searches from KLE Tech. Written next to the pack as city_golden.json.

    python scripts/make_golden_city.py
"""
import json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dhruva.citypack import City, search

PACK = "data/city/city_pack_hubli.json"
C = City(json.load(open(PACK))); rng = random.Random(26168)
KLE = (15.3693, 75.1219)
starts = [KLE] + [(C.lat[i] + rng.uniform(-2e-4, 2e-4), C.lon[i] + rng.uniform(-2e-4, 2e-4)) for i in rng.sample(range(len(C.lat)), 40)]
dests = [next(k for k, p in enumerate(C.places) if p["name"] == "Unakal Kere")] + rng.sample(range(len(C.places)), 40)
routes, t0 = [], time.time()
for (la, lo), k in zip(starts, dests):
    r = C.route(la, lo, C.places[k]["arrive"])
    routes.append(dict(lat=la, lon=lo, place=k, name=C.places[k]["name"], reachable=r is not None,
                       length_m=r[1] if r else None, seconds=r[2] if r else None, n_points=len(r[0]) if r else None,
                       end=list(r[0][-1]) if r else None))
print(f"{len(routes)} routes in {time.time() - t0:.1f} s (python); {sum(not r['reachable'] for r in routes)} unreachable (one-way dead ends)")
queries = ["unkal lake", "unakal kere", "kle tech", "lhc", "railway station", "hospital", "unka", "bvb", "brt", "park", "bank", "vidyanagar"]
searches = [dict(query=q, top=[p["name"] for p, _ in search(C.places, q, *KLE, limit=5)]) for q in queries]
json.dump(dict(pack=os.path.basename(PACK), from_lat=KLE[0], from_lon=KLE[1], routes=routes, searches=searches),
          open("data/city/city_golden.json", "w"), indent=1)
print(f"demo: KLE Tech -> {routes[0]['name']} {routes[0]['length_m']:.0f} m")
for s in searches: print(f"  {s['query']!r:18s} {s['top'][:3]}")
print("wrote data/city/city_golden.json")
