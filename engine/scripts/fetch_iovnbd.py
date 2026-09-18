#!/usr/bin/env python3
"""Download IO-VNBD files via the Git LFS API (no full 1.5 GB clone needed).

    python scripts/fetch_iovnbd.py --list
    python scripts/fetch_iovnbd.py S-S1 S-S2 S-S3a
    python scripts/fetch_iovnbd.py --all-smartphone

NOTE: the URL printed in ISRO's problem statement (github.com/onyekpe/IO-VNBD)
is a TYPO and 404s. The real repo is onyekpe*u*/IO-VNBD.

S-* = smartphone stream (use these -- the PS forbids the OBD-II wheel feed)
V-* = vehicle CAN stream
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.request, urllib.parse

REPO = "onyekpeu/IO-VNBD"
BRANCH = "master"
BASE = "Synchronised V abd S datasets/Uncategorised IOVNB Dataset"
LFS = f"https://github.com/{REPO}.git/info/lfs/objects/batch"


def _get(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    return urllib.request.urlopen(req, timeout=60).read()


def listing(kind: str):
    d = urllib.parse.quote(f"{BASE}/{kind}-Dataset")
    url = f"https://api.github.com/repos/{REPO}/contents/{d}?ref={BRANCH}"
    return [x["name"] for x in json.loads(_get(url)) if x["type"] == "file"]


def pointer(kind: str, name: str):
    p = urllib.parse.quote(f"{BASE}/{kind}-Dataset/{name}")
    raw = _get(f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{p}").decode()
    oid = size = None
    for line in raw.splitlines():
        if line.startswith("oid sha256:"): oid = line.split(":", 1)[1].strip()
        if line.startswith("size "): size = int(line.split()[1])
    if not oid:
        return None, None, raw          # not LFS -- already the real file
    return oid, size, None


def download(kind: str, name: str, outdir: str):
    out = os.path.join(outdir, name)
    if os.path.exists(out) and os.path.getsize(out) > 10_000:
        print(f"  [skip] {name} already here"); return
    oid, size, plain = pointer(kind, name)
    if plain is not None:
        open(out, "w").write(plain); print(f"  [ok]   {name} (plain)"); return
    body = json.dumps({"operation": "download", "transfers": ["basic"],
                       "objects": [{"oid": oid, "size": size}]}).encode()
    r = json.loads(_get(LFS, body, {"Accept": "application/vnd.git-lfs+json",
                                    "Content-Type": "application/vnd.git-lfs+json"}))
    obj = r["objects"][0]
    if "error" in obj:
        print(f"  [FAIL] {name}: {obj['error']}"); return
    data = _get(obj["actions"]["download"]["href"])
    open(out, "wb").write(data)
    print(f"  [ok]   {name}  {len(data)/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="e.g. S-S1 S-S2  (extension optional)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all-smartphone", action="store_true")
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.list:
        for k in ("S", "V"):
            names = listing(k)
            print(f"\n{k}-Dataset ({'smartphone' if k=='S' else 'vehicle CAN'}) — {len(names)} files")
            for i in range(0, len(names), 6):
                print("   " + "  ".join(n.replace('.csv','') for n in names[i:i+6]))
        return

    targets = []
    if a.all_smartphone:
        targets = [("S", n) for n in listing("S")]
    for f in a.files:
        n = f if f.endswith(".csv") else f + ".csv"
        targets.append(("S" if n.startswith("S-") else "V", n))

    if not targets:
        print(__doc__); sys.exit(1)
    print(f"fetching {len(targets)} file(s) -> {a.outdir}/")
    for kind, name in targets:
        try: download(kind, name, a.outdir)
        except Exception as e: print(f"  [FAIL] {name}: {e}")


if __name__ == "__main__":
    main()
