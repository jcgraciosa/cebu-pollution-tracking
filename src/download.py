"""Download the free inputs: CAMS composition and AQI, ICON winds, visibility,
VIIRS active fire.

Everything is keyless except FIRMS history beyond 24 h, which wants a free
MAP_KEY in $FIRMS_MAP_KEY.

    python src/download.py                      # full domain
    python src/download.py --step 2.0           # coarser, ~4x fewer requests
    python src/download.py --parts vis,site     # refresh only these stages
"""
from __future__ import annotations
import argparse, io, json, os, sys, time
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
import config as C

STAGES = ("aq", "met", "vis", "site", "fires")


def build_grid(step: float) -> list[tuple[float, float]]:
    lats = np.arange(C.BBOX["south"], C.BBOX["north"] + 1e-9, step)
    lons = np.arange(C.BBOX["west"], C.BBOX["east"] + 1e-9, step)
    return [(round(float(a), 4), round(float(b), 4)) for a in lats for b in lons]


def fetch_points(url, hourly, pts, past_days, forecast_days,
                 chunk=C.CHUNK, models=None) -> list[dict]:
    """Open-Meteo takes comma-separated coordinates; chunk to stay under the URL
    limit. Returns one record per requested point."""
    out: list[dict] = []
    for i in range(0, len(pts), chunk):
        c = pts[i:i + chunk]
        params = {
            "latitude": ",".join(str(a) for a, _ in c),
            "longitude": ",".join(str(b) for _, b in c),
            "hourly": ",".join(hourly),
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        if models:
            params["models"] = models
        for attempt in range(6):
            try:
                r = requests.get(url, params=params, timeout=180)
                if r.status_code == 429:
                    # metered per minute/hour/day, so a short backoff is useless
                    wait = int(r.headers.get("Retry-After", 0)) or min(90 * 2 ** attempt, 900)
                    print(f"    rate limited; sleeping {wait}s [{attempt+1}/6]",
                          file=sys.stderr, flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                d = r.json()
                out.extend(d if isinstance(d, list) else [d])
                break
            except Exception as e:                      # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
                print(f"    retry {attempt+1}: {e!r}", file=sys.stderr)
        else:
            raise RuntimeError(f"gave up after 6 attempts at offset {i}")
        print(f"  {min(i+chunk, len(pts)):5d}/{len(pts)} points", file=sys.stderr, flush=True)
    return out


def to_cube(records, variables, step) -> dict:
    """Reshape per-point records into (time, lat, lon) arrays.

    Open-Meteo snaps to its own model grid, so bin the returned coordinates back
    onto our lattice rather than assuming they match what we asked for.
    """
    times = records[0]["hourly"]["time"]
    lats = np.arange(C.BBOX["south"], C.BBOX["north"] + 1e-9, step)
    lons = np.arange(C.BBOX["west"], C.BBOX["east"] + 1e-9, step)
    cubes = {v: np.full((len(times), lats.size, lons.size), np.nan, "float32")
             for v in variables}
    for rec in records:
        i = int(np.abs(lats - rec["latitude"]).argmin())
        j = int(np.abs(lons - rec["longitude"]).argmin())
        for v in variables:
            series = rec["hourly"].get(v)
            if series is None:
                continue
            cubes[v][:, i, j] = np.array(
                [np.nan if x is None else x for x in series], dtype="float32")
    return dict(time=np.array(times), lat=lats, lon=lons, **cubes)


def fetch_fires(days: int) -> pd.DataFrame:
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    frames = []
    if key:
        area = f'{C.BBOX["west"]},{C.BBOX["south"]},{C.BBOX["east"]},{C.BBOX["north"]}'
        for src in ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"):
            url = C.FIRMS_AREA.format(key=key, src=src, area=area, days=min(int(days), 10))
            try:
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                if r.text.lstrip().lower().startswith("invalid"):
                    print(f"  FIRMS rejected the MAP_KEY for {src}", file=sys.stderr)
                    continue
                df = pd.read_csv(io.StringIO(r.text))
                df["source"] = src
                frames.append(df)
                print(f"  {src}: {len(df)} detections ({days} d)", file=sys.stderr)
            except Exception as e:                      # noqa: BLE001
                print(f"  {src} failed: {e!r}", file=sys.stderr)
    if not frames:
        print("  keyless 24 h files only -- fires will not be time-resolved",
              file=sys.stderr)
        for name, url in C.FIRMS_24H.items():
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df["source"] = name
            frames.append(df)
            print(f"  {name}: {len(df)} detections (24 h)", file=sys.stderr)

    fires = pd.concat(frames, ignore_index=True)
    fires = fires[fires.latitude.between(C.BBOX["south"], C.BBOX["north"]) &
                  fires.longitude.between(C.BBOX["west"], C.BBOX["east"])]
    # words in the keyless CSVs, single letters from the area API -- match both
    conf = fires.confidence.astype(str).str.lower()
    fires = fires[~conf.isin(["l", "low"])]
    t = fires.acq_time.astype(int).astype(str).str.zfill(4)
    fires["when"] = pd.to_datetime(
        fires.acq_date + " " + t.str[:2] + ":" + t.str[2:], utc=True)
    return fires.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", type=float, default=C.GRID_STEP)
    p.add_argument("--past-days", type=int, default=C.PAST_DAYS)
    p.add_argument("--forecast-days", type=int, default=C.FORECAST_DAYS)
    p.add_argument("--replace-fires", action="store_true",
                   help="overwrite the fire archive instead of appending to it")
    p.add_argument("--parts", default=",".join(STAGES),
                   help=f"stages to run, comma separated: {','.join(STAGES)}")
    a = p.parse_args()

    parts = {x.strip() for x in a.parts.split(",") if x.strip()}
    if unknown := parts - set(STAGES):
        p.error(f"unknown stage(s): {sorted(unknown)}; choose from {STAGES}")

    pts = build_grid(a.step)
    print(f"{len(pts)} grid points at {a.step} deg; stages: {sorted(parts)}",
          file=sys.stderr)

    if "aq" in parts:
        print("composition + AQI (CAMS)...", file=sys.stderr)
        aq = fetch_points(C.AQ_URL, C.AQ_VARS, pts, a.past_days, min(a.forecast_days, 7))
        np.savez_compressed(C.DATA / "aq_grid.npz", **to_cube(aq, C.AQ_VARS, a.step))

    if parts & {"met", "vis"}:
        mp = C.DATA / "met_grid.npz"
        cube = dict(np.load(mp, allow_pickle=True)) if mp.exists() else {}
        if "met" in parts:
            print(f"winds ({C.MET_MODEL_LABEL})...", file=sys.stderr)
            met = fetch_points(C.MET_URL, C.MET_VARS, pts, a.past_days,
                               a.forecast_days, models=C.MET_MODEL)
            cube.update(to_cube(met, C.MET_VARS, a.step))
        if "vis" in parts:
            print(f"visibility ({C.VIS_MODEL_LABEL})...", file=sys.stderr)
            vis = fetch_points(C.MET_URL, C.VIS_VARS, pts, a.past_days,
                               a.forecast_days, models=C.VIS_MODEL)
            vc = to_cube(vis, C.VIS_VARS, a.step)
            # a vis-only refresh must adopt the new axes, or visibility ends up
            # indexed against a stale time axis from the previous pull
            if "met" not in parts and cube.get("time") is not None \
                    and not np.array_equal(cube["time"], vc["time"]):
                print("  time axis moved since the last pull; refreshing 'met' too",
                      file=sys.stderr)
                met = fetch_points(C.MET_URL, C.MET_VARS, pts, a.past_days,
                                   a.forecast_days, models=C.MET_MODEL)
                cube.update(to_cube(met, C.MET_VARS, a.step))
            cube.update({k: vc[k] for k in C.VIS_VARS})
            cube["time"], cube["lat"], cube["lon"] = vc["time"], vc["lat"], vc["lon"]
        np.savez_compressed(C.DATA / "met_grid.npz", **cube)

    if "site" in parts:
        print("receptor series at Cebu City...", file=sys.stderr)
        here = [(C.CEBU["lat"], C.CEBU["lon"])]
        site = fetch_points(C.AQ_URL, C.AQ_VARS, here, a.past_days, min(a.forecast_days, 7))
        s_met = fetch_points(C.MET_URL, C.MET_VARS, here, a.past_days,
                             a.forecast_days, models=C.MET_MODEL)
        s_vis = fetch_points(C.MET_URL, C.VIS_VARS, here, a.past_days,
                             a.forecast_days, models=C.VIS_MODEL)
        df = pd.DataFrame({"time": site[0]["hourly"]["time"],
                           **{v: site[0]["hourly"][v] for v in C.AQ_VARS}})
        m = pd.DataFrame({"time": s_met[0]["hourly"]["time"],
                          **{v: s_met[0]["hourly"][v] for v in C.MET_VARS},
                          **{v: s_vis[0]["hourly"][v] for v in C.VIS_VARS}})
        (df.merge(m, on="time", how="outer").sort_values("time")
           .to_csv(C.DATA / "cebu_timeseries.csv", index=False))

    if "fires" in parts:
        print("VIIRS active fire (FIRMS)...", file=sys.stderr)
        fires = fetch_fires(a.past_days)
        out = C.DATA / "fires.csv"
        if out.exists() and not a.replace_fires:
            # keyless FIRMS only ever serves the last 24 h, so accumulate:
            # overwriting would permanently discard earlier days
            old = pd.read_csv(out, parse_dates=["when"])
            before = len(old)
            fires = (pd.concat([old, fires], ignore_index=True)
                       .drop_duplicates(subset=["latitude", "longitude",
                                                "acq_date", "acq_time", "source"])
                       .sort_values("when").reset_index(drop=True))
            print(f"  archive {before} + new -> {len(fires)} "
                  f"({len(fires)-before} added)", file=sys.stderr)
        fires.to_csv(out, index=False)
        span = f"{fires.when.min():%Y-%m-%d} .. {fires.when.max():%Y-%m-%d}"
        print(f"  {len(fires)} detections in the domain, {span}", file=sys.stderr)

    (C.DATA / "meta.json").write_text(json.dumps(dict(
        bbox=C.BBOX, step=a.step, past_days=a.past_days,
        forecast_days=a.forecast_days, receptor=C.CEBU,
        met_model=C.MET_MODEL, vis_model=C.VIS_MODEL, stages=sorted(parts),
        downloaded_utc=pd.Timestamp.now("UTC").isoformat()), indent=2))
    print(f"wrote -> {C.DATA}", file=sys.stderr)


if __name__ == "__main__":
    main()
