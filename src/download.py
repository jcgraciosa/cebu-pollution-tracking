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

STAGES = ("aq", "met", "vis", "site", "fires", "wind3d", "terrain", "column", "wind3d_big", "wind3d_fine")
# wind3d is a research fetch over a small box; it is not part of the daily run
DEFAULT_STAGES = STAGES[:5]


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
                # 15/30/60/120/240. Nearly every failure clears on the first
                # retry, so attempt 1 stays cheap; the later steps are there to
                # ride out a real outage. Starting at 30 added 11 min to the
                # 26 Sep run and pushed it 20 s past the 90 min cap.
                time.sleep(min(15 * 2 ** attempt, 240))
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


def fetch_days(path, step) -> int:
    """GRID_FETCH_DAYS when a cube is there to splice onto, else the full window.

    The short fetch is only an optimisation; it is correct just when the cache
    already covers the hours the fetch will not. If the cache is absent, stale
    or from a different grid, a short fetch would leave a hole and the animation
    would quietly shorten instead of failing. So it self-heals: first run, cache
    eviction, or a few days skipped all fall back to the full pull.
    """
    if not path.exists():
        return C.GRID_PAST_DAYS
    fp = json.dumps({"bbox": C.BBOX, "step": float(step)}, sort_keys=True)
    try:
        d = np.load(path, allow_pickle=True)
        if str(d["fp"]) != fp:
            return C.GRID_PAST_DAYS
        last = pd.to_datetime(np.asarray(d["time"]).astype(str)).max()
    except Exception:                               # noqa: BLE001
        return C.GRID_PAST_DAYS
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    gap = (now - last) / pd.Timedelta(days=1)
    if gap >= C.GRID_FETCH_DAYS:
        print(f"  cache ends {gap:.1f} days back; pulling the full window",
              file=sys.stderr)
        return C.GRID_PAST_DAYS
    return C.GRID_FETCH_DAYS


def merge_cube(path, new, step, keep_days=None):
    """Splice a short fresh fetch onto the cached cube; the fetch wins on overlap.

    Each run fetches GRID_FETCH_DAYS and inherits the rest from disk, so the
    request payload drops by roughly two thirds. That is what the ReadTimeouts
    scale with -- not the number of requests, which is fixed by the grid.

    Newest wins because CAMS revises its recent analysis hours. The cache is
    discarded outright if the bbox, step or grid shape moved, since a cube whose
    halves disagree about what a pixel means is worse than no cube.
    """
    fp = json.dumps({"bbox": C.BBOX, "step": float(step)}, sort_keys=True)
    old = None
    if path.exists():
        try:
            d = dict(np.load(path, allow_pickle=True))
            if (str(d.get("fp", "")) == fp
                    and np.array_equal(d.get("lat"), new["lat"])
                    and np.array_equal(d.get("lon"), new["lon"])):
                old = d
            else:
                print("  grid definition changed; rebuilding the cube",
                      file=sys.stderr)
        except Exception as e:                      # noqa: BLE001
            print(f"  cached cube unreadable ({type(e).__name__}); rebuilding",
                  file=sys.stderr)

    if old is None:
        out = dict(new)
    else:
        ot = np.asarray(old["time"]).astype(str)
        nt = np.asarray(new["time"]).astype(str)
        keep = ~np.isin(ot, nt)
        times = np.concatenate([ot[keep], nt])
        order = np.argsort(times)           # ISO 8601 sorts chronologically
        ny, nx = new["lat"].size, new["lon"].size
        out = {"time": times[order], "lat": new["lat"], "lon": new["lon"]}
        for v in sorted((set(old) | set(new)) - {"time", "lat", "lon", "fp"}):
            a = (old[v][keep] if v in old
                 else np.full((int(keep.sum()), ny, nx), np.nan, "float32"))
            b = (new[v] if v in new
                 else np.full((nt.size, ny, nx), np.nan, "float32"))
            out[v] = np.concatenate([a, b])[order]
        print(f"  cache: {keep.sum()} hours kept + {nt.size} fetched "
              f"= {times.size}", file=sys.stderr)

    if keep_days is not None:
        t = pd.to_datetime(np.asarray(out["time"]).astype(str))
        cut = (pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
               - pd.Timedelta(days=keep_days))
        m = np.asarray(t >= cut)
        if not m.all():
            out = {k: (v if k in ("lat", "lon") else v[m]) for k, v in out.items()}
    out["fp"] = np.array(fp)
    return out


def fetch_fires(days: int) -> pd.DataFrame:
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    frames = []
    if key:
        area = f'{C.BBOX["west"]},{C.BBOX["south"]},{C.BBOX["east"]},{C.BBOX["north"]}'
        for src in ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"):
            # The area API caps day_range at 5, and its date parameter is the
            # START of the window counting FORWARD -- not the end. Walk back in
            # 5-day steps, asking for each window by its first day.
            got = []
            today = pd.Timestamp.now("UTC").normalize()
            n_win = max(1, int(np.ceil(days / C.FIRMS_MAX_DAYS)))
            # i == 0 asks for the window starting today, which is how the
            # current (partial) day gets in at all
            starts = [today - pd.Timedelta(days=C.FIRMS_MAX_DAYS * i)
                      for i in range(n_win, -1, -1)]
            for st in starts:
                url = (C.FIRMS_AREA.format(key=key, src=src, area=area,
                                           days=C.FIRMS_MAX_DAYS) + f"/{st:%Y-%m-%d}")
                for attempt in range(3):
                    try:
                        r = requests.get(url, timeout=300)
                        r.raise_for_status()
                        txt = r.text.lstrip()
                        if not txt.startswith("latitude"):
                            print(f"  {src} {st:%d %b}: {txt[:70]}", file=sys.stderr)
                            break
                        got.append(pd.read_csv(io.StringIO(r.text)))
                        break
                    except Exception as e:                  # noqa: BLE001
                        if attempt == 2:
                            print(f"  {src} from {st:%d %b} failed after 3 tries: "
                                  f"{type(e).__name__}", file=sys.stderr)
                        else:
                            time.sleep(5 * (attempt + 1))
            if got:
                df = pd.concat(got, ignore_index=True)
                df["source"] = src
                frames.append(df)
                d = pd.to_datetime(df.acq_date)
                print(f"  {src}: {len(df)} detections, {d.min():%d %b}..{d.max():%d %b} "
                      f"({len(got)}/{len(starts)} windows)", file=sys.stderr)

    if not frames:
        print("  keyless 24 h files only -- fires will not be time-resolved",
              file=sys.stderr)
        for name, url in C.FIRMS_24H.items():
            # same 3 attempts as the keyed path above: this feed only serves the
            # last 24 h, so a timeout here loses that day's fires permanently
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=180)
                    r.raise_for_status()
                    df = pd.read_csv(io.StringIO(r.text))
                    df["source"] = name
                    frames.append(df)
                    print(f"  {name}: {len(df)} detections (24 h)", file=sys.stderr)
                    break
                except Exception as e:                  # noqa: BLE001
                    if attempt == 2:
                        print(f"  {name} failed after 3 tries: {type(e).__name__}",
                              file=sys.stderr)
                    else:
                        time.sleep(5 * (attempt + 1))

    if not frames:
        raise RuntimeError("every FIRMS source failed; no fire data this run")
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
    p.add_argument("--parts", default=",".join(DEFAULT_STAGES),
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
        ap = C.DATA / "aq_grid.npz"
        aq = fetch_points(C.AQ_URL, C.AQ_VARS, pts, fetch_days(ap, a.step),
                          min(a.forecast_days, 7))
        np.savez_compressed(ap, **merge_cube(ap, to_cube(aq, C.AQ_VARS, a.step),
                                             a.step, C.GRID_PAST_DAYS))

    if parts & {"met", "vis"}:
        mp = C.DATA / "met_grid.npz"
        mdays = fetch_days(mp, a.step)
        fresh = {}
        if "met" in parts:
            print(f"winds ({C.MET_MODEL_LABEL})...", file=sys.stderr)
            met = fetch_points(C.MET_URL, C.MET_VARS, pts, mdays,
                               a.forecast_days, models=C.MET_MODEL)
            fresh.update(to_cube(met, C.MET_VARS, a.step))
        if "vis" in parts:
            print(f"visibility ({C.VIS_MODEL_LABEL})...", file=sys.stderr)
            vis = fetch_points(C.MET_URL, C.VIS_VARS, pts, mdays,
                               a.forecast_days, models=C.VIS_MODEL)
            vc = to_cube(vis, C.VIS_VARS, a.step)
            # both models must land on one axis before the merge, or visibility
            # ends up indexed against hours that belong to the wind field
            if fresh and not np.array_equal(fresh["time"], vc["time"]):
                print("  met and vis disagree on the time axis; refetching met",
                      file=sys.stderr)
                met = fetch_points(C.MET_URL, C.MET_VARS, pts, mdays,
                                   a.forecast_days, models=C.MET_MODEL)
                fresh = to_cube(met, C.MET_VARS, a.step)
            fresh.update({k: vc[k] for k in C.VIS_VARS})
            fresh["time"], fresh["lat"], fresh["lon"] = vc["time"], vc["lat"], vc["lon"]
        np.savez_compressed(mp, **merge_cube(mp, fresh, a.step, C.GRID_PAST_DAYS))

    if "site" in parts:
        print("receptor series at Cebu City...", file=sys.stderr)
        here = [(C.CEBU["lat"], C.CEBU["lon"])]
        # SITE_FORECAST_DAYS, not FORECAST_DAYS: this CSV is the forecast figures'
        # only source. All three calls use it so the merged frame stays rectangular.
        site = fetch_points(C.AQ_URL, C.AQ_VARS, here, a.past_days, C.SITE_FORECAST_DAYS)
        s_met = fetch_points(C.MET_URL, C.MET_VARS, here, a.past_days,
                             C.SITE_FORECAST_DAYS, models=C.MET_MODEL)
        s_vis = fetch_points(C.MET_URL, C.VIS_VARS, here, a.past_days,
                             C.SITE_FORECAST_DAYS, models=C.VIS_MODEL)
        df = pd.DataFrame({"time": site[0]["hourly"]["time"],
                           **{v: site[0]["hourly"][v] for v in C.AQ_VARS}})
        m = pd.DataFrame({"time": s_met[0]["hourly"]["time"],
                          **{v: s_met[0]["hourly"][v] for v in C.MET_VARS},
                          **{v: s_vis[0]["hourly"][v] for v in C.VIS_VARS}})
        (df.merge(m, on="time", how="outer").sort_values("time")
           .to_csv(C.DATA / "cebu_timeseries.csv", index=False))

    if "column" in parts:
        # The time-height section needs ONE column, not the whole box. 9 points
        # at 12 levels is a single request; the 441-point version was refused.
        print("deep wind column over Cebu...", file=sys.stderr)
        off = [-0.25, 0.0, 0.25]
        ptsc = [(C.CEBU["lat"] + p1, C.CEBU["lon"] + p2) for p1 in off for p2 in off]
        vc = [f"{v}_{lv}hPa" for lv in C.WIND3D_LEVELS
              for v in ("wind_speed", "wind_direction", "vertical_velocity",
                        "geopotential_height")]
        rec = fetch_points(C.MET_URL, vc, ptsc, C.WIND3D_PAST_DAYS, a.forecast_days,
                           models=C.WIND3D_MODEL)
        tt = np.array(rec[0]["hourly"]["time"], dtype="datetime64[s]")
        shape = (len(tt), len(C.WIND3D_LEVELS), 3, 3)
        cube = {k: np.full(shape, np.nan, "float32")
                for k in ("spd", "dir", "w", "z")}
        key = {"wind_speed": "spd", "wind_direction": "dir",
               "vertical_velocity": "w", "geopotential_height": "z"}
        for n, r3 in enumerate(rec):
            i, j = divmod(n, 3)
            for li, lv in enumerate(C.WIND3D_LEVELS):
                for var, short in key.items():
                    ser = r3["hourly"].get(f"{var}_{lv}hPa")
                    if ser is not None:
                        cube[short][:, li, i, j] = [np.nan if x is None else x
                                                    for x in ser]
        np.savez_compressed(C.DATA / "wind_column.npz", time=tt,
                            lat=np.array([C.CEBU["lat"] + o for o in off]),
                            lon=np.array([C.CEBU["lon"] + o for o in off]),
                            level=np.array(C.WIND3D_LEVELS), **cube)
        nn = 100 * np.isfinite(cube["w"]).mean()
        print(f"  wrote wind_column.npz {shape}, w non-null {nn:.0f}%",
              file=sys.stderr)

    if "wind3d_big" in parts:
        # the same levels over the WHOLE map domain, for regional divergence.
        # Coarser on purpose: divergence magnitude scales with grid spacing, so
        # this field is not directly comparable to the 0.25 deg box.
        print("3D wind over the full domain...", file=sys.stderr)
        b, stb = C.BBOX, C.WIND3D_BIG_STEP
        lab = np.round(np.arange(b["south"], b["north"] + 1e-9, stb), 4)
        lob = np.round(np.arange(b["west"], b["east"] + 1e-9, stb), 4)
        ptsb = [(float(x), float(y)) for x in lab for y in lob]
        vb = [f"{v}_{lv}hPa" for lv in C.WIND3D_LEVELS
              for v in ("wind_speed", "wind_direction", "vertical_velocity",
                        "geopotential_height")]
        print(f"  {len(ptsb)} points x {len(C.WIND3D_LEVELS)} levels",
              file=sys.stderr)
        rec = fetch_points(C.MET_URL, vb, ptsb, C.WIND3D_PAST_DAYS, a.forecast_days,
                           models=C.WIND3D_MODEL)
        tb = np.array(rec[0]["hourly"]["time"], dtype="datetime64[s]")
        shape = (len(tb), len(C.WIND3D_LEVELS), lab.size, lob.size)
        cube = {k: np.full(shape, np.nan, "float32")
                for k in ("spd", "dir", "w", "z")}
        key = {"wind_speed": "spd", "wind_direction": "dir",
               "vertical_velocity": "w", "geopotential_height": "z"}
        for n, r3 in enumerate(rec):
            i, j = divmod(n, lob.size)
            for li, lv in enumerate(C.WIND3D_LEVELS):
                for var, short in key.items():
                    ser = r3["hourly"].get(f"{var}_{lv}hPa")
                    if ser is not None:
                        cube[short][:, li, i, j] = [np.nan if x is None else x
                                                    for x in ser]
        np.savez_compressed(C.DATA / "wind3d_region.npz", time=tb, lat=lab,
                            lon=lob, level=np.array(C.WIND3D_LEVELS), **cube)
        print(f"  wrote wind3d_region.npz {shape}, w non-null "
              f"{100*np.isfinite(cube['w']).mean():.0f}%", file=sys.stderr)

    if "wind3d_fine" in parts:
        # 68 requests at the daily quota's edge, so it CHECKPOINTS: each chunk
        # is written to a .part file and a re-run skips what already landed.
        # A 429 halfway through then costs the remaining chunks, not all of it.
        b, stp = C.WIND3D_FINE, C.WIND3D_FINE_STEP
        laf = np.round(np.arange(b["south"], b["north"] + 1e-9, stp), 4)
        lof = np.round(np.arange(b["west"], b["east"] + 1e-9, stp), 4)
        pts = [(float(x), float(y)) for x in laf for y in lof]
        lv = C.WIND3D_LEVELS
        vnames = [f"{v}_{l}hPa" for l in lv for v in C.WIND3D_VARS]
        part = C.DATA / "wind3d_fine.part.npz"
        final = C.DATA / "wind3d_fine.npz"
        nch = int(np.ceil(len(pts) / C.CHUNK))
        print(f"3D wind, {stp} deg over {b['south']:.0f}-{b['north']:.0f}N "
              f"{b['west']:.0f}-{b['east']:.0f}E: {len(pts)} points, "
              f"{len(lv)} levels, {nch} chunks", file=sys.stderr)

        cube, done, times = None, np.zeros(nch, bool), None
        if part.exists():
            z = np.load(part, allow_pickle=True)
            cube = {k: z[k] for k in ("spd", "dir", "w")}
            done, times = z["done"], z["time"]
            print(f"  resuming: {done.sum()}/{nch} chunks already stored",
                  file=sys.stderr)

        for ci in range(nch):
            if done[ci]:
                continue
            chunk = pts[ci * C.CHUNK:(ci + 1) * C.CHUNK]
            rec = fetch_points(C.MET_URL, vnames, chunk, C.WIND3D_PAST_DAYS,
                               a.forecast_days, models=C.WIND3D_MODEL)
            if times is None:
                times = np.array(rec[0]["hourly"]["time"], dtype="datetime64[s]")
                shape = (len(times), len(lv), laf.size, lof.size)
                cube = {k: np.full(shape, np.nan, "float32")
                        for k in ("spd", "dir", "w")}
            key = {"wind_speed": "spd", "wind_direction": "dir",
                   "vertical_velocity": "w"}
            for n, r3 in enumerate(rec):
                gi, gj = divmod(ci * C.CHUNK + n, lof.size)
                for li, l in enumerate(lv):
                    for var, short in key.items():
                        ser = r3["hourly"].get(f"{var}_{l}hPa")
                        if ser is not None:
                            cube[short][:, li, gi, gj] = [
                                np.nan if x is None else x for x in ser]
            done[ci] = True
            np.savez_compressed(part, time=times, done=done, **cube)
            print(f"  chunk {ci+1}/{nch} stored", file=sys.stderr, flush=True)

        z = np.array([C.WIND3D_Z[l] for l in lv], "float32")
        np.savez_compressed(final, time=times, lat=laf, lon=lof,
                            level=np.array(lv), z=z, **cube)
        part.unlink(missing_ok=True)
        print(f"  wrote {final.name} {cube['w'].shape}, w non-null "
              f"{100*np.isfinite(cube['w']).mean():.0f}%", file=sys.stderr)

    if "terrain" in parts:
        print("terrain over the residence box (Open-Meteo elevation)...",
              file=sys.stderr)
        b, st4 = C.WIND3D_BOX, C.TERRAIN_STEP
        tla = np.round(np.arange(b["south"], b["north"] + 1e-9, st4), 4)
        tlo = np.round(np.arange(b["west"], b["east"] + 1e-9, st4), 4)
        gla, glo = np.meshgrid(tla, tlo, indexing="ij")
        flat_la, flat_lo = gla.ravel(), glo.ravel()
        elev = np.full(flat_la.size, np.nan, "float32")
        CH = 100                              # the elevation API caps at 100 coords
        for i in range(0, flat_la.size, CH):
            sl = slice(i, i + CH)
            for attempt in range(5):
                try:
                    r = requests.get(C.ELEV_URL, timeout=90, params=dict(
                        latitude=",".join(f"{x:.4f}" for x in flat_la[sl]),
                        longitude=",".join(f"{x:.4f}" for x in flat_lo[sl])))
                    if r.status_code == 429:
                        time.sleep(60); continue
                    r.raise_for_status()
                    elev[sl] = r.json()["elevation"]
                    break
                except Exception as e:                      # noqa: BLE001
                    if attempt == 4:
                        raise
                    time.sleep(5 * (attempt + 1))
            if (i // CH) % 20 == 0:
                print(f"  {min(i + CH, flat_la.size):6d}/{flat_la.size} points",
                      file=sys.stderr, flush=True)
        np.savez_compressed(C.DATA / "terrain.npz", lat=tla, lon=tlo,
                            elev=elev.reshape(gla.shape))
        print(f"  wrote terrain.npz {gla.shape}, "
              f"{np.nanmax(elev):.0f} m max", file=sys.stderr)

    if "wind3d" in parts:
        print(f"3D wind over the residence box ({C.WIND3D_MODEL_LABEL})...",
              file=sys.stderr)
        b, st3 = C.WIND3D_BOX, C.WIND3D_STEP
        la3 = np.round(np.arange(b["south"], b["north"] + 1e-9, st3), 4)
        lo3 = np.round(np.arange(b["west"], b["east"] + 1e-9, st3), 4)
        pts3 = [(float(a), float(o)) for a in la3 for o in lo3]
        v3 = [f"{v}_{lv}hPa" for lv in C.WIND3D_LEVELS
              for v in ("wind_speed", "wind_direction", "vertical_velocity",
                        "geopotential_height")]
        print(f"  {len(pts3)} points x {len(C.WIND3D_LEVELS)} levels", file=sys.stderr)
        rec = fetch_points(C.MET_URL, v3, pts3, C.WIND3D_PAST_DAYS, a.forecast_days,
                           models=C.WIND3D_MODEL)
        times = np.array(rec[0]["hourly"]["time"], dtype="datetime64[s]")
        shape = (len(times), len(C.WIND3D_LEVELS), la3.size, lo3.size)
        cube3 = {k: np.full(shape, np.nan, "float32")
                 for k in ("spd", "dir", "w", "z")}
        key = {"wind_speed": "spd", "wind_direction": "dir",
               "vertical_velocity": "w", "geopotential_height": "z"}
        for n, r3 in enumerate(rec):
            i, j = divmod(n, lo3.size)
            for li, lv in enumerate(C.WIND3D_LEVELS):
                for var, short in key.items():
                    ser = r3["hourly"].get(f"{var}_{lv}hPa")
                    if ser is None:
                        continue
                    cube3[short][:, li, i, j] = [np.nan if x is None else x
                                                 for x in ser]
        np.savez_compressed(C.DATA / "wind3d.npz", time=times, lat=la3, lon=lo3,
                            level=np.array(C.WIND3D_LEVELS), **cube3)
        nn = 100 * np.isfinite(cube3["w"]).mean()
        print(f"  wrote wind3d.npz {shape}, w non-null {nn:.0f}%", file=sys.stderr)

    if "fires" in parts:
        print("VIIRS active fire (FIRMS)...", file=sys.stderr)
        fires = fetch_fires(a.past_days)
        out = C.FIRES_CSV
        # fall back to the pre-gzip archive once, so the migration keeps it
        legacy = C.DATA / "fires.csv"
        src_old = out if out.exists() else legacy
        if src_old.exists() and not a.replace_fires:
            # keyless FIRMS only ever serves the last 24 h, so accumulate:
            # overwriting would permanently discard earlier days
            old = pd.read_csv(src_old, parse_dates=["when"], low_memory=False)
            before = len(old)
            fires = pd.concat([old, fires], ignore_index=True)
            # The keyless 24 h feed and the area API name the same satellite
            # differently ("VIIRS_SNPP" vs "VIIRS_SNPP_NRT"), so a source-keyed
            # dedupe stored every overlapping detection twice.
            fires["source"] = fires.source.str.replace("_NRT", "", regex=False)
            fires = (fires.drop_duplicates(subset=["latitude", "longitude",
                                                   "acq_date", "acq_time", "source"])
                          .sort_values("when").reset_index(drop=True))
            print(f"  archive {before} + new -> {len(fires)} "
                  f"({len(fires)-before} added)", file=sys.stderr)
        fires = fires.reindex(columns=C.FIRES_COLS)
        fires.to_csv(out, index=False, compression="gzip")
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
