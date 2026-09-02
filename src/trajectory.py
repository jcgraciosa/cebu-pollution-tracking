"""Kinematic back-trajectories on a single pressure level.

No vertical motion: good enough to tell you which way the synoptic fetch runs,
not good enough to resolve flow around Cebu itself, where the driving grid is
far coarser than the island. Use HYSPLIT when the answer has to stand up.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as C

R_EARTH = 6_371_000.0


def load_met(path=None) -> dict:
    return dict(np.load(path or (C.DATA / "met_grid.npz"), allow_pickle=True))


def wind_uv(met, level="850hPa"):
    """Met convention gives the direction wind comes FROM; return the vector it
    blows TOWARD, in m/s."""
    spd = met[f"wind_speed_{level}"] / 3.6
    rad = np.deg2rad(met[f"wind_direction_{level}"])
    return -spd * np.sin(rad), -spd * np.cos(rad)


def _sample(field, lat, lon, ti, lats, lons):
    """Bilinear sample, clamped at the edges, NaN-tolerant."""
    fi = np.clip((lat - lats[0]) / (lats[1] - lats[0]), 0, lats.size - 1.001)
    fj = np.clip((lon - lons[0]) / (lons[1] - lons[0]), 0, lons.size - 1.001)
    i0, j0 = int(fi), int(fj)
    di, dj = fi - i0, fj - j0
    q = field[ti, i0:i0 + 2, j0:j0 + 2]
    m = ~np.isnan(q)
    if not m.any():
        return np.nan
    w = np.array([[(1 - di) * (1 - dj), (1 - di) * dj],
                  [di * (1 - dj), di * dj]])
    return float((q[m] * w[m]).sum() / w[m].sum())


def back_trajectory(met, lat0, lon0, t_index, hours=120, level="850hPa"):
    """Step backwards one hour at a time. Returns (times, lats, lons); may be
    shorter than `hours` if the parcel leaves the domain or the record starts."""
    u, v = wind_uv(met, level)
    lats, lons, times = met["lat"], met["lon"], met["time"]
    lat, lon, ti = float(lat0), float(lon0), int(t_index)
    out = [(times[ti], lat, lon)]
    for _ in range(hours):
        if ti - 1 < 0:
            break
        uu = _sample(u, lat, lon, ti, lats, lons)
        vv = _sample(v, lat, lon, ti, lats, lons)
        if not (np.isfinite(uu) and np.isfinite(vv)):
            break
        lat -= (vv * 3600.0) * 180.0 / (np.pi * R_EARTH)
        lon -= (uu * 3600.0) * 180.0 / (np.pi * R_EARTH *
                                        max(0.2, np.cos(np.deg2rad(lat))))
        ti -= 1
        out.append((times[ti], lat, lon))
        if not (C.BBOX["south"] < lat < C.BBOX["north"]
                and C.BBOX["west"] < lon < C.BBOX["east"]):
            break
    t, la, lo = zip(*out)
    return np.array(t), np.array(la), np.array(lo)


if __name__ == "__main__":
    met = load_met()
    times = met["time"]
    target = sys.argv[1] if len(sys.argv) > 1 else times[-1]
    ti = int(np.argmin(np.abs(times.astype("datetime64[ns]")
                              - np.datetime64(target))))
    t, la, lo = back_trajectory(met, C.CEBU["lat"], C.CEBU["lon"], ti)
    print(f"850 hPa back-trajectory arriving {C.CEBU['name']} {times[ti]}Z ({len(t)-1} h)")
    for k in range(0, len(t), 12):
        print(f"  T-{k:3d}h  {t[k]}Z  {la[k]:6.2f}N {lo[k]:7.2f}E")
    print(f"  END T-{len(t)-1:3d}h  {t[-1]}Z  {la[-1]:6.2f}N {lo[-1]:7.2f}E")
