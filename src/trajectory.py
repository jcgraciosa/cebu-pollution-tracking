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
    DT = 3600.0
    DEG = 180.0 / (np.pi * R_EARTH)

    def step(la, lo, t, frac):
        """Displacement in degrees for `frac` of an hour backwards from (la,lo,t)."""
        uu = _sample(u, la, lo, t, lats, lons)
        vv = _sample(v, la, lo, t, lats, lons)
        if not (np.isfinite(uu) and np.isfinite(vv)):
            return None, None
        dla = -(vv * DT * frac) * DEG
        dlo = -(uu * DT * frac) * DEG / max(0.2, np.cos(np.deg2rad(la)))
        return dla, dlo

    for _ in range(hours):
        if ti - 1 < 0:
            break
        # midpoint (RK2): sample the wind halfway back, not at the arrival
        # point. Plain Euler drifts outward in rotational flow, which is
        # exactly where a 120 h trajectory spends its time.
        dla, dlo = step(lat, lon, ti, 0.5)
        if dla is None:
            break
        dla, dlo = step(lat + dla, lon + dlo, ti, 1.0)   # wind at the midpoint
        if dla is None:
            break
        lat += dla
        lon += dlo
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


# --- 3D ----------------------------------------------------------------------

def load_traj3d(path=None) -> dict:
    return dict(np.load(path or (C.DATA / "traj3d.npz"), allow_pickle=True))


def _sample3(field, zlev, lat, lon, ti, lats, lons, z):
    """Bilinear in lat/lon, linear in height between the two bracketing levels.

    Levels are pressure surfaces whose heights vary, so the bracketing is done
    on the geopotential height at this column rather than on a fixed table.
    """
    col = np.array([_sample(zlev[:, li], lat, lon, ti, lats, lons)
                    for li in range(zlev.shape[1])])
    good = np.isfinite(col)
    if good.sum() < 2:
        return np.nan
    idx = np.flatnonzero(good)
    col_g = col[idx]
    order = np.argsort(col_g)             # ascending height
    idx, col_g = idx[order], col_g[order]
    zc = float(np.clip(z, col_g[0], col_g[-1]))
    k = int(np.clip(np.searchsorted(col_g, zc) - 1, 0, len(col_g) - 2))
    f = (zc - col_g[k]) / max(col_g[k + 1] - col_g[k], 1e-6)
    a = _sample(field[:, idx[k]], lat, lon, ti, lats, lons)
    b = _sample(field[:, idx[k + 1]], lat, lon, ti, lats, lons)
    if not np.isfinite(a):
        return b
    if not np.isfinite(b):
        return a
    return float(a * (1 - f) + b * f)


def back_trajectory_3d(cube, lat0, lon0, t_index, hours=120, start_hpa=None):
    """3D kinematic back-trajectory: the parcel rises and sinks.

    Returns (times, lats, lons, heights_m). Midpoint (RK2) in all three
    components; plain Euler drifts outward in rotational flow.
    """
    start_hpa = start_hpa or C.TRAJ_START_HPA
    lats, lons, times = cube["lat"], cube["lon"], cube["time"]
    lev = list(cube["level"])
    spd, dirn, w, zf = cube["spd"] / 3.6, cube["dir"], cube["w"], cube["z"]
    u = -spd * np.sin(np.deg2rad(dirn))
    v = -spd * np.cos(np.deg2rad(dirn))

    ti = int(t_index)
    li = lev.index(start_hpa) if start_hpa in lev else len(lev) // 2
    z = _sample(zf[:, li], lat0, lon0, ti, lats, lons)
    if not np.isfinite(z):
        z = 1500.0
    lat, lon = float(lat0), float(lon0)
    out = [(times[ti], lat, lon, z)]
    DT, DEG = 3600.0, 180.0 / (np.pi * R_EARTH)

    def step(la, lo, zz, t, frac):
        uu = _sample3(u, zf, la, lo, t, lats, lons, zz)
        vv = _sample3(v, zf, la, lo, t, lats, lons, zz)
        ww = _sample3(w, zf, la, lo, t, lats, lons, zz)
        if not (np.isfinite(uu) and np.isfinite(vv)):
            return None
        ww = 0.0 if not np.isfinite(ww) else ww
        return (-(vv * DT * frac) * DEG,
                -(uu * DT * frac) * DEG / max(0.2, np.cos(np.deg2rad(la))),
                -(ww * DT * frac))

    for _ in range(hours):
        if ti - 1 < 0:
            break
        d1 = step(lat, lon, z, ti, 0.5)
        if d1 is None:
            break
        d2 = step(lat + d1[0], lon + d1[1], z + d1[2], ti, 1.0)
        if d2 is None:
            break
        lat, lon, z = lat + d2[0], lon + d2[1], max(50.0, z + d2[2])
        ti -= 1
        out.append((times[ti], lat, lon, z))
        if not (C.TRAJ_BOX["south"] < lat < C.TRAJ_BOX["north"]
                and C.TRAJ_BOX["west"] < lon < C.TRAJ_BOX["east"]):
            break
    t, la, lo, zz = zip(*out)
    return np.array(t), np.array(la), np.array(lo), np.array(zz)
