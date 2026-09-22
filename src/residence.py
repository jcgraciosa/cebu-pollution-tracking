"""Does air linger over the Visayas? 3D residence time from IFS winds.

Seeds parcels through a control volume around Cebu and integrates them on the
resolved 3D wind until they leave, either through a side or through the lid.
The lid matters: air lofted above it is no longer air Cebu breathes, so vertical
export counts as loss. A single-level trajectory would have called that
"lingering".

Reported as a RELATIVE diagnostic -- episode against background -- because the
absolute hours carry more uncertainty than the contrast does (see LIMITS below).

    python src/download.py --parts wind3d     # once, fetches the box
    python src/residence.py

LIMITS
  - vertical velocity is quantised at 0.01 m/s against signals of 0.02-0.07,
    so a 24 h vertical displacement is uncertain by roughly +/-900 m
  - resolved motion only: no convection at 0.25 deg, and convective lofting
    over the Visayas in September is fast and entirely invisible here
  - deterministic, not dispersive: no turbulent spread. HYSPLIT or FLEXPART in
    particle mode is the tool when the answer has to stand up
  - residence of AIR, not of pollution: no wet deposition, and monsoon
    scavenging can clear the aerosol while the air itself stays
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C

R_EARTH = 6_371_000.0
DT = 900.0                 # s; 15 min steps
MAX_H = 48                 # give up after this many hours inside
WINDOW_H = 24              # the comparison statistic: escaped within this many hours


def load():
    """u, v, w on ascending height levels, plus the axes."""
    d = np.load(C.DATA / "wind3d.npz", allow_pickle=True)
    spd = d["spd"] / 3.6                       # km/h -> m/s
    rad = np.deg2rad(d["dir"])
    u, v, w = -spd * np.sin(rad), -spd * np.cos(rad), d["w"]
    # geopotential height varies by only a few tens of metres here, so one
    # height per level is enough and keeps the vertical axis fixed
    zlev = (np.nanmean(d["z"], axis=(0, 2, 3)) if d["z"].ndim == 4
          else np.asarray(d["z"], float))
    o = np.argsort(zlev)                       # pressure descends, height ascends
    t = pd.to_datetime(d["time"])
    # Open-Meteo serves pressure levels for the last ~8 days only, whatever
    # past_days asks for, so the head of the record is empty
    keep = np.isfinite(u).all(axis=(1, 2, 3))
    if not keep.all():
        i0, i1 = np.where(keep)[0][[0, -1]]
        print(f"  pressure levels present {t[i0]} .. {t[i1]} "
              f"({keep.sum()} of {keep.size} h)", file=sys.stderr)
        sl = slice(i0, i1 + 1)
        t, u, v, w = t[sl], u[sl], v[sl], w[sl]
    return (t, d["lat"].astype(float), d["lon"].astype(float),
            zlev[o], u[:, o], v[:, o], w[:, o])


def _sample(F, t, zi, la, lo, lat, lon, zlev):
    """Trilinear in (z, lat, lon), linear in time, for whole parcel arrays."""
    t0 = np.clip(int(np.floor(t)), 0, F.shape[0] - 2)
    ft = np.clip(t - t0, 0.0, 1.0)

    fz = np.clip(np.interp(zi, zlev, np.arange(zlev.size)), 0, zlev.size - 1.001)
    fi = np.clip((la - lat[0]) / (lat[1] - lat[0]), 0, lat.size - 1.001)
    fj = np.clip((lo - lon[0]) / (lon[1] - lon[0]), 0, lon.size - 1.001)
    k0, i0, j0 = fz.astype(int), fi.astype(int), fj.astype(int)
    dk, di, dj = fz - k0, fi - i0, fj - j0

    out = 0.0
    for tt, wt in ((t0, 1 - ft), (t0 + 1, ft)):
        if wt == 0:
            continue
        acc = 0.0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    q = F[tt, k0 + a, i0 + b, j0 + c]
                    acc = acc + q * ((dk if a else 1 - dk)
                                     * (di if b else 1 - di)
                                     * (dj if c else 1 - dj))
        out = out + wt * acc
    return out


def integrate(t_index, seeds, times, lat, lon, zlev, u, v, w, box, lid):
    """Forward-integrate until each parcel leaves. Returns hours and exit face."""
    la, lo, zz = (s.copy() for s in seeds)
    n = la.size
    hours = np.full(n, np.nan)
    face = np.full(n, "", dtype=object)
    alive = np.ones(n, bool)
    t = float(t_index)
    steps = int(MAX_H * 3600 / DT)

    ran_out = False
    for s in range(steps):
        if not alive.any():
            break
        if t >= times.size - 1.001:
            ran_out = True                      # censored, NOT held
            break
        a = alive
        args = (t, zz[a], la[a], lo[a], lat, lon, zlev)
        # midpoint (RK2): the wind turns appreciably within an hour
        uu, vv, ww = (_sample(F, *args) for F in (u, v, w))
        dl = 180.0 / (np.pi * R_EARTH)
        la_m = la[a] + vv * (DT / 2) * dl
        lo_m = lo[a] + uu * (DT / 2) * dl / np.maximum(0.2, np.cos(np.deg2rad(la[a])))
        z_m = zz[a] + ww * (DT / 2)
        args_m = (t + DT / 7200.0, z_m, la_m, lo_m, lat, lon, zlev)
        uu, vv, ww = (_sample(F, *args_m) for F in (u, v, w))

        la[a] += vv * DT * dl
        lo[a] += uu * DT * dl / np.maximum(0.2, np.cos(np.deg2rad(la[a])))
        zz[a] += ww * DT
        zz[a] = np.maximum(zz[a], 10.0)        # the ground reflects, it is not a sink
        t += DT / 3600.0

        out_side = ((la < box["south"]) | (la > box["north"]) |
                    (lo < box["west"]) | (lo > box["east"])) & alive
        out_top = (zz > lid) & alive
        h = (s + 1) * DT / 3600.0
        for m, nm in ((out_top, "top"), (out_side & ~out_top, "side")):
            if m.any():
                hours[m], face[m], alive[m] = h, nm, False
        bad = alive & ~np.isfinite(la + lo + zz)
        if bad.any():
            hours[bad], face[bad], alive[bad] = np.nan, "nan", False

    # a parcel still inside when the WIND RECORD ends is censored, not held: the
    # episode sits at the end of the record, so calling these "held" would have
    # manufactured exactly the long residence we are testing for
    hours[alive] = np.nan if ran_out else MAX_H
    face[alive] = "censored" if ran_out else "held"
    return hours, face


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=3, help="seed every N hours")
    ap.add_argument("--seed-step", type=float, default=0.5,
                    help="seed spacing in degrees")
    ap.add_argument("--out", default=str(C.DATA / "residence.csv"))
    a = ap.parse_args()

    times, lat, lon, zlev, u, v, w = load()
    box, lid = C.WIND3D_BOX, C.WIND3D_LID
    print(f"{times[0]} .. {times[-1]}  ({times.size} h), levels {np.round(zlev)} m",
          file=sys.stderr)

    sla = np.arange(box["south"] + a.seed_step, box["north"], a.seed_step)
    slo = np.arange(box["west"] + a.seed_step, box["east"], a.seed_step)
    g_la, g_lo, g_z = np.meshgrid(sla, slo, C.WIND3D_SEED_Z, indexing="ij")
    seeds = (g_la.ravel(), g_lo.ravel(), g_z.ravel())
    print(f"{seeds[0].size} parcels per seed hour "
          f"({sla.size}x{slo.size} x {len(C.WIND3D_SEED_Z)} heights)", file=sys.stderr)

    rows = []
    idx = range(0, times.size - 1, a.every)
    for n, ti in enumerate(idx):
        hrs, face = integrate(ti, seeds, times, lat, lon, zlev, u, v, w, box, lid)
        ok = np.isfinite(hrs)
        # escape fraction is defined whenever the record covers the window, even
        # if longer residences are censored -- that is what makes it comparable
        left = times.size - 1 - ti
        esc = float((np.nan_to_num(hrs, nan=np.inf) <= WINDOW_H).mean()) \
            if left >= WINDOW_H else np.nan
        # box-mean wind in the lowest levels, for the flushing-time comparison
        below = zlev <= lid
        spd = np.nanmean(np.hypot(u[ti][below], v[ti][below]))
        rows.append(dict(
            time=times[ti],
            esc24=esc,
            censored=float((face == "censored").mean()),
            residence_h=np.nanmedian(hrs[ok]) if ok.any() else np.nan,
            residence_p25=np.nanpercentile(hrs[ok], 25),
            residence_p75=np.nanpercentile(hrs[ok], 75),
            frac_top=float((face == "top").mean()),
            frac_side=float((face == "side").mean()),
            frac_held=float((face == "held").mean()),
            mean_wind=spd,
            flush_h=(box["north"] - box["south"]) * 111_000.0 / max(spd, 0.1) / 3600.0,
        ))
        if n % 20 == 0:
            print(f"  {n:4d}/{len(idx)}  {times[ti]:%d %b %Hh}  "
                  f"median {rows[-1]['residence_h']:.1f} h", file=sys.stderr, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(df)} seed hours)", file=sys.stderr)
    print(df[["esc24", "residence_h", "frac_top", "frac_side", "censored",
              "mean_wind", "flush_h"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
