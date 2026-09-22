"""Build an analysis cube from the ARL files: subset, convert, save as npz.

Turns 28 GB of global 3-hourly ARL into a small regional cube in the same
layout the rest of the project already reads, so divergence.py and the
plotting scripts work against it unchanged.

Two conversions happen here, both of which bite if skipped:
  WWND is dp/dt in hPa/s, negative upward -> w = -omega/(rho g), m/s, up positive
  the vertical coordinate is hybrid sigma, so height comes from the stored
  PRES and TEMP per level, not from a fixed table

    python src/arl_cube.py --out data/wind3d_arl.npz
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from arl import ArlFile

R_D, G = 287.05, 9.81
ARL_DIR = "/Volumes/JCG_Backup1/pollution-tracking/data/arl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=ARL_DIR)
    ap.add_argument("--out", default=str(C.DATA / "wind3d_arl.npz"))
    ap.add_argument("--south", type=float, default=5.0)
    ap.add_argument("--north", type=float, default=15.0)
    ap.add_argument("--west", type=float, default=118.0)
    ap.add_argument("--east", type=float, default=128.0)
    ap.add_argument("--nlev", type=int, default=40, help="lowest N model levels")
    a = ap.parse_args()
    box = dict(south=a.south, north=a.north, west=a.west, east=a.east)

    files = sorted(glob.glob(os.path.join(a.dir, "*_gfs0p25")))
    if not files:
        raise SystemExit(f"no ARL files in {a.dir}")
    f0 = ArlFile(files[0])
    lv = [l for l in f0.levels if l > 0][:a.nlev]
    print(f"{len(files)} files, {len(lv)} levels, box "
          f"{a.south}-{a.north}N {a.west}-{a.east}E", file=sys.stderr)

    times, U, V, W, T, P, RH = [], [], [], [], [], [], []
    for path in files:
        f = ArlFile(path)
        for ti, t in enumerate(f.times):
            if (t, lv[0], "TEMP") not in f.map:
                continue
            g = lambda v, l: f.field(v, l, ti, box)[0]
            U.append(np.stack([g("UWND", l) for l in lv]))
            V.append(np.stack([g("VWND", l) for l in lv]))
            W.append(np.stack([g("WWND", l) for l in lv]))
            T.append(np.stack([g("TEMP", l) for l in lv]))
            P.append(np.stack([g("PRES", l) for l in lv]))
            RH.append(np.stack([g("RELH", l) for l in lv]))
            times.append(pd.Timestamp(2000 + t[0], t[1], t[2], t[3]))
        print(f"  {os.path.basename(path)}: {len(f.times)} times", file=sys.stderr)

    U, V, W = np.array(U, "float32"), np.array(V, "float32"), np.array(W, "float32")
    T, P, RH = np.array(T, "float32"), np.array(P, "float32"), np.array(RH, "float32")
    _, la, lo = f0.field("TEMP", lv[0], 0, box)

    # hPa/s, negative up  ->  m/s, positive up
    rho = (P * 100.0) / (R_D * T)
    w_ms = -(W * 100.0) / (rho * G)

    # hybrid sigma: height from the hypsometric relation, integrated upward
    # from the lowest level using the layer-mean virtual-ish temperature
    z = np.zeros_like(P)
    z[:, 0] = R_D * T[:, 0] / G * np.log(P[:, 0] / P[:, 0])      # = 0, the anchor
    for k in range(1, P.shape[1]):
        tm = 0.5 * (T[:, k] + T[:, k - 1])
        z[:, k] = z[:, k - 1] + R_D * tm / G * np.log(P[:, k - 1] / P[:, k])
    zl = np.nanmean(z, axis=(0, 2, 3))

    spd = np.hypot(U, V) * 3.6                    # the project stores km/h
    drc = (np.degrees(np.arctan2(-U, -V)) + 360) % 360
    np.savez_compressed(
        a.out, time=np.array(times, dtype="datetime64[s]"), lat=la, lon=lo,
        level=np.array(lv), z=zl.astype("float32"),
        spd=spd.astype("float32"), dir=drc.astype("float32"),
        w=w_ms.astype("float32"), temp=T, pres=P, relh=RH, zfull=z.astype("float32"))
    print(f"wrote {a.out}  {w_ms.shape}", file=sys.stderr)
    print(f"  {times[0]:%d %b %H:%M} .. {times[-1]:%d %b %H:%M} UTC "
          f"({len(times)} times, 3-hourly)", file=sys.stderr)
    print(f"  heights: {np.round(zl[:10]).astype(int)} ... {zl[-1]:.0f} m",
          file=sys.stderr)
    print(f"  w range {np.nanpercentile(w_ms,1):+.4f}..{np.nanpercentile(w_ms,99):+.4f} m/s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
