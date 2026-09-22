"""Integrated smoke transport: the vector field the episode is actually about.

Divergence measures deformation of the flow; it says nothing about how much
smoke moves. Transport is a FLUX, F = c*v, and the standard diagnostic is the
vertically integrated horizontal part -- the same construction as the
integrated vapour transport used for atmospheric rivers.

    column smoke mass  M = AOD / MEE                 [g m-2]
    layer-mean wind    U = sum(u dz) / sum(dz)       over 0-3 km
    transport          F = M * U                     [g m-1 s-1]

MEE is the mass extinction efficiency of biomass-burning aerosol at 550 nm,
~4 m2/g. It carries most of the absolute uncertainty here, so read the pattern
and the ratios, not the absolute number.

    python src/smoke_flux.py --time "2026-09-19 00:00"
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C

WIND = "/Volumes/JCG_Backup1/pollution-tracking/data/wind3d_region.npz"
MEE = 4.0            # m2/g, smoke at 550 nm
LAYER = 3000.0       # m, the depth the smoke is assumed to move with


def build(wind=WIND, layer=LAYER):
    """Co-locate CAMS AOD with the GFS layer-mean wind on the CAMS grid."""
    d = np.load(wind, allow_pickle=True)
    tw = pd.to_datetime(d["time"])
    spd, drc = d["spd"] / 3.6, d["dir"]
    rad = np.deg2rad(drc)
    u, v = -spd * np.sin(rad), -spd * np.cos(rad)
    zf, z = d["zfull"], d["z"]

    # mass-weighted layer mean: weight each level by its own thickness
    dz = np.empty_like(zf)
    dz[:, 1:-1] = 0.5 * (zf[:, 2:] - zf[:, :-2])
    dz[:, 0] = zf[:, 1] - zf[:, 0]
    dz[:, -1] = zf[:, -1] - zf[:, -2]
    wgt = np.where(zf <= layer, dz, 0.0)
    tot = wgt.sum(axis=1)
    ubar = (u * wgt).sum(axis=1) / np.maximum(tot, 1e-6)
    vbar = (v * wgt).sum(axis=1) / np.maximum(tot, 1e-6)

    aq = dict(np.load(C.DATA / "aq_grid.npz", allow_pickle=True))
    ta = pd.to_datetime([str(x) for x in aq["time"]])
    aod = aq["aerosol_optical_depth"]
    alat, alon = aq["lat"], aq["lon"]

    # the CAMS grid is coarser, so bring the wind to it rather than the reverse
    from scipy.interpolate import RegularGridInterpolator
    keep = np.isin(ta, tw)
    ta, aod = ta[keep], aod[keep]
    U = np.empty((len(ta), alat.size, alon.size), "float32")
    V = np.empty_like(U)
    gy, gx = np.meshgrid(alat, alon, indexing="ij")
    pts = np.stack([gy.ravel(), gx.ravel()], axis=1)
    for n, tt in enumerate(ta):
        k = int(np.argmin(abs(tw - tt)))
        for src, dst in ((ubar[k], U), (vbar[k], V)):
            f = RegularGridInterpolator((d["lat"], d["lon"]), src,
                                        bounds_error=False, fill_value=np.nan)
            dst[n] = f(pts).reshape(gy.shape)
    mass = aod / MEE * 1000.0 / 1000.0          # AOD -> g/m2
    return ta, alat, alon, mass * U, mass * V, mass, U, V


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.DATA / "smoke_flux.npz"))
    a = ap.parse_args()
    t, lat, lon, fx, fy, mass, U, V = build()
    np.savez_compressed(a.out, time=t.values.astype("datetime64[s]"), lat=lat,
                        lon=lon, fx=fx.astype("float32"), fy=fy.astype("float32"),
                        mass=mass.astype("float32"), u=U, v=V, mee=MEE, layer=LAYER)
    mag = np.hypot(fx, fy)
    print(f"wrote {a.out}  {fx.shape}")
    print(f"  {t[0]:%d %b %H:%M} .. {t[-1]:%d %b %H:%M} UTC, {len(t)} times")
    print(f"  column mass  p50 {np.nanpercentile(mass,50):.3f}  p99 "
          f"{np.nanpercentile(mass,99):.3f} g/m2")
    print(f"  |F|          p50 {np.nanpercentile(mag,50):.3f}  p99 "
          f"{np.nanpercentile(mag,99):.3f} g/m/s")
    i = int(np.argmin(abs(lat - C.CEBU["lat"]))); j = int(np.argmin(abs(lon - C.CEBU["lon"])))
    tp = t + pd.Timedelta(hours=C.TZ_OFFSET_H)
    epi = tp >= "2026-09-18 10:00"
    print(f"  at Cebu: background {np.nanmean(mag[~epi, i, j]):.3f}  "
          f"episode {np.nanmean(mag[epi, i, j]):.3f} g/m/s "
          f"({np.nanmean(mag[epi,i,j])/np.nanmean(mag[~epi,i,j]):.1f}x)")


if __name__ == "__main__":
    main()
