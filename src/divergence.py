"""3D divergence of the wind field, in spherical coordinates.

On a sphere the horizontal part is NOT du/dx + dv/dy. Meridians converge, so
the cos(lat) metric has to be carried:

    div_h = 1/(r cos@) * [ du/d& + d(v cos@)/d@ ]          @ = lat, & = lon
    div_v = 1/r^2 * d(r^2 w)/dz                             r = a + z

Ignoring the metric overstates the meridional term by 1/cos(lat) and drops the
2w/r sphericity term entirely; both are small at 10 deg N but free to include.

    python src/divergence.py            # writes data/divergence.npz
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C

A_EARTH = 6_371_000.0


def divergence3d(u, v, w, lat, lon, z, zfull=None):
    """(time, level, lat, lon) fields -> divergence in 1/s.

    Centred differences inside, one-sided at the edges (np.gradient), so the
    boundary rows are the least trustworthy part of the result.

    zfull, if given, is the per-column height (t, z, lat, lon). On a hybrid
    sigma grid the layer thickness varies with surface pressure -- 40 to 79 m
    where the domain mean is 59 -- so using the mean for d/dz is wrong by that
    much, worst near the ground and over terrain. Pass it.
    """
    phi = np.deg2rad(lat)[None, None, :, None]
    dphi = np.deg2rad(np.gradient(lat))[None, None, :, None]
    dlam = np.deg2rad(np.gradient(lon))[None, None, None, :]
    r = A_EARTH + z[None, :, None, None]
    cosp = np.cos(phi)

    du_dlam = np.gradient(u, axis=3) / dlam
    dvcos_dphi = np.gradient(v * cosp, axis=2) / dphi
    div_h = (du_dlam + dvcos_dphi) / (r * cosp)

    # 1/r^2 d(r^2 w)/dz, expanded so the uneven level spacing is handled by
    # gradient on the actual heights
    if zfull is None:
        r2w = (r ** 2) * w
        div_v = np.gradient(r2w, z, axis=1) / (r ** 2)
    else:
        rr = A_EARTH + zfull
        r2w = (rr ** 2) * w
        # second-order centred differences on a non-uniform, per-column axis
        dz_up = np.diff(zfull, axis=1)
        div_v = np.empty_like(w)
        du = np.diff(r2w, axis=1)
        div_v[:, 1:-1] = ((du[:, 1:] * dz_up[:, :-1] / dz_up[:, 1:]
                           + du[:, :-1] * dz_up[:, 1:] / dz_up[:, :-1])
                          / (dz_up[:, :-1] + dz_up[:, 1:]))
        div_v[:, 0] = du[:, 0] / dz_up[:, 0]
        div_v[:, -1] = du[:, -1] / dz_up[:, -1]
        div_v /= rr ** 2
    return div_h + div_v, div_h, div_v


def load(src="wind3d.npz"):
    path = src if os.path.isabs(src) else C.DATA / src
    d = np.load(path, allow_pickle=True)
    zl = (np.nanmean(d["z"], axis=(0, 2, 3)) if d["z"].ndim == 4
          else np.asarray(d["z"], float))
    o = np.argsort(zl)
    spd = d["spd"][:, o] / 3.6
    rad = np.deg2rad(d["dir"][:, o])
    u, v, w = -spd * np.sin(rad), -spd * np.cos(rad), d["w"][:, o]
    t = pd.to_datetime(d["time"])
    keep = np.isfinite(u).all(axis=(1, 2, 3))
    zf = d["zfull"][keep][:, o] if "zfull" in d.files else None
    return (t[keep], d["lat"].astype(float), d["lon"].astype(float), zl[o],
            u[keep], v[keep], w[keep], zf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="wind3d.npz", help="wind cube in data/")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = str(a.src if os.path.isabs(a.src)
                    else C.DATA / a.src).replace("wind3d", "divergence")

    t, lat, lon, z, u, v, w, zfull = load(a.src)
    print(f"{a.src}: {u.shape}  lat {lat.min():.1f}..{lat.max():.1f} "
          f"lon {lon.min():.1f}..{lon.max():.1f}  step "
          f"{abs(lat[1]-lat[0]):.2f} deg", file=sys.stderr)
    div, dh, dv = divergence3d(u, v, w, lat, lon, z, zfull)
    if zfull is None:
        print("  WARNING: no per-column heights in this cube; d/dz uses the "
              "domain mean", file=sys.stderr)
    np.savez_compressed(a.out, time=t.values.astype("datetime64[s]"),
                        lat=lat, lon=lon, z=z,
                        div=div.astype("float32"), div_h=dh.astype("float32"),
                        div_v=dv.astype("float32"))
    print(f"wrote {a.out}  {div.shape}")

    # interior only: the edge rows carry one-sided differences
    inner = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    for name, f in (("total", div), ("horizontal", dh), ("vertical", dv)):
        q = f[inner]
        print(f"  {name:11s} rms {np.sqrt(np.nanmean(q**2)):.2e} /s   "
              f"p1 {np.nanpercentile(q, 1):+.2e}  p99 {np.nanpercentile(q, 99):+.2e}")

    # mass balance: horizontal and vertical terms should largely cancel
    r = np.corrcoef(dh[inner].ravel(), dv[inner].ravel())[0, 1]
    print(f"  corr(horizontal, vertical) = {r:+.3f}   "
          f"(continuity implies they oppose; -1 would be exact)")

    i = int(np.argmin(abs(lat - C.CEBU["lat"])))
    k = int(np.argmin(abs(lon - C.CEBU["lon"])))
    tp = t + pd.Timedelta(hours=C.TZ_OFFSET_H)
    col = np.nanmean(div[:, :, i - 1:i + 2, k - 1:k + 2], axis=(2, 3))
    epi = tp >= "2026-09-18 10:00"
    print("\n  Cebu column, mean divergence by level (1e-5 /s):")
    print(f"  {'height':>8}{'background':>12}{'episode':>10}")
    for n, zz in enumerate(z):
        print(f"  {zz:8.0f}{col[~epi, n].mean()*1e5:+12.2f}{col[epi, n].mean()*1e5:+10.2f}")


if __name__ == "__main__":
    main()
