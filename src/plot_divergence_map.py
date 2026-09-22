"""One divergence map: one time, one level. A sample before committing to a series.

    python src/plot_divergence_map.py --time "2026-09-19 00:00" --level 380
    python src/plot_divergence_map.py --smooth 5        # 5x5 boxcar, ~140 km
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config as C

SRC = "/Volumes/JCG_Backup1/pollution-tracking/data/divergence_region.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--field", default="div", choices=["div", "div_h", "div_v"])
    ap.add_argument("--time", default="2026-09-19 00:00")
    ap.add_argument("--level", type=float, default=380.0, help="height in m")
    ap.add_argument("--smooth", type=int, default=0, help="boxcar width in cells")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = np.load(a.src, allow_pickle=True)
    t = pd.to_datetime(d["time"])
    lat, lon, z = d["lat"], d["lon"], d["z"]
    k = int(np.argmin(abs(t - pd.Timestamp(a.time))))
    li = int(np.argmin(abs(z - a.level)))
    f = d[a.field][k, li] * 1e5                       # 1e-5 /s
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = np.nan  # one-sided edge ring

    if a.smooth > 1:
        from scipy.ndimage import uniform_filter
        m = np.isfinite(f)
        g = np.where(m, f, 0.0)
        num = uniform_filter(g, a.smooth, mode="nearest")
        den = uniform_filter(m.astype(float), a.smooth, mode="nearest")
        f = np.where(m, num / np.maximum(den, 1e-6), np.nan)

    lim = float(np.nanpercentile(abs(f), 98))
    try:
        import cartopy.crs as ccrs
        PC = ccrs.PlateCarree()
    except ImportError:
        PC = None

    fig = plt.figure(figsize=(11.2, 9.2), dpi=C.FIG_DPI)
    ax = fig.add_axes([0.07, 0.10, 0.80, 0.78],
                      **(dict(projection=PC) if PC else {}))
    pm = ax.pcolormesh(lon, lat, f, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       shading="nearest", **(dict(transform=PC) if PC else {}))
    if PC:
        ax.coastlines(resolution="50m", linewidth=0.6, color="#1c1917")
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=PC)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#e7e5e4")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": C.INK_MUTED}
    ax.plot(C.CEBU["lon"], C.CEBU["lat"], marker="*", ms=15, color=C.RECEPTOR,
            mec="white", mew=1.2, **(dict(transform=PC) if PC else {}))
    cb = fig.colorbar(pm, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("divergence (10$^{-5}$ s$^{-1}$)  ·  red = spreading out",
                 fontsize=9, color=C.INK)
    cb.ax.tick_params(labelsize=8, colors=C.INK_MUTED)

    tp = t[k] + pd.Timedelta(hours=C.TZ_OFFSET_H)
    lab = {"div": "total", "div_h": "horizontal", "div_v": "vertical"}[a.field]
    sm = f" · {a.smooth}x{a.smooth} boxcar (~{a.smooth*28:.0f} km)" if a.smooth > 1 else " · unsmoothed"
    fig.text(0.07, 0.965, f"{lab.capitalize()} divergence · {z[li]:.0f} m · "
             f"{tp:%a %d %b %H:%M} PHT", fontsize=13.5, color=C.INK,
             weight="bold", va="top")
    fig.text(0.07, 0.932, f"NOAA GFS 0.25° · spherical, cos(lat) metric{sm} · "
             f"{C.WATERMARK.replace('Made by: ', '')}",
             fontsize=9, color=C.INK_MUTED, va="top")
    fig.text(0.07, 0.045, f"p1 {np.nanpercentile(f,1):+.2f}   p99 "
             f"{np.nanpercentile(f,99):+.2f}   rms {np.sqrt(np.nanmean(f**2)):.2f} "
             f"(10⁻⁵/s) · edge ring masked",
             fontsize=7.5, color=C.INK_MUTED)
    out = C.FIGS / (a.out or f"div_map_{a.field}_{z[li]:.0f}m.png")
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"wrote {out}")
    print(f"  {tp:%d %b %H:%M} PHT, {z[li]:.0f} m, rms {np.sqrt(np.nanmean(f**2)):.2f}e-5/s")


if __name__ == "__main__":
    main()
