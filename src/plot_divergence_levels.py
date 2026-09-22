"""Divergence on all six levels at once, animated.

One frame per hour, six map panels, so the vertical structure is read across
panels instead of being faked between them. Three fields available:

    total  = div_h + div_v     what continuity actually constrains
    h      = horizontal        well resolved: 0.25 deg, centred differences
    v      = vertical          d(r^2 w)/dz across SIX uneven levels -- the
                               weakest term, shown so its weakness is visible

Divergence magnitude depends on grid spacing, so a field computed at 1 deg is
not comparable with one at 0.25 deg. The subtitle states the spacing.

    python src/plot_divergence_levels.py --field total
    python src/plot_divergence_levels.py --src divergence_region.npz --field h
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from make_gif import build_mp4

FIELDS = {"total": ("div", "total divergence"),
          "h": ("div_h", "horizontal divergence"),
          "v": ("div_v", "vertical divergence")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="divergence.npz")
    ap.add_argument("--field", default="total", choices=list(FIELDS))
    ap.add_argument("--start", default=None)
    ap.add_argument("--hours", type=int, default=36)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    key, label = FIELDS[a.field]
    stem = a.out or f"div_levels_{a.field}"

    d = np.load(C.DATA / a.src, allow_pickle=True)
    t = pd.to_datetime(d["time"]) + pd.Timedelta(hours=C.TZ_OFFSET_H)
    lat, lon, z = d["lat"], d["lon"], d["z"]
    fld = d[key] * 1e5                          # 1e-5 /s
    step = abs(float(lat[1] - lat[0]))
    # the outer ring is one-sided differences, not a real estimate
    fld[:, :, 0, :] = fld[:, :, -1, :] = np.nan
    fld[:, :, :, 0] = fld[:, :, :, -1] = np.nan
    lim = float(np.nanpercentile(abs(fld), 99))

    i0 = (int(np.argmin(abs(t - pd.Timestamp(a.start) -
                            pd.Timedelta(hours=C.TZ_OFFSET_H)))) if a.start
          else max(0, t.size - a.hours - 1))
    idx = list(range(i0, min(i0 + a.hours, t.size), a.every))
    print(f"{label}: {fld.shape}, {step:.2f} deg, {len(idx)} frames "
          f"from {t[i0]:%d %b %H:%M} PHT", file=sys.stderr)

    try:
        import cartopy.crs as ccrs, cartopy.feature as cfeature
        PC = ccrs.PlateCarree()
    except ImportError:
        PC = None

    frames_dir = C.FIGS / "frames" / stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    paths = []
    for n, k in enumerate(idx):
        fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.0), dpi=C.FIG_DPI,
                                 subplot_kw=dict(projection=PC) if PC else None)
        for li, ax in enumerate(axes.ravel()):
            pm = ax.pcolormesh(lon, lat, fld[k, li], cmap="RdBu_r",
                               vmin=-lim, vmax=lim, shading="nearest",
                               **(dict(transform=PC) if PC else {}))
            if PC:
                ax.coastlines(resolution="50m", linewidth=0.5, color=C.COAST)
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=PC)
                gl = ax.gridlines(draw_labels=(li % 3 == 0 or li >= 3),
                                  linewidth=0.3, color="#e7e5e4")
                gl.top_labels = gl.right_labels = False
                gl.xlabel_style = gl.ylabel_style = {"size": 6.5,
                                                     "color": C.INK_MUTED}
            ax.plot(C.CEBU["lon"], C.CEBU["lat"], marker="*", ms=9,
                    color=C.RECEPTOR, mec="white", mew=0.8,
                    **(dict(transform=PC) if PC else {}))
            ax.set_title(f"{z[li]/1000:.2f} km", fontsize=9, color=C.INK_MUTED,
                         loc="left", pad=3)
        cb = fig.colorbar(pm, ax=axes, pad=0.012, fraction=0.022, aspect=34)
        cb.set_label(f"{label} (10$^{{-5}}$ s$^{{-1}}$)  ·  red = spreading out",
                     fontsize=8.5, color=C.INK)
        cb.ax.tick_params(labelsize=7, colors=C.INK_MUTED)

        fig.text(0.055, 0.972, f"{label.capitalize()} · "
                 f"{t[k]:%a %d %b %H:%M} PHT", fontsize=13,
                 color=C.INK, weight="bold", va="top")
        fig.text(0.055, 0.938,
                 f"spherical, {step:.2f}° grid ({step*111:.0f} km) · magnitude "
                 f"scales with spacing, so do not compare across resolutions · "
                 f"edge ring masked · {C.WATERMARK.replace('Made by: ', '')}",
                 fontsize=8, color=C.INK_MUTED, va="top")
        out = frames_dir / f"{stem}_{n:04d}.png"
        fig.savefig(out, facecolor=C.SURFACE, bbox_inches=None)
        plt.close(fig)
        paths.append(out)
        if n % 6 == 0:
            print(f"  [{n + 1:3d}/{len(idx)}] {out.name}", file=sys.stderr, flush=True)

    rgb = [Image.open(q).convert("RGB") for q in paths]
    master = rgb[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    imgs = [im.quantize(palette=master, dither=Image.Dither.NONE) for im in rgb]
    per = [int(1000 / a.fps)] * len(imgs); per[-1] = 1800
    gif = C.FIGS / f"{stem}.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=per,
                 loop=0, optimize=False, disposal=2)
    print(f"{len(imgs)} frames -> {gif} ({gif.stat().st_size / 1e6:.1f} MB)",
          file=sys.stderr)
    if mp4 := build_mp4(paths, C.FIGS / f"{stem}.mp4", a.fps):
        print(f"            -> {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
