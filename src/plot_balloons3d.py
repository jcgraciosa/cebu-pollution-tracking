"""3D animation of the residence experiment: balloons advecting in the box.

One release of parcels, carried by the IFS 3D wind, drawn until they leave.
Colour marks fate -- still inside, gone out a side, gone up through the lid --
so the thing the numbers report is visible directly.

    python src/plot_balloons3d.py --start "2026-09-19 00:00"   # episode
    python src/plot_balloons3d.py --start "2026-09-15 00:00" --out balloons_bg

The lid at 2.5 km is the point: a parcel that crosses it has left the air Cebu
breathes even though it is still overhead. A 2D animation cannot show that.
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
from residence import load, _sample, DT
from make_gif import build_mp4

INSIDE, SIDE, TOP = "#0e7490", "#78716c", "#7c3aed"
R_EARTH = 6_371_000.0


def coast_segments(box):
    """Natural Earth coastlines clipped to the box, for the floor of the plot."""
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box as sbox
    except ImportError:
        return []
    clip = sbox(box["west"], box["south"], box["east"], box["north"])
    out = []
    path = shpreader.natural_earth(resolution="50m", category="physical",
                                   name="coastline")
    for geom in shpreader.Reader(path).geometries():
        g = geom.intersection(clip)
        if g.is_empty:
            continue
        for part in (g.geoms if hasattr(g, "geoms") else [g]):
            xy = np.asarray(part.coords)
            if len(xy) > 1:
                out.append(xy)
    return out


def advect(ti, seeds, times, lat, lon, zlev, u, v, w, box, lid, hours):
    """Full position history plus a fate code per parcel, one row per step."""
    la, lo, zz = (s.copy() for s in seeds)
    n = la.size
    fate = np.zeros(n, int)                     # 0 inside, 1 side, 2 lid
    alive = np.ones(n, bool)
    t = float(ti)
    steps = int(hours * 3600 / DT)
    hist = np.empty((steps + 1, 3, n), "float32")
    fates = np.empty((steps + 1, n), int)
    hist[0], fates[0] = np.array([la, lo, zz]), fate

    dl = 180.0 / (np.pi * R_EARTH)
    for s in range(steps):
        if alive.any() and t < times.size - 1.001:
            a = alive
            uu, vv, ww = (_sample(F, t, zz[a], la[a], lo[a], lat, lon, zlev)
                          for F in (u, v, w))
            la_m = la[a] + vv * (DT / 2) * dl
            lo_m = lo[a] + uu * (DT / 2) * dl / np.maximum(0.2, np.cos(np.deg2rad(la[a])))
            z_m = zz[a] + ww * (DT / 2)
            uu, vv, ww = (_sample(F, t + DT / 7200.0, z_m, la_m, lo_m, lat, lon, zlev)
                          for F in (u, v, w))
            la[a] += vv * DT * dl
            lo[a] += uu * DT * dl / np.maximum(0.2, np.cos(np.deg2rad(la[a])))
            zz[a] = np.maximum(zz[a] + ww * DT, 10.0)
            t += DT / 3600.0
            gone_top = (zz > lid) & alive
            gone_side = ((la < box["south"]) | (la > box["north"]) |
                         (lo < box["west"]) | (lo > box["east"])) & alive & ~gone_top
            fate[gone_top], fate[gone_side] = 2, 1
            alive &= ~(gone_top | gone_side)
        hist[s + 1], fates[s + 1] = np.array([la, lo, zz]), fate
    return hist, fates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=None, help="release time, UTC")
    ap.add_argument("--hours", type=int, default=36)
    ap.add_argument("--seed-step", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=6)
    ap.add_argument("--out", default="balloons3d")
    ap.add_argument("--spin", type=float, default=35.0, help="degrees of azimuth drift")
    a = ap.parse_args()

    times, lat, lon, zlev, u, v, w = load()
    box, lid = C.WIND3D_BOX, C.WIND3D_LID
    ti = (int(np.argmin(np.abs(times - pd.Timestamp(a.start)))) if a.start
          else max(0, times.size - a.hours - 1))
    print(f"release {times[ti]} UTC ({times[ti] + pd.Timedelta(hours=C.TZ_OFFSET_H)} PHT)",
          file=sys.stderr)

    sla = np.arange(box["south"] + a.seed_step, box["north"], a.seed_step)
    slo = np.arange(box["west"] + a.seed_step, box["east"], a.seed_step)
    g = np.meshgrid(sla, slo, C.WIND3D_SEED_Z, indexing="ij")
    seeds = tuple(x.ravel() for x in g)
    hist, fates = advect(ti, seeds, times, lat, lon, zlev, u, v, w, box, lid, a.hours)

    # wind arrows on a coarse subset of the grid, below the lid only
    li = np.where(zlev <= lid)[0]
    ji, jj = np.meshgrid(np.arange(1, lat.size, 3), np.arange(1, lon.size, 3),
                         indexing="ij")
    coasts = coast_segments(box)
    frames_dir = C.FIGS / "frames" / a.out
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    every = max(1, int(3600 / DT))               # one frame per hour
    idx = range(0, hist.shape[0], every)
    paths = []
    for n, s in enumerate(idx):
        tt = times[min(ti + int(s * DT / 3600), times.size - 1)]
        fig = plt.figure(figsize=(9.6, 7.6), dpi=C.FIG_DPI)
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.84], projection="3d",
                          computed_zorder=False)

        for xy in coasts:
            ax.plot(xy[:, 0], xy[:, 1], 0, color="#a8a29e", lw=0.7, zorder=1)
        # the lid, drawn as a translucent plane: the surface parcels escape through
        xx, yy = np.meshgrid([box["west"], box["east"]], [box["south"], box["north"]])
        ax.plot_surface(xx, yy, np.full_like(xx, lid, dtype=float),
                        color="#7c3aed", alpha=0.07, shade=False, zorder=2)

        k = li[min(1, li.size - 1)]              # arrows from one level, ~800 m
        la_g, lo_g = lat[ji], lon[jj]
        uu, vv = u[min(ti + s // every, u.shape[0] - 1), k][ji, jj], \
                 v[min(ti + s // every, v.shape[0] - 1), k][ji, jj]
        sc = 0.16
        ax.quiver(lo_g, la_g, np.full_like(lo_g, zlev[k]),
                  uu * sc, vv * sc, np.zeros_like(uu), color="#57534e",
                  linewidth=1.0, arrow_length_ratio=0.4, alpha=0.75, zorder=3)

        st = fates[s]
        for code, col, lab, sz in ((0, INSIDE, "inside the box", 13),
                                   (1, SIDE, "left by a side", 7),
                                   (2, TOP, "left through the lid", 16)):
            m = st == code
            if m.any():
                ax.scatter(hist[s, 1, m], hist[s, 0, m], hist[s, 2, m], s=sz,
                           c=col, alpha=0.85 if code == 0 else 0.22,
                           linewidths=0, label=lab, depthshade=False, zorder=4)

        ax.set_xlim(box["west"], box["east"]); ax.set_ylim(box["south"], box["north"])
        ax.set_zlim(0, lid * 1.25)
        ax.set_xlabel("longitude", fontsize=8, color=C.INK_MUTED, labelpad=2)
        ax.set_ylabel("latitude", fontsize=8, color=C.INK_MUTED, labelpad=2)
        ax.set_zlabel("height (m)", fontsize=8, color=C.INK_MUTED, labelpad=2)
        ax.tick_params(labelsize=7, colors=C.INK_MUTED)
        ax.view_init(elev=22, azim=-60 + a.spin * n / max(1, len(idx) - 1))
        ax.set_box_aspect((1, 1, 0.62))
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("white"); pane.pane.set_edgecolor("#e7e5e4")
        ax.grid(False)
        ax.plot([C.CEBU["lon"]], [C.CEBU["lat"]], [0], marker="*", ms=13,
                color=C.RECEPTOR, mec="white", mew=1.0, zorder=5)
        ax.text(C.CEBU["lon"], C.CEBU["lat"], lid * 0.06, " Cebu", fontsize=8,
                color=C.INK, zorder=5)

        n_in = int((st == 0).sum())
        fig.text(0.06, 0.965, f"Balloons over the Visayas · {tt + pd.Timedelta(hours=C.TZ_OFFSET_H):%a %d %b %H:%M} PHT",
                 fontsize=12.5, color=C.INK, weight="bold", va="top")
        fig.text(0.06, 0.928,
                 f"released {times[ti] + pd.Timedelta(hours=C.TZ_OFFSET_H):%d %b %H:%M} PHT · "
                 f"+{s * DT / 3600:.0f} h · {n_in} of {st.size} still inside "
                 f"({100*n_in/st.size:.0f}%) · lid {lid:.0f} m",
                 fontsize=9, color=C.INK_MUTED, va="top")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9,
                  facecolor="white", edgecolor="#e7e5e4")
        fig.text(0.06, 0.045, C.WATERMARK, fontsize=7, color=C.INK_MUTED)
        fig.text(0.06, 0.025, "Resolved 3D wind only: no convection, no turbulent "
                 "dispersion, no deposition.", fontsize=6.4, color=C.INK_MUTED)

        p = frames_dir / f"{a.out}_{n:04d}.png"
        fig.savefig(p, facecolor=C.SURFACE); plt.close(fig)
        paths.append(p)
        print(f"  [{n + 1:3d}/{len(idx)}] {p.name}", file=sys.stderr)

    rgb = [Image.open(p).convert("RGB") for p in paths]
    master = rgb[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    imgs = [im.quantize(palette=master, dither=Image.Dither.NONE) for im in rgb]
    per = [int(1000 / a.fps)] * len(imgs)
    per[-1] = 1800
    gif = C.FIGS / f"{a.out}.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=per,
                 loop=0, optimize=False, disposal=2)
    print(f"{len(imgs)} frames -> {gif} ({gif.stat().st_size / 1e6:.1f} MB)",
          file=sys.stderr)
    if mp4 := build_mp4(paths, C.FIGS / f"{a.out}.mp4", a.fps):
        print(f"            -> {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
