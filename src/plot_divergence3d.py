"""3D divergence of the wind field over the Visayas, animated.

Six horizontal planes, one per pressure level, each coloured by the spherical
divergence at that level and stacked with vertical exaggeration. Deliberately
planes rather than isosurfaces or a volume: with six levels and gaps up to
1.65 km there is no vertical structure to render, and a volume would invent it.

Red = divergence (air spreading out, sinking above), blue = convergence
(air piling in, rising above).

    python src/divergence.py                  # once
    python src/plot_divergence3d.py --start "2026-09-19 00:00" --out div_epi
    python src/plot_divergence3d.py --start "2026-09-15 00:00" --out div_bg
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from make_gif import build_mp4
from plot_balloons3d import coast_segments

DEG_M = 111_000.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=None, help="first frame, UTC")
    ap.add_argument("--hours", type=int, default=36)
    ap.add_argument("--every", type=int, default=1, help="hours per frame")
    ap.add_argument("--exag", type=float, default=42.0)
    ap.add_argument("--fps", type=float, default=6)
    ap.add_argument("--spin", type=float, default=50.0)
    ap.add_argument("--out", default="divergence3d")
    a = ap.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    d = np.load(C.DATA / "divergence.npz", allow_pickle=True)
    t = pd.to_datetime(d["time"])
    lat, lon, z = d["lat"], d["lon"], d["z"]
    div = d["div"] * 1e5                      # 1e-5 /s, a readable unit
    box = C.WIND3D_BOX
    zs = a.exag / DEG_M

    i0 = (int(np.argmin(abs(t - pd.Timestamp(a.start)))) if a.start
          else max(0, t.size - a.hours - 1))
    idx = list(range(i0, min(i0 + a.hours, t.size), a.every))
    t0 = t[i0] + pd.Timedelta(hours=C.TZ_OFFSET_H)
    print(f"from {t0:%d %b %H:%M} PHT, {len(idx)} frames", file=sys.stderr)

    # the outermost ring carries one-sided differences; drop it from the render
    lai, loi = slice(1, -1), slice(1, -1)
    glo, gla = np.meshgrid(lon[loi], lat[lai], indexing="ij")
    lim = float(np.nanpercentile(abs(div[:, :, lai, loi]), 98))
    coasts = coast_segments(box)

    frames_dir = C.FIGS / "frames" / a.out
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    paths = []
    for n, k in enumerate(idx):
        tt = t[k] + pd.Timedelta(hours=C.TZ_OFFSET_H)
        p = pv.Plotter(off_screen=True, window_size=(1280, 960))
        p.set_background("white")

        for xy in coasts:
            p.add_mesh(pv.lines_from_points(
                np.column_stack([xy[:, 0], xy[:, 1],
                                 np.full(len(xy), -0.06)])),
                color="#44403c", line_width=3, show_scalar_bar=False)

        for li, zz in enumerate(z):
            plane = pv.StructuredGrid(glo, gla, np.full_like(glo, zz * zs))
            plane["divergence"] = div[k, li, lai, loi].T.ravel(order="F")
            p.add_mesh(plane, scalars="divergence", cmap="RdBu_r",
                       clim=(-lim, lim), opacity=0.62, lighting=False,
                       show_scalar_bar=(li == 0),
                       scalar_bar_args=dict(
                           title="divergence (1e-5 /s)", vertical=True,
                           position_x=0.90, position_y=0.26, height=0.46,
                           width=0.035, title_font_size=13, label_font_size=11,
                           color="#44403c"))
            p.add_mesh(pv.Sphere(radius=0.035,
                                 center=(C.CEBU["lon"], C.CEBU["lat"], zz * zs)),
                       color="#111827", lighting=False)

        p.add_text(f"{tt:%a %d %b %H:%M} PHT", position=(24, 912),
                   font_size=15, color="#1c1917")
        p.add_text(f"spherical 3D divergence · {len(z)} pressure levels · "
                   f"red = spreading out, blue = piling in · "
                   f"vertical exaggeration x{a.exag:.0f}",
                   position=(24, 886), font_size=9, color="#78716c")
        p.add_text(C.WATERMARK, position=(24, 18), font_size=8, color="#a8a29e")

        fx, fy = (box["west"] + box["east"]) / 2, (box["south"] + box["north"]) / 2
        fz = z.max() * zs * 0.5
        th = np.deg2rad(-118 + a.spin * n / max(1, len(idx) - 1))
        ph, rad = np.deg2rad(34.0), 10.2
        p.camera_position = [
            (fx + rad * np.cos(ph) * np.cos(th), fy + rad * np.cos(ph) * np.sin(th),
             fz + rad * np.sin(ph)), (fx, fy, fz), (0, 0, 1)]
        out = frames_dir / f"{a.out}_{n:04d}.png"
        p.screenshot(str(out)); p.close()
        paths.append(out)
        print(f"  [{n + 1:3d}/{len(idx)}] {out.name}", file=sys.stderr, flush=True)

    rgb = [Image.open(q).convert("RGB") for q in paths]
    master = rgb[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    imgs = [im.quantize(palette=master, dither=Image.Dither.NONE) for im in rgb]
    per = [int(1000 / a.fps)] * len(imgs); per[-1] = 1800
    gif = C.FIGS / f"{a.out}.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=per,
                 loop=0, optimize=False, disposal=2)
    print(f"{len(imgs)} frames -> {gif} ({gif.stat().st_size / 1e6:.1f} MB)",
          file=sys.stderr)
    if mp4 := build_mp4(paths, C.FIGS / f"{a.out}.mp4", a.fps):
        print(f"            -> {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
