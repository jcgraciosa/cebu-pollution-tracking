"""PyVista version of the balloon experiment, over real Philippine terrain.

Parcels advect on the resolved IFS 3D wind above a DEM of the Visayas. Seeding
the whole column and raising the ceiling makes the vertical field legible: in a
rising regime parcels climb and leave through the top, in a subsiding one they
sink back toward the islands.

    python src/download.py --parts wind3d              # once
    python src/plot_balloons_pv.py --start "2026-09-19 00:00" --out pv_epi
    python src/plot_balloons_pv.py --start "2026-09-15 00:00" --out pv_bg

The 2.5 km plane is drawn for reference -- it is the lid the residence numbers
use -- but parcels pass through it and keep going, so ascent and descent stay
visible instead of being truncated.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from residence import load, _sample, DT
from make_gif import build_mp4

R_EARTH = 6_371_000.0
DEG_M = 111_000.0          # metres per degree, for the vertical exaggeration


def advect_free(ti, seeds, times, lat, lon, zlev, u, v, w, box, ceiling, hours):
    """Advect in 3D until a parcel leaves the display volume. Keeps every step."""
    la, lo, zz = (s.copy() for s in seeds)
    alive = np.ones(la.size, bool)
    t = float(ti)
    steps = int(hours * 3600 / DT)
    hist = np.empty((steps + 1, 3, la.size), "float32")
    live = np.empty((steps + 1, la.size), bool)
    hist[0], live[0] = np.array([la, lo, zz]), alive
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
            zz[a] = np.maximum(zz[a] + ww * DT, 20.0)      # the ground reflects
            t += DT / 3600.0
            alive &= ((la >= box["south"]) & (la <= box["north"]) &
                      (lo >= box["west"]) & (lo <= box["east"]) & (zz <= ceiling))
        hist[s + 1], live[s + 1] = np.array([la, lo, zz]), alive
    return hist, live


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=None)
    ap.add_argument("--hours", type=int, default=36)
    ap.add_argument("--ceiling", type=float, default=6000.0, help="display top, m")
    ap.add_argument("--seed-step", type=float, default=0.6)
    ap.add_argument("--seed-z", default="300,1200,2400,3600,4800",
                    help="release heights in m, comma separated")
    ap.add_argument("--exag", type=float, default=22.0, help="vertical exaggeration")
    ap.add_argument("--fps", type=float, default=6)
    ap.add_argument("--spin", type=float, default=50.0)
    ap.add_argument("--out", default="balloons_pv")
    a = ap.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    times, lat, lon, zlev, u, v, w = load()
    box = C.WIND3D_BOX
    zs = a.exag / DEG_M                       # metres -> plot units (degrees)
    ti = (int(np.argmin(np.abs(times - pd.Timestamp(a.start)))) if a.start
          else max(0, times.size - a.hours - 1))
    t0 = times[ti] + pd.Timedelta(hours=C.TZ_OFFSET_H)
    print(f"release {t0:%d %b %H:%M} PHT", file=sys.stderr)

    seed_z = [float(x) for x in a.seed_z.split(",")]
    sla = np.arange(box["south"] + a.seed_step, box["north"], a.seed_step)
    slo = np.arange(box["west"] + a.seed_step, box["east"], a.seed_step)
    g = np.meshgrid(sla, slo, seed_z, indexing="ij")
    seeds = tuple(x.ravel() for x in g)
    hist, live = advect_free(ti, seeds, times, lat, lon, zlev, u, v, w,
                             box, a.ceiling, a.hours)
    print(f"{seeds[0].size} parcels, {len(seed_z)} release heights", file=sys.stderr)

    # --- coastlines, flat --------------------------------------------------
    # Deliberately NOT a DEM. The driving wind is on pressure levels whose
    # lowest is 93 m and whose grid is 28 km, so it has no knowledge of Cebu's
    # 1000 m spine or Kanlaon at 2435 m. Parcels fly straight through where a
    # mountain would be. Drawing one would claim physics the model does not have.
    from plot_balloons3d import coast_segments
    coasts = [np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])
              for xy in coast_segments(box)]
    sea = pv.Plane(center=((box["west"] + box["east"]) / 2,
                           (box["south"] + box["north"]) / 2, -0.002),
                   direction=(0, 0, 1),
                   i_size=box["east"] - box["west"],
                   j_size=box["north"] - box["south"])

    # --- wind glyphs ---------------------------------------------------------
    keep = zlev <= a.ceiling
    zk = zlev[keep]
    ji = np.arange(1, lat.size, 4)
    jj = np.arange(1, lon.size, 4)
    gx, gy, gz = np.meshgrid(lon[jj], lat[ji], zk * zs, indexing="ij")
    wind_pts = pv.StructuredGrid(gx, gy, gz)

    frames_dir = C.FIGS / "frames" / a.out
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()

    every = max(1, int(3600 / DT))
    idx = list(range(0, hist.shape[0], every))
    paths = []
    for n, s in enumerate(idx):
        hr = s * DT / 3600.0
        tt = t0 + pd.Timedelta(hours=hr)
        k = min(ti + int(hr), u.shape[0] - 1)

        p = pv.Plotter(off_screen=True, window_size=(1280, 960))
        p.set_background("white")
        p.add_mesh(sea, color="#e0f2fe", lighting=False, show_scalar_bar=False)
        for xy in coasts:
            p.add_mesh(pv.lines_from_points(xy), color="#57534e", line_width=2,
                       show_scalar_bar=False)

        uu = u[k][np.ix_(keep, ji, jj)].transpose(2, 1, 0)
        vv = v[k][np.ix_(keep, ji, jj)].transpose(2, 1, 0)
        ww = w[k][np.ix_(keep, ji, jj)].transpose(2, 1, 0)
        vec = np.stack([uu.ravel(order="F"), vv.ravel(order="F"),
                        ww.ravel(order="F") * a.exag], axis=1)
        wind_pts["vectors"] = vec * 0.055
        p.add_mesh(wind_pts.glyph(orient="vectors", scale="vectors", factor=1.4,
                                  geom=pv.Arrow(tip_length=0.3, tip_radius=0.12,
                                                shaft_radius=0.03)),
                   color="#44403c", opacity=0.6, lighting=False,
                   show_scalar_bar=False)

        m = live[s]
        pts = np.column_stack([hist[s, 1, m], hist[s, 0, m], hist[s, 2, m] * zs])
        cloud = pv.PolyData(pts)
        cloud["height"] = hist[s, 2, m]
        p.add_mesh(cloud, scalars="height", cmap="turbo", clim=(0, a.ceiling),
                   render_points_as_spheres=True, point_size=11,
                   scalar_bar_args=dict(title="parcel height (m)", vertical=True,
                                        position_x=0.90, position_y=0.28,
                                        height=0.42, width=0.035,
                                        title_font_size=13, label_font_size=11,
                                        color="#44403c"))

        lid = pv.Plane(center=((box["west"] + box["east"]) / 2,
                               (box["south"] + box["north"]) / 2,
                               C.WIND3D_LID * zs), direction=(0, 0, 1),
                       i_size=box["east"] - box["west"],
                       j_size=box["north"] - box["south"])
        p.add_mesh(lid, color="#7c3aed", opacity=0.13, lighting=False,
                   show_scalar_bar=False)
        p.add_mesh(pv.Sphere(radius=0.055, center=(C.CEBU["lon"], C.CEBU["lat"],
                                                   0.02)), color="#1e293b")

        p.add_text(f"{tt:%a %d %b %H:%M} PHT", position=(24, 912), font_size=15,
                   color="#1c1917")
        p.add_text(f"released {t0:%d %b %H:%M} · +{hr:.0f} h · "
                   f"{int(m.sum())} of {m.size} parcels still in view · "
                   f"lid 2.5 km (violet) · vertical exaggeration x{a.exag:.0f}",
                   position=(24, 886), font_size=9, color="#78716c")
        p.add_text(C.WATERMARK, position=(24, 18), font_size=8, color="#a8a29e")

        # explicit orbit: a preset plus azimuth drifts unpredictably between
        # VTK versions, and every frame must be reproducible
        fx, fy = (box["west"] + box["east"]) / 2, (box["south"] + box["north"]) / 2
        fz = a.ceiling * zs * 0.45
        th = np.deg2rad(-118 + a.spin * n / max(1, len(idx) - 1))
        ph = np.deg2rad(27.0)
        rad = 8.6
        p.camera_position = [
            (fx + rad * np.cos(ph) * np.cos(th), fy + rad * np.cos(ph) * np.sin(th),
             fz + rad * np.sin(ph)),
            (fx, fy, fz), (0, 0, 1)]
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
        print(f"            -> {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
