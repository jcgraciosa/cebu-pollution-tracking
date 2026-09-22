"""Animated flux budget: horizontal and vertical transport against the air.

One figure per 3-hourly analysis -- horizontal mass transport through 0-4 km,
vertical exchange through the 4 km lid, and underneath, what Cebu was actually
breathing at that moment. The cursor ties the maps to the concentration, so the
two are read together rather than in sequence.

    python src/plot_flux_anim.py --at "2026-09-19 00:00" --n 1   # preview
    python src/plot_flux_anim.py                                 # all 72 frames
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from compare_ground import prepared
from plot_forecast import simulate
from make_gif import build_mp4
from scipy.ndimage import uniform_filter


def smooth(a, n=5):
    """n x n boxcar, NaN-aware. ~140 km, the scale at which 0.25 deg
    derivatives of the wind carry meaning."""
    m = np.isfinite(a)
    num = uniform_filter(np.where(m, a, 0.0), n, mode="nearest")
    den = uniform_filter(m.astype(float), n, mode="nearest")
    return np.where(m, num / np.maximum(den, 1e-6), np.nan)

WIND = "/Volumes/JCG_Backup1/pollution-tracking/data/wind3d_region.npz"
EPI = pd.Timestamp("2026-09-18 10:00")
TOP = 4000.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=1, help="use every Nth analysis")
    ap.add_argument("--at", default=None, help="start at this UTC time")
    ap.add_argument("--n", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--mode", default="budget", choices=["budget", "vertical"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or ("flux_anim" if a.mode == "budget" else "vflux_anim")

    d = np.load(WIND, allow_pickle=True)
    tutc = pd.to_datetime(d["time"])
    t = tutc + pd.Timedelta(hours=C.TZ_OFFSET_H)
    lat, lon, zf, zl = d["lat"], d["lon"], d["zfull"], d["z"]
    spd, rad = d["spd"] / 3.6, np.deg2rad(d["dir"])
    u, v, w = -spd * np.sin(rad), -spd * np.cos(rad), d["w"]
    rho = (d["pres"] * 100.0) / (287.05 * d["temp"])
    dz = np.empty_like(zf)
    dz[:, 1:-1] = 0.5 * (zf[:, 2:] - zf[:, :-2])
    dz[:, 0] = zf[:, 1] - zf[:, 0]; dz[:, -1] = zf[:, -1] - zf[:, -2]
    wt = np.where(zf <= TOP, dz, 0.0)
    Fx, Fy = (rho * u * wt).sum(1), (rho * v * wt).sum(1)
    ktop = int(np.argmin(abs(zl - TOP)))
    klow = int(np.argmin(abs(zl - 1000.0)))
    ulow, vlow = u[:, klow], v[:, klow]          # flow at the level each panel shows
    utop, vtop = u[:, ktop], v[:, ktop]
    mf = rho * w * 1000.0                       # g m-2 s-1, + = up
    W = np.stack([smooth(x) for x in mf[:, ktop]])
    # net through-layer: what actually leaves 0-4 km, the continuity term
    NET = np.stack([smooth(x) for x in (mf[:, ktop] - mf[:, 0])])
    LOW = np.stack([smooth(x) for x in mf[:, klow]])
    i = int(np.argmin(abs(lat - C.CEBU["lat"])))
    j = int(np.argmin(abs(lon - C.CEBU["lon"])))

    m, sp = prepared("pm25")
    site = pd.read_csv(C.DATA / "cebu_timeseries.csv", parse_dates=["time"])
    f = site[(site.time >= tutc.min()) & (site.time <= tutc.max())].copy()
    f["tl"] = f.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    f["hr"] = f.tl.dt.hour
    S, _, _ = simulate(f, "pm25", np.random.default_rng(0))
    p25, p50, p75 = (np.percentile(S, q, 1) for q in (25, 50, 75))
    st = m[(m.t_pht >= f.tl.min()) & (m.t_pht <= f.tl.max())]

    vmax = float(np.nanpercentile(np.hypot(Fx, Fy), 98))
    wlim = float(np.nanpercentile(abs(W), 98))
    nlim = float(np.nanpercentile(abs(NET), 98))
    llim = float(np.nanpercentile(abs(LOW), 98))
    try:
        import cartopy.crs as ccrs
        PC = ccrs.PlateCarree()
    except ImportError:
        PC = None

    frames_dir = C.FIGS / "frames" / a.out
    frames_dir.mkdir(parents=True, exist_ok=True)
    k0 = int(np.argmin(abs(tutc - pd.Timestamp(a.at)))) if a.at else 0
    idx = list(range(k0, len(t), a.every))
    if a.n:
        idx = idx[:a.n]
    else:
        for old in frames_dir.glob("*.png"):
            old.unlink()

    paths = []
    for n, k in enumerate(idx):
        fig = plt.figure(figsize=(12.6, 8.6), dpi=C.FIG_DPI)
        for col in (0, 1):
            ax = fig.add_axes([0.045 + col * 0.475, 0.355, 0.40, 0.45],
                              **(dict(projection=PC) if PC else {}))
            if a.mode == "vertical":
                fld, lim2 = (NET[k], nlim) if col == 0 else (LOW[k], llim)
                pm = ax.pcolormesh(lon, lat, fld, cmap="RdBu_r", vmin=-lim2,
                                   vmax=lim2, shading="gouraud",
                                   **(dict(transform=PC) if PC else {}))
                name = ("NET through 0–4 km · [ρw]₄ₖₘ − [ρw]ₛfc" if col == 0
                        else "EXCHANGE at 1 km")
                val = f"{fld[i, j]:+.1f} g m⁻² s⁻¹"
                lab = ("net ρw (g m$^{-2}$s$^{-1}$) · red = venting" if col == 0
                       else "ρw at 1 km (g m$^{-2}$s$^{-1}$) · red = rising")
                # each panel gets the flow at ITS OWN level, so the streamlines
                # explain that panel's vertical motion rather than repeating
                sx, sy = ((Fx[k], Fy[k]) if col == 0 else (ulow[k], vlow[k]))
                ax.streamplot(lon, lat, sx, sy, color="#3f3f46", linewidth=0.5,
                              density=0.85, arrowsize=0.65,
                              **(dict(transform=PC) if PC else {}))
            elif col == 0:
                M = np.hypot(Fx[k], Fy[k])
                pm = ax.pcolormesh(lon, lat, M, cmap="PuBuGn", vmin=0, vmax=vmax,
                                   shading="gouraud",
                                   **(dict(transform=PC) if PC else {}))
                ax.streamplot(lon, lat, Fx[k], Fy[k], color="#1c1917",
                              linewidth=0.55, density=0.9, arrowsize=0.7,
                              **(dict(transform=PC) if PC else {}))
                name, val = "HORIZONTAL · 0–4 km", f"{M[i, j]:,.0f} kg m⁻¹ s⁻¹"
                lab = "|∫ρu dz| (kg m$^{-1}$s$^{-1}$)"
            else:
                pm = ax.pcolormesh(lon, lat, W[k], cmap="RdBu_r", vmin=-wlim,
                                   vmax=wlim, shading="gouraud",
                                   **(dict(transform=PC) if PC else {}))
                name, val = "VERTICAL · through 4 km", f"{W[k][i, j]:+.1f} g m⁻² s⁻¹"
                lab = "ρw (g m$^{-2}$s$^{-1}$) · red = rising"
                ax.streamplot(lon, lat, utop[k], vtop[k], color="#3f3f46",
                              linewidth=0.5, density=0.85, arrowsize=0.65,
                              **(dict(transform=PC) if PC else {}))
            if PC:
                ax.coastlines(resolution="50m", linewidth=0.5, color="#44403c")
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=PC)
                gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#e7e5e4")
                gl.top_labels = gl.right_labels = False
                if col == 1:
                    gl.left_labels = False
                gl.xlabel_style = gl.ylabel_style = {"size": 6.5,
                                                     "color": C.INK_MUTED}
            ax.plot(C.CEBU["lon"], C.CEBU["lat"], "*", ms=12, color="#b45309",
                    mec="white", mew=1.0, **(dict(transform=PC) if PC else {}))
            fig.text(0.045 + col * 0.475, 0.815, f"{name} · at Cebu {val}",
                     fontsize=9, color=C.INK_MUTED, va="bottom")
            cax = fig.add_axes([0.045 + col * 0.475 + 0.408, 0.375, 0.009, 0.40])
            cb = fig.colorbar(pm, cax=cax)
            cb.set_label(lab, fontsize=7.5, color=C.INK)
            cb.ax.tick_params(labelsize=6.5, colors=C.INK_MUTED)

        ax = fig.add_axes([0.045, 0.105, 0.855, 0.175])
        ax.fill_between(f.tl, p25, p75, color="#047857", alpha=0.25, lw=0)
        ax.plot(f.tl, p50, lw=1.6, color="#047857", label="corrected CAMS · median")
        ax.plot(st.t_pht, st[sp["obs"]], "o", ms=2.6, color="#1c1917",
                mec="white", mew=0.4, ls="none", label="station (EMB Central Visayas)")
        ax.axvspan(EPI, f.tl.max(), color="#fca5a5", alpha=0.12, lw=0, zorder=0)
        ax.axvline(t[k], color=C.FIRE, lw=2.0, zorder=6)
        ax.set_xlim(f.tl.min(), f.tl.max()); ax.set_ylim(0, None)
        ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)", fontsize=8.5, color=C.INK)
        ax.set_xlabel("Philippine time", fontsize=8.5, color=C.INK)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(labelsize=7.5, colors=C.INK_MUTED)
        ax.grid(alpha=0.18, lw=0.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9, facecolor="white",
                  edgecolor="#e7e5e4", ncol=2)

        head = ("Flux budget" if a.mode == "budget"
                else "Vertical exchange")
        fig.text(0.045, 0.968, f"{head} · {t[k]:%a %d %b %H:%M} PHT",
                 fontsize=13, color=C.INK, weight="bold", va="top")
        fig.text(0.045, 0.936,
                 "NOAA GFS 0.25° · 0–4 km contains the inversion at every Mactan "
                 "sounding · 5x5 boxcar (~140 km) · no tracer · "
                 + C.WATERMARK.replace("Made by: ", ""),
                 fontsize=8, color=C.INK_MUTED, va="top")
        fig.text(0.045, 0.048, "Modelled fields, not measurements · NOAA NCEP GFS "
                 "0.25 deg (public domain) · Contains modified Copernicus "
                 "Atmosphere Monitoring Service information 2026",
                 fontsize=6.3, color=C.INK_MUTED)
        p = frames_dir / f"{a.out}_{(k0 + n * a.every):04d}.png"
        fig.savefig(p, facecolor=C.SURFACE); plt.close(fig)
        paths.append(p)
        if n % 10 == 0:
            print(f"  [{n+1:3d}/{len(idx)}] {p.name}", file=sys.stderr, flush=True)

    if a.n:
        print(f"preview only: {paths[-1]}")
        return
    rgb = [Image.open(q).convert("RGB") for q in paths]
    master = rgb[0].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    imgs = [im.quantize(palette=master, dither=Image.Dither.NONE) for im in rgb]
    per = [int(1000 / a.fps)] * len(imgs); per[-1] = 1800
    gif = C.FIGS / f"{a.out}.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=per,
                 loop=0, optimize=False, disposal=2)
    print(f"{len(imgs)} frames -> {gif} ({gif.stat().st_size/1e6:.1f} MB)",
          file=sys.stderr)
    if mp4 := build_mp4(paths, C.FIGS / f"{a.out}.mp4", a.fps):
        print(f"            -> {mp4} ({mp4.stat().st_size/1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
