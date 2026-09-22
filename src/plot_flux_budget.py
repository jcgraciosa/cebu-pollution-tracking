"""The first-order budget: what comes in the side, what leaves the top.

Horizontal transport integrated 0-4 km -- a cap chosen because it contains the
inversion at every Mactan sounding (1247-3918 m), so it never truncates the
smoke layer. Vertical exchange as a time-height section over Cebu, which needs
no level to be chosen at all, plus one map at 4 km so the two terms close:

    div( int_0^4km rho*u dz ) + [rho w]_4km - [rho w]_sfc = -d/dt int rho dz

    python src/plot_flux_budget.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))
import config as C

WIND = "/Volumes/JCG_Backup1/pollution-tracking/data/wind3d_region.npz"
EPI = pd.Timestamp("2026-09-18 10:00")
TOP = 4000.0
DPI = 200


def _style(ax):
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d3d1")


def main() -> None:
    d = np.load(WIND, allow_pickle=True)
    t = pd.to_datetime(d["time"]) + pd.Timedelta(hours=C.TZ_OFFSET_H)
    lat, lon, zf, zl = d["lat"], d["lon"], d["zfull"], d["z"]
    spd, rad = d["spd"] / 3.6, np.deg2rad(d["dir"])
    u, v, w = -spd * np.sin(rad), -spd * np.cos(rad), d["w"]
    rho = (d["pres"] * 100.0) / (287.05 * d["temp"])

    dz = np.empty_like(zf)
    dz[:, 1:-1] = 0.5 * (zf[:, 2:] - zf[:, :-2])
    dz[:, 0] = zf[:, 1] - zf[:, 0]
    dz[:, -1] = zf[:, -1] - zf[:, -2]
    wt = np.where(zf <= TOP, dz, 0.0)
    Fx, Fy = (rho * u * wt).sum(1), (rho * v * wt).sum(1)
    epi = t >= EPI

    i = int(np.argmin(abs(lat - C.CEBU["lat"])))
    j = int(np.argmin(abs(lon - C.CEBU["lon"])))
    vflux = (rho * w)[:, :, i - 2:i + 3, j - 2:j + 3].mean(axis=(2, 3)) * 1000.0  # g/m2/s
    ktop = int(np.argmin(abs(zl - TOP)))
    Vmap = (rho * w)[:, ktop] * 1000.0

    try:
        import cartopy.crs as ccrs
        PC = ccrs.PlateCarree()
    except ImportError:
        PC = None

    fig = plt.figure(figsize=(14.2, 10.6), dpi=DPI)
    mag = np.hypot(Fx, Fy)
    vmax = float(np.nanpercentile(mag, 98))

    # --- row 1: horizontal transport, before and during ----------------------
    for col, (msk, name, when) in enumerate((
            (~epi, "BEFORE", "13–18 Sep"), (epi, "DURING", "18–21 Sep"))):
        ax = fig.add_axes([0.05 + col * 0.43, 0.545, 0.39, 0.31],
                          **(dict(projection=PC) if PC else {}))
        U, V = np.nanmean(Fx[msk], 0), np.nanmean(Fy[msk], 0)
        M = np.hypot(U, V)
        pm = ax.pcolormesh(lon, lat, M, cmap="PuBuGn", vmin=0, vmax=vmax,
                           shading="gouraud", **(dict(transform=PC) if PC else {}))
        if PC:
            ax.coastlines(resolution="50m", linewidth=0.5, color="#44403c")
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=PC)
            gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#e7e5e4")
            gl.top_labels = gl.right_labels = False
            if col == 1:
                gl.left_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": C.INK_MUTED}
        ax.streamplot(lon, lat, U, V, color="#1c1917", linewidth=0.6, density=1.0,
                      arrowsize=0.75, **(dict(transform=PC) if PC else {}))
        ax.plot(C.CEBU["lon"], C.CEBU["lat"], "*", ms=13, color="#b45309",
                mec="white", mew=1.1, **(dict(transform=PC) if PC else {}))
        fig.text(0.05 + col * 0.43, 0.876,
                 f"{name} · {when} · horizontal, 0–4 km · at Cebu "
                 f"{M[i, j]:,.0f} kg m⁻¹ s⁻¹", fontsize=9.5,
                 color=C.INK_MUTED, va="bottom")
    cax = fig.add_axes([0.885, 0.565, 0.012, 0.27])
    cb = fig.colorbar(pm, cax=cax)
    cb.set_label("|∫ρu dz| (kg m$^{-1}$ s$^{-1}$)", fontsize=8, color=C.INK)
    cb.ax.tick_params(labelsize=7, colors=C.INK_MUTED)

    # --- row 2: vertical exchange, section over Cebu -------------------------
    ax = fig.add_axes([0.05, 0.315, 0.82, 0.155])
    tn = mdates.date2num(t)
    lim = float(np.nanpercentile(abs(vflux), 97))
    pmv = ax.pcolormesh(tn, zl / 1000, vflux.T, cmap="RdBu_r", vmin=-lim, vmax=lim,
                        shading="nearest")
    ax.axvline(mdates.date2num(EPI), color="#111827", lw=1.8)
    ax.axhline(TOP / 1000, color="#111827", lw=0.9, ls=":")
    ax.set_ylim(0, 6); ax.set_xlim(tn[0], tn[-1])
    ax.set_ylabel("height (km)", fontsize=9, color=C.INK)
    ax.set_xlabel("Philippine time", fontsize=9, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    cbv = fig.colorbar(pmv, ax=ax, pad=0.012, fraction=0.028)
    cbv.set_label("ρw (g m$^{-2}$ s$^{-1}$) · red = rising", fontsize=8, color=C.INK)
    cbv.ax.tick_params(labelsize=7, colors=C.INK_MUTED)
    _style(ax)
    lo = zl < 1200
    b = np.nanmean(vflux[~epi][:, lo]); e = np.nanmean(vflux[epi][:, lo])
    fig.text(0.05, 0.487, f"VERTICAL exchange over Cebu (55 km box) · below 1.2 km "
             f"the mean flux reverses, {b:+.2f} → {e:+.2f} g m⁻² s⁻¹",
             fontsize=9.5, color=C.INK_MUTED, va="bottom")

    # --- row 3: the 4 km lid, spatially -------------------------------------
    for col, (msk, name) in enumerate(((~epi, "BEFORE"), (epi, "DURING"))):
        ax = fig.add_axes([0.05 + col * 0.43, 0.075, 0.39, 0.17],
                          **(dict(projection=PC) if PC else {}))
        Vm = np.nanmean(Vmap[msk], 0)
        vl = float(np.nanpercentile(abs(Vmap), 98))
        pm2 = ax.pcolormesh(lon, lat, Vm, cmap="RdBu_r", vmin=-vl, vmax=vl,
                            shading="gouraud", **(dict(transform=PC) if PC else {}))
        if PC:
            ax.coastlines(resolution="50m", linewidth=0.5, color="#44403c")
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=PC)
        ax.plot(C.CEBU["lon"], C.CEBU["lat"], "*", ms=13, color="#b45309",
                mec="white", mew=1.1, **(dict(transform=PC) if PC else {}))
        fig.text(0.05 + col * 0.43, 0.252,
                 f"{name} · vertical flux through the 4 km lid", fontsize=9.5,
                 color=C.INK_MUTED, va="bottom")
    cax2 = fig.add_axes([0.885, 0.085, 0.012, 0.15])
    cb2 = fig.colorbar(pm2, cax=cax2)
    cb2.set_label("ρw at 4 km (g m$^{-2}$ s$^{-1}$)", fontsize=8, color=C.INK)
    cb2.ax.tick_params(labelsize=7, colors=C.INK_MUTED)

    fig.text(0.05, 0.968, "The first-order budget: in the side, out the top",
             fontsize=14, color=C.INK, weight="bold", va="top")
    fig.text(0.05, 0.935,
             f"NOAA GFS 0.25° · 0–4 km contains the inversion at every Mactan "
             f"sounding (1247–3918 m) · no tracer, no optical assumption · "
             f"{C.WATERMARK.replace('Made by: ', '')}",
             fontsize=8.5, color=C.INK_MUTED, va="top")
    notes = ["Horizontal and vertical terms are linked by continuity: div(int rho*u dz) + [rho w]_top - [rho w]_sfc = -d/dt int rho dz.",
             "The section is the honest form for the vertical term: the signal changes sign with height, so any single level would hide it.",
             "One episode, autocorrelated in time. The reversal is suggestive, not significant."]
    for n, line in enumerate(notes + C.attribution(coastlines=False, trajectory=False,
                                                   cams=False, gfs=True)):
        fig.text(0.05, 0.046 - n * 0.0098, line, fontsize=6.3,
                 color=C.INK_MUTED, va="top")

    out = C.FIGS / "flux_budget.png"
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"wrote {out}")
    print(f"  horizontal at Cebu: {np.hypot(Fx,Fy)[~epi,i,j].mean():,.0f} -> "
          f"{np.hypot(Fx,Fy)[epi,i,j].mean():,.0f} kg/m/s")
    print(f"  vertical below 1.2 km: {b:+.3f} -> {e:+.3f} g/m2/s")


if __name__ == "__main__":
    main()
