"""The conveyor: integrated smoke transport, background against episode.

Magnitude shaded, direction as streamlines, fires overlaid. This is the figure
the whole investigation has implied and never drawn -- transport as a flux
rather than as a property of the flow.

    python src/smoke_flux.py && python src/plot_smoke_flux.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config as C

EPI = pd.Timestamp("2026-09-18 10:00")
DPI = 200


def air_mass_flux(layer=3000.0):
    """Integrated air mass flux: the conveyor with no tracer and no MEE."""
    d = np.load("/Volumes/JCG_Backup1/pollution-tracking/data/wind3d_region.npz",
                allow_pickle=True)
    zf = d["zfull"]
    spd, rad = d["spd"] / 3.6, np.deg2rad(d["dir"])
    u, v = -spd * np.sin(rad), -spd * np.cos(rad)
    rho = (d["pres"] * 100.0) / (287.05 * d["temp"])
    dz = np.empty_like(zf)
    dz[:, 1:-1] = 0.5 * (zf[:, 2:] - zf[:, :-2])
    dz[:, 0] = zf[:, 1] - zf[:, 0]; dz[:, -1] = zf[:, -1] - zf[:, -2]
    wt = np.where(zf <= layer, dz, 0.0)
    return (pd.to_datetime(d["time"]), d["lat"], d["lon"],
            (rho * u * wt).sum(1), (rho * v * wt).sum(1))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="smoke", choices=["smoke", "air"])
    a = ap.parse_args()
    d = np.load(C.DATA / "smoke_flux.npz", allow_pickle=True)
    t = pd.to_datetime(d["time"]) + pd.Timedelta(hours=C.TZ_OFFSET_H)
    lat, lon = d["lat"], d["lon"]
    if a.field == "air":
        tt, lat, lon, fx, fy = air_mass_flux()
        t = tt + pd.Timedelta(hours=C.TZ_OFFSET_H)
        unit, cmap, title = "kg m$^{-1}$ s$^{-1}$", "PuBuGn", "air mass"
        fmt = "{:.0f}"
    else:
        fx, fy = d["fx"], d["fy"]
        unit, cmap, title = "g m$^{-1}$ s$^{-1}$", "YlOrBr", "smoke"
        fmt = "{:.2f}"
    epi = t >= EPI

    fires = pd.read_csv(C.FIRES_CSV, parse_dates=["when"])
    fires["tp"] = fires.when.dt.tz_localize(None) + pd.Timedelta(hours=C.TZ_OFFSET_H)

    try:
        import cartopy.crs as ccrs
        PC = ccrs.PlateCarree()
    except ImportError:
        PC = None

    fig = plt.figure(figsize=(14.0, 7.4), dpi=DPI)
    mag_all = np.hypot(fx, fy)
    vmax = float(np.nanpercentile(mag_all, 98))

    for col, (msk, name, when) in enumerate((
            (~epi, "BACKGROUND", "13–18 Sep"), (epi, "EPISODE", "18–21 Sep"))):
        ax = fig.add_axes([0.045 + col * 0.46, 0.10, 0.42, 0.76],
                          **(dict(projection=PC) if PC else {}))
        U, V = np.nanmean(fx[msk], axis=0), np.nanmean(fy[msk], axis=0)
        M = np.hypot(U, V)
        pm = ax.pcolormesh(lon, lat, M, cmap=cmap, vmin=0, vmax=vmax,
                           shading="gouraud", **(dict(transform=PC) if PC else {}))
        if PC:
            ax.coastlines(resolution="50m", linewidth=0.6, color="#44403c")
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=PC)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#e7e5e4")
            gl.top_labels = gl.right_labels = False
            if col == 1:
                gl.left_labels = False
            gl.xlabel_style = gl.ylabel_style = {"size": 7.5, "color": C.INK_MUTED}
        ax.streamplot(lon, lat, U, V, color="#1c1917", linewidth=0.7,
                      density=1.15, arrowsize=0.8,
                      **(dict(transform=PC) if PC else {}))
        f = fires[(fires.tp >= t[msk].min()) & (fires.tp <= t[msk].max())]
        ax.scatter(f.longitude, f.latitude, s=0.7, c=C.FIRE, alpha=0.30,
                   linewidths=0, **(dict(transform=PC) if PC else {}))
        ax.plot(C.CEBU["lon"], C.CEBU["lat"], marker="*", ms=15,
                color="#0e7490", mec="white", mew=1.2,
                **(dict(transform=PC) if PC else {}))
        i = int(np.argmin(abs(lat - C.CEBU["lat"])))
        j = int(np.argmin(abs(lon - C.CEBU["lon"])))
        ax.set_title(f"{name} · {when}  ·  at Cebu {fmt.format(M[i, j])} {unit}"
                     .replace("$^{-1}$", "⁻¹").replace("$^{-2}$", "⁻²"),
                     fontsize=10, color=C.INK_MUTED, loc="left", pad=6)

    cax = fig.add_axes([0.915, 0.13, 0.014, 0.68])
    cb = fig.colorbar(pm, cax=cax)
    cb.set_label(f"integrated {title} transport |F| ({unit})",
                 fontsize=8.5, color=C.INK)
    cb.ax.tick_params(labelsize=7.5, colors=C.INK_MUTED)

    fig.text(0.045, 0.965, f"The conveyor: integrated {title} transport",
             fontsize=14, color=C.INK, weight="bold", va="top")
    fig.text(0.045, 0.928,
             (f"integral of rho*u over 0–3 km from GFS pressure and temperature · "
              f"no tracer, no optical assumption · "
              if a.field == "air" else
              f"column mass from CAMS AOD ÷ {float(d['mee']):.0f} m² g⁻¹, carried by "
              f"the GFS 0–3 km mean wind · ")
             + f"streamlines show direction, red dots are VIIRS fires · "
             f"{C.WATERMARK.replace('Made by: ', '')}",
             fontsize=8.5, color=C.INK_MUTED, va="top")
    notes = ["Transport is a flux, c*v, not a property of the wind alone: divergence measures how the flow deforms and says nothing about how much smoke moves.",
             "Mass extinction efficiency carries most of the absolute uncertainty (literature 3.5-5 m2/g for smoke). Read the pattern and the ratio, not the absolute value.",
             "AOD is a column integral paired with a 0-3 km mean wind, so this assumes the smoke travels with that layer."]
    for n, line in enumerate(notes + C.attribution(coastlines=False, trajectory=False,
                                                   cams=True, gfs=True)):
        fig.text(0.045, 0.072 - n * 0.0118, line, fontsize=6.3,
                 color=C.INK_MUTED, va="top")

    out = C.FIGS / (f"{a.field}_flux.png")
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
