"""The wind column over Cebu: vertical, horizontal magnitude, direction.

Three time-height sections on one set of axes, an 83 km mean centred on the
grid cell nearest Cebu City. Each carries one quantity, so nothing has to be
decoded:

    w          signed, diverging    up or down
    |u,v|      sequential           how fast
    direction  CYCLIC               where from

Direction needs a cyclic colour map. On a linear map 359 deg and 1 deg sit at
opposite ends of the scale though they are one degree apart, which paints a
false discontinuity straight through northerly flow.

    python src/plot_wind_column.py
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
from compare_ground import prepared
from plot_forecast import simulate

EPI = pd.Timestamp("2026-09-18 10:00")
DPI = 200


def _style(ax):
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d3d1")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="wind3d.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--zmax", type=float, default=None, help="top of the sections, m")
    a = ap.parse_args()
    d = np.load(C.DATA / a.src, allow_pickle=True)
    zl = (np.nanmean(d["z"], axis=(0, 2, 3)) if d["z"].ndim == 4
          else np.asarray(d["z"], float)); o = np.argsort(zl)
    zl = zl[o]
    w, spd = d["w"][:, o], d["spd"][:, o] / 3.6
    drc = d["dir"][:, o]
    rad = np.deg2rad(drc)
    uh, vh = -spd * np.sin(rad), -spd * np.cos(rad)
    t = pd.to_datetime(d["time"]) + pd.Timedelta(hours=C.TZ_OFFSET_H)
    keep = np.isfinite(w).all(axis=(1, 2, 3))

    i = int(np.argmin(abs(d["lat"] - C.CEBU["lat"])))
    k = int(np.argmin(abs(d["lon"] - C.CEBU["lon"])))
    sl = (slice(None), slice(None), slice(i - 1, i + 2), slice(k - 1, k + 2))
    t = t[keep]
    wc = np.nanmean(w[sl], axis=(2, 3))[keep] * 100                  # cm/s
    uc = np.nanmean(uh[sl], axis=(2, 3))[keep]
    vc = np.nanmean(vh[sl], axis=(2, 3))[keep]
    mag = np.hypot(uc, vc)
    # meteorological convention: the direction the wind blows FROM
    az = (np.degrees(np.arctan2(-uc, -vc)) + 360) % 360
    if a.zmax:                                  # 40 GFS levels reach 14.7 km
        keepz = zl <= a.zmax
        zl = zl[keepz]
        wc, uc, vc = wc[:, keepz], uc[:, keepz], vc[:, keepz]
        mag, az = mag[:, keepz], az[:, keepz]
    epi = t >= EPI
    lo3 = zl <= 3000
    tn = mdates.date2num(t)

    # corrected CAMS with its predictive band, on the same axis, so the wind
    # panels can be read against what the air actually did
    m, sperr = prepared("pm25")
    site = pd.read_csv(C.DATA / "cebu_timeseries.csv", parse_dates=["time"])
    f = site[site.time.isin(pd.to_datetime(t - pd.Timedelta(hours=C.TZ_OFFSET_H)))].copy()
    f["tl"] = f.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    f["hr"] = f.tl.dt.hour
    S, _, _ = simulate(f, "pm25", np.random.default_rng(0))
    p25, p50, p75 = (np.percentile(S, q, 1) for q in (25, 50, 75))

    fig = plt.figure(figsize=(13.0, 14.4), dpi=DPI)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.8],
                          width_ratios=[1, 0.018], hspace=0.36, wspace=0.012,
                          left=0.075, right=0.925, top=0.905, bottom=0.185)

    panels = [
        (wc, "RdBu_r", "vertical velocity (cm s$^{-1}$)  ·  blue = sinking",
         True, "VERTICAL — up or down"),
        (mag, "YlGnBu", "horizontal speed (m s$^{-1}$)", False,
         "HORIZONTAL MAGNITUDE — how fast"),
        (az, "twilight", "direction the wind comes FROM (° from north)", "cyc",
         "AZIMUTH — where from"),
    ]
    for row, (field, cmap, label, mode, name) in enumerate(panels):
        ax = fig.add_subplot(gs[row, 0])
        if mode is True:                       # signed, symmetric about zero
            lim = np.nanpercentile(abs(field), 98)
            kw = dict(vmin=-lim, vmax=lim)
        elif mode == "cyc":
            kw = dict(vmin=0, vmax=360)
        else:
            kw = dict(vmin=0, vmax=np.nanpercentile(field, 99))
        pm = ax.pcolormesh(tn, zl / 1000, field.T, cmap=cmap,
                           shading="nearest", **kw)
        if len(zl) <= 10:
            ax.set_yticks(zl / 1000)
            ax.set_yticklabels([f"{z/1000:.1f}" for z in zl])
        ax.axvline(mdates.date2num(EPI), color="#111827", lw=1.8)
        ax.axhline(C.WIND3D_LID / 1000, color="#111827", lw=0.8, ls=":")
        ax.set_xlim(tn[0], tn[-1])
        ax.set_ylabel("height (km)", fontsize=9, color=C.INK)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        if row == 0:
            ax.annotate("episode", (mdates.date2num(EPI), 0.95),
                        xycoords=("data", "axes fraction"), fontsize=8.5,
                        color="#111827", ha="left", va="top", style="italic",
                        xytext=(6, 0), textcoords="offset points")
        ax.set_xlabel("")
        cb = fig.colorbar(pm, cax=fig.add_subplot(gs[row, 1]))
        cb.set_label(label, fontsize=8, color=C.INK)
        cb.ax.tick_params(labelsize=7, colors=C.INK_MUTED)
        if mode == "cyc":
            cb.set_ticks([0, 90, 180, 270, 360])
            cb.set_ticklabels(["N", "E", "S", "W", "N"])
        _style(ax)

        b, e = np.nanmean(field[~epi][:, lo3]), np.nanmean(field[epi][:, lo3])
        if mode == "cyc":                      # a circular mean, not an average
            cb_ = lambda m: (np.degrees(np.arctan2(
                np.nanmean(np.sin(np.deg2rad(field[m][:, lo3]))),
                np.nanmean(np.cos(np.deg2rad(field[m][:, lo3]))))) + 360) % 360
            b, e = cb_(~epi), cb_(epi)
            stat = f"circular mean below 3 km: from {b:.0f}° → {e:.0f}°"
        else:
            stat = f"mean below 3 km: {b:+.2f} → {e:+.2f}"
        ax.set_title(f"{name}  ·  {stat}", fontsize=9.5,
                     color=C.INK_MUTED, loc="left", pad=6)

    # --- the air itself ------------------------------------------------------
    ax = fig.add_subplot(gs[3, 0])
    ax.fill_between(f.tl, p25, p75, color="#047857", alpha=0.28, lw=0,
                    label="corrected CAMS · 50% band")
    ax.plot(f.tl, p50, lw=2.0, color="#047857", label="corrected CAMS · median")
    st = m[(m.t_pht >= f.tl.min()) & (m.t_pht <= f.tl.max())]
    ax.plot(st.t_pht, st[sperr["obs"]], "o", ms=3.2, color="#1c1917", mec="white",
            mew=0.5, ls="none", label="station (EMB Central Visayas)")
    ax.axvline(EPI, color="#111827", lw=1.8)
    ax.axvspan(EPI, mdates.num2date(tn[-1]), color="#fca5a5", alpha=0.12,
               lw=0, zorder=0)
    ax.set_xlim(mdates.num2date(tn[0]), mdates.num2date(tn[-1]))
    ax.set_ylim(0, None)
    ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9, color=C.INK)
    ax.set_xlabel("Philippine time", fontsize=9, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9,
              facecolor="white", edgecolor="#e7e5e4")
    _style(ax)
    sb = st[st.t_pht < EPI][sperr["obs"]].mean()
    se = st[st.t_pht >= EPI][sperr["obs"]].mean()
    ax.set_title(f"THE AIR — station PM$_{{2.5}}$ {sb:.0f} → {se:.0f} µg m⁻³ "
                 f"across the same hours the wind barely moved",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=6)

    fig.text(0.075, 0.968, "The wind column over Cebu", fontsize=14,
             color=C.INK, weight="bold", va="top")
    fig.text(0.075, 0.936,
             f"83 km mean centred on {d['lat'][i]:.2f}°N {d['lon'][k]:.2f}°E · "
             f"{'NOAA GFS 0.25 deg ARL' if 'arl' in a.src else C.WIND3D_MODEL_LABEL} · "
             f"{len(zl)} levels · "
             f"{C.WATERMARK.replace('Made by: ', '')}",
             fontsize=9, color=C.INK_MUTED, va="top")
    notes = ["Direction uses the meteorological convention: the bearing the wind blows FROM. Transport goes the opposite way, so a 250° wind carries smoke toward 70°.",
             "Its colour map is cyclic — north wraps at both ends of the bar — because 359° and 1° are one degree apart, not 358.",
             f"{len(zl)} levels, {np.diff(zl).min():.0f}-{np.diff(zl).max():.0f} m apart, drawn as blocks: nothing is interpolated between them.",
             "The dotted line is 2.5 km, the lid used in the residence calculation. Mactan soundings put the real capping inversion at 1.5-1.9 km, so it sits too high.",
             "Resolved motion only. No convection at 0.25 deg, and an 83 km mean cannot see Cebu's own terrain."]
    arl = "arl" in a.src
    for n, line in enumerate(notes + C.attribution(coastlines=False,
                                                   trajectory=False,
                                                   cams=True, gfs=arl)):
        fig.text(0.075, 0.138 - n * 0.0112, line, fontsize=6.4,
                 color=C.INK_MUTED, va="top")

    # locator inset: the sections are an 83 km average, and that is not
    # obvious from a time-height plot
    try:
        import cartopy.crs as ccrs
        PC = ccrs.PlateCarree()
        lo0, lo1 = d["lon"][k - 1], d["lon"][k + 1]
        la0, la1 = d["lat"][i - 1], d["lat"][i + 1]
        inset = fig.add_axes([0.735, 0.022, 0.185, 0.125], projection=PC)
        inset.set_extent([116, 128, 5.5, 15.5], crs=PC)
        inset.coastlines(resolution="50m", linewidth=0.5, color=C.COAST)
        inset.add_patch(plt.Rectangle(
            (lo0, la0), lo1 - lo0, la1 - la0, transform=PC, zorder=5,
            facecolor="#fca5a5", edgecolor="#b91c1c", alpha=0.85, lw=1.4))
        inset.plot(C.CEBU["lon"], C.CEBU["lat"], marker="*", ms=8,
                   color=C.RECEPTOR, mec="white", mew=0.7, transform=PC, zorder=6)
        import matplotlib.ticker as mtick
        gl = inset.gridlines(draw_labels=True, linewidth=0.25, color="#e7e5e4")
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mtick.FixedLocator([118, 122, 126])
        gl.ylocator = mtick.FixedLocator([6, 10, 14])
        gl.xlabel_style = gl.ylabel_style = {"size": 5.5, "color": C.INK_MUTED}
        inset.set_title(f"the sections average this "
                        f"{3*abs(d['lat'][1]-d['lat'][0])*111:.0f} km box",
                        fontsize=6.5, color=C.INK_MUTED, pad=3)
        for sp_ in inset.spines.values():
            sp_.set_edgecolor("#d6d3d1")
    except ImportError:
        pass

    out = C.FIGS / (a.out or "wind_column.png")
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
