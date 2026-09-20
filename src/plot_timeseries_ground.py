"""Station PM against CAMS through time, restricted to hours the station reported.

Exists to show the diurnal mismatch: CAMS PM2.5 bottoms out around 19:00 local,
which is when the station peaks. Day-to-day agreement is decent; within-day
timing is close to anti-phased.

    python src/plot_timeseries_ground.py
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
from compare_ground import load_pairs

STATION_25 = "#b45309"
STATION_10 = "#0e7490"
MODEL = "#7e22ce"


def _style(ax):
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.22, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d6d3d1")


def main() -> None:
    m = load_pairs()
    m = m[m["PM 2.5"].notna() | m["PM 10"].notna()].copy()
    m["t_pht"] = m.time + pd.Timedelta(hours=C.TZ_OFFSET_H)

    fig = plt.figure(figsize=(13, 9.4), dpi=C.FIG_DPI)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.25, 1.0], hspace=0.38,
                          left=0.07, right=0.94, top=0.90, bottom=0.10)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(m.t_pht, m["PM 2.5"], lw=1.3, color=STATION_25, label="station PM$_{2.5}$")
    ax1.plot(m.t_pht, m["PM 10"], lw=1.1, color=STATION_10, alpha=0.8,
             label="station PM$_{10}$")
    ax1.set_ylabel("µg m$^{-3}$", fontsize=9, color=C.INK_MUTED)
    ax1.set_title("Station measurements", fontsize=9.5, color=C.INK_MUTED, loc="left", pad=4)
    ax1.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9,
               facecolor="white", edgecolor="#e7e5e4")
    _style(ax1)

    # CAMS PM2.5 is ~2.7x low, so a shared axis would flatten it -- twin instead
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(m.t_pht, m["PM 2.5"], lw=1.3, color=STATION_25, label="station PM$_{2.5}$")
    ax2b = ax2.twinx()
    ax2b.plot(m.t_pht, m.pm2_5, lw=1.3, color=MODEL, label="CAMS PM$_{2.5}$ (right)")
    ax2.set_ylabel("station µg m$^{-3}$", fontsize=9, color=STATION_25)
    ax2b.set_ylabel("CAMS µg m$^{-3}$", fontsize=9, color=MODEL)
    ax2b.tick_params(labelsize=8, colors=MODEL)
    ax2b.spines["top"].set_visible(False)
    r = m["PM 2.5"].corr(m.pm2_5)
    ax2.set_title(f"Station vs CAMS PM$_{{2.5}}$ · separate axes (CAMS runs "
                  f"{m['PM 2.5'].mean()/m.pm2_5.mean():.1f}× low) · hourly r = {r:+.2f}",
                  fontsize=9.5, color=C.INK_MUTED, loc="left", pad=4)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=2, loc="upper left",
               framealpha=0.9, facecolor="white", edgecolor="#e7e5e4")
    _style(ax2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))

    # mean diurnal cycle, standardised so the shapes are comparable
    ax3 = fig.add_subplot(gs[2])
    m["hr"] = m.t_pht.dt.hour
    d = m.groupby("hr")[["PM 2.5", "PM 10", "pm2_5"]].mean()
    z = lambda s: (s - s.mean()) / s.std()
    ax3.plot(d.index, z(d["PM 2.5"]), "-o", ms=3.5, lw=1.6, color=STATION_25,
             label="station PM$_{2.5}$")
    ax3.plot(d.index, z(d["PM 10"]), "-o", ms=3, lw=1.2, color=STATION_10,
             alpha=0.8, label="station PM$_{10}$")
    ax3.plot(d.index, z(d.pm2_5), "-o", ms=3.5, lw=1.6, color=MODEL,
             label="CAMS PM$_{2.5}$")
    ax3.axhline(0, color="#d6d3d1", lw=0.6)
    ax3.axvline(d["PM 2.5"].idxmax(), color=STATION_25, lw=0.8, ls="--", alpha=0.7)
    ax3.axvline(d.pm2_5.idxmin(), color=MODEL, lw=0.8, ls="--", alpha=0.7)
    ax3.annotate(f"station peak {d['PM 2.5'].idxmax():02d}:00 · "
                 f"CAMS minimum {d.pm2_5.idxmin():02d}:00",
                 (0.5, 0.04), xycoords="axes fraction", ha="center", fontsize=8,
                 color=C.INK)
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xlabel("hour of day (Philippine time)", fontsize=9, color=C.INK)
    ax3.set_ylabel("standardised", fontsize=9, color=C.INK_MUTED)
    ax3.set_title("Mean diurnal cycle", fontsize=9.5, color=C.INK_MUTED, loc="left", pad=4)
    ax3.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9,
               facecolor="white", edgecolor="#e7e5e4")
    _style(ax3)

    fig.text(0.07, 0.968, "Station PM vs CAMS through time, Cebu",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.07, 0.935,
             f"{m.t_pht.min():%d %b} – {m.t_pht.max():%d %b %Y} PHT · "
             f"{m['PM 2.5'].notna().sum()} h PM$_{{2.5}}$, {m['PM 10'].notna().sum()} h PM$_{{10}}$",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    for n, line in enumerate(C.attribution(m.t_pht.max().year)):
        fig.text(0.07, 0.062 - n * 0.0155, line, fontsize=6.2, color=C.INK_MUTED, va="top")

    out = C.FIGS / "station_vs_cams_timeseries.png"
    fig.savefig(out, facecolor=C.SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  station PM2.5 peak {d['PM 2.5'].idxmax():02d}:00  min {d['PM 2.5'].idxmin():02d}:00 PHT")
    print(f"  station PM10  peak {d['PM 10'].idxmax():02d}:00  min {d['PM 10'].idxmin():02d}:00 PHT")
    print(f"  CAMS  PM2.5   peak {d.pm2_5.idxmax():02d}:00  min {d.pm2_5.idxmin():02d}:00 PHT")
    print(f"  corr of the two mean diurnal shapes: {z(d['PM 2.5']).corr(z(d.pm2_5)):+.2f}")


if __name__ == "__main__":
    main()
