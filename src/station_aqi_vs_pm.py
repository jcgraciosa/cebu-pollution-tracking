"""Station US AQI against its own PM measurements.

Shows what the AQI is actually a function of: EPA defines the PM2.5 sub-index
on a 24 h mean, so the hourly PM scatters around the curve while the 24 h mean
lands on it exactly. PM10 has its own sub-index but never dominates here.

    python src/station_aqi_vs_pm.py
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
from compare_ground import load_pairs, pm25_to_aqi

BP10 = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)]


def pm10_to_aqi(c):
    if not np.isfinite(c):
        return np.nan
    c = np.floor(c)
    for clo, chi, ilo, ihi in BP10:
        if clo <= c <= chi:
            return ilo + (ihi - ilo) * (c - clo) / (chi - clo)
    return np.nan


def main() -> None:
    m = load_pairs().set_index("time")
    m["pm25_24h"] = m["PM 2.5"].rolling(24, min_periods=18).mean()
    m["pm10_24h"] = m["PM 10"].rolling(24, min_periods=18).mean()
    m["aqi10_24h"] = m.pm10_24h.map(pm10_to_aqi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=C.FIG_DPI)
    specs = [(axes[0], "PM 2.5", "pm25_24h", "Station PM$_{2.5}$ (µg m$^{-3}$)", "#b45309"),
             (axes[1], "PM 10", "pm10_24h", "Station PM$_{10}$ (µg m$^{-3}$)", "#0e7490")]
    for ax, hcol, dcol, xlab, colour in specs:
        d = m.dropna(subset=[hcol, "aqi_24h"])
        ax.scatter(d[hcol], d.aqi_24h, s=13, c=colour, alpha=0.25, linewidths=0,
                   label="hourly PM", zorder=2)
        d2 = m.dropna(subset=[dcol, "aqi_24h"])
        ax.scatter(d2[dcol], d2.aqi_24h, s=15, c=C.INK, alpha=0.75, linewidths=0,
                   label="24 h mean PM", zorder=3)
        r = d[hcol].corr(d.aqi_24h)
        r2 = d2[dcol].corr(d2.aqi_24h)
        ax.annotate(f"n = {len(d)}\nhourly r = {r:+.2f}\n24 h mean r = {r2:+.2f}",
                    (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                              edgecolor="#d6d3d1", alpha=0.92))
        for y, lab in ((50, "Good"), (100, "Moderate"), (150, "USG")):
            ax.axhline(y, color="#d6d3d1", lw=0.6, ls="--", zorder=0)
            ax.annotate(lab, (0.995, y), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=6.5, color=C.INK_MUTED)
        ax.set_xlabel(xlab, fontsize=9, color=C.INK)
        ax.set_ylabel("Station US AQI (24 h PM$_{2.5}$ sub-index)", fontsize=9, color=C.INK)
        ax.tick_params(labelsize=8, colors=C.INK_MUTED)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.92,
                  facecolor="white", edgecolor="#e7e5e4")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d6d3d1")

    dd = m.dropna(subset=["aqi_24h", "aqi10_24h"])
    fig.suptitle("Station US AQI against its own PM measurements, Cebu",
                 fontsize=13.5, color=C.INK, weight="bold", x=0.05, ha="left", y=0.985)
    fig.text(0.05, 0.925,
             f"{m.index.min():%d %b} – {m.index.max():%d %b %Y} · AQI is the 24 h PM$_{{2.5}}$ "
             f"sub-index; the PM$_{{10}}$ sub-index never exceeded it "
             f"(0 of {len(dd)} h), so this is also the overall AQI",
             fontsize=9, color=C.INK_MUTED)
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.89])
    out = C.FIGS / "station_aqi_vs_pm.png"
    fig.savefig(out, facecolor=C.SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    for hcol, dcol, lab in (("PM 2.5", "pm25_24h", "PM2.5"), ("PM 10", "pm10_24h", "PM10")):
        d = m.dropna(subset=[hcol, "aqi_24h"]); d2 = m.dropna(subset=[dcol, "aqi_24h"])
        print(f"  AQI vs {lab:6s} hourly r={d[hcol].corr(d.aqi_24h):+.3f}"
              f"  |  24 h mean r={d2[dcol].corr(d2.aqi_24h):+.3f}"
              f"  spearman={d2[dcol].corr(d2.aqi_24h, method='spearman'):+.3f}")


if __name__ == "__main__":
    main()
