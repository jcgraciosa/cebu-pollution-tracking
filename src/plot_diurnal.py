"""Each day's PM overlaid on a common hour axis, to test whether the morning
bump / midday minimum / evening peak is a real recurring pattern or an artefact
of averaging.

    python src/plot_diurnal.py
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
from compare_ground import load_pairs


def panel(ax, piv, colour, label):
    days = piv.columns
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(days):
        ax.plot(piv.index, piv[d], lw=0.9, alpha=0.45,
                color=cmap(i / max(len(days) - 1, 1)), zorder=2)
    med = piv.median(axis=1)
    q1, q3 = piv.quantile(0.25, axis=1), piv.quantile(0.75, axis=1)
    ax.fill_between(piv.index, q1, q3, color=colour, alpha=0.18, zorder=3,
                    label="inter-quartile range")
    ax.plot(piv.index, med, lw=2.8, color=colour, zorder=4, label="median")
    ax.plot(piv.index, piv.mean(axis=1), lw=1.4, color=C.INK, ls="--", zorder=5,
            label="mean")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    ax.set_ylabel(f"{label} (µg m$^{{-3}}$)", fontsize=9, color=C.INK)
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.22, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d6d3d1")
    return med


def main() -> None:
    m = load_pairs()
    m["t_pht"] = m.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    m["day"] = m.t_pht.dt.date
    m["hr"] = m.t_pht.dt.hour

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.8), dpi=C.FIG_DPI, sharex=True)
    out = {}
    for ax, col, lab, colour in ((axes[0], "PM 2.5", "PM$_{2.5}$", "#b45309"),
                                 (axes[1], "PM 10", "PM$_{10}$", "#0e7490")):
        piv = m.pivot_table(index="hr", columns="day", values=col)
        piv = piv.loc[:, piv.notna().sum() >= 18]        # drop badly incomplete days
        med = panel(ax, piv, colour, lab)
        out[col] = (piv, med)
        ax.set_title(f"{lab} · {piv.shape[1]} days overlaid, coloured 1 Sep → 18 Sep",
                     fontsize=9.5, color=C.INK_MUTED, loc="left", pad=4)
        ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9,
                  facecolor="white", edgecolor="#e7e5e4")
    axes[1].set_xlabel("hour of day (Philippine time)", fontsize=9.5, color=C.INK)

    fig.text(0.07, 0.972, "Diurnal cycle of station PM, day by day — Cebu",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.07, 0.941,
             "each thin line is one day; bold line is the median across days",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    for n, line in enumerate(C.attribution(2026)):
        fig.text(0.07, 0.058 - n * 0.0155, line, fontsize=6.2, color=C.INK_MUTED, va="top")
    fig.tight_layout(rect=[0.03, 0.07, 0.99, 0.925])
    p = C.FIGS / "station_diurnal.png"
    fig.savefig(p, facecolor=C.SURFACE)
    plt.close(fig)
    print(f"wrote {p}\n")

    for col, (piv, med) in out.items():
        n = piv.shape[1]
        print(f"{col}: {n} days")
        print(f"  median curve  peak {med.idxmax():02d}:00 ({med.max():.1f})"
              f"  min {med.idxmin():02d}:00 ({med.min():.1f})")
        # how consistent is the pattern day to day?
        pk = piv.idxmax(); mn = piv.idxmin()
        ev = ((pk >= 16) | (pk <= 2)).mean() * 100
        mid = ((mn >= 9) & (mn <= 15)).mean() * 100
        print(f"  daily peak in the evening window 16-02h : {ev:.0f}% of days")
        print(f"  daily minimum in the midday window 9-15h: {mid:.0f}% of days")
        mb = med.loc[6:8].max(); prev = med.loc[3:5].mean()
        print(f"  morning bump 06-08h vs 03-05h: {mb:.1f} vs {prev:.1f} "
              f"({(mb/prev-1)*100:+.0f}%)")
        print(f"  days whose own peak is 19:00: {(pk==19).sum()}/{n}")


if __name__ == "__main__":
    main()
