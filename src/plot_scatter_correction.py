"""Model vs observation scatter, before and after the per-hour correction.

Equal axis extents and a 1:1 line, so the correction shows up as the cloud
moving onto the diagonal rather than sitting below it.

    python src/plot_scatter_correction.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from compare_ground import prepared, hour_factors, split, SPECIES

RAW, COR = "#7e22ce", "#047857"

def stats(o, p):
    e = p - o
    return dict(rmse=np.sqrt((e ** 2).mean()), bias=e.mean(),
                r=np.corrcoef(o, p)[0, 1], n=len(o))


LBL = ""


def panel(ax, o, p, oos, colour, title, lim):
    ax.plot(lim, lim, color=C.INK, lw=1.2, ls="--", zorder=2, label="1:1")
    ax.scatter(o[~oos], p[~oos], s=16, c=colour, alpha=0.22, linewidths=0,
               zorder=3, label="in-sample")
    ax.scatter(o[oos], p[oos], s=20, c=colour, alpha=0.75, linewidths=0,
               zorder=4, label="out-of-sample")
    s = stats(o[oos], p[oos])
    ax.annotate(f"out-of-sample, n = {s['n']}\nRMSE = {s['rmse']:.1f}\n"
                f"bias = {s['bias']:+.1f}\nr = {s['r']:+.2f}",
                (0.035, 0.965), xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="#d6d3d1", alpha=0.93))
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Station {LBL} (µg m$^{{-3}}$)", fontsize=9.5, color=C.INK)
    ax.set_title(title, fontsize=10, color=C.INK, loc="left", pad=6)
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.22, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d6d3d1")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="pm25", choices=list(SPECIES))
    a = ap.parse_args()
    m, sp = prepared(a.species)
    globals()["LBL"] = sp["label"]
    tr, _te, cut = split(m)
    f = hour_factors(m, sp, upto=cut)
    m["corrected"] = m[sp["mod"]] * m.hr.map(f)
    oos = (m.t_pht >= cut).values
    o = m[sp["obs"]].values

    hi = float(np.nanmax([o, m[sp["mod"]].values, m.corrected.values])) * 1.05
    lim = (0, hi)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.4), dpi=C.FIG_DPI)
    s0 = panel(axes[0], o, m[sp["mod"]].values, oos, RAW, "CAMS raw", lim)
    s1 = panel(axes[1], o, m.corrected.values, oos, COR,
               "CAMS + per-hour factors", lim)
    axes[0].set_ylabel(f"CAMS {sp['label']} (µg m$^{{-3}}$)", fontsize=9.5, color=C.INK)
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9,
                   facecolor="white", edgecolor="#e7e5e4")

    fig.text(0.055, 0.972, f"Effect of the per-hour bias correction on {sp['label']}, Cebu",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.055, 0.935,
             f"factors fitted on {len(tr)} h before {cut:%d %b}; identical axes with a 1:1 line · "
             f"RMSE {s0['rmse']:.1f} → {s1['rmse']:.1f}, bias {s0['bias']:+.1f} → {s1['bias']:+.1f}",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    for n, line in enumerate(C.attribution(2026)):
        fig.text(0.055, 0.085 - n * 0.016, line, fontsize=6.2, color=C.INK_MUTED, va="top")
    fig.tight_layout(rect=[0.02, 0.135, 0.99, 0.90])
    out = C.FIGS / f"correction_scatter_{a.species}.png"
    fig.savefig(out, facecolor=C.SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    for lab, s in (("raw", s0), ("corrected", s1)):
        print(f"  {lab:10s} n={s['n']}  RMSE {s['rmse']:5.1f}  bias {s['bias']:+6.1f}  r {s['r']:+.2f}")


if __name__ == "__main__":
    main()
