"""Station truth, raw CAMS, and CAMS with per-hour factors, over the whole record.

Factors are fitted on the first 12 days only, so the shaded tail is genuinely
out-of-sample; the earlier part is in-sample and will flatter the method.

    python src/plot_corrected_series.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from compare_ground import prepared, hour_factors, split, SPECIES

OBS, RAW, COR = "#b45309", "#7e22ce", "#047857"

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="pm25", choices=list(SPECIES))
    a = ap.parse_args()
    m, spec = prepared(a.species)
    tr, _te, cut = split(m)
    f = hour_factors(m, spec, upto=cut)
    m["corrected"] = m[spec["mod"]] * m.hr.map(f)

    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=C.FIG_DPI)
    ax.axvspan(cut, m.t_pht.max(), color="#a8a29e", alpha=0.12, zorder=0)
    ax.plot(m.t_pht, m[spec["obs"]], lw=1.9, color=OBS, label="station (truth)", zorder=4)
    ax.plot(m.t_pht, m[spec["mod"]], lw=1.3, color=RAW, label="CAMS raw", zorder=3)
    ax.plot(m.t_pht, m.corrected, lw=1.6, color=COR,
            label="CAMS + per-hour factors", zorder=3.5)
    ax.axvline(cut, color=C.INK, lw=1.2, ls="--", zorder=5)

    def sc(d):
        e = d.corrected - d[spec["obs"]]
        er = d[spec["mod"]] - d[spec["obs"]]
        return (np.sqrt((er ** 2).mean()), np.sqrt((e ** 2).mean()),
                d[spec["mod"]].corr(d[spec["obs"]]), d.corrected.corr(d[spec["obs"]]))
    te = m[m.t_pht >= cut]
    r0, r1, c0, c1 = sc(te)
    ax.annotate(f"factors fitted here\n({len(tr)} h)", (m.t_pht.min(), ax.get_ylim()[1]),
                xytext=(8, -8), textcoords="offset points", fontsize=8.5,
                color=C.INK_MUTED, va="top")
    ax.annotate(f"out-of-sample ({len(te)} h)\nRMSE {r0:.1f} → {r1:.1f}   "
                f"r {c0:+.2f} → {c1:+.2f}",
                (cut, ax.get_ylim()[1]), xytext=(8, -8), textcoords="offset points",
                fontsize=8.5, color=C.INK, va="top")

    ax.set_ylabel(f"{spec['label']} (µg m$^{{-3}}$)", fontsize=9.5, color=C.INK)
    ax.set_xlabel("Philippine time", fontsize=9.5, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.22, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d6d3d1")
    ax.legend(fontsize=8.5, ncol=3, loc="upper center", framealpha=0.9,
              facecolor="white", edgecolor="#e7e5e4")

    fig.text(0.062, 0.975, f"Station vs raw and bias-corrected CAMS {spec['label']}, Cebu",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.062, 0.935,
             f"{m.t_pht.min():%d %b} – {m.t_pht.max():%d %b %Y} PHT · {len(m)} hours",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    for n, line in enumerate(C.attribution(2026)):
        fig.text(0.062, 0.075 - n * 0.019, line, fontsize=6.2, color=C.INK_MUTED, va="top")
    fig.tight_layout(rect=[0.02, 0.10, 0.995, 0.90])
    out = C.FIGS / f"corrected_series_{a.species}.png"
    fig.savefig(out, facecolor=C.SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  out-of-sample: RMSE {r0:.1f} -> {r1:.1f}   r {c0:+.2f} -> {c1:+.2f}")


if __name__ == "__main__":
    main()
