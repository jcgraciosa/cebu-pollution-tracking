"""Bootstrap uncertainty on the per-hour factors and on the skill gain.

Hourly data is autocorrelated, so this resamples whole DAYS with replacement
(a block bootstrap). An i.i.d. bootstrap over hours would badly understate the
intervals.

    python src/bootstrap_correction.py --species pm25 --n 2000
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


def rmse(o, p):
    return float(np.sqrt(((p - o) ** 2).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="pm25", choices=list(SPECIES))
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    m, spec = prepared(a.species)
    m["day"] = m.t_pht.dt.floor("D")
    days = m.day.unique()
    byday = {d: g for d, g in m.groupby("day")}

    boots = []
    for _ in range(a.n):
        s = pd.concat([byday[d] for d in rng.choice(days, len(days), replace=True)])
        boots.append(s.groupby("hr").apply(
            lambda g: g[spec["obs"]].mean() / g[spec["mod"]].mean(), include_groups=False))
    B = pd.DataFrame(boots)
    pt, lo, hi = hour_factors(m, spec), B.quantile(0.025), B.quantile(0.975)
    glob = m[spec["obs"]].mean() / m[spec["mod"]].mean()
    tr, te, cut = split(m)
    te = te.copy()
    te["cor"] = te[spec["mod"]] * te.hr.map(hour_factors(tr, spec))
    tdays = te.day.unique()
    tbd = {d: g for d, g in te.groupby("day")}
    g0, g1 = [], []
    for _ in range(a.n):
        s = pd.concat([tbd[d] for d in rng.choice(tdays, len(tdays), replace=True)])
        g0.append(rmse(s[spec["obs"]], s[spec["mod"]]))
        g1.append(rmse(s[spec["obs"]], s.cor))
    g0, g1 = np.array(g0), np.array(g1)
    d = g0 - g1
    r0, r1 = rmse(te[spec["obs"]], te[spec["mod"]]), rmse(te[spec["obs"]], te.cor)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=C.FIG_DPI,
                             gridspec_kw=dict(width_ratios=[1.55, 1]))
    ax = axes[0]
    ax.errorbar(pt.index, pt.values, yerr=[pt - lo, hi - pt], fmt="o", ms=5,
                color=COR, ecolor=COR, elinewidth=1.4, capsize=3, zorder=3)
    ax.axhline(glob, color=C.INK, ls="--", lw=1.2, label=f"global ×{glob:.2f}")
    out = ((lo > glob) | (hi < glob)).sum()
    ax.set_xticks(range(0, 24, 2)); ax.set_xlim(-0.6, 23.6)
    ax.set_xlabel("hour of day (PHT)", fontsize=9, color=C.INK)
    ax.set_ylabel("station ÷ CAMS", fontsize=9, color=C.INK)
    ax.set_title(f"Per-hour factors, 95% CI from {a.n} day-resamples · "
                 f"{out}/24 hours differ from the global factor",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, facecolor="white",
              edgecolor="#e7e5e4")

    ax = axes[1]
    ax.hist(d, bins=40, color=COR, alpha=0.8)
    ax.axvline(0, color=C.INK, lw=1.4, ls="--")
    ax.axvline(r0 - r1, color=C.FIRE, lw=1.6)
    ax.set_xlabel("RMSE reduction (µg m$^{-3}$)", fontsize=9, color=C.INK)
    ax.set_ylabel("bootstrap samples", fontsize=9, color=C.INK)
    ax.set_title(f"Skill gain on held-out days\n{r0:.1f} → {r1:.1f}; gain "
                 f"{r0-r1:.1f} [{np.percentile(d,2.5):.1f}, {np.percentile(d,97.5):.1f}], "
                 f"P(gain>0) = {(d>0).mean()*100:.0f}%",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)
    for x in fig.axes:
        x.tick_params(labelsize=8, colors=C.INK_MUTED)
        x.grid(alpha=0.22, lw=0.5)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            x.spines[sp].set_color("#d6d3d1")

    fig.text(0.045, 0.975, f"Bootstrap uncertainty on the {spec['label']} correction, Cebu",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.045, 0.938,
             f"{len(m)} hours over {len(days)} days · whole days resampled, because "
             f"neighbouring hours are not independent",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.89])
    p = C.FIGS / f"bootstrap_{a.species}.png"
    fig.savefig(p, facecolor=C.SURFACE)
    plt.close(fig)
    print(f"wrote {p}")
    print(f"  factor CI width: median {(hi-lo).median():.2f}, widest {(hi-lo).max():.2f} "
          f"at {(hi-lo).idxmax():02d}:00")
    print(f"  RMSE {r0:.1f} -> {r1:.1f}; gain {r0-r1:.1f} "
          f"[{np.percentile(d,2.5):.1f}, {np.percentile(d,97.5):.1f}]  "
          f"P(gain>0)={(d>0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
