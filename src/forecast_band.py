"""Corrected forecast with an empirical predictive band, and daily archiving.

The band is Monte Carlo over two terms: bootstrap uncertainty on the per-hour
factors (~8%) and the corrected model's own residual spread (~49%). The second
dominates, so propagating factor uncertainty alone would understate the band
roughly six-fold.

The residuals come from ANALYSIS hours, which are observation-constrained, so
the band is a floor for true forecast hours. `--archive` saves each run so the
growth with lead time can be measured properly once a few weeks accumulate.

    python src/forecast_band.py --species pm25 --archive
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
from compare_ground import prepared, hour_factors, split, SPECIES, forecast_frame

ARCHIVE = C.DATA / "forecast_archive"
# unconditional: the CI cache step saves this path with if:always(), so a run
# that dies before --archive would otherwise fail with a path-validation error
ARCHIVE.mkdir(parents=True, exist_ok=True)
RAW, COR, OBS = "#a78bfa", "#047857", "#1c1917"


def fetch(species):
    sp = SPECIES[species]
    return forecast_frame([sp["mod"]], past_hours=72), sp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="pm25", choices=list(SPECIES))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--archive", action="store_true", help="save this run for later verification")
    ap.add_argument("--band", default="full", choices=["full", "factors", "both"],
                    help="'factors' propagates only the bootstrap factor spread, which "
                         "omits the dominant residual term and is shown for contrast")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    m, spec = prepared(a.species)
    m["day"] = m.t_pht.dt.floor("D")
    days, byday = m.day.unique(), {d: g for d, g in m.groupby("day")}

    boots = [pd.concat([byday[d] for d in rng.choice(days, len(days), replace=True)])
             .groupby("hr").apply(lambda g: g[spec["obs"]].mean() / g[spec["mod"]].mean(),
                                  include_groups=False)
             for _ in range(400)]
    B = pd.DataFrame(boots)
    fac = hour_factors(m, spec)

    # residuals of the corrected model, per hour, from held-out days
    tr, te, cut = split(m)
    te = te.copy()
    te["cor"] = te[spec["mod"]] * te.hr.map(hour_factors(tr, spec))
    te["rel"] = (te[spec["obs"]] - te.cor) / te.cor.replace(0, np.nan)
    pool = te.rel.dropna().values
    by_hr = {h: g.rel.dropna().values for h, g in te.groupby("hr")}

    f, sp = fetch(a.species)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    f = f[f.time >= now - pd.Timedelta(hours=36)].copy()
    f["cor"] = f[sp["mod"]] * f.hr.map(fac)

    # one bootstrap replicate per simulation, shared across all 24 hours
    rep = rng.integers(0, len(B), a.n)

    def simulate(with_residual):
        d = np.empty((len(f), a.n))
        for i, (_, row) in enumerate(f.iterrows()):
            fb = B[row.hr].to_numpy()[rep]
            if with_residual:
                res = by_hr.get(row.hr, pool)
                rb = rng.choice(res if len(res) >= 8 else pool, a.n, replace=True)
            else:
                rb = 0.0
            d[i] = row[sp["mod"]] * fb * (1 + rb)
        return np.clip(d, 0, None)

    def pct(d, tag):
        for q, name in ((5, "p05"), (25, "p25"), (50, "p50"), (75, "p75"), (95, "p95")):
            f[f"{tag}{name}"] = np.percentile(d, q, 1)

    if a.band in ("full", "both"):
        pct(simulate(True), "")
    if a.band in ("factors", "both"):
        pct(simulate(False), "f_")
    if a.band == "factors":
        for k in ("p05", "p25", "p50", "p75", "p95"):
            f[k] = f["f_" + k]

    now_l = now + pd.Timedelta(hours=C.TZ_OFFSET_H)
    if a.band == "both":
        fig, axs = plt.subplots(2, 1, figsize=(12.4, 9.4), dpi=C.FIG_DPI, sharex=True,
                                sharey=True)
    else:
        fig, ax0 = plt.subplots(figsize=(12.4, 5.8), dpi=C.FIG_DPI)
        axs = [ax0]
    ax = axs[0]
    ax.axvspan(now_l, f.t_pht.max(), color="#a8a29e", alpha=0.10, zorder=0)
    lo_c, hi_c = ("f_p05", "f_p95") if a.band == "both" else ("p05", "p95")
    lo_i, hi_i = ("f_p25", "f_p75") if a.band == "both" else ("p25", "p75")
    ax.fill_between(f.t_pht, f[lo_c], f[hi_c], color=COR, alpha=0.16, lw=0, label="90% band")
    ax.fill_between(f.t_pht, f[lo_i], f[hi_i], color=COR, alpha=0.30, lw=0, label="50% band")
    cen = "f_p50" if a.band == "both" else "p50"
    ax.plot(f.t_pht, f[cen], lw=2.2, color=COR, label="predictive median")
    ax.plot(f.t_pht, f.cor, lw=1.3, color=COR, ls="--", alpha=0.8,
            label="per-hour factors only")
    ax.plot(f.t_pht, f[sp["mod"]], lw=1.3, color=RAW, label="CAMS raw")
    st = m[m.t_pht >= f.t_pht.min()]
    if len(st):
        ax.plot(st.t_pht, st[spec["obs"]], lw=1.7, color=OBS, label="station (measured)")
    ax.axvline(now_l, color=C.FIRE, lw=1.6, zorder=6)
    fut = f[f.time > now]
    if a.band == "both":
        ax.set_title("factor uncertainty only (~8%) — what pure error propagation gives",
                     fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)
        ax2 = axs[1]
        ax2.axvspan(now_l, f.t_pht.max(), color="#a8a29e", alpha=0.10, zorder=0)
        ax2.fill_between(f.t_pht, f.p05, f.p95, color=COR, alpha=0.16, lw=0, label="90% band")
        ax2.fill_between(f.t_pht, f.p25, f.p75, color=COR, alpha=0.30, lw=0, label="50% band")
        ax2.plot(f.t_pht, f.p50, lw=2.2, color=COR, label="predictive median")
        ax2.plot(f.t_pht, f.cor, lw=1.3, color=COR, ls="--", alpha=0.8,
                 label="per-hour factors only")
        ax2.plot(f.t_pht, f[sp["mod"]], lw=1.3, color=RAW, label="CAMS raw")
        if len(st):
            ax2.plot(st.t_pht, st[spec["obs"]], lw=1.7, color=OBS, label="station")
        ax2.axvline(now_l, color=C.FIRE, lw=1.6, zorder=6)
        ax2.set_title("factors + residual spread (~49%) — the honest band",
                      fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)
        ax2.set_ylabel(f"{spec['label']} (µg m$^{{-3}}$)", fontsize=9.5, color=C.INK)
    ax.set_ylabel(f"{spec['label']} (µg m$^{{-3}}$)", fontsize=9.5, color=C.INK)
    ax.set_xlabel("Philippine time", fontsize=9.5, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Hh"))
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.22, lw=0.5)
    for axx in axs:
        axx.tick_params(labelsize=8, colors=C.INK_MUTED)
        axx.grid(alpha=0.22, lw=0.5)
        for s_ in ("top", "right"):
            axx.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            axx.spines[s_].set_color("#d6d3d1")
    ax.legend(fontsize=8, ncol=5, loc="upper left", framealpha=0.9,
              facecolor="white", edgecolor="#e7e5e4")

    fig.text(0.06, 0.972, f"Cebu {spec['label']} · corrected forecast with predictive band",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.06, 0.936,
             f"issued {now_l:%a %d %b %Y %H:%M} PHT · next 24 h median "
             f"{fut.p50.max():.0f}, 90% band {fut.p05.min():.0f}–{fut.p95.max():.0f} µg m$^{{-3}}$",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    notes = ["Band = bootstrap factor uncertainty + the corrected model's own residual distribution; the residual dominates.",
             "The per-hour factors still under-predict on held-out days (residual median +0.78), so the predictive median sits ABOVE them.",
             "Residuals come from analysis hours, which are observation-constrained, so the band is a FLOOR for forecast hours.",
             "GFAS holds fire emissions constant through the forecast."]
    for n, line in enumerate(notes + C.attribution(2026)):
        fig.text(0.06, 0.083 - n * 0.0135, line, fontsize=6.2, color=C.INK_MUTED, va="top")
    fig.tight_layout(rect=[0.02, 0.135, 0.99, 0.90])
    suffix = {"full": "", "factors": "_factorsonly", "both": "_compare"}[a.band]
    out = C.FIGS / f"forecast_band_{a.species}{suffix}.png"
    fig.savefig(out, facecolor=C.SURFACE)
    plt.close(fig)

    if a.archive:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        cols = ["time", "hr", sp["mod"], "cor", "p05", "p25", "p50", "p75", "p95"]
        p = ARCHIVE / f"{a.species}_{now:%Y%m%dT%H}.csv"
        f[cols].assign(issued=now, lead_h=(f.time - now).dt.total_seconds() / 3600
                       ).to_csv(p, index=False)
        print(f"archived {p.name} ({len(f)} rows)")

    print(f"wrote {out}")
    if a.band == "both":
        wf = (fut.f_p95 - fut.f_p05).median(); wt = (fut.p95 - fut.p05).median()
        print(f"  median 90% width: factors only {wf:.0f}, full {wt:.0f} "
              f"-> full is {wt/wf:.1f}x wider")
    else:
        print(f"  next 24 h: median {fut.p50.max():.0f}  (factors-only {fut.cor.max():.0f})  90% band "
              f"{fut.p05.min():.0f}-{fut.p95.max():.0f}  "
              f"median band width {(fut.p95-fut.p05).median():.0f}")


if __name__ == "__main__":
    main()
