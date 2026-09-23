"""Cebu 24 h outlook: PM2.5 and PM10 only, each with a predictive band.

AOD and CO are not shown. They cannot be validated here (no ground AOD or CO),
AOD duplicates PM within the forecast window (r = +0.80), and the one
diagnostic use claimed for CO — divergence signalling a persistent plume —
failed when tested against the station.

    python src/plot_forecast.py
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
from compare_ground import prepared, hour_factors, split, forecast_frame

COR, OBS, SAFE_C = "#047857", "#1c1917", "#0891b2"
NSIM, NBOOT, FCST_H = 3000, 400, 24
HIST_H = 168          # 7 days back, so the station record is visible
# above C.FIG_DPI: this one is read closely on a phone and the dot markers and
# 6.5 pt category labels need the pixels. The maps stay at the shared setting.
DPI = 240
# DENR-EMB categories, 24 h mean, verified against the signed DAO 2020-14
# (PM2.5, stated in ug/m3 directly, not on a 0-500 index) and the EMB National
# Air Quality Status Report 2016-2018 Table 3 (PM10, Annex of IRR). Colours are
# the hex codes the DAO itself specifies.
BANDS = {"pm25": [(0, 25, "Good"), (25, 35, "Fair"), (35, 45, "Unhealthy (sens.)"),
                  (45, 55, "Very unhealthy"), (55, 90, "Acutely unhealthy"),
                  (90, None, "Emergency")],
         "pm10": [(0, 54, "Good"), (54, 154, "Fair"), (154, 254, "Unhealthy (sens.)"),
                  (254, 354, "Very unhealthy"), (354, 424, "Acutely unhealthy"),
                  (424, None, "Emergency")]}
FILL = ["#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97", "#7E0023"]
# yellow and green are illegible as text on white, so labels get darker variants
LABEL = ["#15803d", "#a16207", "#c2410c", "#b91c1c", "#6b21a8", "#7e0023"]
THRESH = {k: [(lo, nm) for lo, _, nm in v[1:]] for k, v in BANDS.items()}

SAFE = {"pm25": 15.0, "pm10": 45.0}


def simulate(f, key, rng):
    m, sp = prepared(key)
    m["day"] = m.t_pht.dt.floor("D")
    byday = {d: g for d, g in m.groupby("day")}
    days = m.day.unique()
    B = pd.DataFrame([
        pd.concat([byday[d] for d in rng.choice(days, len(days), replace=True)])
        .groupby("hr").apply(lambda g: g[sp["obs"]].mean() / g[sp["mod"]].mean(),
                             include_groups=False)
        for _ in range(NBOOT)])
    tr, te, _ = split(m)
    te = te.copy()
    te["cor"] = te[sp["mod"]] * te.hr.map(hour_factors(tr, sp))
    te["rel"] = (te[sp["obs"]] - te.cor) / te.cor.replace(0, np.nan)
    pool = te.rel.replace([np.inf, -np.inf], np.nan).dropna().values
    by_hr = {h: g.rel.replace([np.inf, -np.inf], np.nan).dropna().values
             for h, g in te.groupby("hr")}
    # One bootstrap replicate per simulation, reused for every hour: the 24
    # factors are a single unknown set, not 24 independent unknowns. Drawing
    # them per hour biased the 24 h peak high by mixing unrelated replicates.
    rep = rng.integers(0, NBOOT, NSIM)
    out = np.empty((len(f), NSIM))
    for i, (_, row) in enumerate(f.iterrows()):
        fb = B[row.hr].to_numpy()[rep]
        res = by_hr.get(row.hr, pool)
        rb = rng.choice(res if len(res) >= 6 else pool, NSIM, replace=True)
        out[i] = np.clip(row[sp["mod"]] * fb * (1 + rb), 0, None)
    return out, m, sp


def main() -> None:
    rng = np.random.default_rng(0)
    f = forecast_frame(["pm2_5", "pm10"], past_hours=9 * 24)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    f = f[(f.time >= now - pd.Timedelta(hours=HIST_H)) &
          (f.time <= now + pd.Timedelta(hours=FCST_H))].copy()
    f["t"] = f.t_pht
    now_l = now + pd.Timedelta(hours=C.TZ_OFFSET_H)

    fig, axes = plt.subplots(2, 1, figsize=(12.4, 9.6), dpi=DPI, sharex=True)
    summary, panels = [], []
    for ax, key in zip(axes, ("pm25", "pm10")):
        S, m, sp = simulate(f, key, rng)
        p25, p50, p75 = (np.percentile(S, q, 1) for q in (25, 50, 75))
        ax.fill_between(f.t, p25, p75, color=COR, alpha=0.30, lw=0, zorder=3,
                        label="corrected CAMS · 50% band")
        ax.plot(f.t, p50, lw=2.2, color=COR, zorder=4,
                label="corrected CAMS · predictive median")
        st = m[m.t_pht >= f.t.min()]
        if len(st):
            ax.plot(st.t_pht, st[sp["obs"]], "o", ms=3.2, color=OBS,
                    mec="white", mew=0.5, ls="none", zorder=5,
                    label="station (EMB Central Visayas)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(f.t.min(), f.t.max())
        panels.append((ax, key, sp, S))

    for ax, key, sp, S in panels:
        top = ax.get_ylim()[1]
        for (lo, hi, name), fc, lc in zip(BANDS[key], FILL, LABEL):
            hi = top if hi is None else min(hi, top)
            if lo >= top:
                break
            ax.axhspan(lo, hi, color=fc, alpha=0.16, lw=0, zorder=0)
            # Good is labelled low in its band, not at the midpoint, to leave
            # room for the WHO line that sits inside it
            at = lo + (hi - lo) * (0.25 if name == "Good" else 0.5)
            if hi - lo > top * 0.030:
                ax.annotate(name, (1.008, at), xycoords=("axes fraction", "data"),
                            ha="left", va="center", fontsize=6.5, color=lc,
                            annotation_clip=False)
        ax.axhline(SAFE[key], color=SAFE_C, lw=1.0, ls=(0, (4, 2)), zorder=2)
        ax.annotate(f"WHO 24 h {SAFE[key]:g}", (1.008, SAFE[key]),
                    xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=6.5, color=SAFE_C, annotation_clip=False)
        # a grey wash is invisible over the category colours, so lighten the
        # forecast window instead: white over the bands, under the data
        ax.axvspan(now_l, f.t.max(), color="white", alpha=0.55, lw=0, zorder=1)
        ax.axvline(now_l, color="#1c1917", lw=1.4, zorder=6)
        ax.text(now_l + (f.t.max() - now_l) / 2, 0.955, "forecast",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color=C.INK, style="italic", zorder=20)

        peak = S[(f.time > now).values].max(0)
        probs = [(lab, (peak > t).mean() * 100) for t, lab in THRESH[key]]
        ax.set_ylabel(f"{sp['label']} (µg m$^{{-3}}$)", fontsize=9.5, color=C.INK)
        ax.set_title(f"{sp['label']} · next 24 h peak {np.median(peak):.0f} "
                     f"(50% {np.percentile(peak,25):.0f}–{np.percentile(peak,75):.0f}, "
                     f"90% {np.percentile(peak,5):.0f}–{np.percentile(peak,95):.0f})",
                     fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)
        ax.tick_params(labelsize=8, colors=C.INK_MUTED)
        ax.grid(alpha=0.18, lw=0.5, zorder=2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#d6d3d1")
        summary.append((key, np.median(peak), probs))
    axes[0].legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9,
                   facecolor="white", edgecolor="#e7e5e4")
    # seated under the legend rather than in a corner, so a screenshot that
    # crops the figure still carries the credit
    lg = axes[0].get_window_extent().transformed(axes[0].transAxes.inverted())
    axes[0].text(0.013, 0.845, C.WATERMARK, transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=C.WATERMARK_SIZE,
                 color=C.INK_MUTED, zorder=20)
    axes[1].set_xlabel("Philippine time", fontsize=9.5, color=C.INK)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Hh"))

    fig.text(0.065, 0.972, "Cebu City · 24 h outlook", fontsize=13.5,
             color=C.INK, weight="bold", va="top")
    fig.text(0.065, 0.942,
             f"issued {now_l:%a %d %b %Y %H:%M} PHT · shaded = forecast · "
             f"50% band shown; the quoted 90% peak range uses the full distribution",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    notes = [             "Colour bands = DENR-EMB categories (PM2.5 DAO 2020-14, PM10 DAO 2013-13); dashed line = WHO 2021 24 h guideline. Both are 24 h means, drawn against an hourly trace.",
             "GFAS holds fire emissions constant through the forecast, so this assumes the upwind fires keep burning as last observed."]
    # no map and no trajectory on this figure, so neither notice applies
    for n, line in enumerate(notes + C.attribution(coastlines=False,
                                                   trajectory=False)):
        fig.text(0.065, 0.080 - n * 0.0115, line, fontsize=6.2, color=C.INK_MUTED, va="top")
    fig.tight_layout(rect=[0.02, 0.115, 0.945, 0.925])
    out = C.FIGS / "cebu_forecast.png"
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"wrote {out}")
    for k, med, probs in summary:
        print(f"  {k}: peak median {med:.0f}  " +
              "  ".join(f"P(>{l})={p:.0f}%" for l, p in probs))


if __name__ == "__main__":
    main()
