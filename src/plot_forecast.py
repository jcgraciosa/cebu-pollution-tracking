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
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from compare_ground import prepared, hour_factors, split

COR, OBS, SAFE_C = "#047857", "#1c1917", "#0891b2"
NSIM, NBOOT, FCST_H = 3000, 400, 24
HIST_H = 168          # 7 days back, so the station record is visible
# EPA 24 h PM breakpoints, as a rough reference for an hourly trace
THRESH = {"pm25": [(35.4, "AQI 100"), (55.4, "AQI 150"), (125.4, "AQI 200")],
          "pm10": [(154, "AQI 100"), (254, "AQI 150")]}
# WHO 2021 global air quality guidelines, 24 h mean. Compared here against an
# hourly trace, so crossing it for one hour is not yet a guideline exceedance.
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
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=dict(
        latitude=C.CEBU["lat"], longitude=C.CEBU["lon"], hourly="pm2_5,pm10",
        past_days=9, forecast_days=2, timezone="UTC"), timeout=60).json()["hourly"]
    f = pd.DataFrame(r); f["time"] = pd.to_datetime(f.time)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    f = f[(f.time >= now - pd.Timedelta(hours=HIST_H)) &
          (f.time <= now + pd.Timedelta(hours=FCST_H))].copy()
    f["t"] = f.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    f["hr"] = f.t.dt.hour
    now_l = now + pd.Timedelta(hours=C.TZ_OFFSET_H)

    fig, axes = plt.subplots(2, 1, figsize=(12.4, 9.6), dpi=C.FIG_DPI, sharex=True)
    summary = []
    for ax, key in zip(axes, ("pm25", "pm10")):
        S, m, sp = simulate(f, key, rng)
        fut = f.time > now
        p25, p50, p75 = (np.percentile(S, q, 1) for q in (25, 50, 75))
        ax.axvspan(now_l, f.t.max(), color="#a8a29e", alpha=0.10, zorder=0)
        ax.axhspan(0, SAFE[key], color=SAFE_C, alpha=0.13, lw=0, zorder=0,
                   label="within WHO 24 h guideline")
        # 50% only: the 90% band reached ~700 and squashed everything else.
        # Exceedance probabilities below still use the full distribution.
        ax.fill_between(f.t, p25, p75, color=COR, alpha=0.30, lw=0,
                        label="corrected CAMS · 50% band")
        ax.plot(f.t, p50, lw=2.2, color=COR,
                label="corrected CAMS · predictive median")
        st = m[m.t_pht >= f.t.min()]
        if len(st):
            ax.plot(st.t_pht, st[sp["obs"]], lw=1.7, color=OBS,
                    label="station (measured)")
        ax.axvline(now_l, color=C.FIRE, lw=1.6, zorder=6)
        ax.set_ylim(bottom=0)
        ax.set_xlim(f.t.min(), f.t.max())
        ax.annotate(f"WHO 24 h {SAFE[key]:g}", (1.008, SAFE[key]),
                    xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=6.5, color=SAFE_C, annotation_clip=False)
        ax.text(now_l + (f.t.max() - now_l) / 2, 0.955, "forecast",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color=C.INK_MUTED, style="italic", zorder=7)

        peak = S[fut.values].max(0)
        # still reported on stdout, but kept off the figure: a probability that
        # the PEAK HOUR crosses a 24 h breakpoint reads as a 24 h exceedance
        probs = [(lab, (peak > t).mean() * 100) for t, lab in THRESH[key]]
        for t, lab in THRESH[key]:
            if t < ax.get_ylim()[1]:
                ax.axhline(t, color="#d6d3d1", lw=0.7, ls="--", zorder=1)
                ax.annotate(lab, (1.008, t), xycoords=("axes fraction", "data"),
                            ha="left", va="center", fontsize=6.5,
                            color=C.INK_MUTED, annotation_clip=False)
        ax.set_ylabel(f"{sp['label']} (µg m$^{{-3}}$)", fontsize=9.5, color=C.INK)
        ax.set_title(f"{sp['label']} · next 24 h peak {np.median(peak):.0f} "
                     f"(50% {np.percentile(peak,25):.0f}–{np.percentile(peak,75):.0f}, "
                     f"90% {np.percentile(peak,5):.0f}–{np.percentile(peak,95):.0f})",
                     fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)
        ax.tick_params(labelsize=8, colors=C.INK_MUTED); ax.grid(alpha=0.22, lw=0.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#d6d3d1")
        summary.append((key, np.median(peak), probs))
    axes[0].legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.9,
                   facecolor="white", edgecolor="#e7e5e4")
    axes[1].set_xlabel("Philippine time", fontsize=9.5, color=C.INK)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Hh"))

    fig.text(0.065, 0.972, "Cebu City · 24 h outlook", fontsize=13.5,
             color=C.INK, weight="bold", va="top")
    fig.text(0.065, 0.942,
             f"issued {now_l:%a %d %b %Y %H:%M} PHT · shaded = forecast · "
             f"50% band shown; the quoted 90% peak range uses the full distribution",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    notes = ["Shaded safe region = WHO 2021 24-hour guideline (PM2.5 15, PM10 45 ug/m3); it is a 24 h mean, drawn here against an hourly trace.",
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
