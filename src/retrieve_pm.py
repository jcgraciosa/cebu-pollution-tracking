"""Can surface PM be retrieved from AOD and CO (plus meteorology)?

Day-blocked cross-validation: hours within a day are autocorrelated, so a random
split would leak and inflate every score.

    python src/retrieve_pm.py --species pm25
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from compare_ground import load_pairs, SPECIES

MET = ["boundary_layer_height", "RH", "visibility", "wind_speed_10m",
       "precipitation", "Sol. Rad."]


def build(species):
    sp = SPECIES[species]
    m = load_pairs()
    m["t_pht"] = m.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    m["day"] = m.t_pht.dt.floor("D")
    h = m.t_pht.dt.hour
    m["hr_sin"], m["hr_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    m["logaod"] = np.log(m.aerosol_optical_depth.clip(lower=1e-3))
    m["logco"] = np.log(m.carbon_monoxide.clip(lower=1))
    return m.dropna(subset=[sp["obs"], "aerosol_optical_depth", "carbon_monoxide"]), sp


def cv(m, sp, feats, model, k=6, want_pred=False):
    X, y, g = m[feats].values, m[sp["obs"]].values, m.day.values
    ok = np.isfinite(X).all(1)
    X, y, g = X[ok], y[ok], g[ok]
    pred = np.empty_like(y, dtype=float)
    for tr, te in GroupKFold(n_splits=k).split(X, y, g):
        mdl = model()
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
    e = pred - y
    ss = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out = dict(rmse=np.sqrt((e ** 2).mean()), r2=ss,
               r=np.corrcoef(y, pred)[0, 1], n=len(y))
    return (out, y, pred) if want_pred else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="pm25", choices=list(SPECIES))
    a = ap.parse_args()
    m, sp = build(a.species)

    lin = lambda: make_pipeline(StandardScaler(), LinearRegression())
    rf = lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                       random_state=0, n_jobs=-1)
    gb = lambda: HistGradientBoostingRegressor(max_iter=400, random_state=0)

    sets = [
        ("linear, AOD only", ["aerosol_optical_depth"], lin),
        ("linear, log AOD", ["logaod"], lin),
        ("linear, AOD+CO", ["aerosol_optical_depth", "carbon_monoxide"], lin),
        ("linear, logAOD+logCO", ["logaod", "logco"], lin),
        ("linear, +met+hour", ["logaod", "logco"] + MET + ["hr_sin", "hr_cos"], lin),
        ("random forest", ["logaod", "logco"] + MET + ["hr_sin", "hr_cos"], rf),
        ("grad boosting", ["logaod", "logco"] + MET + ["hr_sin", "hr_cos"], gb),
        ("GB, + CAMS PM", ["logaod", "logco", sp["mod"]] + MET + ["hr_sin", "hr_cos"], gb),
    ]
    print(f"{a.species}: day-blocked 6-fold CV, {len(m)} hours over {m.day.nunique()} days\n")
    print(f"  {'model':24s} {'n':>4s} {'RMSE':>7s} {'R2':>7s} {'r':>6s}")
    for name, feats, mk in sets:
        s = cv(m, sp, feats, mk)
        print(f"  {name:24s} {s['n']:4d} {s['rmse']:7.1f} {s['r2']:7.2f} {s['r']:+6.2f}")

    # baseline: the per-hour bias correction of CAMS
    from compare_ground import hour_factors
    mm = m.copy(); mm["hr"] = mm.t_pht.dt.hour
    days = mm.day.unique()
    pred = np.empty(len(mm))
    for tr, te in GroupKFold(n_splits=6).split(mm, groups=mm.day.values):
        f = hour_factors(mm.iloc[tr], sp)
        pred[te] = (mm.iloc[te][sp["mod"]] * mm.iloc[te].hr.map(f)).values
    e = pred - mm[sp["obs"]].values
    r2 = 1 - (e ** 2).sum() / ((mm[sp["obs"]] - mm[sp["obs"]].mean()) ** 2).sum()
    print(f"  {'[per-hour bias corr.]':24s} {len(mm):4d} {np.sqrt((e**2).mean()):7.1f} "
          f"{r2:7.2f} {np.corrcoef(mm[sp['obs']], pred)[0,1]:+6.2f}")

    # ---- figures -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.2), dpi=C.FIG_DPI)

    ax = axes[0, 0]
    d = m.dropna(subset=["aerosol_optical_depth", sp["obs"]])
    ax.scatter(d.aerosol_optical_depth, d[sp["obs"]], s=15, c="#b45309",
               alpha=0.35, linewidths=0)
    b = pd.cut(d.aerosol_optical_depth, [0, .2, .3, .4, .5, .6, .8, 1.2])
    gb_ = d.groupby(b, observed=True).agg(x=("aerosol_optical_depth", "mean"),
                                          y=(sp["obs"], "mean"))
    ax.plot(gb_.x, gb_.y, "-o", color=C.INK, lw=2, ms=6, label="binned mean")
    lo = d[d.aerosol_optical_depth < .5]; hi = d[d.aerosol_optical_depth >= .5]
    for sub, col, lab in ((lo, "#047857", "< 0.5"), (hi, "#dc2626", "≥ 0.5")):
        k_, c_ = np.polyfit(sub.aerosol_optical_depth, sub[sp["obs"]], 1)
        xs = np.linspace(sub.aerosol_optical_depth.min(), sub.aerosol_optical_depth.max(), 20)
        ax.plot(xs, c_ + k_ * xs, color=col, lw=2, label=f"slope {lab} = {k_:.0f}")
    ax.axvline(0.5, color="#a8a29e", ls="--", lw=1)
    ax.set_xlabel("CAMS AOD 550 nm"); ax.set_ylabel(f"Station {sp['label']} (µg m$^{{-3}}$)")
    ax.set_title("Saturation: the slope collapses above AOD 0.5",
                 fontsize=9.5, color=C.INK_MUTED, loc="left")
    ax.legend(fontsize=7.5)

    ax = axes[0, 1]
    names = [n for n, _, _ in sets]
    r2s = [cv(m, sp, f_, k_)["r2"] for _, f_, k_ in sets]
    cols = ["#a8a29e"] * 5 + ["#0e7490", "#047857", "#b45309"]
    ax.barh(range(len(names)), r2s, color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("R² (day-blocked CV)")
    ax.set_title("Skill by model", fontsize=9.5, color=C.INK_MUTED, loc="left")
    for i, v in enumerate(r2s):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5)

    best = sets[-2]
    s_, y_, p_ = cv(m, sp, best[1], best[2], want_pred=True)
    ax = axes[1, 0]
    hi_ = max(y_.max(), p_.max()) * 1.05
    ax.plot([0, hi_], [0, hi_], "--", color=C.INK, lw=1.2)
    ax.scatter(y_, p_, s=16, c="#047857", alpha=0.45, linewidths=0)
    ax.set_xlim(0, hi_); ax.set_ylim(0, hi_); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Station {sp['label']} (µg m$^{{-3}}$)")
    ax.set_ylabel("predicted (µg m$^{-3}$)")
    ax.set_title(f"{best[0]} · R²={s_['r2']:.2f}, RMSE={s_['rmse']:.1f}",
                 fontsize=9.5, color=C.INK_MUTED, loc="left")

    rfm = rf(); feats = ["logaod", "logco"] + MET + ["hr_sin", "hr_cos"]
    d = m.dropna(subset=feats)
    rfm.fit(d[feats], d[sp["obs"]])
    imp = pd.Series(rfm.feature_importances_, index=feats).sort_values(ascending=False)
    ax = axes[1, 1]
    ax.barh(range(len(imp)), imp.values, color="#7e22ce")
    ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("random-forest importance")
    ax.set_title("What the model leans on", fontsize=9.5, color=C.INK_MUTED, loc="left")

    for x in fig.axes:
        x.tick_params(labelsize=8, colors=C.INK_MUTED); x.grid(alpha=0.22, lw=0.5)
        for s_ in ("top", "right"):
            x.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            x.spines[s_].set_color("#d6d3d1")
    fig.text(0.05, 0.977, f"Retrieving station {sp['label']} from AOD and CO, Cebu",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.05, 0.947, f"{len(m)} hours over {m.day.nunique()} days · "
             f"day-blocked 6-fold CV throughout", fontsize=9.5, color=C.INK_MUTED, va="top")
    fig.tight_layout(rect=[0.02, 0.01, 0.99, 0.925])
    out = C.FIGS / f"retrieve_{a.species}.png"
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"\n  wrote {out}")
    print("  random-forest feature importance:")
    for k, v in imp.items():
        print(f"    {k:24s} {v:.3f}  {'#'*int(v*60)}")


if __name__ == "__main__":
    main()
