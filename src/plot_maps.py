"""Render one map frame per time step: field + winds + fires + back-trajectory,
over an optional satellite basemap, with a receptor strip carrying a cursor.

    python src/plot_maps.py                                  # AOD, 3-hourly
    python src/plot_maps.py --var us_aqi --basemap
    python src/plot_maps.py --var carbon_monoxide --stride 1
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm, ListedColormap
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrow
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from trajectory import back_trajectory

PC = ccrs.PlateCarree()
QUIVER_SCALE = 320          # data units per arrow length unit (axes widths)


class _ArrowHandler(HandlerPatch):
    """Draw the wind key as a real arrow inside the legend."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        a = FancyArrow(0, height / 2, width, 0, width=height * 0.06,
                       head_width=height * 0.5, head_length=height * 0.55,
                       length_includes_head=True, color=orig_handle.get_facecolor(),
                       alpha=orig_handle.get_alpha())
        a.set_transform(trans)
        return [a]


def _blank(path, thresh=5.0):
    """True if the tile is effectively empty (GIBS serves black for a miss)."""
    import numpy as np
    a = np.asarray(Image.open(path).convert("L"), dtype="float32")
    return a.mean() < thresh and a.std() < thresh


def fetch_gibs(date, bbox):
    """Daily true-colour snapshot, cached on disk. Returns a path or None."""
    tag = f"{bbox['south']}_{bbox['west']}_{bbox['north']}_{bbox['east']}"
    out = C.DATA / "gibs" / f"{C.GIBS_LAYER}_{date}_{tag}.jpg"
    if out.exists():
        return None if _blank(out) else out
    out.parent.mkdir(parents=True, exist_ok=True)
    px = C.GIBS_WIDTH_PX
    py = int(round(px * (bbox["north"] - bbox["south"]) / (bbox["east"] - bbox["west"])))
    url = C.GIBS_URL.format(layer=C.GIBS_LAYER, date=date, px=px, py=py,
                            s=bbox["south"], w=bbox["west"],
                            n=bbox["north"], e=bbox["east"])
    try:
        import requests
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        out.write_bytes(r.content)
        if _blank(out):
            # GIBS returns a valid, all-black JPEG when a day has no granules
            # yet. Byte size alone does not catch it, so check the pixels.
            out.unlink()
            print(f"  GIBS has no imagery for {date} yet", file=sys.stderr)
            return None
        return out
    except Exception as e:                                  # noqa: BLE001
        print(f"  GIBS fetch failed for {date}: {e!r}", file=sys.stderr)
        return None


def _boundaries(ax, over_imagery=False):
    """Coastlines and national borders. Over imagery they get a light halo so
    they stay legible against both dark ocean and bright smoke."""
    for res in ("50m", "110m"):
        try:
            if not over_imagery:
                ax.add_feature(cfeature.LAND.with_scale(res),
                               facecolor="#f5f5f4", zorder=0)
                ax.add_feature(cfeature.OCEAN.with_scale(res),
                               facecolor="#eef2f5", zorder=0)
            else:
                ax.add_feature(cfeature.COASTLINE.with_scale(res), edgecolor=C.HALO,
                               linewidth=1.9, alpha=0.55, zorder=5)
                ax.add_feature(cfeature.BORDERS.with_scale(res), edgecolor=C.HALO,
                               linewidth=1.6, alpha=0.5, zorder=5)
            ax.add_feature(cfeature.COASTLINE.with_scale(res),
                           edgecolor=C.COAST, linewidth=0.8, zorder=5.1)
            ax.add_feature(cfeature.BORDERS.with_scale(res), edgecolor=C.BORDER,
                           linewidth=0.7, linestyle="--", zorder=5.1)
            return
        except Exception as e:                              # noqa: BLE001
            print(f"  boundaries {res} unavailable ({e!r})", file=sys.stderr)


def pick_cube(aq, met, var):
    """Composition lives in the aq cube, met variables in the met cube."""
    for cube in (aq, met):
        if var in cube:
            return cube
    raise KeyError(f"{var} is in neither cube; re-run download.py")


def load_ground_obs():
    """Measured AQI, hand-entered. Empty frame if the file is absent."""
    cols = ["time", "station", C.GROUND_OBS_VAR, "source"]
    try:
        g = pd.read_csv(C.GROUND_OBS_CSV, parse_dates=["time"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=cols)
    return g.dropna(subset=["time", C.GROUND_OBS_VAR]).sort_values("time")


def load_all():
    aq = dict(np.load(C.DATA / "aq_grid.npz", allow_pickle=True))
    met = dict(np.load(C.DATA / "met_grid.npz", allow_pickle=True))
    try:
        fires = pd.read_csv(C.FIRES_CSV, parse_dates=["when"])
        if "confidence" in fires:
            conf = fires.confidence.astype(str).str.lower()
            fires = fires[~conf.isin(["l", "low"])]
    except FileNotFoundError:
        fires = pd.DataFrame(columns=["latitude", "longitude", "frp", "when"])
    site = (pd.read_csv(C.DATA / "cebu_timeseries.csv", parse_dates=["time"])
              .sort_values("time").reset_index(drop=True))
    return aq, met, fires, site, load_ground_obs()


def _scaled(values, spec):
    return values * spec["scale"] if spec.get("scale") else values


def make_frame(k, aq, met, fires, site, obs, var, spec, bbox, draw_traj, outdir,
               fire_window_h=12, quiver_every=2, show_forecast=False,
               basemap=False, dpi=None, t_start=None):
    times = aq["time"]
    tstamp = pd.Timestamp(str(times[k]))
    src = pick_cube(aq, met, var)
    lat, lon = src["lat"], src["lon"]
    field = _scaled(src[var][k].astype("float32"), spec)
    if basemap:
        if spec.get("mask_below") is not None:
            field = np.where(field < spec["mask_below"], np.nan, field)
        if spec.get("mask_above") is not None:
            field = np.where(field > spec["mask_above"], np.nan, field)

    nrow = 3 if len(obs) else 2
    # size the canvas from the domain aspect: cartopy preserves it, so a tall
    # box on a wide figure leaves a large empty margin
    asp = (bbox["north"] - bbox["south"]) / (bbox["east"] - bbox["west"])
    fig_w = 10.5
    map_h = fig_w * 0.80 * asp                       # map occupies ~80% of width
    fig_h = map_h + (2.6 if nrow == 2 else 4.3)      # + strips, title, footer
    fig = plt.figure(figsize=(fig_w, min(max(fig_h, 7.5), 15)), dpi=dpi or C.FIG_DPI)
    gs = fig.add_gridspec(nrow, 1, height_ratios=[4.0, 1.0, 1.0][:nrow], hspace=0.30,
                          left=0.06, right=0.93,
                          top=0.905 if nrow == 2 else 0.922,
                          bottom=0.15 if nrow == 2 else 0.125)
    ax = fig.add_subplot(gs[0], projection=PC)
    extent = [bbox["west"], bbox["east"], bbox["south"], bbox["north"]]
    ax.set_extent(extent, crs=PC)

    img = fetch_gibs(f"{tstamp:%Y-%m-%d}", bbox) if basemap else None
    if img is not None:
        ax.imshow(plt.imread(img), origin="upper", transform=PC, zorder=0, extent=extent)
    _boundaries(ax, over_imagery=img is not None)

    if spec.get("bands"):
        cmap = ListedColormap(spec["band_colors"])
        norm = BoundaryNorm(spec["bands"], cmap.N)
    else:
        cmap = spec["cmap"]
        norm = (LogNorm(vmin=max(spec["vmin"], 1e-3), vmax=spec["vmax"])
                if spec["log"] else Normalize(vmin=spec["vmin"], vmax=spec["vmax"]))
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), cmap=cmap, norm=norm,
                         shading="nearest", transform=PC, zorder=1,
                         alpha=0.62 if img is not None else 0.88)

    mi = int(np.clip(np.searchsorted(met["time"], times[k]), 0, met["time"].size - 1))
    spd = met["wind_speed_850hPa"][mi] / 3.6
    rad = np.deg2rad(met["wind_direction_850hPa"][mi])
    u, v = -spd * np.sin(rad), -spd * np.cos(rad)
    s = quiver_every
    ax.quiver(met["lon"][::s], met["lat"][::s], u[::s, ::s], v[::s, ::s],
              transform=PC, color="#57534e", alpha=0.55, scale=QUIVER_SCALE,
              width=0.0022, zorder=4)

    if len(fires):
        span_h = (fires.when.max() - fires.when.min()) / pd.Timedelta(hours=1)
        now_utc = tstamp.tz_localize("UTC")
        if span_h < 48:
            # keyless FIRMS is one 24 h file: show it all and say it is not
            # time-resolved, rather than let fires blink on and off
            w, flabel = fires, f"VIIRS thermal anomaly · 24 h snapshot, not time-resolved (n={len(fires)})"
        else:
            win = pd.Timedelta(hours=fire_window_h)
            w = fires[fires.when.between(now_utc - win, now_utc + win)]
            flabel = f"VIIRS thermal anomaly ±{fire_window_h} h (n={len(w)})"
        if len(w):
            # deliberately ON TOP of the raster: the fires are the source being
            # attributed, so they must stay legible through the plume
            ax.scatter(w.longitude, w.latitude, s=np.clip(w.frp * 0.28, 1.5, 42),
                       c=C.FIRE, alpha=C.FIRE_ALPHA, linewidths=0, transform=PC,
                       zorder=6, label=flabel)
        else:
            # No observations for this hour. Say which kind of gap it is: keyless
            # FIRMS only serves 24 h, so a missed day leaves a hole in the middle
            # of the archive, not just at the edges.
            if now_utc < fires.when.min():
                miss = f"no data before {fires.when.min():%d %b %H:%M} UTC"
            elif now_utc > fires.when.max():
                miss = f"no data after {fires.when.max():%d %b %H:%M} UTC"
            else:
                miss = "no data for this hour"
            ax.scatter([], [], c=C.FIRE, s=18, linewidths=0, transform=PC, zorder=6,
                       label=f"VIIRS thermal anomaly · {miss}")

    if draw_traj:
        _, tla, tlo = back_trajectory(met, C.CEBU["lat"], C.CEBU["lon"], mi,
                                      hours=C.TRAJ_HOURS)
        ax.plot(tlo, tla, color="#ffffff", lw=3.4, alpha=0.9, transform=PC, zorder=7)
        # label the length actually integrated: a parcel that leaves the domain
        # or runs out of record gives a shorter track than requested
        ax.plot(tlo, tla, color=C.TRAJ, lw=1.8, transform=PC, zorder=8,
                label=f"850 hPa back-trajectory, {len(tla)-1} h")
        ax.scatter(tlo[::24], tla[::24], s=26, facecolor="#ffffff",
                   edgecolor=C.TRAJ, linewidths=1.4, transform=PC, zorder=9)

    ax.plot(C.CEBU["lon"], C.CEBU["lat"], marker="*", ms=17, color=C.RECEPTOR,
            markeredgecolor="white", markeredgewidth=1.2, transform=PC, zorder=10)
    ax.annotate(C.CEBU["name"], (C.CEBU["lon"], C.CEBU["lat"]), xytext=(7, 7),
                textcoords="offset points", transform=PC, zorder=10,
                fontsize=9, color=C.INK, weight="bold")

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#d6d3d1", alpha=0.7)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": C.INK_MUTED}

    cb = fig.colorbar(mesh, ax=ax, pad=0.015, shrink=0.88, extend="max",
                      spacing="proportional")
    cb.set_label(spec["label"], fontsize=9, color=C.INK)
    cb.ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    cb.outline.set_visible(False)
    if spec.get("band_names"):
        mids = [(spec["bands"][i] + spec["bands"][i + 1]) / 2
                for i in range(len(spec["band_names"]))]
        cb.set_ticks(mids)
        cb.set_ticklabels(spec["band_names"])
        cb.ax.tick_params(labelsize=7, length=0)

    # Wind key inside the legend, sized so its arrow is genuinely WIND_KEY m/s at
    # the map's quiver scale -- a label that is also a scale bar.
    fs = 8
    ax_px = ax.get_window_extent().width
    target_px = (C.WIND_KEY / QUIVER_SCALE) * ax_px
    handlelen = target_px / (fs * fig.dpi / 72.0)
    hs, ls = ax.get_legend_handles_labels()
    hs.append(FancyArrow(0, 0, 1, 0, color="#57534e", alpha=0.75))
    ls.append(f"850 hPa wind · {C.WIND_KEY:g} m s$^{{-1}}$")
    leg = ax.legend(hs, ls, loc="lower right", fontsize=fs, framealpha=1.0,
                    facecolor="white", edgecolor="#a8a29e", borderpad=0.7,
                    handlelength=handlelen, handletextpad=0.7,
                    handler_map={FancyArrow: _ArrowHandler()})
    leg.set_zorder(20)                     # clear of the field and the imagery
    leg.get_frame().set_linewidth(0.8)

    # figure-level, not ax.set_title: cartopy shrinks the GeoAxes to the map
    # aspect and an axes title can end up clipped above the visible frame
    # local time leads: the audience for these is in the Philippines
    local = tstamp + pd.Timedelta(hours=C.TZ_OFFSET_H)
    ytitle = 0.963 if nrow == 2 else 0.969
    fig.text(0.06, ytitle,
             f"{spec['short']} · {local:%a %d %b %Y, %H:%M} {C.TZ_LABEL}",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.06, ytitle - 0.032 * (8.6 / fig.get_figheight()),
             f"{tstamp:%Y-%m-%d %H:%M} UTC",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    if C.WATERMARK:
        # inside the map frame, boxed so it stays legible over imagery
        credit = C.WATERMARK + (f"\n{C.WATERMARK_SUB}" if C.WATERMARK_SUB else "")
        ax.text(*C.WATERMARK_XY, credit, transform=ax.transAxes,
                ha=C.WATERMARK_HA, va="top",
                fontsize=C.WATERMARK_SIZE, color=C.INK, zorder=20,
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                          edgecolor="#a8a29e", linewidth=0.8, alpha=0.5))

    # --- receptor strip -------------------------------------------------------
    axb = fig.add_subplot(gs[1])
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    st = site.dropna(subset=[var])
    if not show_forecast:
        st = st[st.time <= now]
    if t_start is not None:                 # match the strip to the frame window
        st = st[st.time >= t_start]
    series = _scaled(st[var].to_numpy(dtype="float64"), spec)
    # local time, to match the title; the CSV is UTC
    tz = pd.Timedelta(hours=C.TZ_OFFSET_H)
    st_t, cur_t = st.time + tz, tstamp + tz
    axb.plot(st_t, series, color="#78716c", lw=1.6)
    axb.fill_between(st_t, 0, series, color="#78716c", alpha=0.13)
    axb.axvline(cur_t, color=C.FIRE, lw=1.8)
    if len(st):
        # pin the unit: pandas parses to datetime64[us], so a bare astype("int64")
        # would be microseconds while Timestamp.value is nanoseconds
        xp = st.time.to_numpy().astype("datetime64[ns]").astype("int64")
        cur = float(np.interp(np.datetime64(tstamp, "ns").astype("int64"), xp, series))
        axb.plot([cur_t], [cur], "o", ms=6, color=C.FIRE, mec="white", mew=1.2)
        axb.annotate(f" {spec['fmt'].format(cur)}", (cur_t, cur), fontsize=9,
                     color=C.FIRE, weight="bold", va="center")
    if show_forecast and len(st):
        axb.axvspan(now + tz, st_t.max(), color="#a8a29e", alpha=0.10)

    axb.set_ylabel(spec["short"], fontsize=9, color=C.INK_MUTED)
    axb.set_title(f"{spec['label']} at {C.CEBU['name']}", fontsize=9,
                  color=C.INK_MUTED, loc="left", pad=4)
    axb.set_xlabel("Philippine time", fontsize=9, color=C.INK)
    import matplotlib.dates as _md
    axb.xaxis.set_major_formatter(_md.DateFormatter("%d %b %Hh"))
    axb.tick_params(labelsize=8, colors=C.INK_MUTED)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axb.spines[sp].set_color("#d6d3d1")
    axb.margins(x=0.01)

    if len(obs):
        axc = fig.add_subplot(gs[2], sharex=axb)
        o = obs[obs.time <= now]
        if t_start is not None:
            o = o[o.time >= t_start]
        gv = C.GROUND_OBS_VAR
        for st_name, grp in o.groupby("station"):
            axc.plot(grp.time + tz, grp[gv], "-o", ms=4.5, lw=1.4, color="#0f172a",
                     mec="white", mew=0.9, label=st_name)
        axc.axvline(cur_t, color=C.FIRE, lw=1.8)
        # EPA category boundaries, so the panel is readable without a colour key
        for y, lab in ((50, "Good"), (100, "Moderate"), (150, "USG"), (200, "Unhealthy")):
            if y <= max(120, o[gv].max() * 1.15 if len(o) else 120):
                axc.axhline(y, color="#d6d3d1", lw=0.6, ls="--", zorder=0)
                axc.annotate(lab, (0.002, y), xycoords=("axes fraction", "data"),
                             fontsize=6.5, color=C.INK_MUTED, va="bottom")
        axc.set_ylabel("US AQI", fontsize=9, color=C.INK_MUTED)
        axc.set_title(f"Measured AQI · DENR-EMB stations  (n={len(o)})", fontsize=9,
                      color=C.INK_MUTED, loc="left", pad=4)
        axc.tick_params(labelsize=8, colors=C.INK_MUTED)
        if o.station.nunique() > 1:
            axc.legend(loc="upper left", fontsize=7, framealpha=0.9,
                       facecolor="white", edgecolor="#e7e5e4")
        for sp in ("top", "right"):
            axc.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axc.spines[sp].set_color("#d6d3d1")
        axb.tick_params(labelbottom=False)
        axb.set_xlabel("")
        axc.set_xlabel("Philippine time", fontsize=9, color=C.INK)

    lines = C.attribution(tstamp.year, basemap=img is not None,
                          trajectory=draw_traj)
    for n, line in enumerate(lines):
        fig.text(0.06, (0.062 if nrow == 2 else 0.052) - n * 0.0165 * (8.6 / fig.get_figheight()),
                 line, fontsize=6.2,
                 color=C.INK_MUTED, va="top")

    out = outdir / f"{var}_{k:04d}.png"
    fig.savefig(out, facecolor=C.SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--var", default="aerosol_optical_depth", choices=list(C.VARS))
    p.add_argument("--stride", type=int, default=3, help="hours between frames")
    p.add_argument("--local", action="store_true", help="zoom to the Visayas box")
    p.add_argument("--basemap", action="store_true",
                   help="NASA GIBS true-colour imagery under the field")
    p.add_argument("--no-trajectory", action="store_true")
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--dpi", type=int, default=C.FIG_DPI)
    p.add_argument("--no-obs", action="store_true",
                   help="drop the measured-AQI panel even if observations exist")
    p.add_argument("--days", type=float, default=None,
                   help="rolling window: plot the last N days up to the newest "
                        "analysed hour. Preferred over --start for automation, "
                        "since it needs no editing as time passes.")
    p.add_argument("--start", default=None,
                   help="fixed first frame, e.g. 2026-08-31 (UTC). Overrides --days.")
    p.add_argument("--include-forecast", action="store_true",
                   help="also render hours beyond now (off: this is retrospective)")
    a = p.parse_args()

    aq, met, fires, site, obs = load_all()
    if a.no_obs:
        obs = obs.iloc[0:0]
    spec = dict(C.VARS[a.var])
    if a.vmax is not None:
        spec["vmax"] = a.vmax
    bbox = C.BBOX_LOCAL if a.local else C.BBOX
    outdir = C.FRAMES / (a.var + ("_local" if a.local else "")
                         + ("_tc" if a.basemap else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.png"):
        f.unlink()

    n_t = aq["time"].size
    if not a.include_forecast:
        # trim at the real clock: the API window slides, so an offset from the
        # first sample would drift
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
        n_t = max(1, int(np.searchsorted(aq["time"].astype("datetime64[ns]"),
                                         np.datetime64(now, "ns"), "right")))
        print(f"analysis window: {aq['time'][0]}Z .. {aq['time'][n_t-1]}Z "
              f"({aq['time'].size - n_t} future hours dropped)", file=sys.stderr)

    i0 = 0
    first = None
    if a.start:
        first = pd.Timestamp(a.start)
    elif a.days:
        # anchor on the newest analysed hour, not the clock, so a stale download
        # still yields a full window rather than a truncated one
        first = pd.Timestamp(str(aq["time"][n_t - 1])) - pd.Timedelta(days=a.days)
    if first is not None:
        i0 = int(np.searchsorted(aq["time"].astype("datetime64[ns]"),
                                 np.datetime64(first, "ns"), "left"))
        if i0 >= n_t:
            sys.exit(f"window starts {first} which is after the last analysed hour "
                     f"{aq['time'][n_t-1]}Z")
        if i0 == 0 and a.days:
            print(f"warning: only {n_t} h downloaded, less than the {a.days:g} d "
                  f"requested", file=sys.stderr)
        print(f"window: {aq['time'][i0]}Z .. {aq['time'][n_t-1]}Z", file=sys.stderr)
    idx = list(range(i0, n_t, a.stride))
    if idx[-1] != n_t - 1:
        idx.append(n_t - 1)      # always end on the newest analysed hour
    t_start = pd.Timestamp(str(aq["time"][idx[0]]))
    print(f"{len(idx)} frames -> {outdir}", file=sys.stderr)
    for n, k in enumerate(idx, 1):
        out = make_frame(k, aq, met, fires, site, obs, a.var, spec, bbox,
                         not a.no_trajectory, outdir,
                         show_forecast=a.include_forecast, basemap=a.basemap,
                         dpi=a.dpi, t_start=t_start)
        print(f"  [{n:3d}] {out.name}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
