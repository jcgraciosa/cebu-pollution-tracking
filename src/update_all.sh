#!/usr/bin/env bash
# Refresh everything the site displays: data -> frames -> animations -> figures
# -> site/ -> gh-pages branch.  Does NOT push; that stays a deliberate act.
#
#   ./src/update_all.sh              full run (DAYS=5 rolling window)
#   DAYS=4 ./src/update_all.sh       shorter window
#   ./src/update_all.sh --no-render  skip frames/animations (figures only, fast)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
DAYS="${DAYS:-5}"                  # rolling window, in days, for the animations
RENDER=1
[ "${1:-}" = "--no-render" ] && RENDER=0

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

step "download latest data"
$PY src/download.py --step 1.0

if [ "$RENDER" = 1 ]; then
  # AOD (column) and CO (conservative tracer) follow a transported plume;
  # surface fields do not, and the Cebu time series covers those instead.
  for spec in "aerosol_optical_depth:" "carbon_monoxide:"; do
    v=${spec%%:*}; bm=${spec##*:}
    step "render $v $bm"
    # shellcheck disable=SC2086
    $PY src/plot_maps.py --var "$v" --stride 3 --days "$DAYS" --no-obs --no-trajectory $bm
    $PY src/make_gif.py --var "$v" $bm
  done
fi

step "forecast and correction figures"
$PY src/plot_forecast.py
for sp in pm25 pm10; do
  $PY src/forecast_band.py --species "$sp" --band full --archive
  $PY src/forecast_band.py --species "$sp" --band both
  $PY src/plot_scatter_correction.py --species "$sp"
  $PY src/plot_corrected_series.py --species "$sp"
  $PY src/bootstrap_correction.py --species "$sp" --n 1000
  $PY src/retrieve_pm.py --species "$sp" >/dev/null
done
$PY src/compare_ground.py
$PY src/plot_diurnal.py
$PY src/plot_timeseries_ground.py

step "build site and gh-pages branch"
$PY src/build_site.py
./src/publish_site.sh

printf '\n\033[1mdone.\033[0m publish with:  git push -f origin gh-pages\n'
