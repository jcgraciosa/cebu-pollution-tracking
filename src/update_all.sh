#!/usr/bin/env bash
# Refresh everything the site displays: data -> frames -> animations -> figures
# -> site/ -> gh-pages branch.  Does NOT push; that stays a deliberate act.
#
#   ./src/update_all.sh                full run (DAYS=5 rolling window)
#   DAYS=4 ./src/update_all.sh         shorter window
#   ./src/update_all.sh --no-render    skip frames/animations (figures only, fast)
#   ./src/update_all.sh --forecast-only  receptor series + forecast figures, ~2 min
#
# Figure stages run under `try`: one flaky upstream endpoint should not discard
# a run that already produced the maps. Failures are collected and reported at
# the end, and build_site.py marks any figure it could not refresh as stale.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
DAYS="${DAYS:-5}"                  # rolling window, in days, for the animations
RENDER=1
FORECAST_ONLY=0
case "${1:-}" in
  --no-render)     RENDER=0 ;;
  --forecast-only) FORECAST_ONLY=1; RENDER=0 ;;
esac

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

FAILED=""
try() {
  local name
  name=$(basename "${2:-$1}" .py)
  if ! "$@"; then
    # named once however many species it failed for
    case " $FAILED " in *" $name "*) ;; *) FAILED="$FAILED $name" ;; esac
    printf '\033[33m    ^ %s failed; continuing\033[0m\n' "$name" >&2
  fi
}

# The CI cache step saves this path with if:always(); without it a run that
# never reaches forecast_band.py fails the save with a path-validation error.
mkdir -p data/forecast_archive

if [ "$FORECAST_ONLY" = 0 ]; then
  step "download latest data"
  # Split deliberately: the grids are what the maps need, so losing them means
  # there is nothing worth publishing. The receptor series only feeds the
  # forecast figures, so it stales those alone.
  $PY src/download.py --parts aq,fires,met,vis --step 1.0
fi
step "receptor series at Cebu"
try $PY src/download.py --parts site --step 1.0

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
try $PY src/plot_forecast.py
for sp in pm25 pm10; do
  try $PY src/forecast_band.py --species "$sp" --band full --archive
  try $PY src/forecast_band.py --species "$sp" --band both
  try $PY src/plot_scatter_correction.py --species "$sp"
  try $PY src/plot_corrected_series.py --species "$sp"
  try $PY src/bootstrap_correction.py --species "$sp" --n 1000
  if ! $PY src/retrieve_pm.py --species "$sp" >/dev/null; then
    FAILED="$FAILED retrieve_pm"
  fi
done
try $PY src/compare_ground.py
try $PY src/plot_diurnal.py
try $PY src/plot_timeseries_ground.py

step "build site and gh-pages branch"
$PY src/build_site.py
./src/publish_site.sh

printf '\n\033[1mdone.\033[0m publish with:  git push -f origin gh-pages\n'
if [ -n "$FAILED" ]; then
  # the site is already built and the branch written; exit non-zero so CI still
  # goes red and says which figures on it are stale
  printf '\033[31mstages that failed:%s\033[0m\n' "$FAILED" >&2
  exit 1
fi
