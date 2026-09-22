#!/usr/bin/env bash
# Download HYSPLIT meteorology (ARL format) from NOAA ARL, resumably.
#
#   ./src/fetch_arl.sh gfs0p25 20260918 20260920      # daily files
#   ./src/fetch_arl.sh gdas1   sep26.w2 sep26.w3      # weekly files, names not dates
#
# Resumes with curl -C -, then checks the local size against Content-Length so a
# truncated file is never silently handed to HYSPLIT. Files land in data/arl/.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${ARL_DIR:-data/arl}"
mkdir -p "$DEST"
BASE="https://www.ready.noaa.gov/data/archives"
SET="${1:?usage: fetch_arl.sh <gfs0p25|gdas1> <file-or-date> ...}"
shift

# gfs0p25 forward-dated files are FORECAST, replaced by the analysis build at
# ~23:45 UTC on their own day. Refuse anything not yet rebuilt.
for id in "$@"; do
  case "$SET" in
    gfs0p25) name="${id}_gfs0p25" ;;
    gdas1)   name="gdas1.${id}" ;;
    *) echo "unknown set: $SET" >&2; exit 1 ;;
  esac
  url="$BASE/$SET/$name"
  remote=$(curl -sSI --http1.1 -m 60 "$url" | awk -F': ' 'tolower($1)=="content-length"{print $2+0}')
  mtime=$(curl -sSI --http1.1 -m 60 "$url" | awk -F': ' 'tolower($1)=="last-modified"{print $2}')
  if [ -z "$remote" ] || [ "$remote" = "0" ]; then
    echo "MISSING  $name  (not published yet)" >&2; continue
  fi
  if [ "$SET" = gfs0p25 ]; then
    want=$(date -j -f "%Y%m%d" "$id" "+%s" 2>/dev/null || echo 0)
    got=$(date -j -f "%a, %d %b %Y %H:%M:%S GMT" "$mtime" "+%s" 2>/dev/null || echo 0)
    if [ "$got" -ne 0 ] && [ "$want" -ne 0 ] && [ "$got" -lt "$want" ]; then
      echo "FORECAST $name  (written $mtime, before its valid date) -- skipping" >&2
      continue
    fi
  fi
  avail=$(df -k "$DEST" | awk 'NR==2{print $4*1024}')
  local_sz=$([ -f "$DEST/$name" ] && wc -c < "$DEST/$name" || echo 0)
  need=$((remote - local_sz))
  if [ "$need" -gt "$avail" ]; then
    echo "NO SPACE $name needs $((need/1000000)) MB, $((avail/1000000)) MB free" >&2
    exit 1
  fi
  printf '%s  %.2f GB  (modified %s)\n' "$name" "$(echo "$remote/1000000000" | bc -l)" "$mtime"
  curl -# --http1.1 -C - -o "$DEST/$name" "$url"
  got_sz=$(wc -c < "$DEST/$name")
  if [ "$got_sz" -ne "$remote" ]; then
    echo "TRUNCATED $name: $got_sz of $remote bytes -- re-run to resume" >&2; exit 1
  fi
  echo "  ok $name"
done
echo
du -sh "$DEST"
