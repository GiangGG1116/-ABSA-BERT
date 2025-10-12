#!/usr/bin/env bash
set -euo pipefail
# Edit this ID to your real one
FILE_ID="1d7JABk4jViI-USjLsWmhGkvzi8uQIL5C" # placeholder
OUT="data.zip"

gdown --id "$FILE_ID" -O "$OUT"
echo "Downloaded to $OUT"