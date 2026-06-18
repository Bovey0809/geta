#!/usr/bin/env bash
set -euo pipefail
# Find COCO in the AutoDL public mount (path varies per instance).
echo "== Locating COCO under /root/autodl-pub =="
PUB=$(find /root/autodl-pub -maxdepth 3 -type d -iname "*coco*" 2>/dev/null | head -1 || true)
echo "Candidate COCO dir: ${PUB:-<none found>}"
test -n "${PUB}" || { echo "ERROR: set COCO path manually in this script"; exit 1; }

DST=~/datasets/coco
mkdir -p ~/datasets
ln -sfn "${PUB}" "${DST}"
echo "Linked ${DST} -> ${PUB}"
echo "== Layout =="
ls "${DST}" || true
find "${DST}" -maxdepth 2 -iname "instances_val2017.json" 2>/dev/null | head -1
find "${DST}" -maxdepth 2 -type d -name "val2017" 2>/dev/null | head -2
