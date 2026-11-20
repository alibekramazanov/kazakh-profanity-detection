#!/usr/bin/env bash
# Download KazSAnDRA (CC BY 4.0, https://github.com/IS2AI/KazSAnDRA) and unzip the
# score-classification splits used by src/prepare_data.py into $1 (default data/raw).
set -euo pipefail
OUT="${1:-data/raw}"
TMP="$(mktemp -d)"
git clone --depth 1 https://github.com/IS2AI/KazSAnDRA.git "$TMP/KazSAnDRA"
mkdir -p "$OUT"
for f in 06_sc_train_ib 09_sc_valid 10_sc_test; do
  unzip -o -q "$TMP/KazSAnDRA/dataset/$f.zip" -d "$OUT"
done
rm -rf "$TMP"
ls -la "$OUT"
