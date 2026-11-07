#!/usr/bin/env bash
# The quality gate for a numerics-contract change (P2 prefill, fp8 KV, ...): run it on the old and the new contract and
# compare. Against a server already started on a pair:
#
#   bench/rexl3/quality.sh A TAG      writes .work/rexl3-runs/quality-TAG/{gsm8k.json,needle*.txt,summary.txt}
#
# GSM8K: the first 80 test problems, non-thinking, greedy (bench/rexl3/gsm8k.py). Needles: a fresh code in Moby-Dick at
# 8K and 32K tokens and 10/50/90% depth, then one at ~200K tokens and 50% depth (bench/rexl3/needle.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
PAIR=${1:?PAIR A or B} TAG=${2:?TAG}
case $PAIR in A) HOST=192.168.1.121 ;; B) HOST=192.168.1.123 ;; *) echo "PAIR A or B" >&2; exit 2 ;; esac
BASE=http://$HOST:${PORT:-8080}
OUT=.work/rexl3-runs/quality-$TAG
mkdir -p "$OUT"
{
  python3 bench/rexl3/gsm8k.py --base "$BASE" --n 80 --workers 2 --out "$OUT/gsm8k.json" 2>&1 | tail -12
  python3 bench/rexl3/needle.py --base "$BASE" --tokens 8000,32000 --depths 0.1,0.5,0.9 2>&1 | tee "$OUT/needle-short.txt" | tail -8
  python3 bench/rexl3/needle.py --base "$BASE" --tokens 200000 --depths 0.5 2>&1 | tee "$OUT/needle-200k.txt" | tail -3
} | tee "$OUT/summary.txt"
