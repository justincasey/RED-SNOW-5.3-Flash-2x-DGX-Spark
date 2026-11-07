#!/usr/bin/env bash
# The quality gate for a numerics-contract change: run it once per contract on the SAME engine build, then compare
# with bench/rexl3/gsm8k_compare.py (discordant pairs). Against a server already started on a pair:
#
#   bench/rexl3/quality.sh A|B TAG [long]   writes .work/rexl3-runs/quality-TAG/{gsm8k.json,needles.txt,summary.txt}
#
# GSM8K: the first 400 test problems, non-thinking, greedy (bench/rexl3/gsm8k.py). Needles: a fresh code in Moby-Dick at
# 8K and 32K tokens and 10/50/90% depth, one at ~186K and 50% depth (bench/rexl3/needle.py), and with `long` one near
# 1M tokens (bench/rexl3/needle_long.py; needs a window and KV pool of ~1M).
set -euo pipefail
cd "$(dirname "$0")/../.."

main() {
  local pair=${1:?PAIR A or B} tag=${2:?TAG} long=${3:-} host
  case $pair in A) host=192.168.1.121 ;; B) host=192.168.1.123 ;; *) echo "PAIR A or B" >&2; exit 2 ;; esac
  local base=http://$host:${PORT:-8080} out=.work/rexl3-runs/quality-$tag
  mkdir -p "$out"
  {
    python3 bench/rexl3/gsm8k.py --base "$base" --n 400 --workers 6 --out "$out/gsm8k.json" 2>&1 | grep -E 'correct|METRIC'
    python3 bench/rexl3/needle.py --base "$base" --tokens 8000,32000 --depths 0.1,0.5,0.9 2>&1 | tail -8
    python3 bench/rexl3/needle.py --base "$base" --tokens 200000 --depths 0.5 2>&1 | tail -2
    [[ $long != long ]] || python3 bench/rexl3/needle_long.py --base "$base" 2>&1
  } | tee "$out/summary.txt"
}

main "$@"; exit
