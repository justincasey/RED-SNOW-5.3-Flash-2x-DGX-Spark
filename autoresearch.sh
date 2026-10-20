#!/usr/bin/env bash
# The benchmark for RED-SNOW-5.3-Flash on native TensorFold (redsnow-native, $TF_SRC): sync and build the engine on
# one pair of Sparks, boot both ranks fresh (empty prompt cache), run bench/native/run.py's fixed workload, then
# bench/native/conc.py's concurrency workload (C sweep, mixed load, fixed open-loop trace), stop.
# Prints METRIC lines: trace_mean_s (primary since segment 2: mean submit-to-last-token seconds over the trace's 7
# requests), the C sweep, stall under a long fill, prefill, decode, TTFT, boot.
# Exits non-zero on a failed build/boot, an errored reply, a drafted reply that differs from "draft": false, or a
# concurrent reply that differs from its solo one.
set -euo pipefail
cd "$(dirname "$0")"

PAIR=${PAIR:-A}
case $PAIR in A) HOST=192.168.1.123 ;; B) HOST=192.168.1.121 ;; *) echo "PAIR A or B" >&2; exit 2 ;; esac
export PORT=${PORT:-8080} CONTEXT=${CONTEXT:-65536} PARALLEL=${PARALLEL:-8} TENSORFOLD_MEMORY_RESERVE_GIB=${TENSORFOLD_MEMORY_RESERVE_GIB:-6}
export TF_SRC=${TF_SRC:-$HOME/projects/redsnow-port/tf-native}
echo "ASI engine=$(git -C "$TF_SRC" rev-parse --short HEAD)$(git -C "$TF_SRC" diff --quiet || echo +dirty)"
OUT=.work/native-runs/$(date +%Y%m%d-%H%M%S)-$PAIR
mkdir -p "$OUT"

native/cluster.sh -p "$PAIR" sync >&2
native/cluster.sh -p "$PAIR" build >&2
trap 'native/cluster.sh -p "$PAIR" stop >&2 || true' EXIT
boot=$(native/cluster.sh -p "$PAIR" start | sed -n 's/^boot_s=//p')
[[ -n $boot ]] || { echo "boot failed" >&2; exit 1; }

python3 bench/native/run.py --base "http://$HOST:$PORT" --out "$OUT/run.json" | tee "$OUT/metrics.txt"
python3 bench/native/conc.py --base "http://$HOST:$PORT" --out "$OUT/conc.json" | tee -a "$OUT/metrics.txt"
echo "METRIC boot_s=$boot"
N=200 native/cluster.sh -p "$PAIR" logs > "$OUT/ranks.log" 2>&1 || true
