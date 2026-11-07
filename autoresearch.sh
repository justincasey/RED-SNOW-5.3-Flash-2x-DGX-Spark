#!/usr/bin/env bash
# The benchmark for RED-SNOW-5.3-Flash on REXL3 ($REXL3_SRC, a checkout of justincasey/REXL3 branch redsnow): sync and
# build the engine on pair A (dgx-head rank 0 + dgx-worker rank 1), boot both ranks cold (page caches dropped) with
# rexl3/engine.env, run bench/rexl3/bench.py's fixed workload, stop.
#
# Primary metric target_score (higher is better; 1.0 = every target met):
#   (decode_prose_tok_s/70 * prefill_32k_tok_s/3000 * min(1, kv_tokens/2^20) * min(1, 100/boot_s)) ^ (1/4)
# It prints 0 when any gate fails; target_score_raw keeps the number. Secondary: every raw metric of bench.py, boot_s,
# kv_tokens. Gates: no errored reply, drafted == serial (token ids), concurrent == solo, the needle, cold prompts not
# cached, and serial ids equal to bench/rexl3/reference.json (the pinned numerics contract).
# Exits non-zero on a failed sync/build/boot or any failed gate.
set -euo pipefail
cd "$(dirname "$0")"

PAIR=${PAIR:-A}
case $PAIR in A) R0=dgx-head HOST=192.168.1.121 ;; B) R0=dgx-worker-2 HOST=192.168.1.123 ;; *) echo "PAIR A or B" >&2; exit 2 ;; esac
# rexl3/engine.env is the only source of engine settings: drop every engine variable a caller's shell may still carry
# (a stale REXL3_GLM_PREFILL_CONTRACT once overrode the file), and run the main lane on ~/projects/REXL3
while IFS='=' read -r k _; do [[ $k =~ ^[A-Z][A-Z0-9_]*$ ]] && unset "$k"; done < <(grep -vE '^(#|$)' rexl3/engine.env)
for k in $(env | sed -n 's/^\(REXL3_[A-Z0-9_]*\|NCCL_[A-Z0-9_]*\)=.*/\1/p'); do unset "$k"; done
unset OWNER COLD NSYS_ARGS
export REXL3_SRC=$HOME/projects/REXL3
echo "ASI engine=$(git -C "$REXL3_SRC" rev-parse --short=12 HEAD)$(git -C "$REXL3_SRC" diff --quiet HEAD || echo +dirty) env=$(grep -vE '^(#|$)' rexl3/engine.env | tr '\n' ' ')"
OUT=.work/rexl3-runs/$(date +%Y%m%d-%H%M%S)-$PAIR
mkdir -p "$OUT"

rexl3/cluster.sh -p "$PAIR" sync >&2
rexl3/cluster.sh -p "$PAIR" build >&2
trap 'N=300 rexl3/cluster.sh -p "$PAIR" logs > "$OUT/ranks.log" 2>&1 || true; rexl3/cluster.sh -p "$PAIR" stop >&2 || true' EXIT
boot=$(COLD=1 rexl3/cluster.sh -p "$PAIR" start | sed -n 's/^boot_s=//p')
[[ -n $boot ]] || { echo "boot failed" >&2; exit 1; }
rexl3/cluster.sh -p "$PAIR" boot > "$OUT/boot.log" 2>&1 || true
kv=$(sed -n 's/.*event=kv_plan .*kv_tokens=\([0-9]*\).*/\1/p' "$OUT/boot.log" | head -1)
[[ -n $kv ]] || { echo "no kv_plan line in rank 0's log" >&2; exit 1; }

python3 bench/rexl3/bench.py --base "http://$HOST:8080" --boot-s "$boot" --kv-tokens "$kv" \
  --ref bench/rexl3/reference.json --out "$OUT/bench.json" | tee "$OUT/metrics.txt"
