#!/usr/bin/env bash
# The benchmark for RED-SNOW-5.3-Flash on REXL3 ($REXL3_SRC, a checkout of justincasey/REXL3 branch redsnow): sync and
# build the engine on one pair (PAIR=A: dgx-head + dgx-worker, the official run; PAIR=B: dgx-worker-2 + dgx-worker-3, so
# two candidates are measured at once), boot both ranks cold (page caches dropped) with rexl3/engine.env, run
# bench/rexl3/bench.py's fixed workload, stop. CANDIDATE_SRC names the engine checkout (default ~/projects/REXL3);
# CANDIDATE_ENV names a KEY=VALUE file whose settings override rexl3/engine.env for this run only.
#
# Primary metric target_score (higher is better; 1.0 = every target met):
#   (decode_prose_tok_s/70 * prefill_32k_tok_s/3000 * min(1, kv_tokens/2^20) * min(1, 100/boot_s)) ^ (1/4)
# It prints 0 when any gate fails; target_score_raw keeps the number. Secondary: every raw metric of bench.py, boot_s,
# kv_tokens. Gates (byte-exact against the unmodified engine): no errored reply, drafted == serial, concurrent == solo,
# solo == serial prefix, the needle, cold prompts not cached, and every reply pinned in bench/rexl3/reference.json.
# Exits non-zero on a failed sync/build/boot or any failed gate.
#
# The body is one function called on the last line, so bash has parsed all of it before it runs: replacing this file
# while a run is in flight cannot change what that run executes.
set -euo pipefail
cd "$(dirname "$0")"

main() {
  PAIR=${PAIR:-A}
  case $PAIR in A) HOST=192.168.1.121 ;; B) HOST=192.168.1.123 ;; *) echo "PAIR A or B" >&2; exit 2 ;; esac
  # rexl3/engine.env is the only source of engine settings: drop every engine variable a caller's shell may still carry
  # (a stale REXL3_GLM_PREFILL_CONTRACT once overrode the file), then apply the candidate's file if one is named
  local k
  while IFS='=' read -r k _; do [[ $k =~ ^[A-Z][A-Z0-9_]*$ ]] && unset "$k"; done < <(grep -vE '^(#|$)' rexl3/engine.env)
  for k in $(compgen -e | grep -E '^(REXL3|NCCL)_' || true); do unset "$k"; done
  unset COLD NSYS_ARGS
  export OWNER=bench   # its own lock owner and build lane: a manual session (OWNER main) is never taken over by a run
  local cand_env=""
  if [[ -n ${CANDIDATE_ENV:-} ]]; then
    set -a; source "$CANDIDATE_ENV"; set +a
    cand_env=$(grep -vE '^(#|$)' "$CANDIDATE_ENV" | tr '\n' ' ')
  fi
  export REXL3_SRC=${CANDIDATE_SRC:-$HOME/projects/REXL3}
  local ref=${REFERENCE:-bench/rexl3/reference.json}   # the pin the replies are gated against (an older contract's, to finish a comparison)
  echo "ASI engine=$(git -C "$REXL3_SRC" rev-parse --short=12 HEAD)$(git -C "$REXL3_SRC" diff --quiet HEAD || echo +dirty) pair=$PAIR ref=$ref env=$(grep -vE '^(#|$)' rexl3/engine.env | tr '\n' ' ') candidate_env=$cand_env"
  OUT=.work/rexl3-runs/$(date +%Y%m%d-%H%M%S)-$PAIR
  mkdir -p "$OUT"

  # wait for the pair (a lane may hold it for a test session), hold it, and release it (stop) however this run ends
  local _i
  for _i in $(seq "${LOCK_WAIT_S:-1500}"); do rexl3/cluster.sh -p "$PAIR" hold >/dev/null 2>&1 && break; sleep 1; done
  rexl3/cluster.sh -p "$PAIR" hold >&2
  trap 'N=300 rexl3/cluster.sh -p "$PAIR" logs > "$OUT/ranks.log" 2>&1 || true; rexl3/cluster.sh -p "$PAIR" stop >&2 || true' EXIT
  rexl3/cluster.sh -p "$PAIR" sync >&2
  rexl3/cluster.sh -p "$PAIR" build >&2
  local boot kv
  boot=$(COLD=1 rexl3/cluster.sh -p "$PAIR" start | sed -n 's/^boot_s=//p')
  [[ -n $boot ]] || { echo "boot failed" >&2; exit 1; }
  rexl3/cluster.sh -p "$PAIR" boot > "$OUT/boot.log" 2>&1 || true
  kv=$(sed -n 's/.*event=kv_plan .*kv_tokens=\([0-9]*\).*/\1/p' "$OUT/boot.log" | head -1)
  [[ -n $kv ]] || { echo "no kv_plan line in rank 0's log" >&2; exit 1; }

  python3 bench/rexl3/bench.py --base "http://$HOST:8080" --boot-s "$boot" --kv-tokens "$kv" \
    --ref "$ref" --out "$OUT/bench.json" | tee "$OUT/metrics.txt"
}

main "$@"; exit
