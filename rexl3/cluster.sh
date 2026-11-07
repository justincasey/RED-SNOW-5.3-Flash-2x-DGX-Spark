#!/usr/bin/env bash
# RED-SNOW-5.3-Flash on REXL3 (justincasey/REXL3 branch redsnow), TP=2 on two DGX Sparks.
#
#   rexl3/cluster.sh fabric          print the pair's RDMA rails (read-only)
#   rexl3/cluster.sh sync            rsync an engine checkout ($REXL3_SRC) to rank 0
#   rexl3/cluster.sh build           cargo build --release on rank 0, copy the binary to rank 1, compare digests
#   rexl3/cluster.sh start           take the pair's lock, rank 1 then rank 0; returns once /health/ready answers; prints boot_s=
#   rexl3/cluster.sh stop            stop both ranks and release the lock
#   rexl3/cluster.sh hold | release  take / release the pair's lock without a server (single-node GPU tests)
#   rexl3/cluster.sh status | logs | boot
#
# -p A (dgx-head + dgx-worker) or -p B (dgx-worker-2 + dgx-worker-3) select the lab pairs; without -p the nodes are
# RANK0 / RANK1 from the environment or the repo's .env. Every REXL3_* and NCCL_* variable in the caller's environment
# (or rexl3/engine.env) reaches both ranks. COLD=1 drops both nodes' page caches before start.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

PAIR=""
while getopts "p:" o; do case $o in p) PAIR=$OPTARG ;; *) exit 2 ;; esac; done
shift $((OPTIND - 1))
CMD=${1:-status}

# KEY=VALUE files: the caller's environment wins, then rexl3/engine.env, then the repo's .env (its node keys only)
load() { # file keys-regex
  [[ -f $1 ]] || return 0
  local k v
  while IFS='=' read -r k v || [[ -n $k ]]; do
    v=${v%%#*}; v=${v%"${v##*[![:space:]]}"}; v=${v#\"}; v=${v%\"}
    [[ $k =~ ^($2)$ && -n $v && -z ${!k+set} ]] && export "$k=$v"
  done < "$1"
  return 0
}
load "$HERE/engine.env" '[A-Z][A-Z0-9_]*'
load "$HERE/../.env" 'RANK0|RANK1|BASE_DIR|MODELS_DIR|DRAFTS|PORT|HOST|SERVED_NAME'

log() { printf '[rexl3 %s] %s\n' "${PAIR:-env}" "$*" >&2; }
on() { local h=$1; shift; ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$h" "$@"; }

case $PAIR in
  A) R0=dgx-head R1=dgx-worker LAB=1 ;;
  B) R0=dgx-worker-2 R1=dgx-worker-3 LAB=1 ;;
  "") [[ -n ${RANK0:-} && -n ${RANK1:-} ]] || { echo "set RANK0 and RANK1 (ssh targets) in .env or the environment" >&2; exit 2; }
      R0=$RANK0 R1=$RANK1 LAB=0 ;;
  *) echo "-p takes A or B (or omit it: RANK0 / RANK1)" >&2; exit 2 ;;
esac

if [[ $LAB == 1 ]]; then
  REMOTE=${REMOTE:-/home/spark/rexl3} MODELS_DIR=${MODELS_DIR:-/home/spark/models}
else
  if [[ -z ${BASE_DIR:-} ]]; then BASE_DIR="$(on "$R0" 'printf %s "$HOME"')/redsnow" || { log "ssh to $R0 failed"; exit 1; }; fi
  REMOTE=${REMOTE:-$BASE_DIR/rexl3} MODELS_DIR=${MODELS_DIR:-$BASE_DIR/models}
fi
REXL3_SRC=${REXL3_SRC:-$HOME/projects/REXL3}
MODEL=${MODEL:-$MODELS_DIR/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw}   # vcruz305's pack @ f6b89637
DRAFTER=${DRAFTER:-$MODELS_DIR/GLM-5.3-Flash-DFlash2}             # incoai/GLM-5.3-Flash-DFlash2 @ bf582e4e
DRAFTS=${DRAFTS:-off}                                               # on: --draft $DRAFTER
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
TP_PORT=${TP_PORT:-29615}
SERVED=${SERVED_NAME:-${SERVED:-RED-SNOW-5.3-Flash}}
READY_S=${READY_S:-1200}
# One server runs on a pair at a time: start takes the pair's lock for OWNER (default main, the benchmark) and stop
# releases it; another owner's start or stop is refused (FORCE=1 stop overrides). Each owner builds in its own lane
# ($REMOTE/lanes/OWNER: src, target, bin), so a lane's build never replaces the binary another owner is about to run.
OWNER=${OWNER:-main}
[[ $OWNER =~ ^[A-Za-z0-9_-]+$ ]] || { echo "OWNER must be a plain name" >&2; exit 2; }
WORK=$REMOTE; [[ $OWNER == main ]] || WORK=$REMOTE/lanes/$OWNER
BIN=$WORK/bin/rexl3-serve

# The pair's rails: every rank-0 RDMA netdev whose /24 a rank-1 netdev shares, in HCA-name order.
# Prints: R0_IF R0_HCAS R0_IP R1_IF R1_HCAS (comma-joined HCAs; the first netdev carries NCCL's sockets).
fabric() {
  local q='ibdev2netdev 2>/dev/null | while read -r dev _ _ _ net state; do
             [[ $state == "(Up)" ]] || continue
             ip -4 -br addr show "$net" | awk -v d="$dev" -v n="$net" "{split(\$3,a,\"/\"); print d, n, a[1]}"
           done'
  local a b
  a=$(on "$R0" "$q" | sort) b=$(on "$R1" "$q" | sort)
  local r0_if="" r0_hca="" r0_ip="" r1_if="" r1_hca=""
  while read -r d0 n0 ip0; do
    [[ -n $ip0 ]] || continue
    local net=${ip0%.*}
    while read -r d1 n1 ip1; do
      [[ -n $ip1 && ${ip1%.*} == "$net" ]] || continue
      if [[ -z $r0_if ]]; then r0_if=$n0 r0_ip=$ip0 r1_if=$n1; fi
      r0_hca+=${r0_hca:+,}$d0:1 r1_hca+=${r1_hca:+,}$d1:1
    done <<<"$b"
  done <<<"$a"
  [[ -n $r0_if ]] || { log "no shared RDMA subnet between $R0 and $R1"; return 1; }
  echo "$r0_if $r0_hca $r0_ip $r1_if $r1_hca"
}

engine_env() {
  env | grep -E '^(REXL3_[A-Z0-9_]+|NCCL_[A-Z0-9_]+)=' | grep -vE '^(REXL3_SRC|NCCL_IB_HCA|NCCL_SOCKET_IFNAME)=' | sort \
    | while IFS='=' read -r k v; do printf 'export %s=%q; ' "$k" "$v"; done
}

start_rank() { # host rank iface hcas leader_ip
  local host=$1 rank=$2 iface=$3 hcas=$4 leader=$5
  local args="--model $MODEL --served-model-name $SERVED --allow-unvalidated-pack --no-vision --slots $SLOTS"
  args+=" --kv-tokens $KV_TOKENS --max-model-len $MAX_MODEL_LEN --prefill-chunk $PREFILL_CHUNK --spec $SPEC"
  [[ -n ${PREFIX_CACHE:-} ]] && args+=" --prefix-cache $PREFIX_CACHE"
  [[ -n ${PREFIX_CACHE_MIB:-} ]] && args+=" --prefix-cache-mib $PREFIX_CACHE_MIB"
  [[ $DRAFTS == on ]] && args+=" --draft $DRAFTER"
  args+="${EXTRA_ARGS:+ $EXTRA_ARGS}"
  [[ $rank == 0 ]] && args+=" --host $HOST --port $PORT"
  args+=" --tp-ranks 2 --tp-rank $rank --tp-peer $leader:$TP_PORT"
  on "$host" "mkdir -p $REMOTE/logs $REMOTE/run
    [[ -f $REMOTE/logs/rank$rank.log ]] && mv -f $REMOTE/logs/rank$rank.log $REMOTE/logs/rank$rank.prev.log
    $(engine_env) export REXL3_NCCL_LIB=$REMOTE/lib/libnccl.so.2 NCCL_SOCKET_IFNAME==$iface NCCL_IB_HCA==$hcas \
      NCCL_IB_TC=\${NCCL_IB_TC:-106} NCCL_IB_GID_INDEX=\${NCCL_IB_GID_INDEX:-3} RUST_BACKTRACE=1
    prof=''; [[ $rank == 0 && -n \"${NSYS_ARGS:-}\" ]] && prof=\"nsys profile ${NSYS_ARGS:-} -o $REMOTE/logs/rank0-prof --force-overwrite true\"
    nohup \$prof $BIN $args > $REMOTE/logs/rank$rank.log 2>&1 < /dev/null &
    echo \$! > $REMOTE/run/rank$rank.pid"
}

stop_rank() {
  on "$1" "pid=\$(cat $REMOTE/run/rank$2.pid 2>/dev/null) || exit 0
    kill \$pid 2>/dev/null || true
    for _ in \$(seq 60); do kill -0 \$pid 2>/dev/null || break; sleep 1; done
    kill -9 \$pid 2>/dev/null || true; rm -f $REMOTE/run/rank$2.pid"
}

alive() { on "$1" "kill -0 \$(cat $REMOTE/run/rank$2.pid 2>/dev/null) 2>/dev/null"; }

lock() { # take the pair's lock for OWNER, or name its holder and fail
  local held
  held=$(on "$R0" "mkdir -p $REMOTE/run; if mkdir $REMOTE/run/lock 2>/dev/null; then echo $OWNER > $REMOTE/run/lock/owner;
    else cat $REMOTE/run/lock/owner 2>/dev/null || echo unknown; fi")
  [[ -z $held || $held == "$OWNER" ]] || { log "pair held by '$held' (OWNER=$OWNER)"; return 3; }
}
holder() { on "$R0" "cat $REMOTE/run/lock/owner 2>/dev/null" || true; }

case $CMD in
  fabric) fabric ;;
  sync)
    # by checksum, without source times: a changed file lands with a fresh mtime, so cargo always rebuilds what changed
    on "$R0" "mkdir -p $WORK/src" && rsync -rlpc --delete --exclude target/ --exclude .git/ --exclude __pycache__/ "$REXL3_SRC/" "$R0:$WORK/src/" \
      || { log "sync failed"; exit 1; }
    c=$(git -C "$REXL3_SRC" rev-parse --short=12 HEAD)
    if ! git -C "$REXL3_SRC" diff --quiet HEAD; then c+=-d$(git -C "$REXL3_SRC" diff HEAD | shasum | cut -c1-6); fi
    on "$R0" "echo $c > $WORK/src/.rexl3_commit"
    log "synced $c ($OWNER)" ;;
  build) # once, on rank 0: the ranks compare their executables, so rank 1 runs the same bytes. A build eats rank 0's
    # memory (cargo + nvcc), which would shrink another owner's boot plan, so it takes the pair's lock like a server does
    pre=$(holder); lock || exit 3
    unlock_new() { [[ $pre == "$OWNER" ]] || on "$R0" "rm -rf $REMOTE/run/lock"; }
    on "$R0" "cd $WORK/src && export PATH=/usr/local/cuda/bin:\$HOME/.cargo/bin:\$PATH CUDA_HOME=/usr/local/cuda \
        REXL3_CUDA_ARCH=sm_121 REXL3_COMMIT=\$(cat .rexl3_commit) CARGO_TARGET_DIR=$WORK/target &&
      cargo +1.89.0 build --release --locked -p gb-server --bin rexl3-serve > $WORK/build.log 2>&1 ||
      { grep -E '^(error|warning: unused)' -A6 $WORK/build.log | head -60; tail -5 $WORK/build.log; exit 1; }
      mkdir -p $WORK/bin && cp $WORK/target/release/rexl3-serve $BIN.new && mv -f $BIN.new $BIN" || { log "build failed"; unlock_new; exit 1; }
    on "$R0" "cat $BIN" | on "$R1" "mkdir -p $WORK/bin && cat > $BIN.new && chmod +x $BIN.new && mv -f $BIN.new $BIN" \
      || { log "copy to $R1 failed"; unlock_new; exit 1; }
    a=$(on "$R0" "sha256sum $BIN | cut -c1-12") b=$(on "$R1" "sha256sum $BIN | cut -c1-12")
    unlock_new
    [[ $a == "$b" ]] || { log "rank 1's binary differs ($b vs $a)"; exit 1; }
    log "built $(on "$R0" "cat $WORK/src/.rexl3_commit") on $R0, copied to $R1: $a ($OWNER)" ;;
  start)
    lock || exit 3
    read -r r0_if r0_hca r0_ip r1_if r1_hca < <(fabric)
    log "link: $R0 $r0_if [$r0_hca] $r0_ip <-> $R1 $r1_if [$r1_hca]"
    stop_rank "$R0" 0; stop_rank "$R1" 1
    for h in "$R0" "$R1"; do   # another tenant on either GPU would be OOM-killed mid-boot (or kill this one)
      busy=$(on "$h" "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader")
      [[ -z $busy ]] || { log "$h's GPU is busy: $busy"; exit 2; }
    done
    if [[ ${COLD:-0} == 1 ]]; then
      for h in "$R0" "$R1"; do on "$h" 'sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null' & done; wait
    fi
    t0=$(date +%s.%N)
    start_rank "$R1" 1 "$r1_if" "$r1_hca" "$r0_ip"
    start_rank "$R0" 0 "$r0_if" "$r0_hca" "$r0_ip"
    until on "$R0" "curl -sf -m 2 http://127.0.0.1:$PORT/health/ready >/dev/null"; do
      for r in 0 1; do
        h=$R0; [[ $r == 1 ]] && h=$R1
        if ! alive "$h" $r; then log "rank $r exited during startup:"; on "$h" "tail -30 $REMOTE/logs/rank$r.log"; exit 1; fi
      done
      awk -v a="$t0" -v b="$(date +%s.%N)" -v m="$READY_S" 'BEGIN{exit !(b - a < m)}' || { log "not ready after ${READY_S}s"; exit 1; }
      sleep 1
    done
    awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "boot_s=%.1f\n", b - a}' ;;
  stop)
    h=$(holder)
    if [[ -n $h && $h != "$OWNER" && ${FORCE:-0} != 1 ]]; then log "pair held by '$h': not stopping (OWNER=$OWNER, FORCE=1 overrides)"; exit 3; fi
    stop_rank "$R0" 0; stop_rank "$R1" 1; on "$R0" "rm -rf $REMOTE/run/lock"; log "stopped" ;;
  hold) lock || exit 3; log "held by $OWNER (GPU tests may run; release when done)" ;;   # no server: e.g. single-node tests
  release)
    h=$(holder)
    if [[ -n $h && $h != "$OWNER" && ${FORCE:-0} != 1 ]]; then log "pair held by '$h' (OWNER=$OWNER)"; exit 3; fi
    on "$R0" "rm -rf $REMOTE/run/lock"; log "released" ;;
  status)
    for r in 0 1; do
      h=$R0; [[ $r == 1 ]] && h=$R1
      alive "$h" $r && echo "rank $r ($h): running" || echo "rank $r ($h): stopped"
    done
    echo "lock: $(holder)"
    on "$R0" "curl -s -m 2 http://127.0.0.1:$PORT/health" || true; echo ;;
  boot) # the ranks' boot and load phase lines
    for r in 0 1; do h=$R0; [[ $r == 1 ]] && h=$R1; echo "== rank $r"
      on "$h" "grep -E 'event=(boot|glm_load|tp_rank_loaded|tp_group_ready|tp_follower_ready|kv_plan|mtp_vocab)' $REMOTE/logs/rank$r.log | cut -c1-400"; done ;;
  logs) for r in 0 1; do h=$R0; [[ $r == 1 ]] && h=$R1; echo "== rank $r"; on "$h" "tail -${N:-40} $REMOTE/logs/rank$r.log"; done ;;
  *) echo "unknown command $CMD" >&2; exit 2 ;;
esac; exit   # on the esac line: bash never reads past it, so replacing this file under a running command is harmless
