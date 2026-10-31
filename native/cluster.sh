#!/usr/bin/env bash
# RED-SNOW-5.3-Flash on native TensorFold (justincasey/TensorFold tag redsnow-native-v1.1.0), TP=2 on two DGX Sparks.
# ./redsnow.sh calls this (install, start, stop, status, logs); it also runs on its own:
#
#   native/cluster.sh fabric          print the pair's two-rail link (read-only)
#   native/cluster.sh sync            rsync an engine checkout ($TF_SRC, for development) to both nodes
#   native/cluster.sh build           zig build native (sm_121) once on rank 0, copied to rank 1 through this machine
#   native/cluster.sh start           rank 1 then rank 0; returns once /health answers (or fails)
#   native/cluster.sh stop | status | logs
#
# The nodes are RANK0 / RANK1 (ssh targets) from the environment or the repo's .env. On each node the engine lives in
# REMOTE (default <BASE_DIR>/native: src/ the engine checkout, lib/ libnccl.so.2, zig/ Zig 0.17.0) and the models in
# MODELS_DIR (default <BASE_DIR>/models, where ./redsnow.sh download puts them; BASE_DIR default ~/redsnow).
# -p A / -p B select this project's lab pairs instead. Every TF_GLM_*, TF_KV_*, TF_ROCE_*, TF_HEAT_*, TENSORFOLD_* and
# NCCL_* variable in the caller's environment reaches both ranks.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

PAIR=""
while getopts "p:" o; do case $o in p) PAIR=$OPTARG ;; *) exit 2 ;; esac; done
shift $((OPTIND - 1))
CMD=${1:-status}

# KEY=VALUE files: the caller's environment wins, then native/engine.env, then the repo's .env (its node keys only)
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
load "$HERE/../.env" 'RANK0|RANK1|BASE_DIR|MODELS_DIR|DRAFTS|API_KEY|PORT|HOST|SERVED_NAME'

log() { printf '[cluster %s] %s\n' "${PAIR:-env}" "$*" >&2; }
on() { local h=$1; shift; ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$h" "$@"; }

case $PAIR in
  A) R0=dgx-worker-2 R1=dgx-worker-3 LAB=1 ;;   # this project's lab pairs
  B) R0=dgx-head R1=dgx-worker LAB=1 ;;
  "") [[ -n ${RANK0:-} && -n ${RANK1:-} ]] || { echo "set RANK0 and RANK1 (ssh targets) in .env or the environment" >&2; exit 2; }
      R0=$RANK0 R1=$RANK1 LAB=0 ;;
  *) echo "-p takes A or B (or omit it: RANK0 / RANK1)" >&2; exit 2 ;;
esac

if [[ $LAB == 1 ]]; then
  TF_SRC=${TF_SRC:-$HOME/projects/redsnow-port/tf-native-119} REMOTE=${REMOTE:-/home/spark/redsnow-native}
  MODELS_DIR=${MODELS_DIR:-/home/spark/models} ZIG_DIR=${ZIG_DIR:-/home/spark/zig-tool/zig-aarch64-linux-0.17.0}
else
  if [[ -z ${BASE_DIR:-} ]]; then BASE_DIR="$(on "$R0" 'printf %s "$HOME"')/redsnow" || { log "ssh to $R0 failed"; exit 1; }; fi
  REMOTE=${REMOTE:-$BASE_DIR/native} MODELS_DIR=${MODELS_DIR:-$BASE_DIR/models}
  ZIG_DIR=${ZIG_DIR:-$REMOTE/zig/zig-aarch64-linux-0.17.0} TF_SRC=${TF_SRC:-}
fi
MODEL=${MODEL:-$MODELS_DIR/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw}   # vcruz305's pack @ f6b89637 (./redsnow.sh download)
DRAFTER=${DRAFTER:-$MODELS_DIR/GLM-5.3-Flash-DFlash2}             # incoai/GLM-5.3-Flash-DFlash2 @ bf582e4e
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}
MASTER_PORT=${MASTER_PORT:-29741}
CONTEXT=${CONTEXT:-274432}
PARALLEL=${PARALLEL:-2}
DRAFTS=${DRAFTS:-on}
SERVED=${SERVED_NAME:-${SERVED:-RED-SNOW-5.3-Flash}}
READY_S=${READY_S:-1200}

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
      r0_hca+=${r0_hca:+,}$d0 r1_hca+=${r1_hca:+,}$d1
    done <<<"$b"
  done <<<"$a"
  [[ -n $r0_if ]] || { log "no shared RDMA subnet between $R0 and $R1"; return 1; }
  echo "$r0_if $r0_hca $r0_ip $r1_if $r1_hca"
}

engine_env() {
  env | grep -E '^(TF_GLM_[A-Z0-9_]+|TF_KV_[A-Z0-9_]+|TF_ROCE_[A-Z0-9_]+|TF_HEAT_[A-Z0-9_]+|TENSORFOLD_[A-Z0-9_]+|NCCL_[A-Z0-9_]+)=' \
    | grep -vE '^NCCL_(IB_HCA|SOCKET_IFNAME|COMM_ID)=' | sort | while IFS='=' read -r k v; do printf 'export %s=%q; ' "$k" "$v"; done
}

start_rank() { # host rank if hcas
  local host=$1 rank=$2 iface=$3 hcas=$4 master=$5 port=$6 bind=$HOST
  [[ $rank == 1 ]] && bind=127.0.0.1   # only rank 0 serves the API
  local args="serve $MODEL --name $SERVED --host $bind --port $port --context $CONTEXT --parallel $PARALLEL"
  [[ $DRAFTS == on ]] && args+=" --drafter $DRAFTER" || args+=" --no-drafts"
  [[ -n ${API_KEY:-} ]] && args+=" --api-key $API_KEY"
  args+="${EXTRA_ARGS:+ $EXTRA_ARGS}"   # extra serve flags (engine.env: --kv-cache fp8)
  on "$host" "mkdir -p $REMOTE/logs $REMOTE/run
    [[ -f $REMOTE/logs/rank$rank.log ]] && mv -f $REMOTE/logs/rank$rank.log $REMOTE/logs/rank$rank.prev.log
    $(engine_env) export TF_GLM_RANK=$rank TF_GLM_WORLD=2 NCCL_COMM_ID=$master:$MASTER_PORT \
      NCCL_SOCKET_IFNAME=$iface NCCL_IB_HCA=$hcas TF_ROCE_HCA=\${TF_ROCE_HCA:-$hcas} NCCL_DEBUG=\${NCCL_DEBUG:-WARN} \
      LD_LIBRARY_PATH=$REMOTE/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
    prof=''; [[ $rank == 0 && -n \"${NSYS_ARGS:-}\" ]] && prof=\"nsys profile ${NSYS_ARGS:-} -o $REMOTE/logs/rank0-prof --force-overwrite true\"
    nohup \$prof $REMOTE/src/zig-out/native/bin/tensorfold-native $args > $REMOTE/logs/rank$rank.log 2>&1 < /dev/null &
    echo \$! > $REMOTE/run/rank$rank.pid"
}

stop_rank() {
  on "$1" "pid=\$(cat $REMOTE/run/rank$2.pid 2>/dev/null) || exit 0
    kill \$pid 2>/dev/null || true
    for _ in \$(seq 60); do kill -0 \$pid 2>/dev/null || break; sleep 1; done
    kill -9 \$pid 2>/dev/null || true; rm -f $REMOTE/run/rank$2.pid"
}

case $CMD in
  fabric) fabric ;;
  setup) # Zig 0.17.0 on both nodes (./redsnow.sh install does this and the rest)
    for h in "$R0" "$R1"; do
      on "$h" "[[ -x $ZIG_DIR/zig ]] && exit 0; mkdir -p $(dirname $ZIG_DIR) && cd $(dirname $ZIG_DIR) &&
        curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz | tar xJ" && log "$h: zig $(on "$h" "$ZIG_DIR/zig version")"
    done ;;
  sync)
    [[ -n $TF_SRC ]] || { log "sync needs TF_SRC (an engine checkout here); ./redsnow.sh install clones the pinned tag instead"; exit 2; }
    for h in "$R0" "$R1"; do
      { on "$h" "mkdir -p $REMOTE/src" && rsync -a --delete --exclude .zig-cache --exclude zig-out --exclude .git "$TF_SRC/" "$h:$REMOTE/src/"; } &
    done
    fail=0; for j in $(jobs -p); do wait "$j" || fail=1; done
    [[ $fail == 0 ]] || { log "sync failed"; exit 1; }
    log "synced $(git -C "$TF_SRC" rev-parse --short HEAD)$(git -C "$TF_SRC" diff --quiet || echo +dirty)" ;;
  build) # once, on rank 0: the ranks compare their executables' digests, so rank 1 runs the same bytes
    on "$R0" "cd $REMOTE/src && $ZIG_DIR/zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 > $REMOTE/build.log 2>&1 ||
      { tail -40 $REMOTE/build.log; exit 1; }" || { log "build failed"; exit 1; }
    on "$R0" "tar -C $REMOTE/src/zig-out -cf - ." | on "$R1" "rm -rf $REMOTE/src/zig-out && mkdir -p $REMOTE/src/zig-out && tar -C $REMOTE/src/zig-out -xf -" \
      || { log "copy to $R1 failed"; exit 1; }
    a=$(on "$R0" "sha256sum $REMOTE/src/zig-out/native/bin/tensorfold-native | cut -c1-12")
    b=$(on "$R1" "sha256sum $REMOTE/src/zig-out/native/bin/tensorfold-native | cut -c1-12")
    [[ $a == "$b" ]] || { log "rank 1's binary differs ($b vs $a)"; exit 1; }
    log "built on $R0, copied to $R1: $a" ;;
  start)
    read -r r0_if r0_hca r0_ip r1_if r1_hca < <(fabric)
    log "link: $R0 $r0_if [$r0_hca] $r0_ip <-> $R1 $r1_if [$r1_hca]"
    stop_rank "$R0" 0; stop_rank "$R1" 1
    t0=$(date +%s)
    start_rank "$R1" 1 "$r1_if" "$r1_hca" "$r0_ip" $((PORT + 1))
    start_rank "$R0" 0 "$r0_if" "$r0_hca" "$r0_ip" "$PORT"
    until on "$R0" "curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null"; do
      for r in 0 1; do
        h=$R0; [[ $r == 1 ]] && h=$R1
        if ! on "$h" "kill -0 \$(cat $REMOTE/run/rank$r.pid) 2>/dev/null"; then
          log "rank $r exited during startup:"; on "$h" "tail -30 $REMOTE/logs/rank$r.log"; exit 1
        fi
      done
      (( $(date +%s) - t0 < READY_S )) || { log "not ready after ${READY_S}s"; exit 1; }
      sleep 5
    done
    echo "boot_s=$(( $(date +%s) - t0 ))" ;;
  stop) stop_rank "$R0" 0; stop_rank "$R1" 1; log "stopped" ;;
  status)
    for r in 0 1; do
      h=$R0; [[ $r == 1 ]] && h=$R1
      on "$h" "pid=\$(cat $REMOTE/run/rank$r.pid 2>/dev/null) && kill -0 \$pid 2>/dev/null && echo 'rank $r ($h): running' || echo 'rank $r ($h): stopped'"
    done
    on "$R0" "curl -sf -m 2 http://127.0.0.1:$PORT/health" || true ;;
  logs) for r in 0 1; do h=$R0; [[ $r == 1 ]] && h=$R1; echo "== rank $r"; on "$h" "tail -${N:-40} $REMOTE/logs/rank$r.log"; done ;;
  *) echo "unknown command $CMD" >&2; exit 2 ;;
esac
