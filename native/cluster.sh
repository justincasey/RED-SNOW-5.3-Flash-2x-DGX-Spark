#!/usr/bin/env bash
# RED-SNOW-5.3-Flash on native TensorFold (redsnow-native branch), TP=2 on one pair of DGX Sparks.
#
#   native/cluster.sh [-p A|B] fabric          print the pair's two-rail link (read-only)
#   native/cluster.sh [-p A|B] setup           Zig 0.17.0 on both nodes when missing
#   native/cluster.sh [-p A|B] sync            rsync the engine worktree ($TF_SRC) to both nodes
#   native/cluster.sh [-p A|B] build           zig build native (sm_121) on both nodes, in parallel
#   native/cluster.sh [-p A|B] start           rank 1 then rank 0; returns once /health answers (or fails)
#   native/cluster.sh [-p A|B] stop | status | logs
#
# Pair A = dgx-worker-2 (rank 0) + dgx-worker-3; pair B = dgx-head (rank 0) + dgx-worker (this lab's presets).
# Other machines: RANK0=<ssh host> RANK1=<ssh host> RANK1_LAN=<rank 1's LAN address> override the preset.
# Every TF_GLM_*, TF_ROCE_*, TF_HEAT_* and TENSORFOLD_* variable in the caller's environment reaches both ranks.
set -euo pipefail

PAIR=A
while getopts "p:" o; do case $o in p) PAIR=$OPTARG ;; *) exit 2 ;; esac; done
shift $((OPTIND - 1))
CMD=${1:-status}

case $PAIR in
  A) R0=dgx-worker-2 R1=dgx-worker-3 R1_LAN=192.168.1.18 ;;
  B) R0=dgx-head R1=dgx-worker R1_LAN=192.168.1.122 ;;
  *) echo "pair A or B" >&2; exit 2 ;;
esac
R0=${RANK0:-$R0} R1=${RANK1:-$R1} R1_LAN=${RANK1_LAN:-$R1_LAN}
# The recipe's engine settings (native/engine.env, KEY=VALUE lines); the caller's environment wins.
if [[ -f "$(dirname "$0")/engine.env" ]]; then
  while IFS='=' read -r k v; do [[ $k =~ ^[A-Z][A-Z0-9_]*$ && -z ${!k+set} ]] && export "$k=$v"; done < "$(dirname "$0")/engine.env"
fi
TF_SRC=${TF_SRC:-$HOME/projects/redsnow-port/tf-native}
REMOTE=${REMOTE:-/home/spark/redsnow-native}
MODEL=${MODEL:-/home/spark/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw}
DRAFTER=${DRAFTER:-/home/spark/models/GLM-5.3-Flash-DFlash2}   # incoai/GLM-5.3-Flash-DFlash2 @ bf582e4e (the recipe's)
ZIG_DIR=${ZIG_DIR:-/home/spark/zig-tool/zig-aarch64-linux-0.17.0}
PORT=${PORT:-8080}
MASTER_PORT=${MASTER_PORT:-29741}
CONTEXT=${CONTEXT:-131072}
PARALLEL=${PARALLEL:-4}
DRAFTS=${DRAFTS:-on}
SERVED=${SERVED:-RED-SNOW-5.3-Flash}
READY_S=${READY_S:-1200}

log() { printf '[cluster %s] %s\n' "$PAIR" "$*" >&2; }
on() { local h=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "$@"; }

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
  env | grep -E '^(TF_GLM_[A-Z0-9_]+|TF_ROCE_[A-Z0-9_]+|TF_HEAT_[A-Z0-9_]+|TENSORFOLD_[A-Z0-9_]+|NCCL_[A-Z0-9_]+)=' \
    | grep -vE '^NCCL_(IB_HCA|SOCKET_IFNAME|COMM_ID)=' | sort | while IFS='=' read -r k v; do printf 'export %s=%q; ' "$k" "$v"; done
}

start_rank() { # host rank if hcas
  local host=$1 rank=$2 iface=$3 hcas=$4 master=$5 port=$6
  local args="serve $MODEL --name $SERVED --host 0.0.0.0 --port $port --context $CONTEXT --parallel $PARALLEL"
  [[ $DRAFTS == on ]] && args+=" --drafter $DRAFTER" || args+=" --no-drafts"
  args+="${EXTRA_ARGS:+ $EXTRA_ARGS}"   # extra serve flags (e.g. --kv-cache fp8)
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
  setup)
    for h in "$R0" "$R1"; do
      on "$h" "[[ -x $ZIG_DIR/zig ]] && exit 0; mkdir -p $(dirname $ZIG_DIR) && cd $(dirname $ZIG_DIR) &&
        curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz | tar xJ" && log "$h: zig $(on "$h" "$ZIG_DIR/zig version")"
    done ;;
  sync)
    for h in "$R0" "$R1"; do
      { on "$h" "mkdir -p $REMOTE/src" && rsync -a --delete --exclude .zig-cache --exclude zig-out --exclude .git "$TF_SRC/" "$h:$REMOTE/src/"; } &
    done
    fail=0; for j in $(jobs -p); do wait "$j" || fail=1; done
    [[ $fail == 0 ]] || { log "sync failed"; exit 1; }
    log "synced $(git -C "$TF_SRC" rev-parse --short HEAD)$(git -C "$TF_SRC" diff --quiet || echo +dirty)" ;;
  build) # once, on rank 0: the ranks compare their executables' digests, so rank 1 runs the same bytes
    on "$R0" "cd $REMOTE/src && $ZIG_DIR/zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 > $REMOTE/build.log 2>&1 ||
      { tail -40 $REMOTE/build.log; exit 1; }" || { log "build failed"; exit 1; }
    on "$R0" "rsync -a --delete -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' $REMOTE/src/zig-out/ spark@$R1_LAN:$REMOTE/src/zig-out/" \
      || { log "copy to $R1 failed"; exit 1; }
    log "built on $R0, copied to $R1: $(on "$R0" "sha256sum $REMOTE/src/zig-out/native/bin/tensorfold-native | cut -c1-12")" ;;
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
