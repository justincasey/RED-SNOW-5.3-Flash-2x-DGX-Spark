#!/usr/bin/env bash
# RED-SNOW-5.3-Flash (EXL3 SAGE 4.91bpw) on two DGX Sparks joined by one QSFP cable, TensorFold TP=2.
# ENGINE=native (default): the Zig TensorFold 1.1 engine (justincasey/TensorFold tag redsnow-native-v1.0.0).
# ENGINE=python: v1.0.2's Python engine (tag redsnow-v1.0.2), as before.
#
# Run from any machine that reaches both Sparks over key-based ssh. Rank 0 serves the OpenAI-compatible API.
#
#   ./redsnow.sh setup        preflight + install + download (+ view for python) (first time; idempotent)
#   ./redsnow.sh preflight    check both nodes: GPU, CUDA, python, disk, the link between them
#   ./redsnow.sh install      native: engine at the pinned tag, Zig 0.17, NCCL, one build; python: engine + venv + kernels
#   ./redsnow.sh download     model pack and (DRAFTS=on) the DFlash2 drafter (+ the BF16 head shard for python)
#   ./redsnow.sh view         python only: build the serve view (pack + BF16 lm_head) on both nodes
#   ./redsnow.sh start        launch both ranks and wait until the API answers
#   ./redsnow.sh stop | restart | status | smoke | logs [0|1]
#   ./redsnow.sh bench [run|conc|all]      native, on rank 0; results under bench/runs/ (default all)
#   ./redsnow.sh bench [smoke|decode|pin|long|sixcat|all]   python
#
# Configuration: environment > .env > defaults below. .env.example documents every key.
set -euo pipefail
if (( BASH_VERSINFO[0] < 4 )); then
  echo "redsnow.sh needs bash 4 or newer (this is $BASH_VERSION). On macOS: brew install bash, then run it again." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── config: the calling environment wins over .env ───────────────────────────
declare -A CALLER_ENV=()
while IFS= read -r key; do CALLER_ENV[$key]="${!key}"; done < <(compgen -e)
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key=${BASH_REMATCH[2]} value=${BASH_REMATCH[3]}
    if [[ "$value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      value=${BASH_REMATCH[1]}
    else
      value=${value%%#*}; value="${value%"${value##*[![:space:]]}"}"
    fi
    [[ -n "${!key+set}" ]] || export "$key=$value"
  done < "$ENV_FILE"
fi
for key in "${!CALLER_ENV[@]}"; do printf -v "$key" '%s' "${CALLER_ENV[$key]}"; export "$key"; done

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

# ── defaults ─────────────────────────────────────────────────────────────────
ENGINE="${ENGINE:-native}"                               # native: Zig TensorFold 1.1 (default); python: v1.0.2's engine
[[ $ENGINE == native || $ENGINE == python ]] || die "ENGINE must be native or python (got: $ENGINE)"
# what the caller or .env set (native passes only these on; native/engine.env holds the native defaults)
SET_CONTEXT="${CONTEXT:-}" SET_PARALLEL="${PARALLEL:-}" SET_KV_DTYPE="${KV_DTYPE:-}"
SET_RESERVE="${MEMORY_RESERVE_GIB:-}" SET_NCHANNELS="${NCCL_MAX_NCHANNELS:-}"
RANK0="${RANK0:-}"; RANK1="${RANK1:-}"                   # ssh targets, e.g. spark@spark-a.local
BASE_DIR="${BASE_DIR:-}"                                 # empty: $HOME/redsnow of rank 0's ssh user
MODELS_DIR="${MODELS_DIR:-}"                             # empty: $BASE_DIR/models

TF_REPO="${TF_REPO:-https://github.com/justincasey/TensorFold.git}"
if [[ $ENGINE == native ]]; then
  TF_REF="${TF_REF:-redsnow-native-v1.0.0}"
  TF_COMMIT="${TF_COMMIT:-76548777e61cb1402b81058564c34cb8d9af1a74}"   # install refuses any other commit
else
  TF_REF="${TF_REF:-redsnow-v1.0.2}"
  TF_COMMIT="${TF_COMMIT:-f5d727aec09e3d25aac6435458bd3b467b4cf952}"
fi
ZIG_URL="${ZIG_URL:-https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz}"

PACK_REPO="${PACK_REPO:-vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw}"
PACK_REV="${PACK_REV:-f6b8963784254095a54944f2963b2925fcf4b199}"
HEAD_REPO="${HEAD_REPO:-Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16}"
HEAD_REV="${HEAD_REV:-c0a7b6e3309db18e71f9c390e225594feebecafc}"
HEAD_FILE="${HEAD_FILE:-model-00001-of-00120.safetensors}"
HEAD_SHA256="${HEAD_SHA256:-730c19fa750f46a3803a478ac7e698af21431e8426c40417bc8a2136be377371}"
DRAFTER_REPO="${DRAFTER_REPO:-incoai/GLM-5.3-Flash-DFlash2}"
DRAFTER_REV="${DRAFTER_REV:-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
ACCEPT_DFLASH2_LICENSE="${ACCEPT_DFLASH2_LICENSE:-no}"   # yes: you accept CC BY-NC-ND 4.0 (non-commercial)
SIXCAT_REPO="${SIXCAT_REPO:-https://github.com/vcruz305/sixcat-eval.git}"
SIXCAT_COMMIT="${SIXCAT_COMMIT:-841f90a4843745675c771ac4b5ae9205049b42c0}"   # v0.7.0

SERVED_NAME="${SERVED_NAME:-RED-SNOW-5.3-Flash}"
HOST="${HOST:-0.0.0.0}"; PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
CONTEXT="${CONTEXT:-131072}"
KV_DTYPE="${KV_DTYPE:-fp8}"
DRAFTS="${DRAFTS:-on}"                                   # on: DFlash2 drafting; off: serial decode, no drafter
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-2}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
MASTER_PORT="${MASTER_PORT:-29551}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"

# The tuned serving path of the python engine (each measured; README "Benchmarks"). Override in .env to experiment.
# The native engine's settings live in native/engine.env.
if [[ $ENGINE == python ]]; then
  TF_GLM_MTP="${TF_GLM_MTP:-0}"                          # DFlash2 drafts every request; the MTP head stays unloaded
  TF_GLM_EXL3_LOADS="${TF_GLM_EXL3_LOADS:-nc2}"          # 16-byte no-allocate trellis loads, 2 steps in flight
  TF_GLM_HC_SPLIT="${TF_GLM_HC_SPLIT:-1}"                # prefill: row-split hyper-connection glue across ranks
  TF_GLM_PREFILL_OVERLAP="${TF_GLM_PREFILL_OVERLAP:-2}"  # prefill: next block's front under the glue exchanges
  TF_GLM_KDA_CHUNKED="${TF_GLM_KDA_CHUNKED:-1}"          # prefill: KDA prompt chunks in chunked (WY) form
  export TF_GLM_MTP TF_GLM_EXL3_LOADS TF_GLM_HC_SPLIT TF_GLM_PREFILL_OVERLAP TF_GLM_KDA_CHUNKED
fi

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
need_hosts() { [[ -n "$RANK0" && -n "$RANK1" ]] || die "set RANK0 and RANK1 (ssh targets) in .env — see .env.example"; }
host_of() { [[ $1 = 0 ]] && echo "$RANK0" || echo "$RANK1"; }

# remote RANK SCRIPT [VAR=value ...]: run SCRIPT with bash on that rank's node, the variables set first
remote() {
  local rank=$1 script=$2; shift 2
  local prelude="set -euo pipefail"$'\n' kv
  for kv in "$@"; do prelude+="$(printf '%q' "${kv%%=*}")=$(printf '%q' "${kv#*=}")"$'\n'; done
  ssh "${SSH_OPTS[@]}" "$(host_of "$rank")" bash -s <<<"$prelude$script"
}

# both SCRIPT [VAR=value ...]: run on both nodes at once, output prefixed, fail if either fails
both() {
  local script=$1; shift
  local pids=() r rc=0
  for r in 0 1; do
    ( remote "$r" "$script" RANK="$r" "$@" 2>&1 | sed -u "s/^/[rank$r] /" ; exit "${PIPESTATUS[0]}" ) & pids+=($!)
  done
  for r in 0 1; do wait "${pids[$r]}" || { log "rank $r failed"; rc=1; }; done
  return $rc
}

resolve_paths() {
  need_hosts
  if [[ -z "$BASE_DIR" ]]; then
    BASE_DIR="$(remote 0 'printf %s "$HOME"')/redsnow"
  fi
  MODELS_DIR="${MODELS_DIR:-$BASE_DIR/models}"
  SRC_DIR="$BASE_DIR/tensorfold"; VENV="$BASE_DIR/venv"; LOG_DIR="$BASE_DIR/logs"; NATIVE_DIR="$BASE_DIR/native"
  EXT_DIR="$BASE_DIR/torch_extensions"
  PACK_DIR="$MODELS_DIR/${PACK_REPO##*/}"; HEAD_DIR="$MODELS_DIR/${HEAD_REPO##*/}"
  DRAFTER_DIR="$MODELS_DIR/${DRAFTER_REPO##*/}"; SERVE_DIR="$MODELS_DIR/redsnow-serve"
}
paths_kv() {
  echo BASE_DIR="$BASE_DIR" MODELS_DIR="$MODELS_DIR" SRC_DIR="$SRC_DIR" VENV="$VENV" LOG_DIR="$LOG_DIR" \
       EXT_DIR="$EXT_DIR" PACK_DIR="$PACK_DIR" HEAD_DIR="$HEAD_DIR" DRAFTER_DIR="$DRAFTER_DIR" SERVE_DIR="$SERVE_DIR" \
       NATIVE_DIR="$NATIVE_DIR"
}

# ── the link between the Sparks ──────────────────────────────────────────────
# Lists "netdev hca ipv4/prefix mtu" for every RDMA-capable netdev with carrier on this node.
FABRIC_LIST='for d in /sys/class/net/*; do
  n=${d##*/}; [[ -d $d/device/infiniband && "$(cat "$d/carrier" 2>/dev/null)" == 1 ]] || continue
  hca=$(ls "$d/device/infiniband" | head -1)
  for a in $(ip -4 -o addr show dev "$n" | awk "{print \$4}"); do echo "$n $hca $a $(cat "$d/mtu")"; done
done'

# Sets RANK0_IF RANK0_HCA RANK0_FABRIC_IP RANK1_IF RANK1_HCA RANK1_FABRIC_IP: the first pair of RDMA netdevs
# (by HCA name) that share a subnet, unless .env pins all of them. The ports may cross (rank 0 f0 <-> rank 1 f1),
# so each rank keeps its own netdev and HCA.
detect_fabric() {
  if [[ -n "${RANK0_IF:-}" && -n "${RANK0_HCA:-}" && -n "${RANK0_FABRIC_IP:-}" && -n "${RANK1_IF:-}" \
        && -n "${RANK1_HCA:-}" && -n "${RANK1_FABRIC_IP:-}" ]]; then
    return
  fi
  local l0 l1 pair
  l0=$(remote 0 "$FABRIC_LIST") || die "cannot list rank 0's RDMA links"
  l1=$(remote 1 "$FABRIC_LIST") || die "cannot list rank 1's RDMA links"
  pair=$(python3 - "$l0" "$l1" <<'PY'
import ipaddress, sys
side = [[l.split() for l in s.splitlines() if l.strip()] for s in sys.argv[1:3]]
pairs = []
for d0, h0, a0, m0 in side[0]:
    for d1, h1, a1, m1 in side[1]:
        i0, i1 = ipaddress.ip_interface(a0), ipaddress.ip_interface(a1)
        if i0.network == i1.network and i0.ip != i1.ip:
            pairs.append((h0.lower(), d0, h0, str(i0.ip), d1, h1, str(i1.ip), m0, m1))
if pairs:
    print(*sorted(pairs)[0][1:])
PY
) || true
  [[ -n "$pair" ]] || die "no RDMA link with both ends in one subnet. Rank 0 has:
$l0
Rank 1 has:
$l1
Address the QSFP link on both nodes (README: 'The link') or pin RANK0_IF/RANK0_HCA/RANK0_FABRIC_IP/RANK1_* in .env."
  read -r RANK0_IF RANK0_HCA RANK0_FABRIC_IP RANK1_IF RANK1_HCA RANK1_FABRIC_IP MTU0 MTU1 <<<"$pair"
  [[ $MTU0 == 9000 && $MTU1 == 9000 ]] || log "warning: link MTU is $MTU0/$MTU1, not 9000 (README: 'The link')"
}

# ── commands ─────────────────────────────────────────────────────────────────
cmd_preflight() {
  resolve_paths
  log "checking $RANK0 (rank 0) and $RANK1 (rank 1)"
  both '
gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1) || true
[[ -n "$gpu" ]] || { echo "FAIL nvidia-smi: no GPU"; exit 1; }
echo "ok   gpu: $gpu"
[[ -x /usr/local/cuda/bin/nvcc ]] || { echo "FAIL /usr/local/cuda/bin/nvcc missing (DGX OS ships CUDA 13)"; exit 1; }
echo "ok   nvcc: $(/usr/local/cuda/bin/nvcc --version | sed -n "s/.*release \([0-9.]*\).*/\1/p")"
command -v python3.12 >/dev/null || { echo "FAIL python3.12 missing"; exit 1; }
python3.12 -c "import venv, ensurepip" 2>/dev/null || { echo "FAIL python3.12 venv/ensurepip missing: sudo apt install python3.12-venv"; exit 1; }
echo "ok   python3.12 + venv"
for t in git curl; do command -v $t >/dev/null || { echo "FAIL $t missing"; exit 1; }; done
mkdir -p "$MODELS_DIR"
free=$(df -B1G --output=avail "$MODELS_DIR" | tail -1 | tr -d " ")
have=$(du -s -B1G "$MODELS_DIR" 2>/dev/null | cut -f1)
echo "ok   disk: ${free} GiB free at $MODELS_DIR (models need ~200 GiB; ${have} GiB already there)"
[[ $free -ge 20 ]] || { echo "FAIL under 20 GiB free at $MODELS_DIR"; exit 1; }
mem=$(awk "/MemTotal/ {printf \"%d\", \$2/1048576}" /proc/meminfo)
echo "ok   memory: ${mem} GiB"
mine=" $(pgrep -d " " -f "tensorfol[d] serve" || true) "
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | while IFS=", " read -r pid name; do
  [[ "$mine" == *" $pid "* ]] || echo "$pid $name"; done)
[[ -z "$busy" ]] || echo "WARN other GPU processes (the model needs ~103 GiB a node): $busy"
' $(paths_kv)
  detect_fabric
  log "link: rank 0 $RANK0_IF/$RANK0_HCA $RANK0_FABRIC_IP <-> rank 1 $RANK1_IF/$RANK1_HCA $RANK1_FABRIC_IP"
  remote 0 'ping -c 2 -W 2 -I "$IF" "$PEER" >/dev/null' IF="$RANK0_IF" PEER="$RANK1_FABRIC_IP" \
    || die "rank 0 cannot ping $RANK1_FABRIC_IP over $RANK0_IF"
  log "preflight passed"
}

cmd_install() {
  resolve_paths
  log "installing TensorFold $TF_REF from $TF_REPO into $BASE_DIR on both nodes"
  both '
mkdir -p "$BASE_DIR" "$LOG_DIR" "$EXT_DIR"
if [[ -d "$SRC_DIR/.git" ]]; then
  git -C "$SRC_DIR" remote set-url origin "$TF_REPO"
  git -C "$SRC_DIR" fetch --quiet --tags origin "$TF_REF"
else
  git clone --quiet --no-checkout "$TF_REPO" "$SRC_DIR"
  git -C "$SRC_DIR" fetch --quiet --tags origin "$TF_REF"
fi
git -C "$SRC_DIR" -c advice.detachedHead=false checkout --quiet --force FETCH_HEAD
head=$(git -C "$SRC_DIR" rev-parse HEAD)
if [[ -n "$TF_COMMIT" && "$head" != "$TF_COMMIT" ]]; then
  echo "FAIL $TF_REF resolved to $head, expected $TF_COMMIT"; exit 1
fi
echo "engine at $head"
[[ -x "$VENV/bin/python" ]] || python3.12 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -c "$BASE_DIR/constraints.txt" \
  torch triton nvidia-nccl-cu13 ninja cuda-toolkit huggingface_hub
"$VENV/bin/python" -m pip install --quiet -c "$BASE_DIR/constraints.txt" -e "$SRC_DIR"
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), \"torch sees no GPU\"; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)"
echo "building CUDA kernels (first time: several minutes)"
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TORCH_EXTENSIONS_DIR="$EXT_DIR"
"$VENV/bin/python" - <<PY
import importlib, time
for m in ("tensorfold.cuda.exl3.experts", "tensorfold.cuda.exl3.linear", "tensorfold.cuda.kernels.qmm",
          "tensorfold.families.glm5_next.cuda.kda", "tensorfold.families.glm5_next.cuda.kda_chunked"):
    t = time.time(); importlib.import_module(m)._ext(); print(f"  {m}: {time.time() - t:.0f} s", flush=True)
PY
echo "install done"
' $(paths_kv) TF_REPO="$TF_REPO" TF_REF="$TF_REF" TF_COMMIT="$TF_COMMIT" < /dev/null \
    || die "install failed"
}

# copies config/constraints.txt (the measured venv's exact package set) to both nodes for install's pip
push_constraints() {
  local r
  for r in 0 1; do
    ssh "${SSH_OPTS[@]}" "$(host_of "$r")" "mkdir -p $(printf '%q' "$BASE_DIR") && cat > $(printf '%q' "$BASE_DIR/constraints.txt")" \
      < "$ROOT/config/constraints.txt" || die "cannot write constraints.txt on rank $r"
  done
}

cmd_download() {
  resolve_paths
  if [[ $DRAFTS == on && $ACCEPT_DFLASH2_LICENSE != yes ]]; then
    die "the DFlash2 drafter ($DRAFTER_REPO) is CC BY-NC-ND 4.0: non-commercial use only. Set
ACCEPT_DFLASH2_LICENSE=yes in .env to accept it, or DRAFTS=off to serve without drafting (README: 'Licenses')."
  fi
  log "downloading to $MODELS_DIR on both nodes (~200 GiB each; resumes where it stopped)"
  both '
hf="$VENV/bin/hf"; [[ -x "$hf" ]] || { echo "FAIL $hf missing: run install first"; exit 1; }
[[ -z "$HF_TOKEN" ]] || export HF_TOKEN
"$hf" download "$PACK_REPO" --revision "$PACK_REV" --local-dir "$PACK_DIR" >/dev/null
echo "pack: $(du -sh "$PACK_DIR" | cut -f1)"
if [[ $ENGINE == python ]]; then   # python: the serve view swaps in the BF16 head (native reads the pack head)
  "$hf" download "$HEAD_REPO" "$HEAD_FILE" --revision "$HEAD_REV" --local-dir "$HEAD_DIR" >/dev/null
  sum=$(sha256sum "$HEAD_DIR/$HEAD_FILE" | cut -d" " -f1)
  [[ "$sum" == "$HEAD_SHA256" ]] || { echo "FAIL head shard sha256 $sum != $HEAD_SHA256"; exit 1; }
  echo "head shard: sha256 ok"
fi
if [[ $DRAFTS == on ]]; then
  "$hf" download "$DRAFTER_REPO" --revision "$DRAFTER_REV" --local-dir "$DRAFTER_DIR" >/dev/null
  echo "drafter: $(du -sh "$DRAFTER_DIR" | cut -f1)"
fi
' $(paths_kv) PACK_REPO="$PACK_REPO" PACK_REV="$PACK_REV" HEAD_REPO="$HEAD_REPO" HEAD_REV="$HEAD_REV" \
    HEAD_FILE="$HEAD_FILE" HEAD_SHA256="$HEAD_SHA256" DRAFTER_REPO="$DRAFTER_REPO" DRAFTER_REV="$DRAFTER_REV" \
    DRAFTS="$DRAFTS" HF_TOKEN="${HF_TOKEN:-}" ENGINE="$ENGINE" < /dev/null || die "download failed"
}

cmd_view() {
  resolve_paths
  log "building the serve view $SERVE_DIR on both nodes"
  both '
"$VENV/bin/python" "$SRC_DIR/tools/serve_view.py" --pack "$PACK_DIR" --head "$HEAD_DIR/$HEAD_FILE" \
  --out "$SERVE_DIR" --expect-head-sha256 "$HEAD_SHA256"
' $(paths_kv) HEAD_FILE="$HEAD_FILE" HEAD_SHA256="$HEAD_SHA256" < /dev/null || die "view failed"
}

# the boot script both ranks run (rank-specific values passed as variables)
BOOT='
cd "$BASE_DIR"
mkdir -p "$LOG_DIR"
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TORCH_EXTENSIONS_DIR="$EXT_DIR"
export TF_NCCL_LIB=$("$VENV/bin/python" -c "import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], \"lib/libnccl.so.2\"))")
export NCCL_SOCKET_IFNAME="$IF" NCCL_IB_HCA="$HCA" NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS" NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
eval "$ENGINE_ENV"
[[ -f "$LOG_DIR/rank$RANK.log" ]] && mv -f "$LOG_DIR/rank$RANK.log" "$LOG_DIR/rank$RANK.prev.log"
setsid "$VENV/bin/python" -m tensorfold serve "$SERVE_DIR" --tp 2 --rank "$RANK" --master "$MASTER" \
  --master-port "$MASTER_PORT" --context "$CONTEXT" --kv-dtype "$KV_DTYPE" --name "$SERVED_NAME" $ARGS \
  </dev/null >"$LOG_DIR/rank$RANK.log" 2>&1 &
echo "rank $RANK launched (pid $!)"
'

stop_ranks() {
  local r
  for r in 1 0; do remote "$r" 'pkill -f "tensorfol[d] serve" || true' >/dev/null 2>&1 || true; done
  for r in 0 1; do
    remote "$r" 'for _ in $(seq 30); do pgrep -f "tensorfol[d] serve" >/dev/null || exit 0; sleep 1; done
pkill -9 -f "tensorfol[d] serve" || true' >/dev/null 2>&1 || true
  done
}

cmd_start() {
  resolve_paths
  case "$DRAFTS" in
    on) DRAFT_ARGS="--drafter $DRAFTER_DIR" ;;
    off) DRAFT_ARGS="--no-drafts" ;;
    *) die "DRAFTS must be on or off (got: $DRAFTS)" ;;
  esac
  local r
  for r in 0 1; do
    remote "$r" '[[ -x "$VENV/bin/python" ]] || { echo "no venv at $VENV: run ./redsnow.sh install"; exit 1; }
[[ -f "$SERVE_DIR/lm_head.safetensors" ]] || { echo "no serve view at $SERVE_DIR: run ./redsnow.sh view"; exit 1; }
[[ $DRAFTS != on || -f "$DRAFTER_DIR/config.json" ]] || { echo "no drafter at $DRAFTER_DIR: run ./redsnow.sh download"; exit 1; }' \
      $(paths_kv) DRAFTS="$DRAFTS" || die "rank $r is not set up"
  done
  detect_fabric
  # every TF_GLM_*, TF_MOE* and TENSORFOLD_* variable set here reaches both ranks
  local engine_env
  engine_env=$(env | grep -E '^(TF_GLM_[A-Z0-9_]+|TF_MOE[A-Z0-9_]*|TENSORFOLD_[A-Z0-9_]+)=' | sort \
               | while IFS='=' read -r k v; do printf 'export %s=%q\n' "$k" "$v"; done)
  local key_args=""; [[ -n "$API_KEY" ]] && key_args="--api-key $API_KEY"
  stop_ranks
  log "link: rank 0 $RANK0_IF/$RANK0_HCA $RANK0_FABRIC_IP <-> rank 1 $RANK1_IF/$RANK1_HCA $RANK1_FABRIC_IP"
  log "engine env: $(echo "$engine_env" | sed 's/^export //' | tr '\n' ' ')"
  remote 1 "$BOOT" $(paths_kv) RANK=1 IF="$RANK1_IF" HCA="$RANK1_HCA" MASTER="$RANK0_FABRIC_IP" \
    MASTER_PORT="$MASTER_PORT" NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS" MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB" \
    CONTEXT="$CONTEXT" KV_DTYPE="$KV_DTYPE" SERVED_NAME="$SERVED_NAME" ENGINE_ENV="$engine_env" \
    ARGS="$DRAFT_ARGS $key_args --host 127.0.0.1 --port $((PORT + 1))"
  remote 0 "$BOOT" $(paths_kv) RANK=0 IF="$RANK0_IF" HCA="$RANK0_HCA" MASTER="$RANK0_FABRIC_IP" \
    MASTER_PORT="$MASTER_PORT" NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS" MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB" \
    CONTEXT="$CONTEXT" KV_DTYPE="$KV_DTYPE" SERVED_NAME="$SERVED_NAME" ENGINE_ENV="$engine_env" \
    ARGS="$DRAFT_ARGS $key_args --host $HOST --port $PORT"
  log "waiting for the API (load takes about a minute; up to ${WAIT_TIMEOUT}s)"
  local t0=$SECONDS auth=""; [[ -n "$API_KEY" ]] && auth="-H 'Authorization: Bearer $API_KEY'"
  while (( SECONDS - t0 < WAIT_TIMEOUT )); do
    if remote 0 "curl -fsS -m 2 $auth http://127.0.0.1:$PORT/v1/models >/dev/null" >/dev/null 2>&1; then
      log "serving $SERVED_NAME at http://${RANK0#*@}:$PORT/v1 (ready in $((SECONDS - t0)) s)"
      return 0
    fi
    for r in 0 1; do
      if ! remote "$r" 'pgrep -f "tensorfol[d] serve" >/dev/null' >/dev/null 2>&1; then
        remote "$r" 'tail -25 "$LOG_DIR/rank$R.log"' LOG_DIR="$LOG_DIR" R="$r" >&2 || true
        stop_ranks
        die "rank $r exited during startup (log above: $LOG_DIR/rank$r.log)"
      fi
    done
    sleep 5
  done
  for r in 0 1; do remote "$r" 'tail -8 "$LOG_DIR/rank$R.log"' LOG_DIR="$LOG_DIR" R="$r" >&2 || true; done
  die "not healthy after ${WAIT_TIMEOUT}s"
}

cmd_stop() { need_hosts; stop_ranks; log "both ranks stopped"; }

cmd_status() {
  resolve_paths
  local r n
  for r in 0 1; do
    n=$(remote "$r" 'pgrep -f "tensorfol[d] serve" | wc -l' 2>/dev/null || echo "?")
    log "rank $r ($(host_of "$r")): $n serve process(es)"
  done
  local auth=""; [[ -n "$API_KEY" ]] && auth="-H 'Authorization: Bearer $API_KEY'"
  remote 0 "curl -fsS -m 5 $auth http://127.0.0.1:$PORT/v1/models" && echo || log "API not answering on :$PORT"
}

cmd_smoke() {
  need_hosts
  local auth=""; [[ -n "$API_KEY" ]] && auth="-H 'Authorization: Bearer $API_KEY'"
  local out
  out=$(remote 0 "curl -fsS -m 300 $auth http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"$SERVED_NAME\",\"temperature\":0,\"max_tokens\":512,\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with just the number.\"}]}'") \
    || die "request failed"
  echo "$out"
  grep -q '391' <<<"$out" && log "smoke passed (17*23 = 391)" || die "smoke: reply lacks 391"
}

cmd_logs() {
  resolve_paths
  local r=${1:-0}; [[ $r == 0 || $r == 1 ]] || die "logs takes rank 0 or 1"
  remote "$r" 'tail -n "${N:-60}" "$LOG_DIR/rank$R.log"' LOG_DIR="$LOG_DIR" R="$r" N="${N:-60}"
}

# Runs on rank 0 in $BENCH_DIR (bench/ shipped there); every step logs to $OUT/<step>.log, `all` goes on past a failure.
BENCH_SCRIPT='
cd "$BENCH_DIR"; mkdir -p "$OUT"; export API_KEY
rc=0
step() {
  local name=$1; shift
  echo "== $name"
  if "$@" 2>&1 | tee "$OUT/$name.log"; then :; else rc=1; echo "== $name FAILED"; fi
}
api=(--base "$BASE_URL" --model "$SERVED_NAME")
do_smoke()  { step smoke python3 bench_v1.py --smoke "${api[@]}" --out "$OUT/smoke.json"; }
do_decode() { step decode python3 bench_v1.py --reps "$REPS" "${api[@]}" --out "$OUT/decode.json"; }
do_pin() {
  local tok tpl eng
  tok=$(sha256sum "$SERVE_DIR/tokenizer.json" | cut -c1-64); tpl=$(sha256sum "$SERVE_DIR/chat_template.jinja" | cut -c1-64)
  eng=$(git -C "$SRC_DIR" rev-parse HEAD)
  step pin python3 bench_v1.py --ids self --ref sweep6_prompts.json --reps "$REPS" "${api[@]}" --out "$OUT/pin.json" \
    --pin "$OUT/redsnow_reference.json" --meta engine_commit="$eng" --meta pack_rev="$PACK_REV" \
    --meta drafter_rev="$DRAFTER_REV" --meta context="$CONTEXT" --meta kv_dtype="$KV_DTYPE" \
    --meta tokenizer_sha256="$tok" --meta chat_template_sha256="$tpl"
}
do_long() {
  [[ -f longctx_prompts.json ]] || step prompts "$VENV/bin/python" make_prompts.py "$SERVE_DIR" longctx_prompts.json
  step long python3 longctx_run.py "$BASE_URL" "$SERVED_NAME" longctx_prompts.json "$OUT/longctx.json" "$LONG"
}
do_sixcat() {
  local sv="$BASE_DIR/sixcat-venv"
  if [[ "$(cat "$sv/.commit" 2>/dev/null)" != "$SIXCAT_COMMIT" ]]; then
    python3 -m venv "$sv" && "$sv/bin/pip" install -q "git+$SIXCAT_REPO@$SIXCAT_COMMIT" && echo "$SIXCAT_COMMIT" > "$sv/.commit"
  fi
  # the server decodes one request at a time (C>1 only queues: flat 35.9 tok/s aggregate at C=1..8 on 2026-10-10),
  # so the curve stops at C=1 and the three profiles fit the budget
  step sixcat "$sv/bin/python" -m sixcat speed --base-url "$BASE_URL" --model "$SERVED_NAME" --api-key "${API_KEY:-none}" \
    --policy strict --thinking off --candidates 1 --max-seconds 1200 --out "$OUT/sixcat.json"
}
case $WHAT in
  all) do_smoke; do_decode; do_long; do_sixcat ;;
  *)   "do_$WHAT" ;;
esac
exit $rc'

cmd_bench() {
  resolve_paths
  local what=${1:-all}
  case $what in smoke|decode|pin|long|sixcat|all) ;; *) die "bench takes smoke, decode, pin, long, sixcat or all" ;; esac
  [[ $what =~ ^(pin|long|sixcat)$ || -f "$ROOT/bench/redsnow_reference.json" ]] \
    || die "bench/redsnow_reference.json is missing: ./redsnow.sh bench pin writes it"
  local stamp bdir out rc=0
  stamp="$(date +%Y-%m-%d-%H%M%S)-$what"; bdir="$BASE_DIR/bench"; out="$BASE_DIR/bench-runs/$stamp"
  log "bench $what on rank 0; results in bench/runs/$stamp"
  COPYFILE_DISABLE=1 tar -C "$ROOT/bench" --exclude ./records --exclude ./runs --exclude ./books \
    --exclude ./longctx_prompts.json -cf - . | ssh "${SSH_OPTS[@]}" "$RANK0" "mkdir -p '$bdir' && tar -C '$bdir' -xf -"
  remote 0 "$BENCH_SCRIPT" $(paths_kv) BENCH_DIR="$bdir" OUT="$out" WHAT="$what" REPS="${REPS:-3}" \
    LONG="${LONG:-book_128k}" BASE_URL="http://127.0.0.1:$PORT/v1" SERVED_NAME="$SERVED_NAME" API_KEY="$API_KEY" \
    PACK_REV="$PACK_REV" DRAFTER_REV="$DRAFTER_REV" CONTEXT="$CONTEXT" KV_DTYPE="$KV_DTYPE" \
    SIXCAT_REPO="$SIXCAT_REPO" SIXCAT_COMMIT="$SIXCAT_COMMIT" || rc=$?
  mkdir -p "$ROOT/bench/runs/$stamp"
  ssh "${SSH_OPTS[@]}" "$RANK0" "tar -C '$out' -cf - ." | tar -C "$ROOT/bench/runs/$stamp" -xf -
  if [[ $what == pin && $rc == 0 ]]; then
    cp "$ROOT/bench/runs/$stamp/redsnow_reference.json" "$ROOT/bench/redsnow_reference.json"
    log "pinned bench/redsnow_reference.json"
  fi
  return "$rc"
}

# ── the native engine (default; ENGINE=python serves with v1.0.2's engine) ──
cmd_install_native() {
  resolve_paths
  log "installing native TensorFold $TF_REF into $NATIVE_DIR on both nodes"
  both '
mkdir -p "$NATIVE_DIR/lib" "$NATIVE_DIR/zig" "$BASE_DIR"
src="$NATIVE_DIR/src"
[[ -d "$src/.git" ]] || git init --quiet "$src"
git -C "$src" fetch --quiet --depth 1 "$TF_REPO" tag "$TF_REF"
git -C "$src" -c advice.detachedHead=false checkout --quiet --force "$TF_REF"
head=$(git -C "$src" rev-parse "$TF_REF^{commit}")
[[ -z "$TF_COMMIT" || "$head" == "$TF_COMMIT" ]] || { echo "FAIL $TF_REF resolved to $head, expected $TF_COMMIT"; exit 1; }
echo "engine at $head"
zig="$NATIVE_DIR/zig/zig-aarch64-linux-0.17.0/zig"
[[ -x "$zig" ]] || curl -fsSL "$ZIG_URL" | tar -C "$NATIVE_DIR/zig" -xJ
echo "zig $("$zig" version)"
[[ -x "$VENV/bin/python" ]] || python3.12 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -c "$BASE_DIR/constraints.txt" huggingface_hub nvidia-nccl-cu13
nccl=$("$VENV/bin/python" -c "import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], \"lib/libnccl.so.2\"))")
ln -sf "$nccl" "$NATIVE_DIR/lib/libnccl.so.2"
echo "nccl: $nccl"
' $(paths_kv) TF_REPO="$TF_REPO" TF_REF="$TF_REF" TF_COMMIT="$TF_COMMIT" ZIG_URL="$ZIG_URL" < /dev/null \
    || die "install failed"
  log "building the engine on rank 0 (a few minutes), then copying it to rank 1"
  native_cluster build || die "build failed (log: $NATIVE_DIR/build.log on rank 0)"
}

# native/cluster.sh with this script's settings; only the context, streams, KV and reserve you set reach it
# (native/engine.env holds the native defaults: 2 streams x 274,432 tokens, fp8 KV, a 6 GiB reserve)
native_cluster() {
  local kv=(RANK0="$RANK0" RANK1="$RANK1" BASE_DIR="$BASE_DIR" MODELS_DIR="$MODELS_DIR" REMOTE="$NATIVE_DIR"
            MODEL="$PACK_DIR" DRAFTER="$DRAFTER_DIR" HOST="$HOST" PORT="$PORT" DRAFTS="$DRAFTS" API_KEY="$API_KEY"
            SERVED_NAME="$SERVED_NAME" READY_S="$WAIT_TIMEOUT")
  [[ -n $SET_CONTEXT ]] && kv+=(CONTEXT="$SET_CONTEXT")
  [[ -n $SET_PARALLEL ]] && kv+=(PARALLEL="$SET_PARALLEL")
  [[ -n $SET_KV_DTYPE ]] && kv+=(EXTRA_ARGS="--kv-cache $SET_KV_DTYPE")
  [[ -n $SET_RESERVE ]] && kv+=(TENSORFOLD_MEMORY_RESERVE_GIB="$SET_RESERVE")
  [[ -n $SET_NCHANNELS ]] && kv+=(NCCL_MAX_NCHANNELS="$SET_NCHANNELS")
  env -u CONTEXT -u PARALLEL -u EXTRA_ARGS -u TENSORFOLD_MEMORY_RESERVE_GIB -u NCCL_MAX_NCHANNELS -u ZIG_DIR \
    -u TF_SRC "${kv[@]}" "$ROOT/native/cluster.sh" "$@"
}

cmd_start_native() {
  resolve_paths
  [[ $DRAFTS == on || $DRAFTS == off ]] || die "DRAFTS must be on or off (got: $DRAFTS)"
  local r
  for r in 0 1; do
    remote "$r" '[[ -x "$NATIVE_DIR/src/zig-out/native/bin/tensorfold-native" ]] || { echo "no engine build in $NATIVE_DIR: run ./redsnow.sh install"; exit 1; }
[[ -f "$PACK_DIR/config.json" ]] || { echo "no model at $PACK_DIR: run ./redsnow.sh download"; exit 1; }
[[ $DRAFTS != on || -f "$DRAFTER_DIR/config.json" ]] || { echo "no drafter at $DRAFTER_DIR: run ./redsnow.sh download"; exit 1; }' \
      $(paths_kv) DRAFTS="$DRAFTS" || die "rank $r is not set up"
  done
  log "starting both ranks (the model loads in about 90 s)"
  native_cluster start >/dev/null || die "start failed (./redsnow.sh logs)"
  log "serving $SERVED_NAME at http://${RANK0#*@}:$PORT/v1"
}

# bench run: prefill (8K/32K + a needle), TTFT, decode (drafted == serial), C=4; bench conc: C=1/2/4/8, mixed load,
# open-loop trace (every concurrent reply equals its solo one); bench all: both. On rank 0; results in bench/runs/.
cmd_bench_native() {
  resolve_paths
  local what=${1:-all}
  case $what in run|conc|all) ;; *) die "bench takes run, conc or all (ENGINE=python: smoke, decode, pin, long, sixcat)" ;; esac
  local stamp bdir out rc=0 base="http://127.0.0.1:$PORT"
  stamp="$(date +%Y-%m-%d-%H%M%S)-native-$what"; bdir="$BASE_DIR/bench"; out="$BASE_DIR/bench-runs/$stamp"
  log "bench $what on rank 0; results in bench/runs/$stamp"
  COPYFILE_DISABLE=1 tar -C "$ROOT/bench" --exclude ./records --exclude ./runs --exclude ./books \
    --exclude ./longctx_prompts.json -cf - . | ssh "${SSH_OPTS[@]}" "$RANK0" "mkdir -p '$bdir' && tar -C '$bdir' -xf -"
  remote 0 'cd "$BDIR/native"; mkdir -p "$OUT"; export API_KEY
rc=0
if [[ $WHAT == run || $WHAT == all ]]; then python3 run.py --base "$BASE" --out "$OUT/run.json" | tee "$OUT/run.log" || rc=1; fi
if [[ $WHAT == conc || $WHAT == all ]]; then python3 conc.py --base "$BASE" --out "$OUT/conc.json" | tee "$OUT/conc.log" || rc=1; fi
exit $rc' BDIR="$bdir" OUT="$out" WHAT="$what" BASE="$base" API_KEY="$API_KEY" || rc=$?
  mkdir -p "$ROOT/bench/runs/$stamp"
  ssh "${SSH_OPTS[@]}" "$RANK0" "tar -C '$out' -cf - ." | tar -C "$ROOT/bench/runs/$stamp" -xf -
  return "$rc"
}

native() { [[ $ENGINE == native ]]; }
case "${1:-help}" in
  setup)     resolve_paths; cmd_preflight; push_constraints
             if native; then cmd_install_native; cmd_download; else cmd_install; cmd_download; cmd_view; fi
             log "setup complete: ./redsnow.sh start" ;;
  preflight) cmd_preflight ;;
  install)   resolve_paths; push_constraints; if native; then cmd_install_native; else cmd_install; fi ;;
  download)  cmd_download ;;
  view)      native && die "view is for ENGINE=python (the native engine reads the pack as downloaded)"; cmd_view ;;
  start)     if native; then cmd_start_native; else cmd_start; fi ;;
  stop)      if native; then resolve_paths; native_cluster stop; else cmd_stop; fi ;;
  restart)   if native; then cmd_start_native; else cmd_stop; cmd_start; fi ;;   # native start stops both ranks first
  status)    if native; then resolve_paths; native_cluster status; echo; else cmd_status; fi ;;
  smoke)     cmd_smoke ;;
  logs)      if native; then resolve_paths; N="${N:-60}" native_cluster logs; else cmd_logs "${2:-0}"; fi ;;
  bench)     if native; then cmd_bench_native "${2:-all}"; else cmd_bench "${2:-all}"; fi ;;
  help|-h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "unknown command: $1 (./redsnow.sh help)" ;;
esac
