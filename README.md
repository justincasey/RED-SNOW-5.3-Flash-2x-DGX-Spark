# RED-SNOW-5.3-Flash on two DGX Sparks

Serve [RED-SNOW-5.3-FLASH](https://huggingface.co/Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16), Blackfrost-AI's
cyber-operations finetune of GLM-5.3-Flash, from two NVIDIA DGX Sparks joined by one QSFP cable, behind an
OpenAI-compatible API. The model is vcruz305's
[EXL3 SAGE 4.91 bpw pack](https://huggingface.co/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw) (199.5 GB), split
across both GPUs by tensor parallelism (TP=2) in a [TensorFold](https://github.com/ashhart/TensorFold) fork, with
Inco AI's [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) drafter for speculative decoding.

One script sets it up and runs it:

```bash
cp .env.example .env        # set RANK0, RANK1, and the drafter license choice
./redsnow.sh setup          # check both nodes, install, download, build the serve view
./redsnow.sh start          # ~1 minute later: http://<rank0>:8080/v1
```

## Headline numbers

Measured 2026-10-10 on 2× DGX Spark (GB10, 128 GB) with this repo's defaults (engine `redsnow-v1.0.2`), by
`./redsnow.sh bench all`: vcruz305's benchmark suite, described under [Benchmarks](#benchmarks). Greedy decoding,
one request at a time. Decode tok/s counts the server's decode time only.

| Measurement | Result |
|---|---|
| Decode with DFlash2, mean of 6 real prompts (512 tokens each, thinking on) | **39.6 tok/s** (prose 31.9 … math 55.9) |
| Decode without drafting, same prompts and same replies | 24.4 tok/s |
| Time to first token, ~220-token prompt | 0.78 s (0.08–0.21 s when the prompt is repeated) |
| Time to first token, 127,712-token prompt (Moby-Dick) | 336 s (381 tok/s prefill) |
| Decode after that prompt, drafted / not drafted | 34.9 / 22.7 tok/s |
| SixCat decode p50 (synthetic prompt that drafts unusually well: a ceiling, not typical speed) | 67.4 tok/s |
| SixCat prefill p50, ~2,640-token prompt | 379 tok/s |

128K-token context (fp8 KV cache). Needle recall exact at 36K and 90K tokens. The server is ready about 60 s
after `start`. The speed depends on what the model writes: DFlash2's drafts are accepted far more often for
code and structured text than for free prose. More data under [Benchmarks](#benchmarks).

## What you need

- **Two DGX Sparks** (GB10) on DGX OS (Ubuntu 24.04 with CUDA 13.0 at `/usr/local/cuda`), with nothing else on
  their GPUs: each rank holds ~103 GiB.
- **One QSFP cable** between them, addressed on both ends ([The link](#the-link)).
- **~200 GiB free disk on each node.** Each node keeps its own copy of the model.
- `python3.12-venv` and `git` on both nodes (`sudo apt install python3.12-venv git`).
- **A control machine** (either Spark works, or a Linux or macOS box) with bash 4 or newer and python3, and
  key-based ssh to both Sparks. On macOS: `brew install bash`.
- Internet on both nodes for the first setup (PyPI, GitHub, Hugging Face).

## Install

```bash
git clone https://github.com/justincasey/RED-SNOW-5.3-Flash-2x-DGX-Spark.git
cd RED-SNOW-5.3-Flash-2x-DGX-Spark
cp .env.example .env
```

Edit `.env`:

- `RANK0` / `RANK1`: ssh targets of the two Sparks (`user@host`). Rank 0 serves the API.
- `ACCEPT_DFLASH2_LICENSE=yes` if the DFlash2 drafter's CC BY-NC-ND 4.0 terms (non-commercial) work for you.
  Otherwise set `DRAFTS=off`: the server decodes without drafting, at about 25 tok/s on prose.

Then:

```bash
./redsnow.sh setup
```

`setup` runs these steps; each can run on its own and is safe to repeat:

| Step | What it does | Time |
|---|---|---|
| `preflight` | GPU, nvcc, python, disk and memory on both nodes; finds the link and pings across it | seconds |
| `install` | clones the pinned TensorFold fork into `~/redsnow/tensorfold`, builds a venv with the exact package set in `config/constraints.txt`, compiles the CUDA kernels | ~3 min |
| `download` | the pack (pinned revision), one BF16 shard holding `lm_head` (sha256-checked), the drafter; both nodes at once, resumable | depends on your link; ~200 GiB a node |
| `view` | the serve view: the pack's files linked, its 8-bit `lm_head` swapped for the BF16 one | seconds |

Everything goes under `~/redsnow` on each node (`BASE_DIR` and `MODELS_DIR` move it).

## Run

```bash
./redsnow.sh start      # both ranks; waits until the API answers
./redsnow.sh smoke      # one request: 17*23 must come back as 391
./redsnow.sh status     # processes and /v1/models
./redsnow.sh logs 0     # tail of rank 0's log (logs 1 for rank 1)
./redsnow.sh bench      # the benchmarks behind the numbers above (~30 min; bench smoke: checks only)
./redsnow.sh stop
```

Use it like any OpenAI endpoint:

```bash
curl http://<rank0>:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "RED-SNOW-5.3-Flash",
  "messages": [{"role": "user", "content": "Explain what an SSRF vulnerability is."}],
  "max_tokens": 1024
}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://<rank0>:8080/v1", api_key="unused")  # or your API_KEY
print(client.chat.completions.create(model="RED-SNOW-5.3-Flash",
      messages=[{"role": "user", "content": "Hello"}]).choices[0].message.content)
```

The model reasons before it answers. The reasoning comes back separately as `reasoning_content`, and it counts
toward `max_tokens`, so leave room for both.

The API has no authentication unless you set `API_KEY`. Set one if anything beyond your own machines can reach
the port.

## The link

The ranks exchange activations over the QSFP cable using RoCE (NCCL). `preflight` and `start` find the link
themselves: on each node they list the RDMA network devices that have a carrier and an IPv4 address, and pick the
pair whose addresses share a subnet. The cable may join different ports on the two nodes (port 0 on one, port 1
on the other); each rank keeps its own device names.

If the link has no addresses yet, give each end one in a private subnet, with MTU 9000. Find the cabled port with
`cat /sys/class/net/enp1s0f*np*/carrier` (1 = cabled), then, for example:

```bash
# on rank 0 (use the cabled netdev's name)
sudo nmcli con add type ethernet ifname enp1s0f0np0 con-name qsfp \
  ipv4.method manual ipv4.addresses 10.100.0.1/24 802-3-ethernet.mtu 9000
# on rank 1
sudo nmcli con add type ethernet ifname enp1s0f1np1 con-name qsfp \
  ipv4.method manual ipv4.addresses 10.100.0.2/24 802-3-ethernet.mtu 9000
```

If detection picks the wrong pair, pin all six `RANK0_IF`/`RANK0_HCA`/`RANK0_FABRIC_IP`/`RANK1_*` keys in `.env`.
The rendezvous port (29551) and the link are not authenticated: keep the link private.

## Configuration

Precedence: environment, then `.env`, then the defaults in `redsnow.sh`. `.env.example` lists every key.

| Key | Default | Meaning |
|---|---|---|
| `RANK0`, `RANK1` | (required) | ssh targets; rank 0 serves the API |
| `DRAFTS` | `on` | `on`: DFlash2 drafting; `off`: serial decode, no drafter download |
| `ACCEPT_DFLASH2_LICENSE` | `no` | must be `yes` to download the drafter |
| `SERVED_NAME` | `RED-SNOW-5.3-Flash` | model id clients send |
| `HOST`, `PORT` | `0.0.0.0`, `8080` | rank 0's API address; rank 1 takes `PORT+1` on localhost |
| `API_KEY` | (empty) | require this bearer key |
| `CONTEXT` | `131072` | prompt + reply window, in tokens |
| `KV_DTYPE` | `fp8` | latent KV cache: `fp8` (e4m3, per-token scales) or `bf16` |
| `BASE_DIR`, `MODELS_DIR` | `~/redsnow`, `~/redsnow/models` | install and model locations, the same on both nodes |
| `MEMORY_RESERVE_GIB` | `2` | memory each rank leaves free |
| `NCCL_MAX_NCHANNELS` | `4` | fewer NCCL channels leave ~1.5 GiB more for the cache |
| `WAIT_TIMEOUT` | `900` | seconds `start` waits for the API |

The engine settings below are the measured serving path. They are on by default, and `.env` can override each one.
Any other `TF_GLM_*`, `TF_MOE*` or `TENSORFOLD_*` variable you set reaches both ranks too.

| Engine setting | Default | Effect |
|---|---|---|
| `TF_GLM_EXL3_LOADS` | `nc2` | MoE decode kernel reads expert weights with 16-byte no-allocate loads, two steps ahead: +4.5% decode |
| `TF_GLM_HC_SPLIT` | `1` | prefill: the hyper-connection glue of a prompt chunk is split between the ranks |
| `TF_GLM_PREFILL_OVERLAP` | `2` | prefill: the next block's front runs under the glue exchanges |
| `TF_GLM_KDA_CHUNKED` | `1` | prefill: KDA prompt chunks in chunked (WY) form. The three prefill settings together give 6–9% faster TTFT at 4K–16K |
| `TF_GLM_MTP` | `0` | the checkpoint's own MTP head stays unloaded; DFlash2 drafts every request |

## Native engine (TensorFold 1.1, in development)

A second serving path runs RED-SNOW on native TensorFold 1.1 (Zig + CUDA) instead of the Python engine above. Engine:
[`justincasey/TensorFold` tag `redsnow-native-v1.0.0`](https://github.com/justincasey/TensorFold/tree/redsnow-native-v1.0.0)
(`76548777`: TensorFold v1.1.0 plus this project's 55 commits; development continues on branch `redsnow-native`).
Against v1.0.2 on the same pair it measured:
- prefill 1,507 / 1,657 tok/s at 8K / 32K;
- decode 45.1 tok/s drafted;
- 8 concurrent chats at 99 tok/s in total;
- boot in ~90 s.

```bash
git clone -b redsnow-native-v1.0.0 https://github.com/justincasey/TensorFold.git ~/tf-native
export TF_SRC=~/tf-native RANK0=<rank 0 ssh host> RANK1=<rank 1 ssh host> RANK1_LAN=<rank 1 LAN address>
native/cluster.sh setup && native/cluster.sh sync && native/cluster.sh build && native/cluster.sh start
python3 bench/native/run.py --base http://<rank 0>:8080          # prefill, TTFT, decode, C=4
python3 bench/native/conc.py --base http://<rank 0>:8080         # C sweep, mixed load, open-loop trace
```

The launcher expects:
- the model at `/home/spark/models/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw`;
- the drafter at `/home/spark/models/GLM-5.3-Flash-DFlash2`.

`MODEL`, `DRAFTER` and `REMOTE` override these paths. `redsnow.sh` is unchanged.

## Streams vs context (native engine)

The native TensorFold 1.1 port (`native/cluster.sh`, in development) sets aside a fixed number of streams (requests
decoding at once), each with its own context window. Every stream's KV cache is reserved at boot, so more streams
means a shorter window per stream. Memory that the streams leave free becomes the prompt cache, which holds earlier
prompts so that chat follow-ups skip prefill.

`native/engine.env` sets the default, 2 streams with a 274,432-token window (67 × 4,096; the model allows
1,048,576). Override it for one boot:

```bash
PARALLEL=8 CONTEXT=65536 native/cluster.sh -p A start
```

Measured on pair A (2026-10-20, engine `bf010924`, fp8 KV, 6 GiB reserve per rank). Values come from rank 0's
startup line. The short TTFT is a ~220-token chat prompt on an idle server. Boot time does not depend on the stream
count: it is the checkpoint read.

| Streams | Context per stream | GiB per stream | Model + buffers per rank | Prompt cache | Short TTFT |
|---|---|---|---|---|---|
| 2 (default) | 274,432 | 2.02 | 100.19 GiB | 2.6 GiB | 473 ms |
| 4 | 131,072 | 1.09 | 99.19 GiB | 2.4 GiB | 474 ms |
| 8 | 65,536 | 0.66 | 98.73 GiB | 1.4 GiB | 476 ms |

With 2 streams, a third request waits until a stream frees up. Requests are not refused. Measured effects of the
stream count:

- At 4 streams, a short chat sent while 4 others decoded and a 32K prompt filled waited 42.9 s.
- At 8 streams, the same chat got its first token in about 0.7 s.
- At 8 streams, 8 chats started together decode at 99 tok/s in total, 2.2× one stream.

For chat-heavy or multi-user serving, use 4 or 8 streams. Use 2 when single requests need windows past 128K.
`autoresearch.sh` always benchmarks 8 streams at 65,536, whatever `engine.env` says.

## How it is put together

| Piece | What |
|---|---|
| Model | GLM-5.3-Flash architecture: 45 layers (3 dense + 42 MoE, 288 experts, top-8), hybrid KDA linear attention and DSA sparse attention, 1M native context |
| Weights | RED-SNOW EXL3 SAGE MixedK, 4.91 bpw, mul1 codebook, 36 shards; the `lm_head` comes from the BF16 release, pinned by revision and sha256 |
| Engine | TensorFold 0.6.5 + the GLM-5.3 TP=4 CUDA work of `vcruz305/TensorFold` `glm53-tp4-spark` + this project's branch `redsnow-v1` ([justincasey/TensorFold](https://github.com/justincasey/TensorFold), tag `redsnow-v1.0.2`) |
| Split | TP=2, one rank a Spark. Rank 0 runs the API and the rendezvous; NCCL over RoCE on the QSFP link |
| Decode | DFlash2 chain drafting with the engine's adaptive policy; drafted rounds are verified by the target in captured CUDA graphs (up to 8 rows), so drafting changes speed, not the reply: a drafted reply equals the `"draft": false` one token for token; a streamed reply arrives one decode round at a time |
| KV cache | DSA latent cache in fp8 (e4m3, per-token scale); the indexer keys dominate the cache cost (~9.7 KiB a token) |
| Memory | ~102.6 GiB a rank at 128K context, of ~112 GiB usable |

The `redsnow-v1` branch adds to TensorFold 0.6.5 + `glm53-tp4-spark`:

- dense-EXL3 support for the RED-SNOW SAGE pack: split rules, composite projections, half-step bit widths, the
  mul1 codebook, the serve view tool, and fixes for fused groups, trellis words, tail blocks and scratch
- the fp8 (and q4) latent KV cache
- MiaAI-Lab's decode-round windows, KDA chunked prefill, HC row split, prefill overlap, pool-score kernel, DFlash2
  stop rules, and the 16-byte decode loads and expert launch order (ports; see [Credits](#credits))
- off by default and not part of this recipe's path: an fp8 MoE prefill route, tree drafting, a decode profiler
  trace

## Benchmarks

### v1.0.2 baseline (2026-10-10, this repo's defaults)

`./redsnow.sh bench all` on a fresh boot, run on rank 0. The suite is adapted from vcruz305's
[GLM-5.3 recipe](https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe) (`bench/`, MIT) and
[SixCat](https://github.com/vcruz305/sixcat-eval) v0.7.0. Raw output:
[`bench/records/2026-10-10-v1.0.2-baseline/`](bench/records/2026-10-10-v1.0.2-baseline/).

| Step | What it runs |
|---|---|
| `bench smoke` | `/v1/models`, `/health`, `/tokenize` against the pinned prompt ids, thinking on/off channel split and markup leaks, the reply streaming round by round, drafted == `"draft": false` on a short reply, DFlash2 as the default drafter, an invalid `tf_policy` refused |
| `bench decode` | 6 prompts (thinking on, greedy, 512 tokens): one `"draft": false` run and 3 drafted runs each (the first prefills the prompt, the others resume it from the prompt cache); every reply must equal the serial ids pinned in `bench/redsnow_reference.json` |
| `bench long` | a 127.5K-token Project Gutenberg prompt (sha256-pinned book), not drafted then drafted: TTFT, prefill, decode, and drafted == not drafted |
| `bench sixcat` | SixCat `speed --policy strict --thinking off`: decode, balanced and prefill profiles at one request in flight |
| `bench pin` | writes a new `bench/redsnow_reference.json` |

| Prompt | Prompt tokens | Drafted tok/s (median of 3) | Tokens per round | TTFT | Not drafted tok/s |
|---|---|---|---|---|---|
| code | 221 | 39.71 | 3.01 | 0.78 s | 24.62 |
| prose | 214 | 31.93 | 2.19 | 0.77 s | 24.61 |
| math | 228 | 55.85 | 4.71 | 0.81 s | 24.55 |
| chat_explain | 242 | 34.81 | 2.56 | 0.85 s | 24.54 |
| chat_multiturn | 342 | 35.14 | 2.42 | 1.12 s | 24.53 |
| long_doc | 3,011 | 39.95 | 3.13 | 8.24 s | 23.79 |
| **mean** | | **39.57** | | | **24.44** |

Every drafted reply equals the `"draft": false` reply (31/31 checks). The TTFT column is the first request; a
repeat of the same prompt resumes from the prompt cache in 0.08–0.21 s.

| Long prompt (127,712 tokens) | TTFT | Prefill | Decode (512 tokens) |
|---|---|---|---|
| not drafted | 334.3 s | 382.0 tok/s | 22.73 tok/s |
| drafted (same reply) | 335.6 s | 380.6 tok/s | 34.93 tok/s (2.65 tokens per round) |

| SixCat profile (one request in flight) | p50 |
|---|---|
| decode, synthetic word-list prompt, up to 512 tokens | 67.37 tok/s |
| prefill, ~2,640-token prompt | 378.6 tok/s, TTFT 6.98 s |
| balanced, ~400-token prompt, 128 tokens | 36.17 tok/s aggregate, TTFT p95 1.67 s |

Quote the 6-prompt mean as the speed a user sees. SixCat's decode prompt drafts far better than real text, so its
number is a ceiling. An earlier SixCat run with 1, 2, 4 and 8 requests in flight measured the same ~35.9 tok/s
aggregate at each level: requests queue, so `bench sixcat` stops the concurrency curve at one.

The pinned reference is a token-for-token regression check. A change meant to keep outputs (decode-only work) must
reproduce it. A change that alters prefill numerics re-pins with `bench pin`, only after the needle test and
`bench smoke` pass, and its commit says so.

v1.0.1 ([record](bench/records/2026-10-10-v1.0.1-baseline/)) measured a 41.43 tok/s mean, but its drafted replies
were not the `"draft": false` ones: they parted at token 0 to 94. They were other texts, and these happened to draft
better. v1.0.2 makes drafted and serial replies agree. A decode round costs the same (code 76 ms vs 78 ms,
chat_multiturn 69 ms vs 72 ms), and on the six prompts' serial texts DFlash2 accepts fewer tokens per round.

### v1.0.0 run (2026-10-10, this repo's defaults)

Measured with v1.0.0's `bench/bench.py` (replaced by the suite above in v1.0.1). One warm-up request first, then
every request starts with a fresh nonce, except the canonical prose arm, which uses the fixed prompt that every
earlier number in this project used.

| Arm | Runs (decode tok/s) | Tokens per round | TTFT |
|---|---|---|---|
| Prose, canonical prompt (214 tokens → 256) | 35.9 / 36.9 / 36.9 | 2.55 / 2.63 / 2.60 | 0.77 s |
| Prose, four other prompts (→ 256) | 31.9 / 31.2 / 36.7 / 32.7 | 2.02 / 2.16 / 2.63 / 2.20 | 0.75–0.83 s |
| Code (231 tokens → 512) | 46.3 / 50.9 / 47.1 | 3.68 / 4.12 / 3.81 | 0.82 s |
| JSON array (240 tokens → 512) | 48.7 / 49.3 / 52.6 | 3.87 / 3.96 / 4.15 | 0.83 s |
| TTFT ~1,200 tokens | | | 3.42 / 3.43 s (350 tok/s) |
| TTFT ~4,200 tokens | | | 10.98 / 10.89 s (384 tok/s) |
| TTFT ~16,200 tokens | | | 39.55 / 39.57 s (410 tok/s) |
| 4 simultaneous prose requests | 33.0 / 32.2 / 32.7 / 34.9 each | | 1,024 tokens in 34.0 s = 30.1 tok/s aggregate |

A second boot gave prose 30.5–33.1 and code 47.7–50.4 with other nonces, and JSON 49.6–51.8. CUDA serves one
request at a time, so concurrent requests queue. Each one is answered correctly, but there is no aggregate gain.

Needle recall (a code buried in filler, greedy): exact at 90,210 tokens (prefill 215 s, 419 tok/s) and at 36,210
tokens.

### Engine settings, A/B

Each arm is a fresh boot. The protocol is the canonical prose prompt (3 runs), plus ~16K-token prompts for TTFT.

| Arm | Prose tok/s | TTFT ~16K |
|---|---|---|
| Baseline (fp8 KV, no decode loads) | 33.4 / 35.5 / 35.9 | — |
| `TF_GLM_EXL3_LOADS=nc2` | 36.0 / 37.2 / 37.4 (**+4.5%**) | — |
| `TF_GLM_EXL3_LOADS=nc4` | 35.7 / 36.9 / 37.0 | — |
| `TF_GLM_EXL3_SWZ=2` (expert launch order) | 34.9 / 35.8 / 36.0 (noise) | — |
| `TF_GLM_L2PF=bulk` (L2 prefetch; removed from v1) | 35.2 / 36.2 / 36.2 (noise) | — |
| nc2, prefill settings off | 36.1 / 37.4 / 37.3 | 42.0 / 42.4 / 42.4 s |
| nc2, prefill settings on (v1) | 35.9 / 36.8 / 37.0 | 39.3 / 39.6 / 39.6 s (**−6%**) |

Every arm passed the layer parity check (EXL3 groups, both ranks, against a float64 reference) before it was
benchmarked.

### KV cache precision and capacity

| KV | Prose tok/s | Memory at 128K | Notes |
|---|---|---|---|
| bf16 | 31.2 (tpr 2.16) | ~103.4 GiB | needle exact at 39K |
| fp8 (v1) | 33.4–33.8 (tpr 2.36) | 102.55 GiB | needle exact at 39K (same tokens per round as bf16) and at 90K |

Before the nc2 and prefill work. The engine also has a q4 latent cache (not part of this recipe's path).
Measured startup estimates, out of ~113.5 GiB:

| KV | Window | GiB a rank |
|---|---|---|
| q4 | 192K | 102.80 |
| q4 | 256K | 103.39 |
| fp8 | 256K | 104.08 |
| q4 | 512K | 105.76 |
| q4 | 1M (native) | 110.50 |

Each extra token costs ~9.7 KiB, mostly indexer keys and scratch. The latent itself is 2.75 KiB of that, so fp8
and q4 differ little until the window gets very large.

### Where decode time goes

One drafted round takes ~67 ms of wall time: ~4.8 ms drafting, and the rest is the target forward on both ranks.
The GPU kernels inside it sum to 59.1 ms (torch profiler trace, fp8 KV, before nc2):

| Kernel | ms a round | Share |
|---|---|---|
| MoE routed experts (EXL3 trellis, 78 launches) | 22.2 | 38% |
| Dense and shared linears (EXL3) | 14.7 | 25% |
| Per-head attention (`_bmm`) | 4.6 | 8% |
| NCCL all-gather | 2.5 | 4% |
| Rotation glue | 2.3 | 4% |
| Everything else | ~13 | 22% |

A read probe on the loaded server reached ~200–230 GB/s. The expert launches stream weights near that rate, so
decode is bound by memory bandwidth at 4.91 bpw, not by kernel launches. Each extra verify row costs ~6 ms.
Faster decode needs higher draft acceptance or fewer weight bytes; fusing kernels only removes 1–3 ms of gaps.

### Drafting

| Policy | Prose tok/s | Tokens per round |
|---|---|---|
| No drafting (`DRAFTS=off`) | ~25 | 1 |
| Adaptive DFlash2 chain (v1) | 30–37 | 2.0–2.6 |
| MTP head only, depth 1 / 2 / 3 | 30.2 / 31.5 / 30.1 | 1.67 / 1.99 / 2.16 |
| DFlash2 chain, depth 5, confidence cut 0.25 / 0.35 | 30.2–30.5 / 31.7 | 2.18 / 2.12 |
| Noise-aware stop rule `fnc7:0.3` | 32.8–32.9 | 1.99 |
| Tree drafting, 2 branches × depth 2 / 3 | 23.3 / 20.2 | 2.14 / 2.32 |

Measured 2026-10-09/10, before nc2, with an older prose prompt that runs a few tok/s below the canonical one, so
compare rows, not these numbers against the v1 run. DFlash2's side branches are accepted far less often than its
main chain, so tree drafting loses to the wider verify windows it needs. NCCL channel counts (2/4/8), LL128 and
NVLS measured flat.

### Prefill

Prefill is ~60% MoE, ~23% KDA and ~14% DSA (profiled 2.7K-token chunk). The three prefill settings cut 16K TTFT
by 6% (A/B above). The engine's optional fp8 MoE prefill route (`--prefill-fp8`, not in v1) was a further ~5%
faster at 16K in a same-day A/B (453 vs 430 tok/s), slower under 2,048 tokens, and lowered drafting acceptance a
little at long context; v1 keeps bf16 prefill.

### For reference

MiaAI-Lab's vLLM stack on the same hardware and model family (GLM-5.3-Flash EXL3, TP=2) reports 27.1 tok/s on
prose, 35–44 on code and structured output, and up to 65 on highly predictable structured output.

## Known limits

- One request decodes at a time. Concurrent requests queue (each is answered correctly).
- Prose decode depends on the text: 30–37 tok/s across the prompts above.
- A stop string ends the streamed text at once, but the request finishes only when decoding reaches an end token or
  `max_tokens`: both ranks decode to the end.
- The `test_exl3_experts` bit-identity checks fail identically on the pre-graft tree (last-ulp differences in the
  grouped kernel, present before the nc2 work).
- Each start logs `CUDA extension … waits on the build lock …` lines for kernels already built. When the server
  comes up, they are harmless.

## Credits

**Model**

- [GLM-5.3-Flash](https://huggingface.co/zai-org) by Z.AI (zai-org), MIT License: the base model.
- [RED-SNOW-5.3-FLASH-BF16](https://huggingface.co/Blackfrost-AI/RED-SNOW-5.3-FLASH-BF16) by
  [Blackfrost-AI](https://huggingface.co/Blackfrost-AI), MIT License: the finetune. Its card asks that it be used by
  experienced operators in environments they own or are explicitly authorized to assess, with a human operator
  kept in control of scope, target selection, credentials, tool execution and destructive actions. This recipe
  passes that on: read the card before you deploy.
- [RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw](https://huggingface.co/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw) by
  [vcruz305](https://huggingface.co/vcruz305) (Victor Cruz), with Blackfrost-AI, MIT License (copied from the BF16
  release, Copyright 2026 Z.AI Co., Ltd.): the quantized pack this recipe serves.
- [GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) by [Inco AI](https://inco.ai),
  CC BY-NC-ND 4.0: the drafter. Cite as the card asks:

  ```bibtex
  @misc{inco2026dflash2, title={{DFlash 2: Keep Drafting Parallel}}, author={{Inco AI}}, year={2026},
        month={August}, url={https://inco.ai/blog/dflash2/}}
  @inproceedings{chen2026dflash, title={{DFlash: Block Diffusion for Flash Speculative Decoding}},
        author={Chen, Jian and Liang, Yesheng and Liu, Zhijian}, booktitle={ICML}, year={2026}}
  ```

**Engine**

- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and the TensorFold contributors, Apache-2.0
  (code before 0.6.0 also MIT): the engine.
- [vcruz305/TensorFold](https://github.com/vcruz305/TensorFold) `glm53-tp4-spark`: the GLM-5.3 TP=4 CUDA engine
  by drowzeys (prompt experts, decode linears, concurrency, sequence-parallel prefill, NCCL grouping) and fixes by
  vcruz305. The fork carries these commits unchanged, under their original authors.
- [MiaAI-Lab's GLM-5.3-Flash EXL3 TensorFold patch series](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  by MiaAI-Lab and contributors, Apache-2.0. Ported designs: 0010 (HC row split), 0012/0039/0014 (KDA chunked
  prefill), 0013 (decode rounds, kv_b formats), 0018 (DFlash2 noise-aware stop rules), 0033 (prefill overlap),
  0086 (pool scores), 0047 (decode loads) and 0090 (expert launch order). The last two are adapted from
  [Jay Leaton's glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache-2.0).
  Per-file detail: the fork's `THIRD_PARTY_NOTICES.md`.
- [MiaAI-Lab's GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
  vLLM recipe: the reference for this stack's layout and numbers. No code from it is included.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp, MIT: the EXL3 format. TensorFold's
  decoders are separate implementations checked bit for bit against it. vcruz305's ExLlamaV3 fork was the parity
  oracle for this port.
- [z-lab/dflash](https://github.com/z-lab/dflash), MIT: the DFlash drafter architecture.

**Benchmarks**

- `bench/bench_v1.py`, `bench/longctx_run.py`, `bench/make_prompts.py` and the six prompts in
  `bench/sweep6_prompts.json` are adapted from vcruz305's
  [GLM-5.3-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe) (`ed2f31d`), MIT,
  © 2026 Victor Cruz ([notice](bench/LICENSE-vcruz305)). Each file's header lists the changes.
- [SixCat](https://github.com/vcruz305/sixcat-eval) v0.7.0 by Victor Cruz, MIT: `bench sixcat` installs it on
  rank 0 in its own venv.

## Licenses

This repository's scripts and documentation: Apache-2.0 ([LICENSE](LICENSE)), except the benchmark files adapted from
vcruz305's recipe, which stay MIT ([notice](bench/LICENSE-vcruz305)). The engine fork: Apache-2.0, with
the MIT notice for TensorFold code written before 0.6.0, and third-party notices in its `THIRD_PARTY_NOTICES.md`.
The model files are not part of this repository. `download` fetches them from Hugging Face under their own
licenses: MIT for the RED-SNOW pack and the BF16 release, and **CC BY-NC-ND 4.0 for the DFlash2 drafter** (research
and non-commercial use; commercial licensing: contact@inco.ai). With `DRAFTS=off` the drafter is never
downloaded.
