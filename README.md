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

Measured 2026-10-10 on 2× DGX Spark (GB10, 128 GB) with this repo's defaults, by `./redsnow.sh bench`. Greedy
decoding, single request. Decode tok/s counts the server's decode time only.

| Workload | Decode tok/s | Tokens per round |
|---|---|---|
| Code (Python module, 512 tokens) | **46–51** | 3.7–4.1 |
| Structured output (JSON array, 512 tokens) | **49–53** | 3.9–4.2 |
| Prose (stories, letters, explanations; 256 tokens) | **30–37** | 2.0–2.6 |

| Prompt length | Time to first token |
|---|---|
| ~200 tokens | 0.77 s |
| ~1,200 tokens | 3.4 s |
| ~4,200 tokens | 10.9 s |
| ~16,200 tokens | 39.6 s |

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
./redsnow.sh bench      # the benchmark behind the numbers above
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

## How it is put together

| Piece | What |
|---|---|
| Model | GLM-5.3-Flash architecture: 45 layers (3 dense + 42 MoE, 288 experts, top-8), hybrid KDA linear attention and DSA sparse attention, 1M native context |
| Weights | RED-SNOW EXL3 SAGE MixedK, 4.91 bpw, mul1 codebook, 36 shards; the `lm_head` comes from the BF16 release, pinned by revision and sha256 |
| Engine | TensorFold 0.6.5 + the GLM-5.3 TP=4 CUDA work of `vcruz305/TensorFold` `glm53-tp4-spark` + this project's branch `redsnow-v1` ([justincasey/TensorFold](https://github.com/justincasey/TensorFold), tag `redsnow-v1.0.0`) |
| Split | TP=2, one rank a Spark. Rank 0 runs the API and the rendezvous; NCCL over RoCE on the QSFP link |
| Decode | DFlash2 chain drafting with the engine's adaptive policy; drafted rounds are verified by the target in captured CUDA graphs (up to 8 rows), so drafting changes speed, not the reply |
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

### v1 run (2026-10-10, this repo's defaults)

`./redsnow.sh bench` on rank 0. One warm-up request first, then every request starts with a fresh nonce, except
the canonical prose arm, which uses the fixed prompt that every earlier number in this project used.

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
- Two engine test groups fail identically on the pre-graft tree, so the graft did not introduce them:
  `test_exl3_experts` bit-identity checks (last-ulp differences in the grouped kernel, present before the nc2
  work) and three `test_glm_tree_draft` round tests (tree drafting is off).
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

## Licenses

This repository's scripts and documentation: Apache-2.0 ([LICENSE](LICENSE)). The engine fork: Apache-2.0, with
the MIT notice for TensorFold code written before 0.6.0, and third-party notices in its `THIRD_PARTY_NOTICES.md`.
The model files are not part of this repository. `download` fetches them from Hugging Face under their own
licenses: MIT for the RED-SNOW pack and the BF16 release, and **CC BY-NC-ND 4.0 for the DFlash2 drafter** (research
and non-commercial use; commercial licensing: contact@inco.ai). With `DRAFTS=off` the drafter is never
downloaded.
