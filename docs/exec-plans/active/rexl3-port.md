# RED-SNOW-5.3-Flash on REXL3 (fork justincasey/REXL3, branch `redsnow`)

Autoresearch session 2026-11-07, recipe branch `autoresearch/session-20261107`. Harness `autoresearch.sh` (pair A
official, `PAIR=B CANDIDATE_SRC=...` for a parallel candidate), primary
`target_score = (prose/70 * prefill32k/3000 * min(1,kv/2^20) * min(1,100/boot))^(1/4)`, 0 on any gate failure.

## Contract (user rule, 2026-11-07)
Every reply byte-exact against the unmodified engine (REXL3 478d7da, P1 prefill, fp16 latent KV):
`bench/rexl3/reference.json` pins the six serial replies, the 8K/32K prompts' 128-token continuations and the needle
reply; `bench.py` also requires drafted == serial, concurrent == solo, solo == serial prefix (`ref_checked` = 9).
P2 prefill, fp8 KV and any reassociated sum are out, however small their quality cost.

## Targets (TensorFold native v1.1.0 in brackets)
- C1 prose decode >= 70 tok/s [35.5]; mean of 6 [44.8]; serial [27.8]
- cold 32K prefill >= 3000 tok/s [1,660]
- KV pool >= 1,048,576 tokens (fp16) without thinning rank 0's headroom
- boot < 100 s [~100]; C8 [97.8 tok/s, 2.1 s TTFT]; trace mean [25.8 s]; TTFT [0.47 s]

## Physics
- Decode: serial floor ~25-29 ms, 3-row verify window ~37-42 ms, +8 ms per extra row on two Sparks, so prose at
  ~2.05 tokens/round caps near 41-45 tok/s exactly. 70 tok/s needs a lossy change.
- Prefill under P1: decode-identical KDA recurrence and 128-row MLA slabs bound it well below 3,000 tok/s.

## Merged into `redsnow` (all byte-exact under P1/fp16)
GLM DFlash2 drafter (q4 at load, rank 0), exact decode levers (PDL, host uploads, unified MoE launches, fused
router, graphed draft round, stacked drafter linears, GEMV cp.async ring, dense MLP split, pipelined small linears),
TensorFold-style expert width split, multi-prompt prefill forward + layer-part forward (concurrency), planner
reserve knob. P2/fp8 code is present but not selected.

## Lanes
| Lane | Branch | Pair | State |
|---|---|---|---|
| Memory + decode | redsnow-decmem | A | rank-0 headroom (2.30 GiB min today, target >= 4 GiB), then KV >= 1M, then decode |
| P1 prefill | redsnow-prefill3 | B | parallel plan + row-split width finish (783/795 tok/s on B, pre-merge); next: layer-44 dead tail |

## Open issues
- `PREFIX_CACHE=off` in `rexl3/engine.env` ships to `ENGINE=rexl3` users (multi-turn chats re-prefill the history).
  Ship it on with a cached == uncached gate, or document the regression.
- The 15 ms arrival wait adds ~14 ms TTFT on an idle server.
- DFlash2 at C=2 (46.4 tok/s) is slower than C=1 (50.6).
- 1M max-model-len: run one near-pool prompt while watching MemAvailable on both ranks before shipping.
- The pair lock is a `mkdir` lock (stale after a crash); REXL3's own convention is `flock` on /tmp/rexl3-gpu.lock.

## Displaced deployment
DeepSeek-V4.1-Flash-1017 TP4 (container `spark-ds41f` + unit `spark-ds41f-memguard`) was stopped on all four nodes
on 2026-11-07 for this work. The restore command must be verified before it is handed over.
