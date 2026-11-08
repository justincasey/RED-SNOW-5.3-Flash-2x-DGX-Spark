# RED-SNOW-5.3-Flash on REXL3 (fork justincasey/REXL3, branch `redsnow`)

Autoresearch session 2026-11-07, recipe branch `autoresearch/session-20261107`. Harness `autoresearch.sh` (pair A
official, `PAIR=B CANDIDATE_SRC=... [CANDIDATE_ENV=...]` for a parallel candidate), primary
`target_score = (prose/70 * prefill32k/3000 * min(1,kv/2^20) * min(1,100/boot))^(1/4)`, 0 on any gate failure.

Shipped: tag `redsnow-rexl3-v1.0.0` = `49cbd0461a71`, pinned by `REXL3_REF`/`REXL3_COMMIT` in `redsnow.sh`. The
`ENGINE=rexl3 ./redsnow.sh install/start/smoke/status/stop` path was run end to end from the tag on pair B.

## Contract B (user rule, 2026-11-07; replaced "byte-exact vs the unmodified engine")
P2 prefill (chunk 4096, sequence-parallel residual), fp8 latent KV, expert width split, prompt cache on. Gates in
`bench/rexl3/bench.py`: deterministic, drafted == serial, concurrent == solo, solo == serial prefix, cached == uncached,
replies == `bench/rexl3/reference.json` (re-pinned from facec7f; `ref_checked` = 9). Nothing lossy.
Quality gate against the exact P1/fp16 path on one build: GSM8K 1,319 items 1,248 vs 1,256 (discordant 13 vs 21,
exact McNemar p = 0.23); needles 8K/32K/186K pass on both; 1,027,057-token needle passes (rank 0 min 5.21 GiB).

## Result (run 131, 49cbd04, TensorFold native v1.1.0 in brackets)
prose 38.1 [35.5], mean of 6 47.8 [44.8], serial 28.7 [27.8], prefill 8K/32K 1,614/1,685 [1,513/1,660], TTFT 0.46 s
[0.47], cached follow-up 0.39 s, C2/C4/C8 52.9/67.0/100.4 [48.9/66.2/97.8], stall 0.34 s [0.46], under-load TTFT
0.70 s [0.82], trace 25.4 s [25.8], KV 1,179,648 tokens, window 1,048,576 [274,432], boot ~37 s [~100].
Best score 0.7460 (run 130); 131 is within noise.

## Targets not met (physics)
- Prose 70 / 40: serial floor ~25-29 ms, 3-row verify window ~50 ms, draft ~3.8 ms; at ~3 tokens/round prose sits
  near 38. The oracle draft cut gives a 53.9 mean; each remaining exact lever is ~1% (see tech-debt tracker).
- Prefill 3,000 / 2,000: a 4,096-row chunk takes ~2.4 s; routed gate/up, down, rotation and combine run at 180-230
  GB/s against a 232 GB/s DRAM read ceiling, dense GEMMs at ~59% of MMA peak. 7,168-row chunks reach 1,680/1,759
  on pair B but cost the trace (27.2 s) and KV (1,048,576): opt-in only.

## Displaced deployment
DeepSeek-V4.1-Flash-1017 TP4 (`spark-ds41f`) was stopped on all four nodes for this work. Restore (plan verified as
a dry run, not executed): stop REXL3 on both pairs, then
`ssh dgx-head 'cd ~/projects/spark-ds41f && bin/spark --cluster-config config/cluster.json --shape tp4-switched cluster start --apply'`.
At hand-over pair A serves the tagged build (lock `main` held by the launcher); pair B is stopped and unlocked.
