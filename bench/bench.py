#!/usr/bin/env python3
"""Benchmark a running RED-SNOW server (stdlib only; `./redsnow.sh bench` runs it on rank 0).

Every request except the canonical prose arm starts with a fresh nonce, so it reuses no prompt cache. Greedy
decoding throughout. Decode tok/s = completion tokens / the server's own decode time (its `tensorfold` usage
block), so network and prefill time are excluded; TTFT is the server's prefill time. Drafting acceptance (tpr,
tokens per round) depends on the text being written, so speed varies with content: compare arms, not single runs.

  python3 bench.py [--url http://127.0.0.1:8080] [--model RED-SNOW-5.3-Flash] [--api-key KEY]
                   [--arms prose,prose-mix,code,json,ttft,concurrent] [--runs 3] [--json out.json]
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

PROSE = ("Write a short story about a lighthouse keeper who discovers a message in a bottle. "
         "Aim for about two hundred words of vivid prose.")
PROSE_MIX = (
    "Describe a thunderstorm rolling over a mountain village at dusk, in about two hundred words.",
    "Write a short letter from a retired sailor to his granddaughter about the sea.",
    "Explain to a curious ten-year-old why the sky is blue, in a few friendly paragraphs.",
    "Write the opening scene of a mystery novel set in a snowed-in train station.",
)
CODE = ("Write a Python module that parses an Apache combined access log from stdin and prints the ten client IPs "
        "with the most requests, their request counts and total bytes sent. Include type hints and a main guard.")
JSON = ("Return only a JSON array of 12 objects describing fictional network hosts. Each object has the keys "
        "\"hostname\", \"ipv4\", \"os\", \"open_ports\" (array of integers) and \"role\". No prose, no code fences.")
FILLER = "The harbor lights flickered over the water while merchants argued about grain prices. "


def chat(args, prompt, max_tokens):
    body = {"model": args.model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    wall = time.time() - t0
    t, u = d["tensorfold"], d["usage"]
    return {"ptk": u["prompt_tokens"], "ctk": u["completion_tokens"],
            "cached": t.get("cached", (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0,
            "ttft_s": round(t["prefill_s"], 3), "decode_s": round(t["decode_s"], 3),
            "tpr": round(t["tokens_per_round"], 2),
            "tok_s": round(u["completion_tokens"] / t["decode_s"], 1) if t["decode_s"] else 0.0,
            "wall_s": round(wall, 2)}


def nonce(state):
    state["n"] += 1
    return f"[variant {state['salt']}-{state['n']:03d}] "


def show(label, rows, key="tok_s"):
    print(f"\n{label}")
    print("  run   ptk   ctk  cached  ttft_s  decode_s   tpr   tok/s")
    for i, r in enumerate(rows, 1):
        print(f"  {i:>3} {r['ptk']:>5} {r['ctk']:>5} {r['cached']:>7} {r['ttft_s']:>7.3f} {r['decode_s']:>9.3f}"
              f" {r['tpr']:>5.2f} {r['tok_s']:>7.1f}")
    print(f"  median {key}: {statistics.median(r[key] for r in rows)}")


def decode_arm(args, state, label, prompt, max_tokens, out):
    rows = [chat(args, nonce(state) + prompt, max_tokens) for _ in range(args.runs)]
    show(label, rows)
    out[label] = rows


def canonical_prose_arm(args, out):
    """The protocol every number in this repo's history uses: '[variant 000i] ' + the lighthouse prompt (214 prompt
    tokens), 256 greedy tokens. Fixed bytes, so a second bench on the same boot reuses the prompt cache (see the
    cached column; decode speed is unaffected)."""

    rows = [chat(args, f"[variant {i:04d}] " + PROSE, 256) for i in range(1, args.runs + 1)]
    show("prose, canonical nonce-214 prompt (decode, 256 tokens)", rows)
    out["prose_canonical"] = rows


def prose_mix_arm(args, state, out):
    rows = [chat(args, nonce(state) + p, 256) for p in PROSE_MIX]
    show(f"prose, {len(PROSE_MIX)} other prompts (decode, 256 tokens)", rows)
    out["prose_mix"] = rows


def ttft_arm(args, state, out):
    for size in (1000, 4000, 16000):
        prompt = nonce(state) + FILLER * (size // 15) + "\n\nIn one sentence, what do the merchants argue about?"
        rows = [chat(args, prompt if i == 0 else nonce(state) + prompt, 24) for i in range(max(1, args.runs - 1))]
        show(f"ttft ~{size} prompt tokens", rows, key="ttft_s")
        out[f"ttft_{size}"] = rows


def concurrent_arm(args, state, out, n=4):
    prompts = [nonce(state) + PROSE for _ in range(n)]
    rows, errors = [None] * n, []

    def one(i):
        try:
            rows[i] = chat(args, prompts[i], 256)
        except Exception as e:  # report, don't hide: a failed stream fails the arm
            errors.append(repr(e))

    t0 = time.time()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0
    if errors:
        raise SystemExit(f"concurrent arm failed: {errors}")
    show(f"{n} simultaneous prose requests (CUDA serves one at a time; the rest queue)", rows)
    agg = round(sum(r["ctk"] for r in rows) / wall, 1)
    print(f"  aggregate: {sum(r['ctk'] for r in rows)} tokens in {wall:.2f} s wall = {agg} tok/s")
    out[f"concurrent_{n}"] = {"rows": rows, "wall_s": round(wall, 2), "aggregate_tok_s": agg}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="RED-SNOW-5.3-Flash")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--arms", default="prose,prose-mix,code,json,ttft,concurrent")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json", default="", help="also write every row to this file")
    args = ap.parse_args()
    state = {"salt": f"{random.randrange(16**6):06x}", "n": 0}
    out = {}
    arms = args.arms.split(",")
    chat(args, nonce(state) + "Say hi.", 8)  # warm-up: first-request graph capture stays out of the numbers
    if "prose" in arms:
        canonical_prose_arm(args, out)
    if "prose-mix" in arms:
        prose_mix_arm(args, state, out)
    if "code" in arms:
        decode_arm(args, state, "code (decode, 512 tokens)", CODE, 512, out)
    if "json" in arms:
        decode_arm(args, state, "structured JSON (decode, 512 tokens)", JSON, 512, out)
    if "ttft" in arms:
        ttft_arm(args, state, out)
    if "concurrent" in arms:
        concurrent_arm(args, state, out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
