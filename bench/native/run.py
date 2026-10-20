#!/usr/bin/env python3
"""RED-SNOW-5.3-Flash on native TensorFold: one fixed workload, printed as METRIC lines (stdlib only).

Phases, in order, on a freshly started server (empty prompt cache):
  prefill   pinned 8K and 32K Moby-Dick slices, cold, 8 tokens out: tok/s = prompt tokens / server prefill_seconds
  ttft      a ~220-token chat prompt cold, then again (cached prefix): client time to the first streamed token
  decode    bench/sweep6_prompts.json, greedy, 512 tokens, drafted and "draft": false; replies must hash equal
  conc      4 different decode prompts at once, 256 tokens each: completion tokens / wall seconds

Exit 1 when any reply errors or a drafted reply differs from its serial one (token_sha).
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def stream(base, body, timeout=3600):
    """POST a streamed chat completion; returns client timings and the final runtime fields."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    out = {"usage": None, "tensorfold": None, "speculative": None, "error": None}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if "error" in ev:
                out["error"] = ev["error"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    first = time.perf_counter()
            for k in ("usage", "tensorfold", "speculative"):
                if ev.get(k):
                    out[k] = ev[k]
    out["ttft_s"] = (first or time.perf_counter()) - t0
    out["wall_s"] = time.perf_counter() - t0
    return out


def need(cond, what):
    if not cond:
        print(f"FAIL {what}", file=sys.stderr)
        sys.exit(1)


def greedy(p, max_tokens=None, **extra):
    b = {"model": "RED-SNOW-5.3-Flash", "messages": p["messages"], "temperature": 0,
         "max_tokens": max_tokens or p.get("max_tokens", 512)}
    b.update(extra)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", "http://192.168.1.123:8080"))
    ap.add_argument("--out", default="")
    ap.add_argument("--phases", default="prefill,ttft,decode,conc")
    a = ap.parse_args()
    phases = set(a.phases.split(","))
    rec = {"base": a.base, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    m = {}

    decode_prompts = json.load(open(os.path.join(HERE, "..", "sweep6_prompts.json")))["prompts"]
    r = stream(a.base, greedy(decode_prompts[0], max_tokens=8))  # warm-up: kernels, allocations, first graphs
    need(r["error"] is None, f"warm-up: {r['error']}")

    if "prefill" in phases:
        rates = []
        for p in json.load(open(os.path.join(HERE, "prefill_prompts.json")))["prompts"]:
            r = stream(a.base, greedy(p))
            need(r["error"] is None and r["tensorfold"] and r["usage"], f"{p['name']}: {r['error']}")
            n = r["usage"]["prompt_tokens"]
            cached = (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            need(cached == 0, f"{p['name']}: cached {cached} on a cold prompt")
            ps = r["tensorfold"].get("prefill_seconds") or r["ttft_s"]
            rate = n / ps
            rates.append(rate)
            key = p["name"].replace("prefill_", "")
            m[f"prefill_{key}_tok_s"] = rate
            m[f"ttft_{key}_s"] = r["ttft_s"]
            rec[p["name"]] = {"prompt_tokens": n, "prefill_seconds": ps, "ttft_s": r["ttft_s"]}
        m["prefill_tok_s"] = math.exp(sum(math.log(x) for x in rates) / len(rates))
        # correctness of the prompt path: a code buried mid-way through ~6K tokens must come back (greedy, no thinking)
        book = json.load(open(os.path.join(HERE, "prefill_prompts.json")))["prompts"][0]["messages"][0]["content"]
        body = book[: len(book) * 3 // 4]
        mid = body.rfind("\n", 0, len(body) // 2)
        code = "7319-KESTREL-0482"
        text = ("You will be asked for an access code hidden in the text below.\n\n" + body[:mid] +
                f"\nThe access code for the archive is {code}.\n" + body[mid:] +
                "\n\nWhat is the access code for the archive? Reply with the code only.")
        req = urllib.request.Request(a.base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": "x", "messages": [{"role": "user", "content": text}],
                                                      "temperature": 0, "max_tokens": 24,
                                                      "chat_template_kwargs": {"enable_thinking": False}}).encode())
        reply = json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"].get("content", "")
        need(code in reply, f"needle: {code!r} not in {reply!r}")
        rec["needle"] = reply

    if "ttft" in phases:
        p = decode_prompts[3]  # chat_explain: a short chat turn
        cold = stream(a.base, greedy(p, max_tokens=8))
        warm = stream(a.base, greedy(p, max_tokens=8))
        need(cold["error"] is None and warm["error"] is None, "ttft requests")
        m["ttft_short_ms"] = cold["ttft_s"] * 1000
        m["ttft_cached_ms"] = warm["ttft_s"] * 1000
        rec["ttft"] = {"cold": cold["ttft_s"], "cached": warm["ttft_s"], "cached_tokens": (warm["usage"] or {}).get("prompt_tokens_details")}

    if "decode" in phases:
        rows = []
        for p in decode_prompts:
            d = stream(a.base, greedy(p))
            s = stream(a.base, greedy(p, draft=False))
            for x, what in ((d, "drafted"), (s, "serial")):
                need(x["error"] is None and x["tensorfold"], f"{p['name']} {what}: {x['error']}")
            sha_d, sha_s = d["tensorfold"].get("token_sha"), s["tensorfold"].get("token_sha")
            need(sha_d == sha_s, f"{p['name']}: drafted reply {sha_d} != serial {sha_s}")
            rows.append({"prompt": p["name"], "tok_s": d["tensorfold"]["tokens_per_second"],
                         "serial_tok_s": s["tensorfold"]["tokens_per_second"],
                         "tpr": (d["speculative"] or {}).get("tokens_per_round"), "sha": sha_d,
                         "tokens": (d["usage"] or {}).get("completion_tokens")})
        rec["decode"] = rows
        m["decode_tok_s"] = sum(x["tok_s"] for x in rows) / len(rows)
        m["serial_tok_s"] = sum(x["serial_tok_s"] for x in rows) / len(rows)
        tprs = [x["tpr"] for x in rows if x["tpr"]]
        if tprs:
            m["tokens_per_round"] = sum(tprs) / len(tprs)

    if "conc" in phases:
        ps = decode_prompts[:4]
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(len(ps)) as ex:
            outs = list(ex.map(lambda p: stream(a.base, greedy(p, max_tokens=256)), ps))
        wall = time.perf_counter() - t0
        for o in outs:
            need(o["error"] is None and o["usage"], f"concurrent: {o['error']}")
        m["agg_c4_tok_s"] = sum(o["usage"]["completion_tokens"] for o in outs) / wall
        m["ttft_c4_max_ms"] = max(o["ttft_s"] for o in outs) * 1000
        rec["conc"] = [{"ttft_s": o["ttft_s"], "wall_s": o["wall_s"], "tokens": o["usage"]["completion_tokens"]} for o in outs]

    rec["metrics"] = m
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    for k, v in m.items():
        print(f"METRIC {k}={v:.4f}")


if __name__ == "__main__":
    main()
