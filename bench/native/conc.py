#!/usr/bin/env python3
"""RED-SNOW native: does a request wait on the others? Concurrency and scheduling metrics (stdlib only).

Phases:
  sweep  C = 1, 2, 4, 8 chat requests started together (different prompts, 256 tokens, greedy): aggregate tok/s,
         per-stream decode tok/s, TTFT p50/max. scale_cN = aggregate at C=N / aggregate at C=1.
  mixed  4 chat streams decoding; 1 s in, a cold ~32K prompt arrives. Records the decoding streams' largest gap
         between two streamed chunks while the long prompt fills (stall_ms), the long prompt's TTFT, and a short
         chat's TTFT sent while the long prompt fills (ttft_under_load_ms).
  trace  a fixed open-loop arrival schedule (TRACE below: 5 chats, a cold ~32K and a cold ~8K prompt, 7 requests over
         6 s): trace_mean_s = mean submit-to-last-token seconds over every request (what queueing, stalls and
         throughput cost a user), trace_chat_ttft_max_s, trace_wall_s.
Long prompts are salted (cold) so no kept prompt state from earlier phases or runs matches them.
Every reply must succeed; the chats are also checked against their solo replies (concurrent == solo) by token_sha.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run  # noqa: E402  (stream, greedy, need)


def timed_stream(base, body):
    """Like run.stream, but records the arrival time of every streamed chunk (absolute perf_counter)."""
    import urllib.request
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, out = [], {"usage": None, "tensorfold": None, "error": None, "t0": t0}
    with urllib.request.urlopen(req, timeout=3600) as r:
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
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    stamps.append(time.perf_counter())
            for k in ("usage", "tensorfold"):
                if ev.get(k):
                    out[k] = ev[k]
    out["stamps"] = stamps
    out["ttft_s"] = (stamps[0] if stamps else time.perf_counter()) - t0
    out["end"] = time.perf_counter()
    return out


def chat(p, n=256):
    return run.greedy(p, max_tokens=n)


# (arrival s, kind, index): chats are sweep6 chats (256 tokens); longs are prefill_prompts.json's, salted (64 tokens)
TRACE = [(0.0, "chat", 0), (0.5, "chat", 1), (1.0, "long", 1), (2.0, "chat", 2), (3.0, "long", 0), (4.0, "chat", 3),
         (6.0, "chat", 4)]


def cold(p, salt):
    """The prompt with a salt line first: no kept prompt state matches any prefix of it."""
    msgs = [dict(m) for m in p["messages"]]
    msgs[0]["content"] = f"[request {salt}]\n" + msgs[0]["content"]
    return dict(p, messages=msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", "http://192.168.1.123:8080"))
    ap.add_argument("--out", default="")
    ap.add_argument("--phases", default="sweep,mixed,trace")
    ap.add_argument("--levels", default="1,2,4,8")
    a = ap.parse_args()
    prompts = json.load(open(os.path.join(HERE, "..", "sweep6_prompts.json")))["prompts"]
    chats = [p for p in prompts if p["name"] != "long_doc"]
    m, rec = {}, {}
    run.need(run.stream(a.base, chat(chats[0], 8))["error"] is None, "warm-up")

    solo = {}
    for p in chats:  # each chat alone: its tokens are the reference for concurrent == solo
        r = run.stream(a.base, chat(p))
        run.need(r["error"] is None, f"solo {p['name']}")
        solo[p["name"]] = r["tensorfold"]["token_sha"]

    if "sweep" in a.phases:
        agg1 = None
        for c in [int(x) for x in a.levels.split(",")]:
            ps = [chats[i % len(chats)] for i in range(c)]
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(c) as ex:
                outs = list(ex.map(lambda p: timed_stream(a.base, chat(p)), ps))
            wall = time.perf_counter() - t0
            for p, o in zip(ps, outs):
                run.need(o["error"] is None and o["usage"], f"C={c} {p['name']}: {o['error']}")
                run.need(o["tensorfold"]["token_sha"] == solo[p["name"]], f"C={c} {p['name']}: concurrent != solo")
            toks = sum(o["usage"]["completion_tokens"] for o in outs)
            agg = toks / wall
            agg1 = agg1 or agg
            ttfts = sorted(o["ttft_s"] for o in outs)
            per = [o["usage"]["completion_tokens"] / max(1e-9, o["end"] - o["stamps"][0]) for o in outs]
            m[f"sweep_c{c}_tok_s"] = agg  # sweep_ prefix: run.py prints its own agg_c4_tok_s / ttft_c4_max_ms
            m[f"scale_c{c}"] = agg / agg1
            m[f"sweep_c{c}_ttft_max_ms"] = ttfts[-1] * 1000
            m[f"stream_c{c}_min_tok_s"] = min(per)
            rec[f"c{c}"] = {"wall": wall, "ttft": ttfts, "per_stream_tok_s": per}

    if "mixed" in a.phases:
        long_p = cold(json.load(open(os.path.join(HERE, "prefill_prompts.json")))["prompts"][1], f"mixed-{time.time_ns()}")
        res = {}

        def bg(i, p):
            res[i] = timed_stream(a.base, chat(p, 512))

        threads = [threading.Thread(target=bg, args=(i, chats[i])) for i in range(4)]
        for t in threads:
            t.start()
        time.sleep(1.0)
        lt = threading.Thread(target=lambda: res.__setitem__("long", timed_stream(a.base, run.greedy(long_p))))
        lt.start()
        time.sleep(1.0)
        short = timed_stream(a.base, chat(chats[3], 16))  # a new chat while the long prompt fills
        lt.join()
        for t in threads:
            t.join()
        long_r = res["long"]
        run.need(long_r["error"] is None and short["error"] is None, "mixed: errors")
        t_long0, t_long1 = long_r["t0"], long_r["t0"] + long_r["ttft_s"]
        stall = 0.0
        for i in range(4):
            o = res[i]
            run.need(o["error"] is None, f"mixed chat {i}")
            s = [x for x in o["stamps"]]
            gaps = [b - a_ for a_, b in zip(s, s[1:]) if b > t_long0 and a_ < t_long1]
            stall = max([stall] + gaps)
        m["stall_ms"] = stall * 1000
        m["ttft_long_s"] = long_r["ttft_s"]
        m["ttft_under_load_ms"] = short["ttft_s"] * 1000
        rec["mixed"] = {"stall_s": stall, "long_ttft": long_r["ttft_s"], "short_ttft": short["ttft_s"],
                        "long_tokens": (long_r["usage"] or {}).get("prompt_tokens")}

    if "trace" in a.phases:
        longs = json.load(open(os.path.join(HERE, "prefill_prompts.json")))["prompts"]
        salt = f"trace-{time.time_ns()}"
        res, t0 = {}, time.perf_counter()

        def go(i, at, kind, k):
            time.sleep(max(0.0, t0 + at - time.perf_counter()))
            body = chat(chats[k]) if kind == "chat" else run.greedy(cold(longs[k], f"{salt}-{i}"), max_tokens=64)
            res[i] = (kind, k, time.perf_counter(), timed_stream(a.base, body))

        threads = [threading.Thread(target=go, args=(i, *t)) for i, t in enumerate(TRACE)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lat, chat_ttft, rows = [], [], []
        for i in range(len(TRACE)):
            run.need(i in res, f"trace {i}: no reply")
            kind, k, sub, o = res[i]
            run.need(o["error"] is None and o["usage"], f"trace {i} {kind}: {o['error']}")
            if kind == "chat":
                run.need(o["tensorfold"]["token_sha"] == solo[chats[k]["name"]], f"trace {i}: concurrent != solo")
                chat_ttft.append(o["ttft_s"])
            lat.append(o["end"] - sub)
            rows.append({"kind": kind, "k": k, "ttft": o["ttft_s"], "latency": o["end"] - sub,
                         "prompt": o["usage"]["prompt_tokens"], "out": o["usage"]["completion_tokens"]})
        m["trace_mean_s"] = sum(lat) / len(lat)
        m["trace_chat_ttft_max_s"] = max(chat_ttft)
        m["trace_wall_s"] = max(r[3]["end"] for r in res.values()) - t0
        rec["trace"] = rows

    if "large" in a.phases:  # opt-in (not in the harness default): C distinct cold large prompts at once, 256 tokens each
        base8k = json.load(open(os.path.join(HERE, "prefill_prompts.json")))["prompts"][0]
        salt = f"large-{time.time_ns()}"

        def sized(i, chars):
            p = cold(base8k, f"{salt}-{chars}-{i}")
            p["messages"][0]["content"] = p["messages"][0]["content"][:chars]
            return run.greedy(p, max_tokens=256)

        for chars, tag in ((8000, "2k"), (10 ** 9, "8k")):
            for c in (4, 8):
                bodies = [sized(i, chars) for i in range(c)]
                ref = {i: run.stream(a.base, bodies[i])["tensorfold"]["token_sha"] for i in (0, c - 1)}
                t0 = time.perf_counter()
                with cf.ThreadPoolExecutor(c) as ex:
                    outs = list(ex.map(lambda b: timed_stream(a.base, b), bodies))
                wall = time.perf_counter() - t0
                for i, o in enumerate(outs):
                    run.need(o["error"] is None and o["usage"], f"large {tag} C={c} #{i}: {o['error']}")
                    if i in ref:
                        run.need(o["tensorfold"]["token_sha"] == ref[i], f"large {tag} C={c} #{i}: concurrent != solo")
                toks = sum(o["usage"]["completion_tokens"] for o in outs)
                ptoks = sum(o["usage"]["prompt_tokens"] for o in outs)
                ttfts = sorted(o["ttft_s"] for o in outs)
                m[f"large{tag}_c{c}_out_tok_s"] = toks / wall
                m[f"large{tag}_c{c}_wall_s"] = wall
                m[f"large{tag}_c{c}_ttft_max_s"] = ttfts[-1]
                m[f"large{tag}_c{c}_ttft_mean_s"] = sum(ttfts) / c
                rec[f"large{tag}_c{c}"] = {"wall": wall, "ttft": ttfts, "prompt_tokens": ptoks, "out_tokens": toks}

    rec["metrics"] = m
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    for k, v in m.items():
        print(f"METRIC {k}={v:.4f}")


if __name__ == "__main__":
    main()
