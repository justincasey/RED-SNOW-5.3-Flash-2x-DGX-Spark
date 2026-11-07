#!/usr/bin/env python3
"""RED-SNOW-5.3-Flash on REXL3: one fixed workload, printed as METRIC lines (stdlib only).

Phases, in order, on a freshly started server (empty prefix cache):
  warm     one drafted and one serial 64-token reply (kernels, first graphs)
  prefill  pinned 8K and 32K Moby-Dick slices (../native/prefill_prompts.json), cold, 64 tokens out (pinned):
           tok/s = prompt tokens / client time to the first streamed token
  needle   a code buried mid-way through ~6K tokens must come back
  ttft     a ~220-token chat prompt, client time to the first streamed token
  decode   ../sweep6_prompts.json, greedy, 512 tokens, drafted and "draft": false, thinking per the server default;
           tok/s = (tokens - 1) / (last chunk - first chunk); the two replies' token ids must be equal
  conc     each chat alone (256 tokens), then C = 1, 2, 4, 8 at once: aggregate tok/s, worst TTFT; every concurrent
           reply's token ids must equal its solo reply's
  mixed    4 chats decoding, a cold ~32K prompt arrives, then a short chat: stall_ms, ttft_long_s, ttft_under_load_ms
  trace    a fixed open-loop arrival schedule (5 chats, cold 32K and 8K prompts): trace_mean_s
(the GLM family serves no prompt_logprobs, so quality is held by exactness: serial ids against a pinned reference)

Gates (each failure prints FAIL and voids target_score): errors, drafted != serial, concurrent != solo, a missed needle,
a warm prompt reported as cached, and, when --ref names a pinned reference (--pin writes one), serial ids or the long
prompts' and the needle's replies that differ from it (byte-exact: no output may drift from the reference engine).

target_score = (decode_prose/70 * prefill_32k/3000 * min(1, kv_tokens/2^20) * min(1, 100/boot_s)) ^ (1/4)
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HEADERS = {"Content-Type": "application/json"}
MODEL = os.environ.get("SERVED_NAME", "RED-SNOW-5.3-Flash")
FAILS = []


def fail(what):
    print(f"FAIL {what}", file=sys.stderr)
    FAILS.append(what)


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream(base, body, timeout=3600):
    """POST a streamed chat completion with token ids; returns ids, usage, chunk arrival stamps, client timings."""
    body = dict(body, stream=True, stream_options={"include_usage": True}, return_token_ids=True)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=HEADERS)
    out = {"ids": [], "usage": None, "error": None, "stamps": [], "fingerprint": None, "t0": time.perf_counter()}
    try:
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
                out["fingerprint"] = ev.get("system_fingerprint") or out["fingerprint"]
                for ch in ev.get("choices") or []:
                    ids = (ch.get("delta") or {}).get("token_ids") or ch.get("token_ids") or []
                    if ids:
                        out["ids"].extend(ids)
                        out["stamps"].append(time.perf_counter())
                if ev.get("usage"):
                    out["usage"] = ev["usage"]
    except Exception as e:  # noqa: BLE001 - every failure is a gate failure, reported by the caller
        out["error"] = repr(e)
    out["end"] = time.perf_counter()
    out["ttft_s"] = (out["stamps"][0] if out["stamps"] else out["end"]) - out["t0"]
    n = len(out["ids"])
    span = out["stamps"][-1] - out["stamps"][0] if len(out["stamps"]) > 1 else 0.0
    out["decode_tok_s"] = (n - 1) / span if span > 0 else 0.0
    out["sha"] = hashlib.sha256(json.dumps(out["ids"]).encode()).hexdigest()[:16]
    return out


def ok(r, what):
    if r["error"] is not None or not r["usage"] or not r["ids"]:
        fail(f"{what}: {r['error'] or 'empty reply'}")
        return False
    return True


def greedy(p, max_tokens=None, **extra):
    b = {"model": MODEL, "messages": p["messages"], "temperature": 0, "max_tokens": max_tokens or p.get("max_tokens", 512)}
    b.update(extra)
    return b


def cold(p, salt):
    """The prompt with a salt line first: no kept prompt state matches any prefix of it."""
    msgs = [dict(m) for m in p["messages"]]
    msgs[0]["content"] = f"[request {salt}]\n" + msgs[0]["content"]
    return dict(p, messages=msgs)


# (arrival s, kind, index): chats are sweep6 chats (256 tokens); longs are prefill_prompts.json's, salted (64 tokens)
TRACE = [(0.0, "chat", 0), (0.5, "chat", 1), (1.0, "long", 1), (2.0, "chat", 2), (3.0, "long", 0), (4.0, "chat", 3),
         (6.0, "chat", 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:8080"))
    ap.add_argument("--out", default="")
    ap.add_argument("--phases", default="warm,prefill,needle,ttft,decode,conc,mixed,trace")
    ap.add_argument("--boot-s", type=float, default=0.0)
    ap.add_argument("--kv-tokens", type=float, default=0.0)
    ap.add_argument("--ref", default="", help="pinned reference to gate serial ids against")
    ap.add_argument("--pin", default="", help="write this run's serial ids as a reference")
    a = ap.parse_args()
    phases = set(a.phases.split(","))
    prompts = json.load(open(os.path.join(HERE, "..", "sweep6_prompts.json")))["prompts"]
    longs = json.load(open(os.path.join(HERE, "..", "native", "prefill_prompts.json")))["prompts"]
    chats = [p for p in prompts if p["name"] != "long_doc"]
    ref = json.load(open(a.ref)) if a.ref and os.path.exists(a.ref) else None
    rec = {"base": a.base, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    m = {}
    pinned_ids = {}  # the long prompts' and the needle's replies: the prefill path's numerics, pinned like the serial ids

    if "warm" in phases:
        for d in (True, False):
            ok(stream(a.base, greedy(prompts[0], max_tokens=64, draft=d)), f"warm-up draft={d}")

    if "prefill" in phases:
        for p in longs:
            r = stream(a.base, greedy(p, max_tokens=64))  # 64 tokens: a reply long enough to fingerprint the prompt's state
            if not ok(r, p["name"]):
                continue
            n = r["usage"]["prompt_tokens"]
            cached = (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            if cached:
                fail(f"{p['name']}: {cached} cached tokens on a cold prompt")
            key = p["name"].replace("prefill_", "")
            m[f"prefill_{key}_tok_s"] = n / r["ttft_s"]
            m[f"ttft_{key}_s"] = r["ttft_s"]
            rec[p["name"]] = {"prompt_tokens": n, "ttft_s": r["ttft_s"]}
            pinned_ids[p["name"]] = r["sha"]

    if "needle" in phases:
        book = longs[0]["messages"][0]["content"]
        body = book[: len(book) * 3 // 4]
        mid = body.rfind("\n", 0, len(body) // 2)
        code = "7319-KESTREL-0482"
        text = ("You will be asked for an access code hidden in the text below.\n\n" + body[:mid] +
                f"\nThe access code for the archive is {code}.\n" + body[mid:] +
                "\n\nWhat is the access code for the archive? Reply with the code only.")
        try:
            r = post(a.base, "/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": text}],
                                                      "temperature": 0, "max_tokens": 24,
                                                      "chat_template_kwargs": {"enable_thinking": False}})
            reply = r["choices"][0]["message"].get("content") or ""
        except Exception as e:  # noqa: BLE001
            reply = repr(e)
        if code not in reply:
            fail(f"needle: {code!r} not in {reply!r}")
        rec["needle"] = reply
        pinned_ids["needle"] = hashlib.sha256(reply.encode()).hexdigest()[:16]

    if "ttft" in phases:
        r = stream(a.base, greedy(prompts[3], max_tokens=8))
        if ok(r, "ttft"):
            m["ttft_short_ms"] = r["ttft_s"] * 1000

    serial, serial_ids = {}, {}
    if "decode" in phases:
        rows = []
        for p in prompts:
            d = stream(a.base, greedy(p))
            s = stream(a.base, greedy(p, draft=False))
            if not (ok(d, f"{p['name']} drafted") and ok(s, f"{p['name']} serial")):
                continue
            if d["ids"] != s["ids"]:
                k = next((i for i, (x, y) in enumerate(zip(d["ids"], s["ids"])) if x != y), min(len(d["ids"]), len(s["ids"])))
                fail(f"{p['name']}: drafted reply differs from serial at token {k}")
            serial[p["name"]], serial_ids[p["name"]] = s["sha"], s["ids"]
            spec = d["usage"].get("spec") or {}
            rows.append({"prompt": p["name"], "tok_s": d["decode_tok_s"], "serial_tok_s": s["decode_tok_s"],
                         "tpr": len(d["ids"]) / spec["rounds"] if spec.get("rounds") else None, "sha": s["sha"],
                         "tokens": len(d["ids"]), "spec": spec})
            m[f"decode_{p['name']}_tok_s"] = d["decode_tok_s"]
        rec["decode"] = rows
        rec["fingerprint"] = d["fingerprint"]
        if rows:
            m["decode_tok_s"] = sum(x["tok_s"] for x in rows) / len(rows)
            m["serial_tok_s"] = sum(x["serial_tok_s"] for x in rows) / len(rows)
            tprs = [x["tpr"] for x in rows if x["tpr"]]
            if tprs:
                m["tokens_per_round"] = sum(tprs) / len(tprs)

    solo = {}
    if phases & {"conc", "mixed", "trace"}:
        for p in chats:  # each chat alone: the reference for concurrent == solo, and itself the serial reply's first tokens
            r = stream(a.base, greedy(p, max_tokens=256))
            if ok(r, f"solo {p['name']}"):
                solo[p["name"]] = r["ids"]
                if p["name"] in serial_ids and r["ids"] != serial_ids[p["name"]][:len(r["ids"])]:
                    fail(f"solo {p['name']}: reply differs from the serial reply's first {len(r['ids'])} tokens")

    def same_as_solo(p, o, what):
        if p["name"] in solo and o["ids"] != solo[p["name"]]:
            fail(f"{what} {p['name']}: concurrent reply differs from solo")

    if "conc" in phases:
        agg1 = None
        for c in (1, 2, 4, 8):
            ps = [chats[i % len(chats)] for i in range(c)]
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(c) as ex:
                outs = list(ex.map(lambda p: stream(a.base, greedy(p, max_tokens=256)), ps))
            wall = time.perf_counter() - t0
            good = [o for p, o in zip(ps, outs) if ok(o, f"C={c} {p['name']}")]
            for p, o in zip(ps, outs):
                same_as_solo(p, o, f"C={c}")
            if len(good) != c:
                continue
            agg = sum(len(o["ids"]) for o in outs) / wall
            agg1 = agg1 or agg
            m[f"sweep_c{c}_tok_s"] = agg
            m[f"sweep_c{c}_ttft_max_ms"] = max(o["ttft_s"] for o in outs) * 1000
            rec[f"c{c}"] = {"wall": wall, "ttft": [o["ttft_s"] for o in outs], "per_stream_tok_s": [o["decode_tok_s"] for o in outs]}
        if agg1 and "sweep_c8_tok_s" in m:
            m["scale_c8"] = m["sweep_c8_tok_s"] / agg1

    if "mixed" in phases:
        long_p = cold(longs[1], f"mixed-{time.time_ns()}")
        res = {}
        threads = [threading.Thread(target=lambda i=i: res.__setitem__(i, stream(a.base, greedy(chats[i], max_tokens=512))))
                   for i in range(4)]
        for t in threads:
            t.start()
        time.sleep(1.0)
        lt = threading.Thread(target=lambda: res.__setitem__("long", stream(a.base, greedy(long_p))))
        lt.start()
        time.sleep(1.0)
        short = stream(a.base, greedy(chats[3], max_tokens=16))
        lt.join()
        for t in threads:
            t.join()
        long_r = res["long"]
        if ok(long_r, "mixed long") and ok(short, "mixed short") and all(ok(res[i], f"mixed chat {i}") for i in range(4)):
            t_long0, t_long1 = long_r["t0"], long_r["t0"] + long_r["ttft_s"]
            stall = 0.0
            for i in range(4):
                s = res[i]["stamps"]
                stall = max([stall] + [b - x for x, b in zip(s, s[1:]) if b > t_long0 and x < t_long1])
            m["stall_ms"] = stall * 1000
            m["ttft_long_s"] = long_r["ttft_s"]
            m["ttft_under_load_ms"] = short["ttft_s"] * 1000

    if "trace" in phases:
        salt = f"trace-{time.time_ns()}"
        res, t0 = {}, time.perf_counter()

        def go(i, at, kind, k):
            time.sleep(max(0.0, t0 + at - time.perf_counter()))
            body = greedy(chats[k], max_tokens=256) if kind == "chat" else greedy(cold(longs[k], f"{salt}-{i}"), max_tokens=64)
            res[i] = (kind, k, time.perf_counter(), stream(a.base, body))

        threads = [threading.Thread(target=go, args=(i, *t)) for i, t in enumerate(TRACE)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lat, chat_ttft, good = [], [], True
        for i in range(len(TRACE)):
            kind, k, sub, o = res[i]
            good &= ok(o, f"trace {i} {kind}")
            if kind == "chat":
                same_as_solo(chats[k], o, f"trace {i}")
                chat_ttft.append(o["ttft_s"])
            lat.append(o["end"] - sub)
        if good:
            m["trace_mean_s"] = sum(lat) / len(lat)
            m["trace_chat_ttft_max_s"] = max(chat_ttft)

    if ref:
        for name, sha in ref.get("serial", {}).items():
            if name in serial and serial[name] != sha:
                fail(f"{name}: serial reply {serial[name]} differs from the pinned reference {sha}")
        for name, sha in ref.get("prompt_replies", {}).items():
            if name in pinned_ids and pinned_ids[name] != sha:
                fail(f"{name}: reply {pinned_ids[name]} differs from the pinned reference {sha}")
    if a.pin:
        json.dump({"pinned": rec["started"], "fingerprint": rec.get("fingerprint"), "serial": serial,
                   "prompt_replies": pinned_ids}, open(a.pin, "w"), indent=1)

    if a.boot_s:
        m["boot_s"] = a.boot_s
    if a.kv_tokens:
        m["kv_tokens"] = a.kv_tokens
    need = ("decode_prose_tok_s", "prefill_32k_tok_s")
    if all(k in m for k in need) and a.boot_s and a.kv_tokens:
        raw = (m["decode_prose_tok_s"] / 70 * m["prefill_32k_tok_s"] / 3000 * min(1.0, a.kv_tokens / 2 ** 20)
               * min(1.0, 100 / a.boot_s)) ** 0.25
        m["target_score_raw"] = raw
        m["target_score"] = 0.0 if FAILS else raw
    else:
        m["target_score"] = 0.0
    m["gates_failed"] = len(FAILS)
    rec["metrics"], rec["fails"] = m, FAILS
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    for k, v in m.items():
        print(f"METRIC {k}={v:.4f}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
