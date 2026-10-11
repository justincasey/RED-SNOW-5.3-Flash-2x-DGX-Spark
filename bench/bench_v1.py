#!/usr/bin/env python3
# Adapted from vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe@ed2f31d bench/bench_v1.py (MIT, (c) 2026 Victor Cruz);
# see bench/LICENSE-vcruz305. Changes: RED-SNOW defaults and reference, server tok/s from the stats block, the
# "tf_policy" checks of this engine, a streaming check, --pin.
"""/v1 smoke test and decode benchmark for the RED-SNOW TensorFold TP=2 server (stdlib only: runs on rank 0's
system python3 or anywhere that reaches the API). `./redsnow.sh bench smoke|decode|pin` runs it on rank 0.

  python3 bench_v1.py --smoke
  python3 bench_v1.py [--reps 3] [--no-serial] [--ids reference|self] [--pin OUT.json] [--base http://127.0.0.1:8080/v1]

Benchmark: six prompts (code, prose, math, chat_explain, chat_multiturn, long_doc) as the 2026-10-05 GLM-5.3 sweep
sent them (thinking on, greedy, 512 tokens), streamed with "return_token_ids": true. For each prompt /tokenize must
give the pinned prompt ids. With --ids reference (default) the "draft": false reply and every drafted reply must
equal the pinned serial ids token for token, whether the drafted request prefilled the prompt whole or resumed it
from the prompt cache: a regression oracle for engine changes. With --ids self every drafted reply must equal this
server's own "draft": false reply; --pin then writes this run's prompt and serial ids as a new reference. Speed: the
server's decode rate (tokens after the first over the stats block's decode_s) and the client's (tokens after the
first over the first to last delta), medians over --reps. Exit 1 on any failed check.

Smoke: /v1/models, /health, /tokenize parity (thinking on ends <|assistant|><think>, off ends <|assistant|>),
a thinking-off reply that must stop at a turn-end id with clean content, a thinking-on streamed reply whose reasoning
and content stay split with no markup in either channel, the reply streaming round by round, drafted == "draft":
false ids, the default request drafting, and the refusal of an invalid "tf_policy".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

ASSISTANT_ID = 154828
THINK_OPEN_ID, THINK_END_ID = 154841, 154842          # this pack's tokenizer (sha256 19e77364...)
EOS_IDS = (154820, 154827, 154829)                     # generation_config.json eos_token_id
LEAKS = ("</think>", "<think>", "<|user|>", "<|observation|>", "<|assistant|>", "<|endoftext|>", "<|system|>")


class Client:
    def __init__(self, base: str, model: str, key: str | None = None, timeout: float = 1800.0):
        self.base, self.model, self.key, self.timeout = base.rstrip("/"), model, key, timeout
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base

    def _req(self, url: str, body: dict | None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        return urllib.request.urlopen(req, timeout=self.timeout)

    def get(self, path: str) -> dict:
        with self._req(self.base + path, None) as r:
            return json.loads(r.read())

    def post(self, path: str, body: dict, root: bool = False) -> tuple[int, dict]:
        try:
            with self._req((self.root if root else self.base) + path, body) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read())
            except ValueError:
                return e.code, {}

    def stream(self, body: dict) -> dict:
        """POST a streamed chat completion; per-chunk deltas with arrival times, the final stats and usage."""
        body = dict(body, stream=True, model=self.model)
        t0 = time.perf_counter()
        chunks, final, err = [], None, None
        with self._req(self.base + "/chat/completions", body) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                now = time.perf_counter() - t0
                if "error" in obj:
                    err = obj["error"]
                    continue
                if obj.get("choices"):
                    ch = obj["choices"][0]
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        chunks.append({"t": now, "content": d.get("content") or "",
                                       "reasoning": d.get("reasoning_content") or ""})
                    if ch.get("finish_reason"):
                        final = obj
                elif obj.get("usage") and final is not None:
                    final["usage"] = obj["usage"]
        total = time.perf_counter() - t0
        if err is not None:
            return {"error": err, "total_s": total}
        stats = (final or {}).get("tensorfold") or {}
        ids = stats.get("token_ids") or []
        ts = [c["t"] for c in chunks]
        n = len(ids)
        stats = {k: v for k, v in stats.items() if k != "token_ids"}
        dec = stats.get("decode_s")
        if dec and n > 1:                     # the engine's own rate: DecodeResult.tokens_per_second
            stats["tok_s"] = round((n - 1) / dec, 2)
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        return {"ids": ids, "n": n, "finish": ((final or {}).get("choices") or [{}])[0].get("finish_reason"),
                "usage": (final or {}).get("usage"), "stats": stats,
                "content": "".join(c["content"] for c in chunks), "reasoning": "".join(c["reasoning"] for c in chunks),
                "chunks": len(chunks), "both_in_one_chunk": sum(1 for c in chunks if c["content"] and c["reasoning"]),
                "max_gap_s": max(gaps, default=None),
                "ttft_s": ts[0] if ts else None, "total_s": total,
                "client_tok_s": (n - 1) / (ts[-1] - ts[0]) if n > 1 and len(ts) > 1 and ts[-1] > ts[0] else None}


def chat_body(messages, max_tokens, thinking: bool | None, **extra) -> dict:
    b = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True}
    if thinking is not None:
        b["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    b.update(extra)
    return b


def first_diff(a, b):
    for i in range(max(len(a), len(b))):
        if i >= len(a) or i >= len(b) or a[i] != b[i]:
            return i
    return None


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name: str, ok: bool, detail="") -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
        return ok

    @property
    def ok(self) -> bool:
        return all(i["ok"] for i in self.items)


def tokenize(cl: Client, messages, thinking: bool) -> list[int]:
    code, r = cl.post("/tokenize", {"model": cl.model, "messages": messages,
                                    "chat_template_kwargs": {"enable_thinking": thinking}}, root=True)
    if code != 200:
        raise RuntimeError(f"/tokenize HTTP {code}: {r}")
    return r["tokens"]


def smoke(cl: Client, ref: dict, ck: Checks) -> dict:
    out = {}
    models = cl.get("/models")
    ids = [m["id"] for m in models.get("data", [])]
    ck.add("/v1/models lists the served name", cl.model in ids, f"{ids}")
    with urllib.request.urlopen(cl.root + "/health", timeout=30) as r:
        ck.add("/health 200", r.status == 200, r.read().decode()[:200])
    eos = set(EOS_IDS)
    for p in ref["prompts"]:
        got = tokenize(cl, p["messages"], True)
        ck.add(f"/tokenize {p['name']} thinking on == pinned prompt ids", got == p["prompt_ids"],
               f"{len(got)} vs {p['prompt_len']} tokens, first diff {first_diff(got, p['prompt_ids'])}")
    q = [{"role": "user", "content": "Reply with exactly the word OK and nothing else."}]
    off = tokenize(cl, q, False)
    on = tokenize(cl, q, True)
    # this pack's template (sha256 f987a112...) ends a thinking-off prompt at <|assistant|>, with no </think>
    ck.add("thinking off prompt ends <|assistant|>", off[-1] == ASSISTANT_ID and THINK_END_ID not in off[-2:],
           f"{off[-3:]}")
    ck.add("thinking on prompt ends <|assistant|><think>", on[-2:] == [ASSISTANT_ID, THINK_OPEN_ID], f"{on[-3:]}")
    r = cl.stream(chat_body(q, 64, False))
    out["thinking_off"] = r
    ck.add("thinking off: no error", "error" not in r, str(r.get("error", ""))[:200])
    if "error" not in r:
        ck.add("thinking off: finish stop at a turn-end id", r["finish"] == "stop" and r["ids"] and r["ids"][-1] in eos,
               f"finish {r['finish']}, last id {r['ids'][-1:]}, n {r['n']}")
        ck.add("thinking off: empty reasoning, content has OK", not r["reasoning"] and "OK" in r["content"],
               f"content {r['content'][:80]!r}")
        ck.add("thinking off: no markup leak", not any(t in r["content"] for t in LEAKS), r["content"][:80])
    r = cl.stream(chat_body([{"role": "user", "content": "What is 17 * 23? Answer with the number."}], 1024, True))
    out["thinking_on"] = r
    ck.add("thinking on: no error", "error" not in r, str(r.get("error", ""))[:200])
    if "error" not in r:
        ck.add("thinking on: reasoning non-empty", bool(r["reasoning"].strip()), f"{len(r['reasoning'])} chars")
        ck.add("thinking on: no markup leak in either channel",
               not any(t in r["content"] or t in r["reasoning"] for t in LEAKS), "")
        if r["finish"] == "stop":
            ck.add("thinking on: stops at a turn-end id with an answer", r["ids"][-1] in eos and "391" in r["content"],
                   f"content {r['content'][:80]!r}")
        else:
            ck.add("thinking on: finish", r["finish"] == "length", f"{r['finish']} (1024 tokens still thinking?)")
        # one delta a decode round (~70 ms), not one burst when decoding ends
        ck.add("thinking on: streams round by round", r["chunks"] >= r["n"] // 8 and (r["max_gap_s"] or 0) < 1.0,
               f"{r['chunks']} deltas for {r['n']} tokens, largest gap {r['max_gap_s'] or 0:.3f} s")
    short = [{"role": "user", "content": "Name three prime numbers above 100."}]
    a = cl.stream(chat_body(short, 96, False, draft=False))
    b = cl.stream(chat_body(short, 96, False))
    ok = "error" not in a and "error" not in b and a["ids"] == b["ids"] and a["n"] > 0
    ck.add("drafted ids == \"draft\": false ids", ok,
           f"n {a.get('n')} / {b.get('n')}, first diff {first_diff(a.get('ids') or [], b.get('ids') or [])}")
    st = b.get("stats") or {}
    # the default policy "auto" resolves to DFlash2 fc5:0.3 on this EXL3 pack; the auto loop would report "drafters"
    ck.add("default request drafts with DFlash2", st.get("drafts") is True and (st.get("drafted") or 0) > 0
           and "drafters" not in st,
           f"policy {st.get('policy')}, drafted {st.get('drafted')}, accepted {st.get('accepted')}, "
           f"tokens/round {st.get('tokens_per_round')}")
    bad = cl.stream(chat_body(short, 8, False, tf_policy="bogus"))
    ck.add("invalid tf_policy refused", "error" in bad, str(bad.get("error", ""))[:160])
    return out


def bench(cl: Client, ref: dict, ck: Checks, reps: int, serial: bool, names: list[str] | None,
          ids_mode: str = "reference") -> dict:
    """ids_mode "reference": the serial reply and every drafted reply must equal the pinned serial ids. "self": every
    drafted reply must equal this server's own "draft": false reply (a new reference, or another numeric path)."""
    rows = []
    if ids_mode == "self" and not serial:
        raise SystemExit("--ids self needs the serial run (drop --no-serial)")
    warm = cl.stream(chat_body([{"role": "user", "content": "Hello"}], 16, True))
    ck.add("warm-up request", "error" not in warm, str(warm.get("error", ""))[:200])
    for p in ref["prompts"]:
        if names and p["name"] not in names:
            continue
        got = tokenize(cl, p["messages"], True)
        if ids_mode == "reference":
            ck.add(f"{p['name']}: prompt ids == reference", got == p["prompt_ids"],
                   f"first diff {first_diff(got, p['prompt_ids'])}")
        body = chat_body(p["messages"], p["max_tokens"], True)
        row = {"prompt": p["name"], "prompt_len": len(got), "prompt_ids": got, "runs": []}
        want = p.get("serial_ids")
        if serial:
            r = cl.stream(dict(body, draft=False))
            row["serial_ids"] = r.get("ids") or []
            if ids_mode == "reference":
                ck.add(f"{p['name']}: serial ids == pinned serial", r.get("ids") == p["serial_ids"],
                       f"n {r.get('n')}, first diff {first_diff(r.get('ids') or [], p['serial_ids'])}, "
                       f"tok/s {r.get('stats', {}).get('tok_s')}")
            else:
                want = row["serial_ids"]
                ck.add(f"{p['name']}: serial reply complete", "error" not in r and bool(want),
                       f"n {r.get('n')}, tok/s {r.get('stats', {}).get('tok_s')}")
            row["serial"] = {"tok_s": r.get("stats", {}).get("tok_s"), "client_tok_s": r.get("client_tok_s"),
                             "finish": r.get("finish"), "ttft_s": r.get("ttft_s"), "n": r.get("n")}
        for rep in range(reps):
            r = cl.stream(body)
            st = r.get("stats") or {}
            ids = r.get("ids") or []
            cached = st.get("cached") or 0           # > 0: resumed from the prompt cache
            ck.add(f"{p['name']} rep {rep}: drafted ids == {'pinned' if ids_mode == 'reference' else 'this server'} "
                   f"serial", ids == want,
                   f"n {r.get('n')}, cached {cached}, first diff {first_diff(ids, want or [])}, tok/s "
                   f"{st.get('tok_s')}, tokens/round {st.get('tokens_per_round')}")
            row["runs"].append({"cached": cached, "tok_s": st.get("tok_s"), "client_tok_s": r.get("client_tok_s"),
                                "tokens_per_round": st.get("tokens_per_round"), "ttft_s": r.get("ttft_s"),
                                "max_gap_s": r.get("max_gap_s"), "n": r.get("n"), "finish": r.get("finish"),
                                "identical": ids == want, "policy": st.get("policy"), "drafted": st.get("drafted"),
                                "accepted": st.get("accepted"), "stages_ms": st.get("stages_ms")})
        med = lambda k: statistics.median([x[k] for x in row["runs"] if x.get(k) is not None]) if row["runs"] else None  # noqa: E731
        row["tok_s"], row["client_tok_s"], row["tpr"] = med("tok_s"), med("client_tok_s"), med("tokens_per_round")
        row["fresh_ttft_s"] = next((x["ttft_s"] for x in row["runs"] if not x["cached"]), None)
        rows.append(row)
    print("\n| prompt | tokens | server tok/s (median) | client tok/s | tokens/round | TTFT s (uncached) | serial tok/s | "
          "exact |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        ex = all(x["identical"] for x in r["runs"])
        print(f"| {r['prompt']} | {r['prompt_len']} | {fmt(r['tok_s'])} | {fmt(r['client_tok_s'])} | {fmt(r['tpr'])} | "
              f"{fmt(r['fresh_ttft_s'])} | {fmt((r.get('serial') or {}).get('tok_s'))} | {'yes' if ex else 'NO'} |")
    full = [r for r in rows if r["tok_s"]]
    out = {"rows": rows}
    if full and len(full) == len(ref["prompts"]):
        out["mean_tok_s"] = round(sum(r["tok_s"] for r in full) / len(full), 2)
        sr = [r["serial"]["tok_s"] for r in full if (r.get("serial") or {}).get("tok_s")]
        out["mean_serial_tok_s"] = round(sum(sr) / len(sr), 2) if len(sr) == len(full) else None
        print(f"\nmean over all {len(full)} prompts: drafted {out['mean_tok_s']:.2f} tok/s, serial "
              f"{fmt(out['mean_serial_tok_s'])} tok/s")
    return out


def pin(ref: dict, res: dict, path: str, meta: dict) -> None:
    """A new reference from a full --ids self run (every drafted reply already equal to the serial one): the prompts
    with this server's prompt ids and serial reply ids."""
    by = {r["prompt"]: r for r in res["rows"]}
    missing = [p["name"] for p in ref["prompts"] if p["name"] not in by or "serial_ids" not in by[p["name"]]]
    if missing:
        raise SystemExit(f"--pin needs every prompt with its serial run; missing {missing}")
    sha = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]  # noqa: E731
    prompts = []
    for p in ref["prompts"]:
        r = by[p["name"]]
        prompts.append({"name": p["name"], "messages": p["messages"], "max_tokens": p["max_tokens"],
                        "prompt_len": r["prompt_len"], "prompt_ids": r["prompt_ids"], "prompt_sha": sha(r["prompt_ids"]),
                        "serial_ids": r["serial_ids"], "serial_sha": sha(r["serial_ids"]),
                        "tok_s": r["tok_s"], "serial_tok_s": (r.get("serial") or {}).get("tok_s")})
    keep = {"eos_ids": list(EOS_IDS), "think_open_id": THINK_OPEN_ID, "think_end_id": THINK_END_ID}
    out = {"what": "RED-SNOW reference: the six sweep prompts' served prompt ids and serial (\"draft\": false) reply "
                   "ids, which every drafted reply equals; greedy, 512 tokens, thinking on",
           "pinned": time.strftime("%Y-%m-%d %H:%M:%S"), **meta, **keep,
           "mean_tok_s": res.get("mean_tok_s"), "mean_serial_tok_s": res.get("mean_serial_tok_s"), "prompts": prompts}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"pinned {path}")


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="RED-SNOW-5.3-Flash")
    ap.add_argument("--ref", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "redsnow_reference.json"),
                    help="the 6 prompts with their pinned prompt, serial and drafted ids (default: next to this file)")
    ap.add_argument("--ids", choices=("reference", "self"), default="reference",
                    help="reference: replies must equal the pinned ids; self: no comparison to the pins")
    ap.add_argument("--pin", default=None, help="with --ids self: write this run as a new reference to PIN")
    ap.add_argument("--meta", action="append", default=[], help="KEY=VALUE recorded in a pinned reference")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-serial", action="store_true", help="skip the \"draft\": false run per prompt")
    ap.add_argument("--prompts", default="", help="comma list (default: all 6)")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY") or None)
    a = ap.parse_args()
    if a.pin and (a.ids != "self" or a.no_serial or a.prompts or a.reps < 1):
        raise SystemExit("--pin needs --ids self, the serial run, every prompt and at least one rep")
    with open(a.ref) as f:
        ref = json.load(f)
    cl = Client(a.base, a.model, a.api_key)
    ck = Checks()
    t0 = time.time()
    if a.smoke:
        res = smoke(cl, ref, ck)
    else:
        res = bench(cl, ref, ck, a.reps, not a.no_serial, [n for n in a.prompts.split(",") if n] or None, a.ids)
    res.update(checks=ck.items, ok=ck.ok, base=a.base, model=a.model, ids_mode=None if a.smoke else a.ids,
               started=t0, wall_s=round(time.time() - t0, 1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {a.out}")
    if a.pin and ck.ok:
        pin(ref, res, a.pin, dict(kv.split("=", 1) for kv in a.meta))
    print(f"{'ALL CHECKS PASSED' if ck.ok else 'FAILED'}: {sum(i['ok'] for i in ck.items)}/{len(ck.items)}")
    return 0 if ck.ok else 1


if __name__ == "__main__":
    sys.exit(main())
