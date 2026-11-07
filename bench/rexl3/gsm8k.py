#!/usr/bin/env python3
"""GSM8K accuracy on the live server (stdlib only): the quality gate for any change that is not bit-exact.

  gsm8k.py --base http://HOST:8000 [--n 100] [--offset 0] [--think] [--workers 2] [--out FILE]

The first --n problems from --offset of the GSM8K test split (openai/gsm8k, MIT; bench/data/gsm8k_test.json), greedy,
one commit per item. Default is non-thinking chat (a step-by-step answer ending `#### <number>`, 700 tokens); --think
turns the reasoning mode on (12,000 tokens, temperature 0.6, repetition-loop items count as uncommitted, not wrong).
Reference points are NOT comparable across harnesses: the pack card's 96.7% (1271/1315 committed) used its own.
Prints METRIC gsm8k_acc and gsm8k_committed.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = "{q}\n\nSolve it step by step. End your reply with a final line `#### <number>` holding only the answer."


def number(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def ask(base, q, think):
    body = {"model": os.environ.get("MODEL", "RED-SNOW-5.3-Flash"), "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
            "max_tokens": 12000 if think else 700, "temperature": 0.6 if think else 0}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    ch = r["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason"), r["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", "http://192.168.1.121:8080"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = json.load(open(os.path.join(HERE, "data", "gsm8k_test.json")))[a.offset:a.offset + a.n]
    t0 = time.time()

    def one(i):
        item = data[i]
        gold = number(item["answer"].split("####")[-1])
        text, why, toks = ask(a.base, item["question"], a.think)
        got = number(text.split("####")[-1]) if "####" in text else None
        return {"i": a.offset + i, "gold": gold, "got": got, "finish": why, "tokens": toks,
                "committed": got is not None and why == "stop", "ok": got is not None and abs(got - gold) < 1e-6}

    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, range(len(data))))
    done = [r for r in rows if r["committed"]]
    ok = sum(r["ok"] for r in done)
    print(f"{ok}/{len(done)} correct of {len(rows)} asked ({time.time() - t0:.0f} s, {sum(r['tokens'] for r in rows)} tokens)", file=sys.stderr)
    for r in rows:
        if not r["ok"]:
            print(f"  wrong/uncommitted #{r['i']}: gold {r['gold']} got {r['got']} finish {r['finish']}", file=sys.stderr)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    print(f"METRIC gsm8k_acc={ok / max(len(done), 1):.4f}")
    print(f"METRIC gsm8k_committed={len(done) / len(rows):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
