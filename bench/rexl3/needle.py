#!/usr/bin/env python3
"""Needle in a book (stdlib only): does the model find a code buried in real text at a given length and depth?

  needle.py --base http://HOST:8000 --tokens 8000,32000,120000 [--depths 0.1,0.5,0.9]

The filler is Project Gutenberg's Moby-Dick (bench/data/moby_dick_pg2701.txt, sha256 b5e88b71…), never repeated text:
a sparse (top-2048) attention that misplaces its keys fails here where a repeated sentence would not. The code is a fresh
7-character string per item; the reply is non-thinking and 24 tokens. Prints one line per item and METRIC needle_pass
(fraction). Exit 1 on any miss. Prompt tokens come from the server's usage, not from this script's estimate.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS_PER_TOKEN = 3.7  # measured: the server counted 9,341 tokens for 34,400 characters of this book


def code_for(tokens, depth):
    h = hashlib.sha256(f"{tokens}:{depth}".encode()).hexdigest()
    return h[:3].upper() + "-" + h[3:7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", "http://192.168.1.121:8080"))
    ap.add_argument("--tokens", default="8000,32000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--max-tokens", type=int, default=24)
    a = ap.parse_args()
    book = open(os.path.join(HERE, "data", "moby_dick_pg2701.txt"), encoding="utf-8").read()
    ok = n = 0
    for tokens in (int(x) for x in a.tokens.split(",")):
        body = book[: int(tokens * CHARS_PER_TOKEN)]
        for depth in (float(x) for x in a.depths.split(",")):
            code = code_for(tokens, depth)
            at = body.rfind("\n", 0, int(len(body) * depth))
            text = ("You will be asked for an access code hidden in the text below.\n\n" + body[:at] +
                    f"\nThe access code for the archive is {code}.\n" + body[at:] +
                    "\n\nWhat is the access code for the archive? Reply with the code only.")
            req = urllib.request.Request(a.base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"model": "RED-SNOW-5.3-Flash", "messages": [{"role": "user", "content": text}],
                                                          "temperature": 0, "max_tokens": a.max_tokens,
                                                          "chat_template_kwargs": {"enable_thinking": False}}).encode())
            t0 = time.time()
            r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
            reply = r["choices"][0]["message"].get("content") or ""
            hit = code in reply
            ok += hit
            n += 1
            print(f"{r['usage']['prompt_tokens']:>7} tokens depth {depth:.1f}: {'PASS' if hit else 'MISS'} "
                  f"({time.time() - t0:.0f} s) {reply.strip()[:40]!r}", flush=True)
    print(f"METRIC needle_pass={ok / n:.4f}")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
