#!/usr/bin/env python3
"""One needle near the window's end (stdlib only): needle.py's prompt and code over non-repeated filler long enough for
~1M tokens (Project Gutenberg: Moby-Dick #2701, War and Peace #2600, Pride and Prejudice #1342, in bench/rexl3/data/),
cut to --chars characters (about 4.1 per token: 4,200,000 measured 1,027,057 tokens). Greedy, thinking off, 24 tokens.

  needle_long.py --base http://HOST:8080 [--chars 4200000] [--depth 0.5]

Prints one line and METRIC needle_long_pass / needle_long_tokens / needle_long_tok_s. Exit 1 on a miss.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BOOKS = ("moby_dick_pg2701.txt", "pg2600.txt", "pg1342.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE", "http://192.168.1.121:8080"))
    ap.add_argument("--chars", type=int, default=4_200_000)
    ap.add_argument("--depth", type=float, default=0.5)
    a = ap.parse_args()
    book = "".join(open(os.path.join(HERE, "data", b), encoding="utf-8").read() for b in BOOKS)
    if a.chars > len(book):
        sys.exit(f"--chars {a.chars} exceeds the filler ({len(book)} characters)")
    body = book[:a.chars]
    code = hashlib.sha256(f"{a.chars}:{a.depth}".encode()).hexdigest()
    code = code[:3].upper() + "-" + code[3:7]
    at = body.rfind("\n", 0, int(len(body) * a.depth))
    text = ("You will be asked for an access code hidden in the text below.\n\n" + body[:at] +
            f"\nThe access code for the archive is {code}.\n" + body[at:] +
            "\n\nWhat is the access code for the archive? Reply with the code only.")
    body = {"model": os.environ.get("SERVED_NAME", "RED-SNOW-5.3-Flash"), "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    dt = time.time() - t0
    reply = r["choices"][0]["message"].get("content") or ""
    n = r["usage"]["prompt_tokens"]
    ok = code in reply
    print(f"{n:>8} tokens depth {a.depth:.2f}: {'PASS' if ok else 'MISS'} ({dt:.0f} s) want {code!r} got {reply.strip()[:40]!r}",
          file=sys.stderr)
    print(f"METRIC needle_long_pass={1.0 if ok else 0.0:.4f}")
    print(f"METRIC needle_long_tokens={n}")
    print(f"METRIC needle_long_tok_s={n / dt:.1f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
