#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single File + CLI Version
"""

import os, json, gzip, random, hashlib, time
import argparse
from pathlib import Path

from datasets import load_dataset
from langdetect import detect
from transformers import AutoTokenizer


# ------------------------------------------------
# CLI PARSER
# ------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="FineWeb data collector with filters, sampling, and .gz output"
    )

    p.add_argument("--repo", "-r", type=str,
                   default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset ID")

    p.add_argument("--split", "-s", type=str, default="train",
                   help="Dataset split")

    p.add_argument("--out", "-o", type=str,
                   default="collected_output.jsonl.gz",
                   help="Output file")

    p.add_argument("--tokens", "-t", type=int,
                   default=50_000_000,
                   help="Target token count")

    p.add_argument("--sample", "-p", type=float,
                   default=0.20,
                   help="Sampling probability (0-1)")

    p.add_argument("--lang", "-l", type=str, default="pt",
                   help="Expected language (or None to disable)")

    p.add_argument("--min-chars", type=int, default=500)
    p.add_argument("--max-chars", type=int, default=200_000)

    p.add_argument("--checkpoint", type=str,
                   default="checkpoint_state.json",
                   help="Checkpoint file")

    return p.parse_args()


# ------------------------------------------------
# INTERNAL FUNCTIONS
# ------------------------------------------------

def deterministic_sample(text, p):
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xffffffff
    return v < p


def detect_lang_safe(text):
    try:
        return detect(text[:1000])
    except:
        return "unknown"


def clean_text(text):
    return text.replace("\r", " ").replace("\t", " ").strip()


def token_count(tok, text):
    try:
        enc = tok([text], add_special_tokens=False, return_length=True)
        return enc["length"][0]
    except:
        return 0


def load_state(path):
    if not os.path.exists(path):
        return {"tokens": 0, "docs": 0}
    return json.load(open(path))


def save_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f)


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

def main():
    args = get_args()

    print("== Collector (CLI) ==")
    print(f"[INFO] Dataset:       {args.repo}")
    print(f"[INFO] Split:         {args.split}")
    print(f"[INFO] Output:        {args.out}")
    print(f"[INFO] Target Tokens: {args.tokens:,}")
    print(f"[INFO] Language:      {args.lang}")
    print("-" * 50)

    state = load_state(args.checkpoint)

    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    tokens_total = state["tokens"]
    docs_total = state["docs"]

    with gzip.open(args.out, "at", encoding="utf-8") as out:
        for row in ds:
            text = clean_text(row.get("text", ""))

            # Basic filters
            if len(text) < args.min_chars or len(text) > args.max_chars:
                continue

            if not deterministic_sample(text, args.sample):
                continue

            if args.lang is not None:
                lg = detect_lang_safe(text)
                if lg != args.lang:
                    continue

            # Tokenization
            ntok = token_count(tok, text)
            if ntok == 0:
                continue

            # Save
            obj = {
                "text": text,
                "n_tokens": ntok,
                "repo": args.repo,
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

            tokens_total += ntok
            docs_total += 1

            # Checkpoint
            if tokens_total // 5_000_000 != state["tokens"] // 5_000_000:
                state = {"tokens": tokens_total, "docs": docs_total}
                save_state(args.checkpoint, state)
                print(f"[CHECKPOINT] {tokens_total:,} tokens | {docs_total:,} docs")

            if tokens_total >= args.tokens:
                break

    print("\n== FINISHED ==")
    print(f"Tokens: {tokens_total:,}")
    print(f"Docs:   {docs_total:,}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
