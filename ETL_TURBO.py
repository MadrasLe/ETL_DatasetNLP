#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETL Fast- Minimalistic & Fast Data Collector
Converted to CLI for simple and direct use.

- Streaming / Parquet loading
- Simple filtering (min/max chars)
- Optional sampling
- Token counting with chosen tokenizer
- Saves clean text lines (no EOS, no packing)
"""

import argparse
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm


# -------------------------------------------------------------
# CLI PARSER
# -------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Fast ETL collector for NLP datasets (minimalistic, no checkpoint, no dedupe)."
    )

    p.add_argument("--repo", type=str, default="HuggingFaceFW/fineweb-edu",
                   help="Dataset repo ID on HuggingFace")

    p.add_argument("--tokenizer", type=str, default="gpt2",
                   help="Tokenizer used for token counting")

    p.add_argument("--split", type=str, default="train",
                   help="Dataset split")

    p.add_argument("--quota", type=int, default=1_200_000_000,
                   help="Target token quota (stops after reaching this amount)")

    p.add_argument("--min-chars", type=int, default=2000,
                   help="Minimum document length")

    p.add_argument("--max-chars", type=int, default=100_000,
                   help="Maximum document length")

    p.add_argument("--sample", type=float, default=0.15,
                   help="Sampling probability (0–1). Use 1.0 to disable sampling.")

    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size for tokenization")

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")

    p.add_argument("--out", type=str, default="dataset_out.txt",
                   help="Output file (clean text, one document per line)")

    return p.parse_args()


# -------------------------------------------------------------
# BATCHER
# -------------------------------------------------------------
def batcher(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


# -------------------------------------------------------------
# TRY TO LOAD PARQUET OR STREAM
# -------------------------------------------------------------
def load_dataset_flexible(repo, split):
    patterns = [
        f"hf://datasets/{repo}/data/**/*.parquet",
        f"hf://datasets/{repo}/**/*.parquet",
    ]

    print("Attempting to load via parquet...")
    for pat in patterns:
        try:
            ds = load_dataset(
                "parquet",
                data_files=pat,
                columns=["text"],
                split=split,
                streaming=True
            )
            print(f"Loaded via parquet: {pat}")
            return ds
        except Exception as e:
            print(f"[parquet] failed at {pat}: {type(e).__name__}: {e}")

    print("Parquet failed. Attempting official script...")
    return load_dataset(repo, split=split, streaming=True, verification_mode="no_checks")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    args = get_args()

    random.seed(args.seed)

    print(f"\n== ETL Fast Mistral ==")
    print(f"Dataset.........: {args.repo}")
    print(f"Tokenizer.......: {args.tokenizer}")
    print(f"Quota...........: {args.quota:,} tokens")
    print(f"Output..........: {args.out}")
    print("---------------------------\n")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    ds = load_dataset_flexible(args.repo, args.split)

    collected_tokens = 0
    pbar = tqdm(total=args.quota, unit=" token", desc="Collecting tokens")

    with open(args.out, "w", encoding="utf-8") as fout:
        for batch in batcher(ds, args.batch_size):
            texts = []
            for ex in batch:
                txt = ex.get("text", "")
                if not txt:
                    continue
                if len(txt) < args.min_chars or len(txt) > args.max_chars:
                    continue

                if args.sample < 1.0 and random.random() > args.sample:
                    continue

                clean = txt.strip().replace("\n", " ")
                texts.append(clean)

            if not texts:
                continue

            # Token counting
            token_ids = tok(texts, add_special_tokens=False)["input_ids"]
            token_counts = [len(t) for t in token_ids]

            for text_line, ntok in zip(texts, token_counts):
                if ntok == 0:
                    continue

                fout.write(text_line + "\n")
                collected_tokens += ntok
                pbar.update(ntok)

                if collected_tokens >= args.quota:
                    pbar.close()
                    print(f"\nDone! Total: {collected_tokens:,} tokens.")
                    print(f"File saved: {args.out}")
                    return

    pbar.close()
    print(f"\nFinished with {collected_tokens:,} tokens.")
    print(f"File saved: {args.out}")


if __name__ == "__main__":
    main()