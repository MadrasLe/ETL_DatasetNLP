![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![HF Datasets](https://img.shields.io/badge/HuggingFace-Datasets-yellow.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

# ETL_DatasetNLP
A collection of high-quality ETL pipelines for NLP and LLM training datasets.  
This repository provides **simple, scalable and production-grade** data collectors using:

- Streaming extraction from HuggingFace datasets  
- Text cleaning and normalization  
- Deterministic sampling  
- Language filtering  
- Tokenization  
- Deduplication strategies  
- Checkpointing for long runs  
- Output in `.jsonl.gz` format (LLM-friendly)


---

##  Features

- **Single-file ETL pipelines** (easy to extend, inspect and maintain)  
- **HuggingFace streaming mode** for massive datasets  
- **Deterministic sampling** based on content hashes  
- **Optional language filtering** via lightweight detection  
- **Token counting** using `transformers`  
- **Automatic checkpoints** (resume-friendly)  
- **Output compatible with LLaMA, Gemma, Mistral, JAX, PyTorch, HuggingFace, Megatron, DeepSpeed, etc.**  
- **Minimal dependencies** — easy environment setup 

---

## 📁 Current Pipelines

### `ETL.py` (or the name you choose)
A fully CLI-enabled extractor for NLP datasets using HuggingFace streaming, with:

- cleaning  
- sampling  
- token counting  
- language filtering  
- progress checkpoints  
- JSONL output compressed with gzip  

This file is designed as a **base pipeline**:  
you can duplicate and adapt it for any dataset you want to ETL.

---

## Installation

```bash
pip install -r requirements.txt
```

 How to Run (CLI Examples)

Run with defaults:
```bash
python ETL_complete.py
```

Specify a different dataset:
```bash
python ETL_complete.py --repo allenai/c4 --split train
```

Collect 100M tokens:
```bash
python ETL_complete.py --tokens 100000000
```

Change sampling probability:
```bash
python ETL_complete.py --sample 0.10
```

Disable language filtering:
```bash
python ETL_complete.py --lang None
```

Set minimum text length:
```bash
python ETL_complete.py --min-chars 300
```

Full example:
```bash
python ETL_complete.py \
  --repo HuggingFaceFW/fineweb-edu \
  --split train \
  --tokens 50000000 \
  --lang pt \
  --sample 0.20 \
  --min-chars 500 \
  --out dataset_pt.jsonl.gz
```
