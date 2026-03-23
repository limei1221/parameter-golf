# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**OpenAI Model Craft Challenge: Parameter Golf** — a competition to train the best language model that fits in a **16MB artifact** and trains in under **10 minutes on 8xH100s**, scored by bits-per-byte (bpb) compression on the FineWeb validation set (lower is better). Baseline: 1.2244 bpb. Current SOTA: ~1.1428 bpb.

The challenge optimizes L(N) — lowest loss given fixed model size — with no constraints on architecture, tokenizer, or training recipe.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download dataset (FineWeb with 1024-token SentencePiece vocab):
```bash
# Full dataset (8B tokens, ~80 shards)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Smoke-test subset (1 shard)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

## Running Training

**Local smoke test (Mac/MLX):**
```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

**Single GPU (1xH100):**
```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Multi-GPU (8xH100, leaderboard submission):**
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful env vars:
- `MAX_WALLCLOCK_SECONDS=0` — disable 10-minute cap (for experiments)
- `VAL_LOSS_EVERY=200` — print periodic validation during training
- `ITERATIONS=N` — override step count

## Key Files

- **`train_gpt.py`** — Primary PyTorch training script (hard limit: ≤1500 lines). This is the submission artifact. All submission code must live here.
- **`train_gpt_mlx.py`** — MLX version for Apple Silicon local iteration.
- **`data/cached_challenge_fineweb.py`** — Downloads preprocessed FineWeb dataset from HuggingFace.
- **`data/download_hf_docs_and_tokenize.py`** — Rebuilds tokenizer from source (standalone, no local imports).

## Architecture Notes

`train_gpt.py` contains:
- `Hyperparameters` class — all config, overridable via environment variables
- Muon optimizer with Newton-Schulz orthogonalization (from modded-nanogpt)
- DDP (DistributedDataParallel) multi-GPU training
- Artifact size calculated as: `len(script_bytes) + len(zlib_compressed_model_bytes)`

## Submission Rules

1. All submission code must fit in `train_gpt.py`
2. Artifact limit: **16,000,000 bytes** (decimal MB, not MiB) = code + zlib-compressed model weights
3. Training must complete in ≤10 minutes on 8xH100 SXM; evaluation gets an additional 10 minutes
4. No network access during evaluation; model must be fully self-contained
5. Cannot access validation data during training
6. New SOTA records must beat the prior record by ≥0.005 nats with `p < 0.01` statistical significance (multiple seeds required)
7. External packages are allowed (e.g., FlashAttention) as long as they don't sneak in extra compute or effective code size

## Submission Structure

Each record goes in `records/track_10min_16mb/<date>_<name>/` containing:
- `README.md` — technique description and ablations
- `submission.json` — metadata (author, score, bytes, date)
- `train_gpt.py` — the submission script
- Training logs (multiple seeds for reproducibility)

Non-leaderboard experiments go in `records/track_non_record_16mb/`.

## Techniques That Have Worked

Proven improvements over baseline (cumulative contributions):
- **Int5/Int6 QAT** — quantization-aware training (~0.04+ bpb)
- **3× MLP expansion** — wider MLP layers (~0.029 bpb)
- **Sliding window evaluation** — longer effective context at eval (~0.032 bpb, zero training cost)
- **10-11 layers** — deeper models (~0.01–0.02 bpb)
- **BigramHash embeddings** — bigram hashing for embeddings (~0.005–0.01 bpb)
- **SmearGate** — activation modification (~0.005 bpb)
- **Stochastic Weight Averaging (SWA)** — averaging weights near end of training (~0.002–0.004 bpb)
- **FP16 tied embeddings** — half-precision tied embed/unembed (~0.005 bpb)
- **Muon weight decay** — WD on Muon optimizer (~0.002–0.005 bpb)
- **Orthogonal initialization** — spectral/orthogonal weight init (~0.002 bpb)
- **Longer sequences** (2048–4096 tokens) — longer context during training (~0.02 bpb)
