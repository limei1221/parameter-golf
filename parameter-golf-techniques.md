# Parameter Golf Leaderboard Techniques (Most to Least Promising)

Updated 2026-03-23. Current top 5: **1.1228 / 1.1249 / 1.1271 / 1.1307 / 1.1428 bpb** (baseline: 1.2244).

---

## Tier 1: Foundational (used by all top-5 entries)

### 1. Aggressive Quantization (Int5/Int6 QAT) — ~0.04+ bpb saved indirectly

The single most enabling technique. STE fake quantization during training makes models robust to low-precision weights. Frees ~2-3MB of artifact budget for more layers and wider MLPs. Top entries use **int6 per-row** for MLP/attention, **int8** for embeddings, with **zstd-22** compression. #5 uses int5 for MLPs to save even more space.

### 2. 3x MLP Expansion — ~0.029 bpb

Widening MLP hidden dim from 2x to 3x (1536) is the largest direct quality contributor. Only feasible because quantization frees the size budget. Every top-5 entry uses it.

### 3. Sliding Window Evaluation — ~0.032 bpb (free at training time)

Zero training cost. Overlapping windows (stride=64) give every token ~960+ tokens of context instead of ~512 average. Pure eval-time trick. Every competitive entry uses it.

### 4. 11 Layers with U-Net Skip Connections — ~0.01-0.02 bpb

Going deeper (9 -> 11 layers) adds capacity funded by quantization savings. U-Net skip connections (encoder half feeds decoder half) stabilize training in deeper models. Top 4 entries all use 11 layers.

### 5. Weight Averaging (EMA or SWA) — ~0.002-0.006 bpb

All top entries use some form of weight averaging. **EMA** (decay=0.997, every step) has replaced SWA in the top 3 entries as it provides smoother, continuous weight smoothing. SWA (averaging checkpoints from last 40% of warmdown) still works well (#4, #5). Both improve generalization and quantization robustness.

### 6. SmearGate + BigramHash — ~0.005-0.01 bpb combined

**BigramHash** (4096-10240 buckets, dim=128) maps consecutive token pairs via hashing for explicit bigram context. **SmearGate** blends current/previous token embeddings via a learned gate. Together they give strong local context before self-attention. All top-5 entries use both.

---

## Tier 2: High-Impact Differentiators (separate top-3 from the rest)

### 7. Exclusive Self Attention (XSA) — ~0.005 bpb

Applied to the last 3-4 layers only. Subtracts the self-value component from attention output, forcing the model to learn purely from context rather than self-reinforcing. Efficient GQA-aware implementation using reshape + broadcasting keeps overhead to ~2ms/step. Present in top 3 entries (#1, #2, #3).

### 8. GPTQ-lite (Per-Layer Optimal Clipping) — ~0.0006 bpb (free)

Post-training optimization: search 5 clip percentile candidates (0.999 to 1.0) per layer for int6 quantization, picking the one that minimizes reconstruction MSE. Zero training cost, pure eval-time improvement. Used by #1 entry.

### 9. Partial RoPE (16 of 64 dims) — ~0.002 bpb

Apply rotary position embeddings to only 25% of head dimensions, letting the remaining 75% learn position-invariant patterns. Simple change with meaningful improvement. Used by #1 and #2.

### 10. LN Scale Factor (1/sqrt(layer+1)) — ~0.0005 bpb

Damp RMSNorm outputs by layer depth to stabilize training in deep (11L) models. Small but consistent. Used by #1 and #2.

### 11. EMA replacing SWA — incremental over SWA

Exponential moving average (decay=0.997) updated every step provides smoother weight distributions than discrete SWA snapshots. The top 3 entries all switched from SWA to EMA.

---

## Tier 3: Solid Contributors (widely used, proven gains)

### 12. FP16 Tied Embeddings — ~0.005 bpb

Keep tied embedding in FP16 instead of quantizing to int8. Embeddings serve double duty (input + output), so quantization errors compound. Costs ~1MB extra but worth it.

### 13. Muon Weight Decay (0.04) — ~0.002-0.005 bpb

Decoupled weight decay on Muon optimizer keeps weight magnitudes small, benefiting generalization and quantization. Top entries converged on WD=0.04.

### 14. Orthogonal Weight Initialization — ~0.002 bpb

Initialize all linear layers with orthogonal matrices (singular values = 1). Uniform gradient flow and faster convergence within the limited ~12k step budget.

### 15. Extended Warmdown (3000-3500 iters) — ~0.001-0.003 bpb

Longer warmdown schedules give weights more time to settle before final quantization. #1 entry uses 3500 iters.

### 16. Longer Sequence Length (2048-4096) — ~0.02 bpb

Richer context per sample during training. 2048 is the sweet spot; 4096 helps but costs more steps.

---

## Tier 4: Promising but Unproven in Top Runs

### 17. Test-Time Training (LoRA TTT) — ~0.037 bpb but orthogonal

Per-document LoRA adaptation at eval time. Significant improvement in isolation but expensive at eval time and **not yet combined** with current top techniques (XSA, EMA, GPTQ-lite). Potentially the biggest remaining opportunity if successfully stacked.

### 18. FlashAttention 3 — speed, not bpb

Not a quality improvement per se, but enables fitting more computation in the 10-minute budget. Used by several top entries.

### 19. Muon Momentum Warmup (0.92 -> 0.99) — small

Gradually increasing Muon momentum stabilizes early training. Standard practice, marginal isolated impact.

### 20. Magnitude Pruning (3%) — small

Prune smallest 3% of weights. Used by #5. Minor savings and quality is ambiguous at this scale.

---

## Key Insights

1. **The top 3 are separated by architecture innovations** (XSA, Partial RoPE, GPTQ-lite) layered on top of the now-standard stack of QAT + 3x MLP + sliding eval + 11L + EMA + SmearGate + BigramHash.

2. **EMA has replaced SWA** in the best entries — continuous averaging outperforms discrete snapshots.

3. **GPTQ-lite is free performance** — a post-training clipping search that costs nothing during training.

4. **LoRA TTT remains the biggest untapped opportunity** — 0.037 bpb in isolation, not yet combined with the current SOTA stack. If combined with the #1 entry's techniques, could potentially push below 1.09 bpb.

5. **Diminishing returns from hyperparameter tuning** — the gap between #1 and #2 is only 0.002 bpb, suggesting architecture innovations matter more than tuning at this point.
