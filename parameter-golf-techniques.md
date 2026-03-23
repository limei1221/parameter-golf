# Parameter Golf Leaderboard Techniques (Most to Least Promising)

Based on the current leaderboard (best: **1.1428**, baseline: **1.2244**) as of 2026-03-23.

## 1. Aggressive Quantization (Int5/Int6 QAT) — ~0.04+ bpb saved indirectly

The single most enabling technique. By compressing weights from int8 → int6 → int5 using **Straight-Through Estimator (STE) fake quantization** during training, models learn to be robust to low-precision. This frees up massive budget (~2-3MB) to add more layers and wider MLPs. The top run uses **int5 for MLP weights** and **int6 for attention weights**, with zstd-22 compression on top.

## 2. 3x MLP Expansion — ~0.029 bpb

Widening the MLP hidden dim from 2x to 3x (1536) is the single largest direct contributor to quality. Only feasible because quantization savings free the size budget. Every top-5 entry uses 3x MLP.

## 3. Sliding Window Evaluation — ~0.032 bpb (free)

Zero cost at training time. By evaluating with overlapping windows (stride=64), every token gets ~960+ tokens of context instead of ~512 average. Pure eval-time trick with no artifact cost. Every competitive entry uses it.

## 4. More Layers (10-11L) — ~0.01-0.02 bpb

Going from 9→10→11 layers adds depth/capacity. Again funded by quantization savings. The top entry uses 10 layers.

## 5. BigramHash Embedding — ~0.005-0.01 bpb

A hash table (4096-10240 buckets, dim=128) that maps consecutive token pairs via `(prev*92821 + cur) % buckets`. Gives the model explicit bigram context cheaply. Larger tables (10240) reduce collisions and help more.

## 6. SmearGate — ~0.005 bpb

A per-dimension learned gate that blends the current token embedding with the previous token's embedding. Provides bigram signal without self-attention. Complements BigramHash — together they give strong local context.

## 7. Stochastic Weight Averaging (SWA) — ~0.002-0.004 bpb

Average checkpoints from the last 40-50% of training. Produces smoother weight distributions that quantize better and generalize better. Small but consistent gain.

## 8. FP16 Tied Embeddings — ~0.005 bpb

Keep the tied embedding in FP16 instead of quantizing to int8. Embeddings serve double duty (input + output), so quantization errors compound. Costs ~1MB extra but is well worth it.

## 9. Muon Weight Decay (0.02-0.04) — ~0.002-0.005 bpb

Decoupled weight decay on the Muon optimizer keeps weight magnitudes small, directly benefiting both generalization and quantization robustness.

## 10. Orthogonal Weight Initialization — ~0.002 bpb

Initialize all linear layers with orthogonal matrices (singular values = 1). Gives uniform gradient flow and faster convergence within the limited ~12k step budget.

## 11. Longer Sequence Length (2048-4096) — ~0.02 bpb

Training with longer sequences gives richer context per sample. The tradeoff is fewer steps per wallclock. 2048 is the sweet spot; 4096 helps but costs more steps.

## 12. Muon Momentum Warmup (0.92→0.99) — small but helpful

Gradually increasing Muon momentum stabilizes early training and allows higher final momentum for better convergence.

## 13. Test-Time Training (LoRA TTT) — ~0.037 bpb but orthogonal

Per-document LoRA adaptation at eval time. Significant improvement but expensive at eval time and hasn't been combined with the top quantization techniques yet. Potentially very promising if combined with the other wins.

## 14. Spectral/Overtone Embedding Init — ~0.002 bpb

SVD power-law spectrum shaping for embeddings. Marginal gains, mostly superseded by other init techniques.

## 15. Extended Warmdown / LR Tuning — ~0.001-0.003 bpb

Longer warmdown schedules (1200→3000 steps) and tuned learning rates. Standard hyperparameter optimization — necessary but incremental.

---

## Key Insight

The top entries stack 5-7 of these techniques together. The biggest unlocks are *quantization-aware training* (which funds more capacity) + *sliding window eval* (free quality) + *wider MLPs* (best use of freed budget). LoRA TTT is notably absent from the top entries and could be a major opportunity if combined with the current best practices.
