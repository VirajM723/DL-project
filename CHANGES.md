# Changes vs. the paper: motivation, trade-offs, measured effect

**Base paper:** H. Tang, Y. Chen, L. Jiang, *TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking*, arXiv:2512.02789 (Dec 2025).

This document lists every place where our implementation differs from the paper:

1. **Necessary deviations**: these apply to *all* models equally, so the comparison stays fair.
2. **Our five modifications (M1–M5)**, which together turn TrackNetV5 into **TrackNetV5-Lite+**.

> **Status of the numbers.**
> All numbers in this document are **measured and confirmed** from the completed Kaggle dual Tesla T4 run (`gpu` profile, batch 16, 3,000 pre-train + 1,500 fine-tune steps). Full tables and raw metrics are saved in `results/changes_measured_gpu.md`, `results/changes_summary_gpu.csv`, and `results/real_changes_summary_gpu.csv`.

---

## 0. Fidelity check of the re-implementation
Our TrackNetV2 backbone at the paper's 512×288 input and ½ channel width has **2.836 M params / 56.5 GFLOPs**. Scaled to full width (params and FLOPs both ∝ width²), that is **≈ 11.34 M params / ≈ 113 GMACs**. The paper reports V2 at **11.33 M params / 112.89 G**. So the backbone size matches the paper's. On the Kaggle T4 the faithful V5 (MDD + R-STR) costs **66.5 GFLOPs, +17.8 % over V2**; almost all of that is the R-STR head's attention. *(An earlier CPU measurement gave +3.3 %: PyTorch's FLOP counter does not count the CPU attention kernel, so it missed most of the head.)*

## 1. Necessary deviations (apply to every model)
| Deviation | Paper | Ours | Why | Effect on conclusions |
|---|---|---|---|---|
| Data | TrackNetV2 badminton dataset (1280×720 broadcast) + private Loveall set | **Two stages.** (1) Pre-training and controlled ablation on procedural synthetic broadcasts (perspective court, ballistic ball with bounces, motion blur, occluding players, static look-alike distractors, dead-ball segments). (2) **Fine-tuning on the real TrackNet tennis dataset, games 1–8, and testing on unseen games 9–10.** | Synthetic data gives exact labels for occlusion and speed analysis and a large, controllable training set. The real tennis set (19.8k frames) tests whether the changes hold on broadcast footage. | Tennis instead of badminton, so absolute scores are **not** comparable to the paper's; *relative* effects are. |
| Channel width | 1.0 | 0.5 (`gpu` profile; `gpu_full` = 1.0) / 0.25 (`cpu`) | Free-tier GPU time | All variants are shrunk identically. |
| Schedule | AdamW 1e-4, batch 2, 30 epochs, ×0.1 at epochs 20/25 | AdamW 1e-3, batch 16, 3,000 steps (+1,500 real fine-tune), ×0.1 at 2/3 and 5/6 (dual T4 GPUs auto-double batch and halve steps from 6,000 / 3,000) | Shorter compute budget; disk-target models need higher initial lr to form initial peaks | Same proportional decay as the paper; identical for every variant. |
| Tolerance τ | 4 px at 1280×720 | 5.33 px at 512 px (≈ 13 px at 1280) + the full F1-vs-τ curve | 4 px at 1280 is 1.6 px at 512, about the size of label rounding | The τ-curve (Fig. 9) shows strict-τ behaviour too. |
| TSATTHead size | Not specified (layers / heads / dim) | patch 8, d = 64, 2 divided space-time blocks, 4 heads | Paper omits it | This is our best reading of "lightweight". |
| Fusion "Draft ⊕ MDD" | Not specified | Draft + w·A_t, with a learnable scalar w (init 1) and A_t the per-frame max-polarity motion map | Paper omits it | Used identically in V5 and in our variants. |

---

## 2. Our modifications

### M1: Ball-size-aware Gaussian targets + sub-pixel soft-argmax decoding
* **What changed.** `make_heatmap(kind="gauss")`: the target is $\exp(-d^2/2\sigma^2)$ with $\sigma=\max(1, 0.8\,r_{ball})$, replacing V5's binary disk of radius 30 px (at 1280). Decoding uses `decode_softargmax`: global peak → probability-weighted centroid in a 5×5 window. It replaces the contour / bounding-box centre used by V2/V3/V5.
* **Why.** The disk is **≈ 60 px wide while the ball is 5–15 px**. Every pixel inside the disk is an equally correct answer, so the network has no reason to put its maximum on the centre. A bbox centre of a thresholded blob is biased whenever the blob is clipped by an occluder or merges with a racket.
* **Trade-off.** Less positive "mass" per frame can slow early training and cost recall on the faintest balls. Cost: **zero extra parameters or FLOPs**; the soft-argmax is cheaper than contour extraction.
* **Expected / where to look.** Lower `err_mean`, and much higher F1 at strict τ (Fig. 9, "F1@τ" table).

### M2: Per-pair, per-polarity MDD + explicit direction channel
* **What changed.** `MDD(per_channel=True, direction=True)`: separate learnable (α, β) for each of {Δ(t−1→t), Δ(t→t+1)} × {P⁺, P⁻} (4 slopes, 4 thresholds instead of 1 + 1). Two extra input channels $A^{+}-A^{-}$ also give the network a signed "dipole" pointing along the direction of travel.
* **Why.** In the paper's Eq. 2, $|x|$ is applied to $P^{\pm}$, which are already ≥ 0 after the ReLU, and one shared (α, β) thresholds both polarities. A ball *arriving* on a dark court and *leaving* a bright line produce differences of different magnitude. One threshold can't fit both, and the direction (the point of V5's "decoupling") is left for the backbone to rediscover.
* **Trade-off.** **+6 scalars, +2 input channels**: +0.0006 M params, **+0.17 GFLOPs (+0.3 %)**, +0.02 s per CPU training step (+1.5 %). Risk: the thresholds could overfit to the synthetic intensity statistics.
* **Expected / where to look.** Better hit-rate on the fastest speed bin (Fig. 11). Learned k and m values and the dipole maps are in Fig. 14.

### M3: Consistent, slim R-STR refinement head
* **What changed.**
  1. `TrackNet(consistent=True)`: the head receives **Dropout(Draft_MDD) during training and Draft_MDD at inference**.
  2. `RSTR_CFG["slim"]`: patch 16, d = 32, 1 block, 2 heads, versus the full head's patch 8, d = 64, 2 blocks, 4 heads. That is **4× fewer tokens** per frame.
* **Why.**
  1. As written, the paper trains the Transformer on Dropout(Draft) but runs it on Draft_MDD = Draft fused with the motion maps. The head never sees its test-time input distribution during training. The fusion weight never receives a gradient either, so motion-heavy regions (moving players) get an untrained logit boost at test time.
  2. Self-attention cost is quadratic in tokens. The head was the single most expensive addition of V5.
* **Measured cost change (exact).** Head **0.262 M → 0.058 M params (−78 %)**; whole model **66.71 → 57.28 GFLOPs on the T4**, i.e. the head's compute falls by more than 90 % (4× fewer tokens make attention ~16× cheaper). CPU training step: 1.32 → 0.99 s (**−25 %**).
* **Trade-off.** A 16-px residual patch refines less finely than an 8-px one. The consistency fix itself is free.
* **Expected / where to look.** Higher precision and fewer FP2 (Fig. 7). The draft → residual → final decomposition is in Fig. 15.

### M4: Depthwise-separable backbone
* **What changed.** `UNetV2(dw=True)`: every 3×3 conv except the first stage becomes a depthwise 3×3 followed by a pointwise 1×1 (MobileNet-style).
* **Measured cost change (exact).** **2.894 M → 0.410 M params (−86 %, 7.1× smaller)**, **57.28 → 10.92 GFLOPs (−81 %, 5.2× fewer)**.
* **Trade-off.** **FLOPs ≠ latency.** On the laptop CPU the training step went only from 0.99 s to 0.96 s (−3 %), and batch-1 inference FPS rose only 9–23 % (depending on resolution). Depthwise kernels are memory-bandwidth-bound and poorly optimised compared with dense 3×3 convs. The gain is real for model size and storage, and on accelerators with good depthwise kernels (mobile NPUs, TensorRT), but not on a generic CPU. The capacity loss is measured as ΔF1.
* **Where to look.** Fig. 8 (accuracy vs. GFLOPs and vs. FPS).

### M5: Physics-aware temporal post-processing (no training)
* **What changed.** `kalman_refine`:
  - A constant-acceleration Kalman filter (state x, y, ẋ, ẏ, ẍ, ÿ) whose measurement noise scales with detection confidence.
  - χ²(2 dof, 99.9 %) gating rejects outlier detections. Two consecutive rejected detections that agree with each other re-initialise the track (a hit or bounce).
  - Gaps of ≤ 4 frames are filled with a quadratic-in-time fit to ≤ 3 detections on each side (ballistic flight). Fills that would leave the frame are discarded.
* **Why.** V5's authors name *false positives in complex backgrounds* as their remaining limitation. V5 also has no occlusion recovery; V3 had a learned rectification network, but V5 dropped it. A ball obeys simple physics between contacts, so that prior is free to use.
* **Trade-off.** **0 parameters, 0 GFLOPs**, microseconds per frame. Risks: true detections right after an abrupt hit can be rejected (lower recall), and a hit *inside* a gap is interpolated wrongly.
* **Where to look.** §8 table (raw / gating only / gating + filling), occluded-frame hit-rate, Fig. 12.

---

## 3. Efficiency summary (measured, exact)
`gpu` profile, 512×288, width 0.5, one 3-frame MIMO forward pass, counted on the Kaggle Tesla T4:

| Model | Params (M) | GFLOPs | vs. faithful V5 |
|---|---|---|---|
| TrackNetV2 (baseline) | 2.836 | 56.51 | − |
| **TrackNetV5 (faithful)** | **3.098** | **66.54** | 1.00× |
| V5 + M1 | 3.098 | 66.54 | 1.00× |
| V5 + M1 + M2 | 3.098 | 66.71 | 1.00× |
| V5 + M1 + M2 + M3 | 2.894 | 57.28 | 0.86× |
| **V5-Lite+ (M1–M4)** (+ M5, which is free) | **0.410** | **10.92** | **0.13× params, 0.16× FLOPs** |

Training step time on the laptop CPU (`cpu` profile, batch 8): V2 0.77 s · V5 1.27 s · +M1 1.30 s · +M2 1.32 s · +M3 0.99 s · +M4 0.96 s.

## 4. Validation
`tests/` (48 checks, see the README) validates every component against ground truth. Validation found and fixed these bugs before any training run:
- **Occlusion labels.** "Occluded" used to mean "any player pixel covers the ball centre". A 1-px racket string or a covered blur *tail* was enough, so 71 % of the frames labelled occluded showed a clearly visible ball. Now a frame counts as occluded only when a player hides more than half of the ball disc at the labelled position. The ball centre is now hidden in 86 % of occluded frames, with a median of 21 % of the ball still visible (anti-aliased edges).
- **Camera shake.** The jitter moved only the ball and players while the court stayed fixed. The whole frame is now translated, which (realistically) makes static court lines and crowd specks appear in the frame differences.
- **M1 decoder bias.** The plain centroid in a 5×5 window around the integer peak was biased by ≈ 0.24 px. Value-thresholded weights max(p − 0.1·peak, 0) in a resolution-scaled window give < 0.05 px error.
- **Visual-analysis crash.** §9.2/9.3 took the frames t−1…t+1 and failed when t was the last frame of a clip.
- **Smaller fixes.** Resolution-dependent constants (τ, robustness bins, Kalman noise) now scale with `SC = W/288`. The real-data loader also finds datasets attached on Kaggle. Figure legends and axis limits were adjusted.

## 5. Real-footage protocol
| | |
|---|---|
| Dataset | TrackNet tennis (Huang et al., 2019): 10 broadcast matches, 95 clips, 19,835 frames at 1280×720, resized to the working resolution |
| Split | **By match.** Fine-tune on games 1–8, using the last clip of each game as validation (checkpoint selection only). Test on games 9–10, which are never seen before evaluation. |
| Labels | Visibility 0 → no ball; 1 (easy) and 2 (hard) → visible; 3 → occluded. TrackNet's original classes are kept for a per-class breakdown (Fig. 17). The Gaussian width for M1 assumes a ball ≈ 8 px wide at 1280. |
| Fine-tuning | Every variant starts from its own synthetic checkpoint: 3,000 steps × batch 8 at lr 3e‑4 (`gpu` profile), the same WBCE loss and augmentation, and the same ×0.1 decay at 2/3 and 5/6 of training. The starting checkpoint is itself a candidate, so fine-tuning can never end up worse than zero-shot on validation. |
| Reported | Zero-shot (synthetic only), fine-tuned, and fine-tuned + M5, all on games 9–10 at the same τ. The per-change table is computed on the fine-tuned models. |

Why not a random frame split? Neighbouring frames are nearly identical, so a random split leaks the test set into training and inflates scores. TrackNet V1's own 99.7 % precision was on such a split; its 10-fold cross-validation over videos dropped to F1 84.3 %.

## 6. Accuracy deltas
These are produced by the Kaggle run: see `results/changes_measured_gpu.md`, or §10 of the executed notebook, which states for each change whether it improved, hurt or left F1 unchanged. When reporting, treat |ΔF1| < 0.01 as noise (single seed).
