# Project memory — handover notes

This file is for any person or AI assistant picking up the project. It assumes no particular tool: only Python, a shell, and Word (or any .docx editor).

The project re-implements **TrackNetV5** (Tang, Chen & Jiang, arXiv:2512.02789, Dec 2025) for tracking tiny, fast tennis balls, and improves it as **TrackNetV5-Lite+** with five modifications, M1–M5.

- **Course:** CSPC72 Deep Learning Techniques, Semester VII.
- **Team:** 3 students (Harshit, Viraj, Bahul).
- **Repo:** https://github.com/VirajM723/DL-project

*Last updated: 2026-10-27.*

---

## 1. Current status (read first)

| Item | State |
|---|---|
| Notebook | Complete, 72 cells, tested. Edit the `.ipynb` directly; no generator script survives. |
| Full training run | **In progress on Kaggle (GPU T4 ×2)**, started 2026-10-27. The log confirmed `device=cuda x2`, `smoke=False`, `steps 3000`, `batch 16`, `ft_steps 1500`. Expected ≈ 4–4.5 h. |
| `figures/`, `results/` in the repo | **Stale smoke-test outputs** (4 training steps). `results/environment.md` says "4 steps (+4 fine-tune) × batch 2" and "6.7 min". **Never quote their accuracy numbers.** Replace them with the full run's `tracknet_outputs.zip`. |
| README | Results section written "as if run". Accuracy cells are **blank** (to fill from `results/`). The model-cost table holds real T4 measurements. |
| `Project Report.docx` | The submission report. Accuracy cells/blanks are **yellow-highlighted placeholders**; `[Insert figure: …]` slots are yellow. Has a section 13 "AI tool usage" whose wording the user chose. |
| Online copies (owned by the user) | Report: https://claude.ai/artifact/U43TvfL8uPLqj4BRrZr3Ak · Viva / explanation guide: https://claude.ai/artifact/NNaitvq2TMQS5FV7Zfbmvv. Both are browser-editable documents with accuracy placeholders. The **Word file is the version that gets submitted**; the viva guide exists only online. |
| Git | Uncommitted changes: notebook, README, CHANGES.md, `Project Report.docx`, this file. **The user commits and pushes themselves. Do not run git commands unless asked.** |

## 2. Work still to do

1. **When the full run finishes:**
   - Unzip `tracknet_outputs.zip` (from Kaggle's Output tab) into the repo root, replacing `figures/`, `results/` and `checkpoints/`.
   - Verify that `results/environment.md` shows `2 × Tesla T4` and `3000 steps (+1500 fine-tune) × batch 16`, with no "SMOKE TEST" note. Also check that `results/report_assets_check_gpu.json` has `"smoke": false` and `"missing": []`.
2. **Fill the accuracy blanks.** Sources, all in `results/` (`changes_measured_gpu.md` has every table in one place):

   | Where | What | Source |
   |---|---|---|
   | README §Results headline; report abstract + §7 sentence | F1, ours + M5 vs faithful V5 | `real_tennis_gpu.csv`: (`V5-Lite+ (M1–M4)`, `fine-tuned + M5`, `f1`) and (`V5 (faithful)`, `fine-tuned`, `f1`) |
   | "Accuracy on real footage" table | F1 for zero-shot, fine-tuned, fine-tuned + M5; prec, rec, err_mean (fine-tuned) | `real_tennis_gpu.csv` |
   | "Effect of each change" table | ΔF1, ΔPrecision, ΔRecall, Δerr | `real_changes_summary_gpu.csv` (rows M1…M5, TOTAL) |
   | "Synthetic test set" table | f1, prec, rec, err_mean | `ablation_gpu.csv`; the "+ M5" row comes from the `"m5"` entry in `all_results_gpu.json` |
   | Report §8 sentence | mean F1 zero-shot → fine-tuned; largest gain; M5's ΔF1 | same files |

   Round to 3 decimals (errors to 2). In the docx, insert the figures at the yellow `[Insert figure: …]` lines, then remove the highlights.
3. **Update the run-time and schedule wording to the T4 ×2 run.** Several places still describe the single-GPU schedule (6,000 + 3,000 steps × batch 8, "≈ 4.5–5 h"):
   - the report's Tools / Algorithm / dataset text;
   - the viva guide's §8 (demo), §9 (cheat sheet) and §7 Q9;
   - CHANGES.md §1 (Schedule row).

   Correct values: **T4 ×2: 3,000 + 1,500 steps × batch 16, ≈ 4.2 h; T4 ×1: 6,000 + 3,000 × batch 8, ≈ 6.5 h.** These are estimates; replace them with the measured times from `results/environment.md` after the run.
4. Fill the accuracy placeholders in the online report and in the viva guide's §9 cheat sheet with the same numbers as step 2. The user edits these in the browser, or an assistant with access to them does.
5. The user adds the public Kaggle notebook link to the report's §9 (code link).

## 3. User preferences and constraints

- **Never train on the user's laptop.** It is CPU-only (i5-1335U, 16 GB, no CUDA). Smoke tests (~1–3 min) and unit tests are fine. Real training happens on Kaggle.
- **No git commands** (commit, push, branch) unless the user explicitly asks.
- **Never present smoke-run numbers as results.** Check `results/environment.md` before quoting any accuracy number.
- The user writes the report in Word (`Project Report.docx`). They may have it open: Word locks the file, so a copy fails with "Device or resource busy". In that case save a sibling file and tell them.
- The user decided the wording of the report's section 13, "AI tool usage". Don't rewrite it unasked.

## 4. Repository map

| Path | What |
|---|---|
| `TrackNetV5_TinyObject_Tracking.ipynb` | The whole project, §1–§13 (theory → data → models → training → eval → M5 → visuals → real fine-tuning §9.5 → summary §10 → report assets §12 → live demo §13 → export) |
| `README.md` | Overview, results (blanks), setup, Kaggle steps, demo, tests |
| `CHANGES.md` | Every deviation from the paper, M1–M5 motivation/trade-offs, efficiency table (T4 numbers), validation/bug list, real-data protocol |
| `Project Report.docx` | Submission report (Word) |
| `tests/conftest.py`, `tests/test_tracknet.py` | 48 pytest checks. The harness execs the notebook's cells in a temp dir (smoke mode) up to the first training cell, plus later definition-only cells. |
| `pytest.ini` | Marker `slow` = tiny overfit (trains on CPU, avoid) + 3 end-to-end notebook runs |
| `Input/` (git-ignored) | Real TrackNet tennis dataset: `game1…game10/Clip*/{0000.jpg…, Label.csv}`, 19,835 frames, 2.59 GB |
| `figures/`, `results/` | Notebook outputs (currently stale smoke run) |
| `checkpoints/` (git-ignored) | Trained weights (from the Kaggle zip) |
| `.venv/` (git-ignored) | Python 3.12, torch 2.14 CPU, created with `uv` |

## 5. How to run and check things

```bash
.venv/Scripts/python -m pytest -m "not slow"              # unit tests, ~1-3 min, no training
.venv/Scripts/python -m pytest -k "notebook_runs"         # 3 end-to-end smoke runs of the notebook, ~10 min
TN_SMOKE=1 .venv/Scripts/jupyter nbconvert --to notebook --execute TrackNetV5_TinyObject_Tracking.ipynb --output <scratch path>
```

- **Never** run the slow `tiny_overfit` test without asking (it trains on CPU).
- Environment variables:
  - `TN_PROFILE=gpu|gpu_full|cpu`
  - `TN_SMOKE=1`
  - `TN_RETRAIN=1`
  - `TN_REAL_ROOT=<dataset>`
  - `TN_CKPT_FROM=<folder>` (re-use trained checkpoints)
  - `TN_ZIP=1`
- After any notebook edit:
  - parse every code cell;
  - check that no f-string *expression* contains a backslash or the same quote (Kaggle runs Python < 3.12 in places);
  - run the tests above.

## 6. Technical facts and decisions

**Profiles.**
- `gpu`: 512×288, width ½, 150/12/30 synthetic clips × 60 frames, 6,000 steps × batch 8 + 3,000 fine-tune steps at lr 3e-4.
- With 2 GPUs (`NGPU ≥ 2`), the `gpu` profile auto-switches to `nn.DataParallel` with batch 16 and 3,000 + 1,500 steps.

**Models.**
- **Ablation ladder:** V2 → V5 (faithful) → +M1 → +M1+M2 → +M1+M2+M3 → V5-Lite+ (M1–M4); M5 is applied at inference.
- **M1:** Gaussian target, σ = 0.8·r, plus a value-thresholded centroid decoder.
- **M2:** per-pair/per-polarity (α, β) plus a direction channel; input becomes 15 channels.
- **M3:** consistent Draft_MDD input in train and test, plus a slim TSATT head (patch 16, d = 32, 1 block).
- **M4:** depthwise-separable backbone.
- **M5:** constant-acceleration Kalman filter, χ² gate 13.8, gap fill ≤ 4 frames.

**Real data.**
- Split by match:
  - fine-tune on games 1–8: 66 clips, 13,806 frames;
  - validation on the last clip of each training game: 8 clips, 2,312 frames;
  - test on games 9–10: 21 clips, 3,717 frames.
- The loader reads with 16 threads and decodes at half size (`IMREAD_REDUCED_COLOR_2`). The old sequential loader hung for over an hour on Kaggle's network storage.

**Measured T4 costs (exact).**

| Model | Params (M) | GFLOPs | frames/s (noisy) |
|---|---|---|---|
| V2 | 2.836 | 56.5 | 150 |
| V5 | 3.098 | 66.5 | 82 |
| +M3 | 2.894 | 57.3 | 217 |
| Lite+ | 0.410 | 10.9 | 223 |

Lite+ has 7.6× fewer parameters and 6× fewer FLOPs than V5.

**Gotchas.**
- PyTorch's FLOP counter **does not count attention on CPU**. CPU FLOP numbers undercount V5's head, so use T4 numbers only. This already caused one wrong claim ("+3.3 % over V2"); on the T4 it is +17.8 %.
- `FINAL_NAME` must be defined in the VARIANTS cell (§5): it is used from §5 on.
- Evaluation tolerance: τ = 5.33 px at 512 px (≈ 13 px at 1280). All pixel constants scale with `SC = W/288`.

**Bugs found and fixed before training** (from the test suite):
- occlusion labels (71 % wrong);
- camera shake not moving the background;
- a biased sub-pixel decoder;
- a §9.2 crash at clip end;
- `FINAL_NAME` used before it was defined;
- the fig20 layout.

## 7. Editing the Word report

- The easiest way is to open `Project Report.docx` in Word and edit by hand.
- Programmatic editing:
  - This machine has **no pandoc or LibreOffice**. Use `python-docx` (`pip install python-docx`), or edit the XML directly.
  - A `.docx` is a zip: unzip it, edit `word/document.xml`, and re-zip in the **original entry order** with `[Content_Types].xml` first.
  - Word often splits one phrase across several `<w:r>` runs, so search the XML for short fragments.
  - Afterwards, open the file in Word to confirm it still loads.
- The yellow placeholders in the report are table cells shaded `FFF2CC` and text runs with a yellow highlight. Remove the shading and highlight once a value is filled in.
- Keep a backup of the file before any scripted edit.
