# Report assets (profile `gpu`, generated 2026-10-27 13:22)

| Report section | Asset | File | Status |
|---|---|---|---|
| 1. Problem statement | notebook §1.1 (text) | `—` | text |
| 2. Gaps identified | notebook §1.4 (table) + CHANGES.md in the repo | `—` | text |
| 3. Proposed architecture diagram | block diagram | `figures/fig02_pipeline_v5_vs_ours.png` | ✅ |
| 3. Proposed architecture diagram | detailed diagram | `figures/fig19_architecture_detailed.png` | ✅ |
| 3. Proposed architecture diagram | layer table | `results/model_summary_gpu.md` | ✅ |
| 4. Algorithmic steps | notebook §5.2 (text) | `—` | text |
| 5. Dataset used | synthetic samples | `figures/fig03_dataset_samples.png` | ✅ |
| 5. Dataset used | synthetic statistics | `figures/fig04_dataset_stats.png` | ✅ |
| 5. Dataset used | training targets | `figures/fig05_targets_disk_vs_gaussian.png` | ✅ |
| 5. Dataset used | dataset table | `figures/tab00_datasets.png` | ✅ |
| 5. Dataset used | dataset summary | `results/dataset_summary_gpu.md` | ✅ |
| 5. Dataset used | real samples | `figures/fig20_real_dataset.png` | ✅ |
| 6. Tools used | environment table | `figures/tab06_environment.png` | ✅ |
| 6. Tools used | environment | `results/environment.md` | ✅ |
| 7. Screenshots of results | synthetic ablation table | `figures/tab02_synthetic_ablation.png` | ✅ |
| 7. Screenshots of results | training curves | `figures/fig06_training_curves.png` | ✅ |
| 7. Screenshots of results | ablation bars | `figures/fig07_ablation_bars.png` | ✅ |
| 7. Screenshots of results | heatmaps per model | `figures/fig13_heatmaps_per_model.png` | ✅ |
| 7. Screenshots of results | tracking strip | `figures/fig16_tracking_strip.png` | ✅ |
| 7. Screenshots of results | tracking GIF | `figures/tracking_demo.gif` | ✅ |
| 7. Screenshots of results | M5 before/after | `figures/fig12_kalman_m5.png` | ✅ |
| 7. Screenshots of results | real results table | `figures/tab03_real_results.png` | ✅ |
| 7. Screenshots of results | real results chart | `figures/fig17_real_tennis.png` | ✅ |
| 7. Screenshots of results | real example frames | `figures/fig18_real_examples.png` | ✅ |
| 8. Analysis & difference from existing approaches | comparison table | `figures/tab05_comparison.png` | ✅ |
| 8. Analysis & difference from existing approaches | per-change effect (synthetic) | `figures/tab04_changes_synthetic.png` | ✅ |
| 8. Analysis & difference from existing approaches | accuracy vs cost | `figures/fig08_accuracy_vs_cost.png` | ✅ |
| 8. Analysis & difference from existing approaches | tolerance / error CDF | `figures/fig09_tolerance_and_error_cdf.png` | ✅ |
| 8. Analysis & difference from existing approaches | robustness | `figures/fig11_robustness.png` | ✅ |
| 8. Analysis & difference from existing approaches | MDD maps | `figures/fig14_mdd_polarity_maps.png` | ✅ |
| 8. Analysis & difference from existing approaches | R-STR decomposition | `figures/fig15_rstr_decomposition.png` | ✅ |
| 8. Analysis & difference from existing approaches | all numbers | `results/changes_measured_gpu.md` | ✅ |
| 8. Analysis & difference from existing approaches | per-change effect (real) | `figures/tab04_changes_real.png` | ✅ |
| 9. Code link | GitHub repo / public Kaggle notebook URL (added by the student) | `—` | text |
| 10. Novel contribution | notebook §1.4.2 (text) | `—` | text |
| 10. Novel contribution | model cost table | `figures/tab01_model_cost.png` | ✅ |
| 11. Individual contribution | written by the student | `—` | text |
| 12. Learning outcome | notebook §11 conclusions (basis) | `—` | text |

## Key numbers
- Synthetic test F1 @ τ=5.3px: faithful V5 0.000 → ours (M1–M5) 0.125
- Params: V5 3.098 M → ours 0.410 M; GFLOPs 66.54 → 10.92
- Real games 9–10 F1: faithful V5 (fine-tuned) 0.000 → ours + M5 0.301
