# Embedding pipeline report

## How to read this report

- **MAE loss**: Mean squared error (MSE) between predicted and true 8×8×12 piece planes, averaged only over **masked** squares. Unit: squared error per masked position per channel; targets are 0/1 so scale is 0–1. Lower is better; typical range 0.01–0.05 after training.

- **Probe losses**: Regression probes (piece count, elo) report **MSE** (piece count in count², elo in Elo²). Classification probes (in_check, elo top vs bottom) report **log loss** (nats; random guessing ≈ 0.69). Lower is better.

- **Baseline**: Loss of predicting the **mean** (constant predictor). **Improvement %** = (1 − model_loss / baseline_loss) × 100; higher means the model is better than guessing the average.

## 1. MAE training

| Epoch | Train loss | Val loss |
|-------|------------|----------|
| 1 | 0.020599 | 0.017732 |
| 2 | 0.017283 | 0.016880 |
| 3 | 0.016574 | 0.016211 |
| 4 | 0.016020 | 0.015827 |
| 5 | 0.016109 | 0.016315 |
| 6 | 0.015817 | 0.015675 |
| 7 | 0.015579 | 0.015548 |
| 8 | 0.015493 | 0.015393 |
| 9 | 0.015362 | 0.015412 |
| 10 | 0.015290 | 0.015419 |
| 11 | 0.015228 | 0.015197 |
| 12 | 0.015129 | 0.015154 |
| 13 | 0.015166 | 0.015081 |
| 14 | 0.015026 | 0.015157 |
| 15 | 0.015020 | 0.014996 |
| 16 | 0.014961 | 0.014953 |
| 17 | 0.015086 | 0.015008 |
| 18 | 0.014934 | 0.014950 |
| 19 | 0.014870 | 0.014912 |
| 20 | 0.014842 | 0.014931 |
| 21 | 0.014807 | 0.014856 |
| 22 | 0.014868 | 0.014848 |
| 23 | 0.014778 | 0.014844 |
| 24 | 0.014739 | 0.014772 |
| 25 | 0.014729 | 0.014777 |
| 26 | 0.015120 | 0.014945 |
| 27 | 0.014844 | 0.014830 |
| 28 | 0.014779 | 0.014790 |
| 29 | 0.014706 | 0.014754 |
| 30 | 0.014685 | 0.014742 |
| 31 | 0.014660 | 0.014728 |
| 32 | 0.014655 | 0.014777 |
| 33 | 0.014635 | 0.014699 |
| 34 | 0.014621 | 0.014723 |
| 35 | 0.014601 | 0.014707 |
| 36 | 0.014604 | 0.014670 |
| 37 | 0.014580 | 0.014667 |
| 38 | 0.014568 | 0.014662 |
| 39 | 0.014566 | 0.014646 |
| 40 | 0.014549 | 0.014645 |
| 41 | 0.014588 | 0.014732 |
| 42 | 0.014569 | 0.014642 |
| 43 | 0.014531 | 0.014622 |
| 44 | 0.014509 | 0.014606 |
| 45 | 0.014503 | 0.014608 |
| 46 | 0.014492 | 0.014596 |
| 47 | 0.014482 | 0.014588 |
| 48 | 0.014472 | 0.014601 |
| 49 | 0.014525 | 0.014607 |
| 50 | 0.014479 | 0.014587 |

- **MAE baseline (predict mean):** 0.026552
- **Improvement over baseline:** 44.8%
- **Final test loss (MAE):** 0.014657

![MAE loss curve](embedding_report_figs/mae_loss_curve.png)

## 2. Linear probes (subset)

Test loss comparison: **final (trained) model** vs **random embedding**, **random-weights model**, **raw input** (flattened 8×8×19).

| Probe | Final (test) | Random emb | Random model | Raw input | Baseline (test) | Improvement % (final) |
|-------|--------------|------------|--------------|----------|-----------------|------------------------|
| piece_count_white | 1.1320 | 109.0042 | 63.0109 | 0.7494 | 15.2345 | 92.6% |
| piece_count_black | 1.1346 | 109.7113 | 63.4325 | 0.8000 | 15.0332 | 92.5% |
| in_check | 0.2544 | 0.4377 | 0.2966 | 0.2550 | 0.2945 | 13.6% |
| elo_regression | 1851076.3750 | 2923919.7500 | 2913676.0000 | 2597910.7500 | 67191.9258 | -2654.9% |
| elo_top_vs_bottom | 0.6948 | 0.6949 | 0.6929 | 0.6993 | 0.6933 | -0.2% |

![Probe test loss: final vs baselines](embedding_report_figs/probe_progress.png)

## Summary

MAE is **44.8%** better than baseline (predict mean). Probe improvements over baseline (test): 92.6%, 92.5%, 13.6%, -2654.9%, -0.2%.
