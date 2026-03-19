# Embedding pipeline report

## How to read this report

- **MAE loss**: Mean squared error (MSE) between predicted and true 8×8×12 piece planes, averaged only over **masked** squares. Unit: squared error per masked position per channel; targets are 0/1 so scale is 0–1. Lower is better; typical range 0.01–0.05 after training.

- **Probe losses**: Regression probes (piece count, elo) report **MSE** (piece count in count², elo in Elo²). Classification probes (in_check, elo top vs bottom) report **log loss** (nats; random guessing ≈ 0.69). Lower is better.

- **Baseline**: Loss of predicting the **mean** (constant predictor). **Improvement %** = (1 − model_loss / baseline_loss) × 100; higher means the model is better than guessing the average.

## 1. MAE training

| Epoch | Train loss | Val loss |
|-------|------------|----------|
| 1 | 0.020687 | 0.017531 |
| 2 | 0.016464 | 0.015740 |
| 3 | 0.015261 | 0.014841 |
| 4 | 0.014590 | 0.014363 |
| 5 | 0.014167 | 0.014025 |
| 6 | 0.013865 | 0.013804 |
| 7 | 0.013639 | 0.013682 |
| 8 | 0.013453 | 0.013427 |
| 9 | 0.013288 | 0.013346 |
| 10 | 0.013189 | 0.013240 |
| 11 | 0.013079 | 0.013104 |
| 12 | 0.013007 | 0.013062 |
| 13 | 0.012925 | 0.013032 |
| 14 | 0.012881 | 0.013005 |
| 15 | 0.012811 | 0.012933 |
| 16 | 0.012760 | 0.012895 |
| 17 | 0.012732 | 0.012854 |
| 18 | 0.012685 | 0.012873 |
| 19 | 0.012635 | 0.012814 |
| 20 | 0.012630 | 0.012836 |
| 21 | 0.012576 | 0.012691 |
| 22 | 0.012537 | 0.012669 |
| 23 | 0.012503 | 0.012663 |
| 24 | 0.012472 | 0.012631 |
| 25 | 0.012454 | 0.012622 |
| 26 | 0.012423 | 0.012579 |
| 27 | 0.012451 | 0.012593 |
| 28 | 0.012379 | 0.012552 |
| 29 | 0.012356 | 0.012541 |
| 30 | 0.012323 | 0.012522 |
| 31 | 0.012313 | 0.012508 |
| 32 | 0.012270 | 0.012508 |
| 33 | 0.012256 | 0.012472 |
| 34 | 0.012257 | 0.012504 |
| 35 | 0.012223 | 0.012462 |
| 36 | 0.012197 | 0.012432 |
| 37 | 0.012177 | 0.012422 |
| 38 | 0.012150 | 0.012405 |
| 39 | 0.012150 | 0.012411 |
| 40 | 0.012144 | 0.012377 |
| 41 | 0.012118 | 0.012378 |
| 42 | 0.012120 | 0.012354 |
| 43 | 0.012084 | 0.012375 |
| 44 | 0.012063 | 0.012379 |
| 45 | 0.012062 | 0.012366 |
| 46 | 0.012051 | 0.012352 |
| 47 | 0.012047 | 0.012343 |
| 48 | 0.012035 | 0.012345 |
| 49 | 0.012022 | 0.012323 |
| 50 | 0.012012 | 0.012326 |

- **MAE baseline (predict mean):** 0.026546
- **Improvement over baseline:** 53.3%
- **Final test loss (MAE):** 0.012385

![MAE loss curve](embedding_report_figs/mae_loss_curve.png)

## 2. Linear probes (subset)

Test loss comparison: **final (trained) model** vs **random embedding**, **random-weights model**, **raw input** (flattened 8×8×19).

| Probe | Final (test) | Random emb | Random model | Raw input | Baseline (test) | Improvement % (final) |
|-------|--------------|------------|--------------|----------|-----------------|------------------------|
| piece_count_white | 0.5955 | 109.0042 | 63.0109 | 0.7494 | 15.2345 | 96.1% |
| piece_count_black | 0.6245 | 109.7113 | 63.4325 | 0.8000 | 15.0332 | 95.8% |
| in_check | 0.2620 | 0.4377 | 0.2966 | 0.2550 | 0.2945 | 11.1% |
| elo_regression | 2078057.6250 | 2923919.7500 | 2913676.0000 | 2597910.7500 | 67191.9258 | -2992.7% |
| elo_top_vs_bottom | 0.6908 | 0.6949 | 0.6929 | 0.6993 | 0.6933 | 0.4% |

![Probe test loss: final vs baselines](embedding_report_figs/probe_progress.png)

## Summary

MAE is **53.3%** better than baseline (predict mean). Probe improvements over baseline (test): 96.1%, 95.8%, 11.1%, -2992.7%, 0.4%.
