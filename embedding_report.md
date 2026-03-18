# Embedding pipeline report

## How to read this report

- **MAE loss**: Mean squared error (MSE) between predicted and true 8×8×12 piece planes, averaged only over **masked** squares. Unit: squared error per masked position per channel; targets are 0/1 so scale is 0–1. Lower is better; typical range 0.01–0.05 after training.

- **Probe losses**: Regression probes (piece count, elo) report **MSE** (piece count in count², elo in Elo²). Classification probes (in_check, elo top vs bottom) report **log loss** (nats; random guessing ≈ 0.69). Lower is better.

- **Baseline**: Loss of predicting the **mean** (constant predictor). **Improvement %** = (1 − model_loss / baseline_loss) × 100; higher means the model is better than guessing the average.

## 1. MAE training

| Epoch | Train loss | Val loss |
|-------|------------|----------|
| 1 | 0.021387 | 0.018274 |
| 2 | 0.017875 | 0.017393 |
| 3 | 0.016975 | 0.016535 |
| 4 | 0.016349 | 0.016031 |
| 5 | 0.016281 | 0.015918 |
| 6 | 0.015941 | 0.015522 |
| 7 | 0.015539 | 0.015306 |
| 8 | 0.015276 | 0.015225 |
| 9 | 0.015165 | 0.015252 |
| 10 | 0.015004 | 0.014937 |

- **MAE baseline (predict mean):** 0.027128
- **Improvement over baseline:** 43.9%
- **Final test loss (MAE):** 0.015228

![MAE loss curve](embedding_report_figs/mae_loss_curve.png)

## 2. Linear probes (subset)

| Probe | Train loss (final) | Val loss (final) | Test loss (epoch 1) | Test loss (final) | Baseline (test) | Improvement % |
|-------|-------------------|------------------|---------------------|-------------------|-----------------|----------------|
| piece_count_white | 0.6542 | 0.6524 | 1.1180 | 0.7223 | 14.3035 | 95.0% |
| piece_count_black | 0.6349 | 0.7023 | 1.3476 | 0.6780 | 14.0923 | 95.2% |
| in_check | 0.3124 | 0.2806 | 0.3335 | 0.3268 | 0.3662 | 10.8% |
| elo_regression | 8410.9648 | 8390.0850 | 7778.7764 | 7545.8994 | 7947.0777 | 5.0% |
| elo_top_vs_bottom | 0.6151 | 0.7163 | 0.7054 | 0.7340 | 0.6939 | -5.8% |

![Probe test loss: epoch 1 vs final](embedding_report_figs/probe_progress.png)

## Summary

MAE is **43.9%** better than baseline (predict mean). Probe improvements over baseline (test): 95.0%, 95.2%, 10.8%, 5.0%, -5.8%.
