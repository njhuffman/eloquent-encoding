# Embedding pipeline report

## How to read this report

- **MAE loss**: Mean squared error (MSE) between predicted and true 8×8×12 piece planes, averaged only over **masked** squares. Unit: squared error per masked position per channel; targets are 0/1 so scale is 0–1. Lower is better; typical range 0.01–0.05 after training.

- **Probe losses**: Regression probes (piece count, elo) report **MSE** (piece count in count², elo in Elo²). Classification probes (in_check, elo top vs bottom) report **log loss** (nats; random guessing ≈ 0.69). Lower is better.

- **Baseline**: Loss of predicting the **mean** (constant predictor). **Improvement %** = (1 − model_loss / baseline_loss) × 100; higher means the model is better than guessing the average.

## 1. MAE training

| Epoch | Train loss | Val loss |
|-------|------------|----------|
| 1 | 0.021288 | 0.018267 |
| 2 | 0.018095 | 0.017914 |
| 3 | 0.017747 | 0.017583 |
| 4 | 0.017465 | 0.017321 |
| 5 | 0.017083 | 0.016801 |
| 6 | 0.016719 | 0.016531 |
| 7 | 0.016531 | 0.016425 |
| 8 | 0.016461 | 0.016321 |
| 9 | 0.016345 | 0.016282 |
| 10 | 0.016270 | 0.016200 |
| 11 | 0.016563 | 0.016245 |
| 12 | 0.016248 | 0.016151 |
| 13 | 0.016190 | 0.016096 |
| 14 | 0.016141 | 0.016057 |
| 15 | 0.016131 | 0.016058 |
| 16 | 0.016092 | 0.016006 |
| 17 | 0.016067 | 0.015976 |
| 18 | 0.016329 | 0.016870 |
| 19 | 0.016329 | 0.016079 |
| 20 | 0.016118 | 0.016009 |
| 21 | 0.016057 | 0.015979 |
| 22 | 0.016032 | 0.015934 |
| 23 | 0.016003 | 0.015922 |
| 24 | 0.015983 | 0.015922 |
| 25 | 0.015960 | 0.015904 |
| 26 | 0.015944 | 0.015893 |
| 27 | 0.015932 | 0.015874 |
| 28 | 0.015912 | 0.015867 |
| 29 | 0.015900 | 0.015844 |
| 30 | 0.015890 | 0.015844 |
| 31 | 0.015909 | 0.015834 |
| 32 | 0.015874 | 0.015832 |
| 33 | 0.015856 | 0.015815 |
| 34 | 0.015842 | 0.015795 |
| 35 | 0.015830 | 0.015809 |
| 36 | 0.015811 | 0.015786 |
| 37 | 0.015811 | 0.015779 |
| 38 | 0.015806 | 0.015778 |
| 39 | 0.015790 | 0.015782 |
| 40 | 0.015778 | 0.015771 |
| 41 | 0.015783 | 0.015758 |
| 42 | 0.015765 | 0.015748 |
| 43 | 0.015753 | 0.015754 |
| 44 | 0.015747 | 0.015742 |
| 45 | 0.015736 | 0.015735 |
| 46 | 0.015730 | 0.015729 |
| 47 | 0.015722 | 0.015726 |
| 48 | 0.015717 | 0.015720 |
| 49 | 0.015710 | 0.015729 |
| 50 | 0.015704 | 0.015717 |

- **MAE baseline (predict mean):** 0.026570
- **Improvement over baseline:** 40.7%
- **Final test loss (MAE):** 0.015757

![MAE loss curve](embedding_report_figs/mae_loss_curve.png)

## 2. Linear probes (subset)

Test loss comparison: **final (trained) model** vs **random embedding**, **random-weights model**, **raw input** (flattened 8×8×19).

| Probe | Final (test) | Random emb | Random model | Raw input | Baseline (test) | Improvement % (final) |
|-------|--------------|------------|--------------|----------|-----------------|------------------------|
| piece_count_white | 1.4391 | 109.0042 | 77.2248 | 0.7494 | 15.2345 | 90.6% |
| piece_count_black | 1.5436 | 109.7113 | 77.7321 | 0.8000 | 15.0332 | 89.7% |
| in_check | 0.2846 | 0.4377 | 0.3056 | 0.2550 | 0.2945 | 3.4% |
| elo_regression | 219913.5156 | 2923919.7500 | 2917393.0000 | 2597910.7500 | 67191.9258 | -227.3% |
| elo_top_vs_bottom | 0.5785 | 0.6949 | 0.6928 | 0.6993 | 0.6933 | 16.6% |

![Probe test loss: final vs baselines](embedding_report_figs/probe_progress.png)

## Summary

MAE is **40.7%** better than baseline (predict mean). Probe improvements over baseline (test): 90.6%, 89.7%, 3.4%, -227.3%, 16.6%.
