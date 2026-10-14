# Benchmark run 2026-10-14: UCI

Status label: `Verified (run 2026-10-14, seeds 0, 1, 2, 3, 4)`. Produced by `scripts/run_benchmark.py --dataset uci`. These are new numbers from an independent re-implementation; they are not the result reported for the earlier deployment on the CV.

## Settings

* Python 3.11.9, numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1
* Hardware: AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, 16 logical cores, Windows; 6 worker processes; runtime 3.9 minutes
* Units: 60 (60 usable for matched recall: at least 2 calibration events and one test event; 60 usable for alert budgets)
* Model seeds: 0, 1, 2, 3, 4 (deterministic methods are run once)
* Primary recall target 0.8; all targets (0.5, 0.65, 0.8); persistence k in (1, 2, 3); alert budgets (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0) alerts/day
* Threshold rules: matched recall = highest threshold whose calibration event recall >= target; alert budget = lowest threshold that keeps all alerts on the calibration window within the budget (no labels).
* FA = false alert = an alert overlapping no labelled event. Event recall = share of labelled events with at least one flagged point. Time to detect (TTD) = minutes from event start to first flagged point, median over detected events.
* Percent changes are the pooled change in test false alerts per day relative to the global z-score at the same calibration recall (negative = fewer false alerts); CIs are 95% percentile bootstrap intervals over series (UCI: 12 series, each averaged over its injection seeds; SMD: machines).
* Data: UCI Individual Household Electric Power Consumption (CC BY 4.0); 12 gap-free 120-day hourly windows; injected anomalies (spike, level shift, drift, stuck, missing); window halves are 60 days; fit window = calibration half incl. warm-up (as in the first run).

## Development notes (full disclosure)

* A first complete run of this benchmark exposed a bug in the one-class SVM score: it was `-decision_function`, which is negative for normal points, so missing values (which score 0 by contract) ranked above normal points and low thresholds flagged every missing burst for free. The score is now shifted to be non-negative, a test checks that every detector's scores are non-negative, and both benchmarks were re-run. This log is from the re-run; the first run was discarded.
* Thresholds, hyper-parameters, rule settings and the primary comparisons were fixed before the test windows of the final run were read; the SMD primary recall target (0.65) was chosen from calibration-window diagnostics only.
* The alert budget counts alert segments per day, so a threshold low enough that a detector stays flagged for a very long stretch counts as one alert. The column `flagged_share` in `budget.csv.gz` records the share of test points flagged.

## Hyper-parameters chosen on calibration windows

| Method | Parameters | Units | Share not reaching target | Calibration FA/day | Chosen |
|---|---|---|---|---|---|
| Autoencoder | {} | 60 | 0.20 | 2.52 | False |
| Autoencoder | {'hidden': 16, 'bottleneck': 4} | 60 | 0.13 | 2.24 | True |
| Seasonal residual + EWMA | {} | 60 | 0.25 | 1.39 | True |
| Voting ensemble | {} | 60 | 0.27 | 1.42 | False |
| Voting ensemble | {'min_votes': 3} | 60 | 0.18 | 1.28 | True |
| EWMA control chart | {} | 60 | 0.33 | 0.81 | False |
| EWMA control chart | {'lam': 0.1} | 60 | 0.75 | 0.40 | False |
| EWMA control chart | {'lam': 0.5} | 60 | 0.13 | 1.19 | True |
| Isolation Forest | {} | 60 | 0.22 | 1.11 | False |
| Isolation Forest | {'n_estimators': 200, 'max_samples': 1024} | 60 | 0.22 | 1.03 | True |
| Local outlier factor | {} | 60 | 0.35 | 1.89 | True |
| Local outlier factor | {'n_neighbors': 50} | 60 | 0.38 | 1.81 | False |
| One-class SVM | {} | 60 | 0.00 | 1.37 | True |
| One-class SVM | {'nu': 0.01} | 60 | 0.03 | 1.50 | False |
| PCA reconstruction error | {} | 60 | 0.20 | 2.17 | False |
| PCA reconstruction error | {'variance': 0.8} | 60 | 0.17 | 2.29 | True |
| Rolling median/MAD | {} | 60 | 0.85 | 1.79 | True |
| Rolling median/MAD | {'window_days': 3} | 60 | 0.90 | 1.71 | False |
| Rolling median/MAD | {'window_days': 1} | 60 | 0.90 | 1.81 | False |
| Seasonal residual | {} | 60 | 0.07 | 1.91 | False |
| Seasonal residual | {'profile_days': 3} | 60 | 0.17 | 1.88 | False |
| Seasonal residual | {'scale_days': 7} | 60 | 0.02 | 1.81 | True |
| Seasonal residual | {'profile_days': 3, 'scale_days': 7} | 60 | 0.10 | 2.01 | False |
| Robust STL residual | {} | 60 | 0.10 | 1.85 | False |
| Robust STL residual | {'trend_days': 1} | 60 | 0.10 | 1.62 | True |
| Global z-score (baseline) | {} | 60 | 0.90 | 1.67 | True |

Rule-layer settings (z-score + all rules, same criterion):

| Rule settings | Share not reaching target | Calibration FA/day | Chosen |
|---|---|---|---|
| {'range_margin': 0.5, 'max_flat_fraction': 0.5} | 0.00 | 0.04 | True |
| {'range_margin': 0.5, 'max_flat_fraction': 0.05} | 0.00 | 0.04 | False |
| {'range_margin': 2.0, 'max_flat_fraction': 0.5} | 0.00 | 0.04 | False |
| {'range_margin': 2.0, 'max_flat_fraction': 0.05} | 0.00 | 0.04 | False |
| {'range_margin': 5.0, 'max_flat_fraction': 0.05} | 0.00 | 0.04 | False |

## Primary comparison: Detector alone, k=1, calibration recall target 0.8

| Method | Matched units | Change in FA/day | 95% CI | FA/day (method vs z-score) | Test event recall | Median time to detect, min | Series better / worse |
|---|---|---|---|---|---|---|---|
| Rolling median/MAD | 5 | +14% | [-18%, +51%] | 3.41 vs 2.99 | 0.64 vs 0.64 | 0 vs 0 | 1 / 3 |
| Seasonal residual | 6 | -63% | [-76%, -50%] | 1.11 vs 2.98 | 0.73 vs 0.63 | 0 vs 0 | 4 / 0 |
| EWMA control chart | 6 | -76% | [-87%, -69%] | 0.73 vs 2.98 | 0.68 vs 0.63 | 60 vs 0 | 4 / 0 |
| Seasonal residual + EWMA | 6 | -62% | [-80%, -48%] | 1.13 vs 2.98 | 0.73 vs 0.63 | 0 vs 0 | 4 / 0 |
| Robust STL residual | 6 | -58% | [-66%, -52%] | 1.24 vs 2.98 | 0.72 vs 0.63 | 0 vs 0 | 4 / 0 |
| Isolation Forest | 6 | -70% | [-77%, -58%] | 0.89 vs 2.98 | 0.74 vs 0.63 | 0 vs 0 | 4 / 0 |
| Local outlier factor | 5 | -36% | [-49%, -20%] | 1.54 vs 2.40 | 0.76 vs 0.64 | 0 vs 0 | 3 / 0 |
| One-class SVM | 6 | -58% | [-70%, -48%] | 1.26 vs 2.98 | 0.73 vs 0.63 | 0 vs 0 | 4 / 0 |
| PCA reconstruction error | 6 | -16% | [-38%, +14%] | 2.52 vs 2.98 | 0.79 vs 0.63 | 0 vs 0 | 2 / 2 |
| Autoencoder | 6 | -38% | [-49%, -16%] | 1.84 vs 2.98 | 0.76 vs 0.63 | 0 vs 0 | 4 / 0 |
| Voting ensemble | 6 | -67% | [-80%, -54%] | 1.00 vs 2.98 | 0.73 vs 0.63 | 0 vs 0 | 4 / 0 |

Operating points on the test window (all methods, including the baseline):

| Method | Target reached (calibration) | Test FA/day [95% CI] | Test event recall [95% CI] | Event precision [95% CI] | Median TTD, min |
|---|---|---|---|---|---|
| Global z-score (baseline) | 0.10 | 1.65 [1.37, 1.98] | 0.63 [0.60, 0.65] | 0.30 [0.24, 0.37] | 0 |
| Rolling median/MAD | 0.15 | 1.94 [1.62, 2.31] | 0.62 [0.61, 0.63] | 0.20 [0.17, 0.23] | 0 |
| Seasonal residual | 0.98 | 1.99 [1.55, 2.42] | 0.75 [0.74, 0.76] | 0.21 [0.16, 0.27] | 0 |
| EWMA control chart | 0.87 | 1.24 [1.03, 1.45] | 0.77 [0.75, 0.78] | 0.23 [0.19, 0.27] | 0 |
| Seasonal residual + EWMA | 0.75 | 1.49 [1.23, 1.78] | 0.75 [0.75, 0.76] | 0.20 [0.16, 0.25] | 0 |
| Robust STL residual | 0.90 | 1.83 [1.38, 2.31] | 0.76 [0.74, 0.77] | 0.24 [0.21, 0.26] | 0 |
| Isolation Forest | 0.78 | 1.18 [1.07, 1.30] | 0.77 [0.76, 0.78] | 0.23 [0.20, 0.25] | 0 |
| Local outlier factor | 0.65 | 2.18 [1.97, 2.39] | 0.77 [0.75, 0.78] | 0.21 [0.19, 0.23] | 0 |
| One-class SVM | 1.00 | 1.47 [1.30, 1.64] | 0.76 [0.74, 0.77] | 0.31 [0.27, 0.34] | 0 |
| PCA reconstruction error | 0.83 | 2.47 [2.27, 2.68] | 0.77 [0.76, 0.79] | 0.20 [0.19, 0.21] | 0 |
| Autoencoder | 0.87 | 2.32 [2.12, 2.52] | 0.76 [0.75, 0.78] | 0.22 [0.22, 0.23] | 0 |
| Voting ensemble | 0.82 | 1.44 [1.15, 1.79] | 0.75 [0.73, 0.76] | 0.23 [0.18, 0.28] | 0 |

## Primary comparison: Detector + rule layer (missing, range, stuck), k=1, calibration recall target 0.8

| Method | Matched units | Change in FA/day | 95% CI | FA/day (method vs z-score) | Test event recall | Median time to detect, min | Series better / worse |
|---|---|---|---|---|---|---|---|
| Rolling median/MAD | 60 | +310% | [+112%, +1047%] | 0.20 vs 0.05 | 0.80 vs 0.78 | 0 vs 60 | 0 / 8 |
| Seasonal residual | 60 | +178% | [+37%, +636%] | 0.13 vs 0.05 | 0.83 vs 0.78 | 0 vs 60 | 1 / 6 |
| EWMA control chart | 60 | +21% | [-51%, +242%] | 0.06 vs 0.05 | 0.76 vs 0.78 | 120 vs 60 | 3 / 5 |
| Seasonal residual + EWMA | 60 | +44% | [-49%, +323%] | 0.07 vs 0.05 | 0.77 vs 0.78 | 60 vs 60 | 3 / 5 |
| Robust STL residual | 60 | +419% | [+207%, +1274%] | 0.25 vs 0.05 | 0.76 vs 0.78 | 0 vs 60 | 0 / 8 |
| Isolation Forest | 60 | +124% | [+0%, +584%] | 0.11 vs 0.05 | 0.80 vs 0.78 | 0 vs 60 | 3 / 5 |
| Local outlier factor | 60 | +1001% | [+545%, +2999%] | 0.53 vs 0.05 | 0.79 vs 0.78 | 0 vs 60 | 0 / 8 |
| One-class SVM | 60 | +350% | [+114%, +1359%] | 0.22 vs 0.05 | 0.83 vs 0.78 | 0 vs 60 | 0 / 8 |
| PCA reconstruction error | 60 | +679% | [+286%, +2290%] | 0.38 vs 0.05 | 0.79 vs 0.78 | 0 vs 60 | 0 / 8 |
| Autoencoder | 60 | +902% | [+399%, +3202%] | 0.48 vs 0.05 | 0.78 vs 0.78 | 0 vs 60 | 0 / 8 |
| Voting ensemble | 60 | +171% | [+23%, +731%] | 0.13 vs 0.05 | 0.80 vs 0.78 | 0 vs 60 | 3 / 5 |

Operating points on the test window (all methods, including the baseline):

| Method | Target reached (calibration) | Test FA/day [95% CI] | Test event recall [95% CI] | Event precision [95% CI] | Median TTD, min |
|---|---|---|---|---|---|
| Global z-score (baseline) | 1.00 | 0.05 [0.02, 0.09] | 0.78 [0.74, 0.81] | 0.90 [0.82, 0.96] | 60 |
| Rolling median/MAD | 1.00 | 0.20 [0.11, 0.31] | 0.80 [0.75, 0.85] | 0.65 [0.55, 0.75] | 0 |
| Seasonal residual | 1.00 | 0.13 [0.07, 0.24] | 0.83 [0.79, 0.88] | 0.77 [0.68, 0.85] | 0 |
| EWMA control chart | 1.00 | 0.06 [0.02, 0.11] | 0.76 [0.72, 0.80] | 0.88 [0.79, 0.94] | 120 |
| Seasonal residual + EWMA | 1.00 | 0.07 [0.02, 0.14] | 0.77 [0.71, 0.82] | 0.85 [0.76, 0.93] | 60 |
| Robust STL residual | 1.00 | 0.25 [0.18, 0.34] | 0.76 [0.74, 0.79] | 0.55 [0.48, 0.61] | 0 |
| Isolation Forest | 1.00 | 0.11 [0.06, 0.18] | 0.80 [0.75, 0.85] | 0.77 [0.70, 0.82] | 0 |
| Local outlier factor | 1.00 | 0.53 [0.39, 0.69] | 0.79 [0.75, 0.84] | 0.45 [0.39, 0.50] | 0 |
| One-class SVM | 1.00 | 0.22 [0.14, 0.30] | 0.83 [0.79, 0.87] | 0.64 [0.57, 0.69] | 0 |
| PCA reconstruction error | 1.00 | 0.38 [0.27, 0.52] | 0.79 [0.75, 0.84] | 0.45 [0.40, 0.50] | 0 |
| Autoencoder | 1.00 | 0.48 [0.35, 0.64] | 0.78 [0.74, 0.82] | 0.42 [0.39, 0.46] | 0 |
| Voting ensemble | 1.00 | 0.13 [0.08, 0.21] | 0.80 [0.75, 0.85] | 0.75 [0.68, 0.81] | 0 |

## Other recall targets, detector alone, k=1

Target 0.5:

| Method | Matched units | Change in FA/day | 95% CI | FA/day (method vs z-score) | Test event recall | Median time to detect, min | Series better / worse |
|---|---|---|---|---|---|---|---|
| Rolling median/MAD | 60 | +205% | [+72%, +604%] | 0.29 vs 0.09 | 0.50 vs 0.46 | 0 vs 90 | 2 / 10 |
| Seasonal residual | 60 | +186% | [+64%, +552%] | 0.27 vs 0.09 | 0.52 vs 0.46 | 0 vs 90 | 0 / 12 |
| EWMA control chart | 60 | +36% | [-30%, +222%] | 0.13 vs 0.09 | 0.48 vs 0.46 | 165 vs 90 | 6 / 6 |
| Seasonal residual + EWMA | 60 | +51% | [-15%, +241%] | 0.14 vs 0.09 | 0.48 vs 0.46 | 15 vs 90 | 5 / 7 |
| Robust STL residual | 60 | +354% | [+185%, +876%] | 0.42 vs 0.09 | 0.50 vs 0.46 | 0 vs 90 | 0 / 12 |
| Isolation Forest | 60 | +119% | [+2%, +477%] | 0.21 vs 0.09 | 0.52 vs 0.46 | 0 vs 90 | 3 / 9 |
| Local outlier factor | 60 | +816% | [+440%, +2088%] | 0.86 vs 0.09 | 0.57 vs 0.46 | 0 vs 90 | 0 / 12 |
| One-class SVM | 60 | +232% | [+60%, +812%] | 0.31 vs 0.09 | 0.58 vs 0.46 | 0 vs 90 | 3 / 9 |
| PCA reconstruction error | 60 | +655% | [+330%, +1818%] | 0.71 vs 0.09 | 0.55 vs 0.46 | 0 vs 90 | 0 / 12 |
| Autoencoder | 60 | +730% | [+380%, +1963%] | 0.78 vs 0.09 | 0.54 vs 0.46 | 0 vs 90 | 0 / 12 |
| Voting ensemble | 60 | +142% | [+25%, +490%] | 0.23 vs 0.09 | 0.50 vs 0.46 | 0 vs 90 | 3 / 9 |

Target 0.65:

| Method | Matched units | Change in FA/day | 95% CI | FA/day (method vs z-score) | Test event recall | Median time to detect, min | Series better / worse |
|---|---|---|---|---|---|---|---|
| Rolling median/MAD | 49 | -11% | [-31%, +17%] | 1.26 vs 1.42 | 0.60 vs 0.64 | 0 vs 0 | 8 / 4 |
| Seasonal residual | 50 | -63% | [-73%, -45%] | 0.54 vs 1.44 | 0.64 vs 0.64 | 0 vs 0 | 11 / 1 |
| EWMA control chart | 50 | -77% | [-83%, -67%] | 0.33 vs 1.44 | 0.62 vs 0.64 | 60 vs 0 | 11 / 1 |
| Seasonal residual + EWMA | 50 | -66% | [-75%, -52%] | 0.49 vs 1.44 | 0.65 vs 0.64 | 0 vs 0 | 11 / 1 |
| Robust STL residual | 50 | -38% | [-54%, -13%] | 0.89 vs 1.44 | 0.63 vs 0.64 | 0 vs 0 | 11 / 1 |
| Isolation Forest | 50 | -67% | [-76%, -52%] | 0.47 vs 1.44 | 0.65 vs 0.64 | 0 vs 0 | 11 / 1 |
| Local outlier factor | 50 | +8% | [-16%, +42%] | 1.55 vs 1.44 | 0.71 vs 0.64 | 0 vs 0 | 5 / 7 |
| One-class SVM | 50 | -68% | [-79%, -49%] | 0.46 vs 1.44 | 0.63 vs 0.64 | 0 vs 0 | 10 / 2 |
| PCA reconstruction error | 50 | -4% | [-24%, +29%] | 1.39 vs 1.44 | 0.67 vs 0.64 | 0 vs 0 | 7 / 5 |
| Autoencoder | 50 | -2% | [-23%, +32%] | 1.42 vs 1.44 | 0.67 vs 0.64 | 0 vs 0 | 6 / 6 |
| Voting ensemble | 50 | -62% | [-72%, -45%] | 0.54 vs 1.44 | 0.64 vs 0.64 | 0 vs 0 | 11 / 1 |

## Alert-fatigue curves (alert budget, labels not used for thresholds)

Test-window event recall (mean over series, 95% CI) when the threshold keeps the calibration-window alert rate within the budget, detector alone, k=1.

Rule layer: none

| Method | 0.1/day | 0.25/day | 0.5/day | 1/day | 2/day | 4/day | 8/day |
|---|---|---|---|---|---|---|---|
| Global z-score (baseline) | 0.18 [0.15, 0.21] | 0.37 [0.33, 0.41] | 0.53 [0.50, 0.56] | 0.59 [0.57, 0.61] | 0.64 [0.62, 0.65] | 0.77 [0.74, 0.80] | 0.80 [0.80, 0.80] |
| Rolling median/MAD | 0.24 [0.17, 0.30] | 0.35 [0.27, 0.43] | 0.48 [0.41, 0.55] | 0.57 [0.54, 0.60] | 0.63 [0.62, 0.64] | 0.77 [0.74, 0.80] | 0.80 [0.80, 0.80] |
| Seasonal residual | 0.22 [0.16, 0.29] | 0.39 [0.32, 0.46] | 0.57 [0.51, 0.63] | 0.69 [0.65, 0.72] | 0.75 [0.73, 0.76] | 0.79 [0.79, 0.80] | 0.80 [0.80, 0.80] |
| EWMA control chart | 0.19 [0.12, 0.25] | 0.39 [0.33, 0.46] | 0.60 [0.54, 0.66] | 0.73 [0.71, 0.75] | 0.78 [0.77, 0.79] | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] |
| Seasonal residual + EWMA | 0.22 [0.15, 0.30] | 0.39 [0.33, 0.45] | 0.61 [0.54, 0.67] | 0.72 [0.70, 0.75] | 0.77 [0.75, 0.78] | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] |
| Robust STL residual | 0.19 [0.15, 0.23] | 0.32 [0.28, 0.36] | 0.47 [0.43, 0.49] | 0.63 [0.59, 0.67] | 0.76 [0.74, 0.78] | 0.79 [0.79, 0.80] | 0.80 [0.80, 0.80] |
| Isolation Forest | 0.18 [0.13, 0.23] | 0.38 [0.32, 0.44] | 0.56 [0.51, 0.62] | 0.72 [0.70, 0.75] | 0.79 [0.79, 0.80] | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] |
| Local outlier factor | 0.17 [0.14, 0.21] | 0.30 [0.25, 0.36] | 0.49 [0.41, 0.56] | 0.62 [0.56, 0.68] | 0.73 [0.70, 0.77] | 0.80 [0.79, 0.80] | 0.80 [0.80, 0.80] |
| One-class SVM | 0.22 [0.17, 0.28] | 0.37 [0.31, 0.44] | 0.60 [0.55, 0.65] | 0.69 [0.64, 0.73] | 0.75 [0.73, 0.77] | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] |
| PCA reconstruction error | 0.12 [0.08, 0.17] | 0.27 [0.21, 0.34] | 0.44 [0.38, 0.50] | 0.59 [0.53, 0.64] | 0.70 [0.67, 0.73] | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] |
| Autoencoder | 0.14 [0.09, 0.18] | 0.26 [0.19, 0.32] | 0.42 [0.35, 0.50] | 0.56 [0.50, 0.62] | 0.69 [0.66, 0.73] | 0.79 [0.79, 0.80] | 0.80 [0.80, 0.80] |
| Voting ensemble | 0.21 [0.14, 0.27] | 0.37 [0.30, 0.44] | 0.57 [0.51, 0.62] | 0.71 [0.67, 0.74] | 0.77 [0.75, 0.78] | 0.80 [0.80, 0.80] | 0.80 [0.80, 0.80] |

Rule layer: all

| Method | 0.1/day | 0.25/day | 0.5/day | 1/day | 2/day | 4/day | 8/day |
|---|---|---|---|---|---|---|---|
| Global z-score (baseline) | 0.40 [0.40, 0.40] | 0.62 [0.58, 0.66] | 0.87 [0.83, 0.91] | 0.97 [0.95, 0.99] | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Rolling median/MAD | 0.40 [0.40, 0.40] | 0.65 [0.58, 0.72] | 0.83 [0.75, 0.90] | 0.94 [0.90, 0.98] | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Seasonal residual | 0.40 [0.40, 0.40] | 0.63 [0.57, 0.70] | 0.86 [0.81, 0.91] | 0.97 [0.96, 0.99] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| EWMA control chart | 0.40 [0.40, 0.40] | 0.61 [0.55, 0.68] | 0.89 [0.84, 0.93] | 0.98 [0.97, 0.99] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Seasonal residual + EWMA | 0.40 [0.40, 0.40] | 0.66 [0.59, 0.73] | 0.90 [0.86, 0.95] | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Robust STL residual | 0.40 [0.40, 0.40] | 0.60 [0.56, 0.64] | 0.77 [0.75, 0.80] | 0.93 [0.90, 0.95] | 0.99 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Isolation Forest | 0.40 [0.40, 0.40] | 0.61 [0.55, 0.66] | 0.86 [0.82, 0.90] | 0.98 [0.97, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Local outlier factor | 0.40 [0.40, 0.40] | 0.58 [0.55, 0.62] | 0.77 [0.71, 0.83] | 0.88 [0.83, 0.93] | 0.96 [0.94, 0.99] | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] |
| One-class SVM | 0.40 [0.40, 0.40] | 0.64 [0.59, 0.70] | 0.88 [0.83, 0.92] | 0.95 [0.92, 0.98] | 0.98 [0.96, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| PCA reconstruction error | 0.40 [0.40, 0.40] | 0.54 [0.49, 0.59] | 0.76 [0.69, 0.82] | 0.89 [0.85, 0.93] | 0.97 [0.96, 0.98] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Autoencoder | 0.40 [0.40, 0.40] | 0.54 [0.50, 0.59] | 0.72 [0.65, 0.79] | 0.86 [0.82, 0.90] | 0.94 [0.92, 0.97] | 0.99 [0.99, 1.00] | 1.00 [1.00, 1.00] |
| Voting ensemble | 0.40 [0.40, 0.40] | 0.63 [0.57, 0.71] | 0.85 [0.80, 0.89] | 0.98 [0.96, 0.99] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |

False alerts per day, precision and median time to detect at three budgets (detector alone, k=1):

| Method | 0.25/day: FA/day, precision, TTD min | 1/day: FA/day, precision, TTD min | 4/day: FA/day, precision, TTD min |
|---|---|---|---|
| Global z-score (baseline) | 0.05, 0.84, 120 | 0.61, 0.38, 0 | 1.05, 0.66, 0 |
| Rolling median/MAD | 0.14, 0.55, 0 | 0.71, 0.32, 0 | 0.96, 0.66, 0 |
| Seasonal residual | 0.09, 0.72, 0 | 0.71, 0.36, 0 | 1.18, 0.57, 0 |
| EWMA control chart | 0.05, 0.83, 240 | 0.71, 0.31, 0 | 0.11, 0.82, 0 |
| Seasonal residual + EWMA | 0.07, 0.76, 105 | 0.75, 0.29, 0 | 0.01, 0.85, 0 |
| Robust STL residual | 0.15, 0.45, 0 | 0.76, 0.27, 0 | 1.38, 0.56, 0 |
| Isolation Forest | 0.09, 0.67, 0 | 0.79, 0.30, 0 | 0.01, 0.85, 0 |
| Local outlier factor | 0.14, 0.48, 0 | 0.97, 0.26, 0 | 0.51, 0.74, 0 |
| One-class SVM | 0.15, 0.53, 0 | 0.59, 0.44, 15 | 0.87, 0.62, 0 |
| PCA reconstruction error | 0.16, 0.37, 0 | 0.86, 0.24, 0 | 2.36, 0.34, 0 |
| Autoencoder | 0.19, 0.32, 0 | 0.83, 0.28, 15 | 1.95, 0.43, 0 |
| Voting ensemble | 0.09, 0.68, 0 | 0.77, 0.31, 0 | 0.11, 0.83, 0 |

## Persistence rule (k consecutive flagged points), threshold re-calibrated per k

| Method | k | Target reached | Test FA/day | Test event recall | Median TTD, min |
|---|---|---|---|---|---|
| Global z-score (baseline) | 1 | 0.10 | 1.65 | 0.63 | 0 |
| Global z-score (baseline) | 2 | 0.05 | 1.17 | 0.57 | 60 |
| Global z-score (baseline) | 3 | 0.00 | 0.74 | 0.51 | 120 |
| Rolling median/MAD | 1 | 0.15 | 1.94 | 0.62 | 0 |
| Rolling median/MAD | 2 | 0.10 | 1.18 | 0.55 | 60 |
| Rolling median/MAD | 3 | 0.00 | 0.79 | 0.48 | 120 |
| Seasonal residual | 1 | 0.98 | 1.99 | 0.75 | 0 |
| Seasonal residual | 2 | 0.23 | 1.06 | 0.64 | 60 |
| Seasonal residual | 3 | 0.02 | 0.67 | 0.58 | 120 |
| EWMA control chart | 1 | 0.87 | 1.24 | 0.77 | 0 |
| EWMA control chart | 2 | 0.27 | 0.90 | 0.67 | 60 |
| EWMA control chart | 3 | 0.02 | 0.67 | 0.59 | 120 |
| Seasonal residual + EWMA | 1 | 0.75 | 1.49 | 0.75 | 0 |
| Seasonal residual + EWMA | 2 | 0.23 | 0.70 | 0.65 | 60 |
| Seasonal residual + EWMA | 3 | 0.00 | 0.51 | 0.57 | 120 |
| Robust STL residual | 1 | 0.90 | 1.83 | 0.76 | 0 |
| Robust STL residual | 2 | 0.22 | 1.26 | 0.68 | 60 |
| Robust STL residual | 3 | 0.03 | 0.76 | 0.57 | 120 |
| Isolation Forest | 1 | 0.78 | 1.18 | 0.77 | 0 |
| Isolation Forest | 2 | 0.22 | 0.82 | 0.69 | 60 |
| Isolation Forest | 3 | 0.00 | 0.64 | 0.60 | 120 |
| Local outlier factor | 1 | 0.65 | 2.18 | 0.77 | 0 |
| Local outlier factor | 2 | 0.18 | 1.46 | 0.70 | 15 |
| Local outlier factor | 3 | 0.00 | 1.10 | 0.64 | 75 |
| One-class SVM | 1 | 1.00 | 1.47 | 0.76 | 0 |
| One-class SVM | 2 | 0.32 | 1.04 | 0.67 | 60 |
| One-class SVM | 3 | 0.02 | 0.69 | 0.58 | 180 |
| PCA reconstruction error | 1 | 0.83 | 2.47 | 0.77 | 0 |
| PCA reconstruction error | 2 | 0.18 | 1.43 | 0.66 | 60 |
| PCA reconstruction error | 3 | 0.00 | 0.92 | 0.56 | 90 |
| Autoencoder | 1 | 0.87 | 2.32 | 0.76 | 0 |
| Autoencoder | 2 | 0.12 | 1.31 | 0.67 | 60 |
| Autoencoder | 3 | 0.00 | 0.96 | 0.60 | 120 |
| Voting ensemble | 1 | 0.82 | 1.44 | 0.75 | 0 |
| Voting ensemble | 2 | 0.22 | 0.81 | 0.66 | 60 |
| Voting ensemble | 3 | 0.00 | 0.62 | 0.58 | 120 |

## Rule-layer ablation (matched recall 0.8, k=1)

Test false alerts per day and event recall of each detector with each subset of the three rules. `all-X` removes rule X from the full layer.

| Method | Rules | Target reached | Test FA/day | Test event recall | Alerts |
|---|---|---|---|---|---|
| Global z-score (baseline) | none | 0.10 | 1.65 | 0.63 | 118.0 |
| Global z-score (baseline) | all | 1.00 | 0.05 | 0.78 | 18.1 |
| Global z-score (baseline) | missing | 1.00 | 0.17 | 0.71 | 28.4 |
| Global z-score (baseline) | range | 0.10 | 1.65 | 0.63 | 118.0 |
| Global z-score (baseline) | stuck | 1.00 | 0.17 | 0.71 | 28.7 |
| Global z-score (baseline) | all-missing | 1.00 | 0.17 | 0.71 | 28.7 |
| Global z-score (baseline) | all-range | 1.00 | 0.05 | 0.78 | 18.1 |
| Global z-score (baseline) | all-stuck | 1.00 | 0.17 | 0.71 | 28.4 |
| Rolling median/MAD | none | 0.15 | 1.94 | 0.62 | 135.6 |
| Rolling median/MAD | all | 1.00 | 0.20 | 0.80 | 28.1 |
| Rolling median/MAD | missing | 1.00 | 0.42 | 0.74 | 44.9 |
| Rolling median/MAD | range | 0.15 | 1.94 | 0.62 | 135.6 |
| Rolling median/MAD | stuck | 1.00 | 0.44 | 0.74 | 46.7 |
| Rolling median/MAD | all-missing | 1.00 | 0.44 | 0.74 | 46.7 |
| Rolling median/MAD | all-range | 1.00 | 0.20 | 0.80 | 28.1 |
| Rolling median/MAD | all-stuck | 1.00 | 0.42 | 0.74 | 44.9 |
| Seasonal residual | none | 0.98 | 1.99 | 0.75 | 143.8 |
| Seasonal residual | all | 1.00 | 0.13 | 0.83 | 25.8 |
| Seasonal residual | missing | 1.00 | 0.35 | 0.77 | 40.7 |
| Seasonal residual | range | 0.98 | 1.99 | 0.75 | 143.8 |
| Seasonal residual | stuck | 1.00 | 0.45 | 0.73 | 46.3 |
| Seasonal residual | all-missing | 1.00 | 0.45 | 0.73 | 46.3 |
| Seasonal residual | all-range | 1.00 | 0.13 | 0.83 | 25.8 |
| Seasonal residual | all-stuck | 1.00 | 0.35 | 0.77 | 40.7 |
| EWMA control chart | none | 0.87 | 1.24 | 0.77 | 91.7 |
| EWMA control chart | all | 1.00 | 0.06 | 0.76 | 18.3 |
| EWMA control chart | missing | 1.00 | 0.20 | 0.74 | 27.5 |
| EWMA control chart | range | 0.87 | 1.24 | 0.77 | 91.7 |
| EWMA control chart | stuck | 1.00 | 0.25 | 0.70 | 30.2 |
| EWMA control chart | all-missing | 1.00 | 0.25 | 0.70 | 30.2 |
| EWMA control chart | all-range | 1.00 | 0.06 | 0.76 | 18.3 |
| EWMA control chart | all-stuck | 1.00 | 0.20 | 0.74 | 27.5 |
| Seasonal residual + EWMA | none | 0.75 | 1.49 | 0.75 | 107.2 |
| Seasonal residual + EWMA | all | 1.00 | 0.07 | 0.77 | 18.0 |
| Seasonal residual + EWMA | missing | 1.00 | 0.25 | 0.76 | 29.9 |
| Seasonal residual + EWMA | range | 0.75 | 1.49 | 0.75 | 107.2 |
| Seasonal residual + EWMA | stuck | 1.00 | 0.28 | 0.72 | 31.0 |
| Seasonal residual + EWMA | all-missing | 1.00 | 0.28 | 0.72 | 31.0 |
| Seasonal residual + EWMA | all-range | 1.00 | 0.07 | 0.77 | 18.0 |
| Seasonal residual + EWMA | all-stuck | 1.00 | 0.25 | 0.76 | 29.9 |
| Robust STL residual | none | 0.90 | 1.83 | 0.76 | 139.8 |
| Robust STL residual | all | 1.00 | 0.25 | 0.76 | 29.1 |
| Robust STL residual | missing | 1.00 | 0.61 | 0.77 | 53.8 |
| Robust STL residual | range | 0.90 | 1.83 | 0.76 | 139.8 |
| Robust STL residual | stuck | 0.97 | 0.75 | 0.73 | 63.7 |
| Robust STL residual | all-missing | 0.97 | 0.75 | 0.73 | 63.7 |
| Robust STL residual | all-range | 1.00 | 0.25 | 0.76 | 29.1 |
| Robust STL residual | all-stuck | 1.00 | 0.61 | 0.77 | 53.8 |
| Isolation Forest | none | 0.78 | 1.18 | 0.77 | 89.2 |
| Isolation Forest | all | 1.00 | 0.11 | 0.80 | 22.5 |
| Isolation Forest | missing | 1.00 | 0.33 | 0.78 | 36.8 |
| Isolation Forest | range | 0.78 | 1.18 | 0.77 | 89.2 |
| Isolation Forest | stuck | 1.00 | 0.36 | 0.73 | 37.4 |
| Isolation Forest | all-missing | 1.00 | 0.36 | 0.73 | 37.4 |
| Isolation Forest | all-range | 1.00 | 0.11 | 0.80 | 22.5 |
| Isolation Forest | all-stuck | 1.00 | 0.33 | 0.78 | 36.8 |
| Local outlier factor | none | 0.65 | 2.18 | 0.77 | 162.7 |
| Local outlier factor | all | 1.00 | 0.53 | 0.79 | 48.1 |
| Local outlier factor | missing | 1.00 | 1.18 | 0.85 | 94.5 |
| Local outlier factor | range | 0.65 | 2.18 | 0.77 | 162.7 |
| Local outlier factor | stuck | 0.85 | 1.47 | 0.74 | 111.6 |
| Local outlier factor | all-missing | 0.85 | 1.47 | 0.74 | 111.6 |
| Local outlier factor | all-range | 1.00 | 0.53 | 0.79 | 48.1 |
| Local outlier factor | all-stuck | 1.00 | 1.18 | 0.85 | 94.5 |
| One-class SVM | none | 1.00 | 1.47 | 0.76 | 121.5 |
| One-class SVM | all | 1.00 | 0.22 | 0.83 | 31.1 |
| One-class SVM | missing | 1.00 | 0.37 | 0.81 | 45.0 |
| One-class SVM | range | 1.00 | 1.47 | 0.76 | 121.5 |
| One-class SVM | stuck | 1.00 | 0.72 | 0.75 | 67.2 |
| One-class SVM | all-missing | 1.00 | 0.72 | 0.75 | 67.2 |
| One-class SVM | all-range | 1.00 | 0.22 | 0.83 | 31.1 |
| One-class SVM | all-stuck | 1.00 | 0.37 | 0.81 | 45.0 |
| PCA reconstruction error | none | 0.83 | 2.47 | 0.77 | 184.8 |
| PCA reconstruction error | all | 1.00 | 0.38 | 0.79 | 36.9 |
| PCA reconstruction error | missing | 1.00 | 0.99 | 0.81 | 80.4 |
| PCA reconstruction error | range | 0.83 | 2.47 | 0.77 | 184.8 |
| PCA reconstruction error | stuck | 1.00 | 1.20 | 0.74 | 93.7 |
| PCA reconstruction error | all-missing | 1.00 | 1.20 | 0.74 | 93.7 |
| PCA reconstruction error | all-range | 1.00 | 0.38 | 0.79 | 36.9 |
| PCA reconstruction error | all-stuck | 1.00 | 0.99 | 0.81 | 80.4 |
| Autoencoder | none | 0.87 | 2.32 | 0.76 | 177.8 |
| Autoencoder | all | 1.00 | 0.48 | 0.78 | 46.2 |
| Autoencoder | missing | 1.00 | 1.07 | 0.81 | 89.0 |
| Autoencoder | range | 0.87 | 2.32 | 0.76 | 177.8 |
| Autoencoder | stuck | 0.98 | 1.38 | 0.72 | 107.8 |
| Autoencoder | all-missing | 0.98 | 1.38 | 0.72 | 107.8 |
| Autoencoder | all-range | 1.00 | 0.48 | 0.78 | 46.2 |
| Autoencoder | all-stuck | 1.00 | 1.07 | 0.81 | 89.0 |
| Voting ensemble | none | 0.82 | 1.44 | 0.75 | 106.3 |
| Voting ensemble | all | 1.00 | 0.13 | 0.80 | 24.4 |
| Voting ensemble | missing | 1.00 | 0.33 | 0.76 | 37.0 |
| Voting ensemble | range | 0.82 | 1.44 | 0.75 | 106.3 |
| Voting ensemble | stuck | 1.00 | 0.40 | 0.72 | 40.6 |
| Voting ensemble | all-missing | 1.00 | 0.40 | 0.72 | 40.6 |
| Voting ensemble | all-range | 1.00 | 0.13 | 0.80 | 24.4 |
| Voting ensemble | all-stuck | 1.00 | 0.33 | 0.76 | 37.0 |

## Best method chosen on calibration windows only

Candidates must reach the target on at least 90% of units; ranking by mean calibration false alerts per day (detector alone, k=1):

One-class SVM (1.37), Robust STL residual (1.62), Seasonal residual (1.81)
