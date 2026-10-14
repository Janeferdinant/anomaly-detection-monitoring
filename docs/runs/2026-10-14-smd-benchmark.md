# Benchmark run 2026-10-14: SMD

Status label: `Verified (run 2026-10-14, seeds 0, 1, 2, 3, 4)`. Produced by `scripts/run_benchmark.py --dataset smd`. These are new numbers from an independent re-implementation; they are not the result reported for the earlier deployment on the CV.

## Settings

* Python 3.11.9, numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1
* Hardware: AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, 16 logical cores, Windows; 8 worker processes; runtime 15.4 minutes
* Units: 28 (22 usable for matched recall: at least 2 calibration events and one test event; 28 usable for alert budgets)
* Model seeds: 0, 1, 2, 3, 4 (deterministic methods are run once)
* Primary recall target 0.65; all targets (0.5, 0.65, 0.8); persistence k in (1, 2, 3); alert budgets (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0) alerts/day
* Threshold rules: matched recall = highest threshold whose calibration event recall >= target; alert budget = lowest threshold that keeps all alerts on the calibration window within the budget (no labels).
* FA = false alert = an alert overlapping no labelled event. Event recall = share of labelled events with at least one flagged point. Time to detect (TTD) = minutes from event start to first flagged point, median over detected events.
* Percent changes are the pooled change in test false alerts per day relative to the global z-score at the same calibration recall (negative = fewer false alerts); CIs are 95% percentile bootstrap intervals over series (UCI: 12 series, each averaged over its injection seeds; SMD: machines).
* Data: Server Machine Dataset (MIT), commit 7fb0e0acf8; real server metrics with expert labels; 5-minute means (labels: a bin is anomalous if any minute is); fit window = first (unlabelled) part, calibration = first half of the labelled part, test = second half; 192 constant-in-training channels dropped over 28 machines; univariate methods are run per channel and alert on the largest normalised channel score.

## Development notes (full disclosure)

* A first complete run of this benchmark exposed a bug in the one-class SVM score: it was `-decision_function`, which is negative for normal points, so missing values (which score 0 by contract) ranked above normal points and low thresholds flagged every missing burst for free. The score is now shifted to be non-negative, a test checks that every detector's scores are non-negative, and both benchmarks were re-run. This log is from the re-run; the first run was discarded.
* Thresholds, hyper-parameters, rule settings and the primary comparisons were fixed before the test windows of the final run were read; the SMD primary recall target (0.65) was chosen from calibration-window diagnostics only.
* The alert budget counts alert segments per day, so a threshold low enough that a detector stays flagged for a very long stretch counts as one alert. The column `flagged_share` in `budget.csv.gz` records the share of test points flagged.

## Hyper-parameters chosen on calibration windows

| Method | Parameters | Units | Share not reaching target | Calibration FA/day | Chosen |
|---|---|---|---|---|---|
| Autoencoder | {} | 22 | 0.00 | 2.11 | True |
| Autoencoder | {'hidden': 16, 'bottleneck': 4} | 22 | 0.00 | 2.22 | False |
| Seasonal residual + EWMA | {} | 22 | 0.00 | 1.81 | True |
| Voting ensemble | {} | 22 | 0.00 | 3.02 | False |
| Voting ensemble | {'min_votes': 3} | 22 | 0.00 | 2.63 | True |
| EWMA control chart | {} | 22 | 0.14 | 3.22 | False |
| EWMA control chart | {'lam': 0.1} | 22 | 0.32 | 2.00 | False |
| EWMA control chart | {'lam': 0.5} | 22 | 0.00 | 3.61 | True |
| Isolation Forest | {} | 22 | 0.00 | 3.49 | False |
| Isolation Forest | {'n_estimators': 200, 'max_samples': 1024} | 22 | 0.00 | 2.56 | True |
| Local outlier factor | {} | 22 | 0.23 | 7.52 | False |
| Local outlier factor | {'n_neighbors': 50} | 22 | 0.09 | 7.12 | True |
| One-class SVM | {} | 22 | 0.00 | 1.23 | True |
| One-class SVM | {'nu': 0.01} | 22 | 0.00 | 1.52 | False |
| PCA reconstruction error | {} | 22 | 0.00 | 3.43 | False |
| PCA reconstruction error | {'variance': 0.8} | 22 | 0.00 | 2.28 | True |
| Rolling median/MAD | {} | 22 | 0.05 | 1.62 | False |
| Rolling median/MAD | {'window_days': 3} | 22 | 0.05 | 1.67 | False |
| Rolling median/MAD | {'window_days': 1} | 22 | 0.00 | 2.66 | True |
| Seasonal residual | {} | 22 | 0.00 | 2.62 | False |
| Seasonal residual | {'profile_days': 3} | 22 | 0.00 | 3.33 | False |
| Seasonal residual | {'scale_days': 7} | 22 | 0.00 | 1.80 | True |
| Seasonal residual | {'profile_days': 3, 'scale_days': 7} | 22 | 0.00 | 2.95 | False |
| Robust STL residual | {} | 22 | 0.00 | 2.12 | True |
| Robust STL residual | {'trend_days': 1} | 22 | 0.00 | 2.36 | False |
| Global z-score (baseline) | {} | 22 | 0.09 | 1.62 | True |

Rule-layer settings (z-score + all rules, same criterion):

| Rule settings | Share not reaching target | Calibration FA/day | Chosen |
|---|---|---|---|
| {'range_margin': 0.5, 'max_flat_fraction': 0.5} | 0.00 | 13.83 | False |
| {'range_margin': 0.5, 'max_flat_fraction': 0.05} | 0.00 | 4.52 | True |
| {'range_margin': 2.0, 'max_flat_fraction': 0.5} | 0.05 | 12.78 | False |
| {'range_margin': 2.0, 'max_flat_fraction': 0.05} | 0.05 | 2.13 | False |
| {'range_margin': 5.0, 'max_flat_fraction': 0.05} | 0.05 | 2.04 | False |

## Primary comparison: Detector alone, k=1, calibration recall target 0.65

| Method | Matched units | Change in FA/day | 95% CI | FA/day (method vs z-score) | Test event recall | Median time to detect, min | Series better / worse |
|---|---|---|---|---|---|---|---|
| Rolling median/MAD | 20 | -42% | [-74%, +106%] | 1.64 vs 2.82 | 0.74 vs 0.71 | 0 vs 0 | 5 / 7 |
| Seasonal residual | 20 | -40% | [-71%, +100%] | 1.69 vs 2.82 | 0.63 vs 0.71 | 1 vs 0 | 5 / 7 |
| EWMA control chart | 20 | +51% | [-31%, +526%] | 4.25 vs 2.82 | 0.59 vs 0.71 | 0 vs 0 | 4 / 10 |
| Seasonal residual + EWMA | 20 | -46% | [-74%, +71%] | 1.51 vs 2.82 | 0.68 vs 0.71 | 0 vs 0 | 5 / 8 |
| Robust STL residual | 20 | -22% | [-67%, +200%] | 2.18 vs 2.82 | 0.68 vs 0.71 | 1 vs 0 | 4 / 8 |
| Isolation Forest | 20 | +48% | [-35%, +412%] | 4.18 vs 2.82 | 0.49 vs 0.71 | 1 vs 0 | 5 / 9 |
| Local outlier factor | 19 | +199% | [+26%, +1292%] | 6.17 vs 2.06 | 0.64 vs 0.71 | 0 vs 0 | 2 / 11 |
| One-class SVM | 20 | -24% | [-69%, +200%] | 2.14 vs 2.82 | 0.53 vs 0.71 | 1 vs 0 | 5 / 8 |
| PCA reconstruction error | 20 | -0% | [-50%, +218%] | 2.82 vs 2.82 | 0.55 vs 0.71 | 0 vs 0 | 3 / 10 |
| Autoencoder | 20 | -16% | [-59%, +189%] | 2.37 vs 2.82 | 0.57 vs 0.71 | 6 vs 0 | 6 / 8 |
| Voting ensemble | 20 | +1% | [-53%, +239%] | 2.85 vs 2.82 | 0.58 vs 0.71 | 0 vs 0 | 4 / 9 |

Operating points on the test window (all methods, including the baseline):

| Method | Target reached (calibration) | Test FA/day [95% CI] | Test event recall [95% CI] | Event precision [95% CI] | Median TTD, min |
|---|---|---|---|---|---|
| Global z-score (baseline) | 0.91 | 2.56 [0.61, 4.92] | 0.69 [0.57, 0.81] | 0.63 [0.48, 0.78] | 0 |
| Rolling median/MAD | 1.00 | 2.25 [1.01, 3.85] | 0.76 [0.65, 0.87] | 0.48 [0.37, 0.60] | 0 |
| Seasonal residual | 1.00 | 2.28 [1.00, 3.89] | 0.65 [0.55, 0.75] | 0.47 [0.34, 0.60] | 4 |
| EWMA control chart | 1.00 | 4.36 [2.78, 6.08] | 0.60 [0.48, 0.71] | 0.22 [0.14, 0.29] | 0 |
| Seasonal residual + EWMA | 1.00 | 1.71 [0.86, 2.67] | 0.66 [0.54, 0.78] | 0.46 [0.34, 0.60] | 0 |
| Robust STL residual | 1.00 | 2.45 [1.36, 3.69] | 0.68 [0.57, 0.80] | 0.48 [0.35, 0.62] | 4 |
| Isolation Forest | 1.00 | 3.86 [1.86, 6.35] | 0.50 [0.35, 0.65] | 0.31 [0.18, 0.44] | 2 |
| Local outlier factor | 0.91 | 5.54 [2.93, 8.50] | 0.63 [0.50, 0.75] | 0.32 [0.22, 0.44] | 0 |
| One-class SVM | 1.00 | 1.98 [0.88, 3.39] | 0.55 [0.40, 0.69] | 0.40 [0.29, 0.52] | 2 |
| PCA reconstruction error | 1.00 | 2.78 [1.51, 4.11] | 0.57 [0.43, 0.70] | 0.29 [0.21, 0.37] | 0 |
| Autoencoder | 1.00 | 2.43 [1.41, 3.54] | 0.60 [0.45, 0.73] | 0.35 [0.26, 0.46] | 6 |
| Voting ensemble | 1.00 | 2.62 [1.26, 4.15] | 0.58 [0.46, 0.71] | 0.35 [0.25, 0.46] | 2 |

## Primary comparison: Detector + rule layer (missing, range, stuck), k=1, calibration recall target 0.65

| Method | Matched units | Change in FA/day | 95% CI | FA/day (method vs z-score) | Test event recall | Median time to detect, min | Series better / worse |
|---|---|---|---|---|---|---|---|
| Rolling median/MAD | 22 | +0% | [-3%, +3%] | 7.45 vs 7.44 | 0.90 vs 0.89 | 0 vs 0 | 1 / 3 |
| Seasonal residual | 22 | +1% | [-2%, +8%] | 7.54 vs 7.44 | 0.89 vs 0.89 | 0 vs 0 | 1 / 2 |
| EWMA control chart | 22 | -5% | [-19%, +5%] | 7.08 vs 7.44 | 0.90 vs 0.89 | 0 vs 0 | 2 / 5 |
| Seasonal residual + EWMA | 22 | -2% | [-9%, +4%] | 7.30 vs 7.44 | 0.90 vs 0.89 | 0 vs 0 | 1 / 2 |
| Robust STL residual | 22 | +1% | [-1%, +4%] | 7.50 vs 7.44 | 0.90 vs 0.89 | 0 vs 0 | 2 / 3 |
| Isolation Forest | 22 | +9% | [+0%, +22%] | 8.09 vs 7.44 | 0.90 vs 0.89 | 0 vs 0 | 4 / 5 |
| Local outlier factor | 22 | -1% | [-19%, +20%] | 7.38 vs 7.44 | 0.91 vs 0.89 | 0 vs 0 | 3 / 4 |
| One-class SVM | 22 | -1% | [-10%, +10%] | 7.39 vs 7.44 | 0.90 vs 0.89 | 0 vs 0 | 2 / 3 |
| PCA reconstruction error | 22 | -3% | [-17%, +9%] | 7.24 vs 7.44 | 0.90 vs 0.89 | 0 vs 0 | 3 / 5 |
| Autoencoder | 22 | -2% | [-15%, +10%] | 7.29 vs 7.44 | 0.91 vs 0.89 | 0 vs 0 | 4 / 6 |
| Voting ensemble | 22 | +1% | [-10%, +16%] | 7.54 vs 7.44 | 0.90 vs 0.89 | 0 vs 0 | 5 / 4 |

Operating points on the test window (all methods, including the baseline):

| Method | Target reached (calibration) | Test FA/day [95% CI] | Test event recall [95% CI] | Event precision [95% CI] | Median TTD, min |
|---|---|---|---|---|---|
| Global z-score (baseline) | 1.00 | 7.44 [3.64, 12.69] | 0.89 [0.83, 0.95] | 0.32 [0.19, 0.47] | 0 |
| Rolling median/MAD | 1.00 | 7.45 [3.66, 12.67] | 0.90 [0.84, 0.96] | 0.32 [0.19, 0.47] | 0 |
| Seasonal residual | 1.00 | 7.54 [3.71, 12.83] | 0.89 [0.83, 0.94] | 0.32 [0.19, 0.48] | 0 |
| EWMA control chart | 1.00 | 7.08 [3.49, 12.33] | 0.90 [0.84, 0.96] | 0.32 [0.20, 0.48] | 0 |
| Seasonal residual + EWMA | 1.00 | 7.30 [3.63, 12.54] | 0.90 [0.84, 0.95] | 0.32 [0.19, 0.48] | 0 |
| Robust STL residual | 1.00 | 7.50 [3.67, 12.78] | 0.90 [0.84, 0.96] | 0.32 [0.19, 0.47] | 0 |
| Isolation Forest | 1.00 | 8.09 [3.87, 13.62] | 0.90 [0.84, 0.95] | 0.32 [0.19, 0.48] | 0 |
| Local outlier factor | 1.00 | 7.38 [3.54, 12.80] | 0.91 [0.85, 0.96] | 0.33 [0.20, 0.48] | 0 |
| One-class SVM | 1.00 | 7.39 [3.74, 12.65] | 0.90 [0.84, 0.96] | 0.32 [0.19, 0.47] | 0 |
| PCA reconstruction error | 1.00 | 7.24 [3.55, 12.51] | 0.90 [0.84, 0.96] | 0.32 [0.19, 0.47] | 0 |
| Autoencoder | 1.00 | 7.29 [3.59, 12.55] | 0.91 [0.85, 0.96] | 0.32 [0.19, 0.48] | 0 |
| Voting ensemble | 1.00 | 7.54 [3.76, 12.86] | 0.90 [0.84, 0.96] | 0.32 [0.19, 0.47] | 0 |

## Other recall targets, detector alone, k=1

Target 0.5:

| Method | Matched units | Change in FA/day | 95% CI | FA/day (method vs z-score) | Test event recall | Median time to detect, min | Series better / worse |
|---|---|---|---|---|---|---|---|
| Rolling median/MAD | 22 | -51% | [-85%, +181%] | 1.30 vs 2.65 | 0.61 vs 0.61 | 5 vs 5 | 4 / 5 |
| Seasonal residual | 22 | -39% | [-82%, +244%] | 1.62 vs 2.65 | 0.56 vs 0.61 | 5 vs 5 | 6 / 4 |
| EWMA control chart | 22 | +17% | [-52%, +615%] | 3.11 vs 2.65 | 0.49 vs 0.61 | 0 vs 5 | 4 / 8 |
| Seasonal residual + EWMA | 22 | -57% | [-83%, +122%] | 1.15 vs 2.65 | 0.56 vs 0.61 | 5 vs 5 | 6 / 5 |
| Robust STL residual | 22 | -21% | [-72%, +372%] | 2.08 vs 2.65 | 0.60 vs 0.61 | 5 vs 5 | 3 / 9 |
| Isolation Forest | 22 | +27% | [-52%, +509%] | 3.35 vs 2.65 | 0.44 vs 0.61 | 4 vs 5 | 4 / 8 |
| Local outlier factor | 22 | +35% | [-53%, +704%] | 3.58 vs 2.65 | 0.54 vs 0.61 | 2 vs 5 | 5 / 7 |
| One-class SVM | 22 | -30% | [-76%, +288%] | 1.86 vs 2.65 | 0.50 vs 0.61 | 8 vs 5 | 4 / 7 |
| PCA reconstruction error | 22 | -42% | [-75%, +149%] | 1.53 vs 2.65 | 0.49 vs 0.61 | 5 vs 5 | 5 / 7 |
| Autoencoder | 22 | -39% | [-75%, +198%] | 1.63 vs 2.65 | 0.49 vs 0.61 | 6 vs 5 | 6 / 6 |
| Voting ensemble | 22 | -34% | [-73%, +200%] | 1.74 vs 2.65 | 0.52 vs 0.61 | 5 vs 5 | 5 / 6 |

Target 0.8:

| Method | Matched units | Change in FA/day | 95% CI | FA/day (method vs z-score) | Test event recall | Median time to detect, min | Series better / worse |
|---|---|---|---|---|---|---|---|
| Rolling median/MAD | 19 | -35% | [-68%, +33%] | 1.99 vs 3.08 | 0.79 vs 0.80 | 0 vs 0 | 9 / 7 |
| Seasonal residual | 19 | +35% | [-40%, +248%] | 4.16 vs 3.08 | 0.72 vs 0.80 | 0 vs 0 | 7 / 8 |
| EWMA control chart | 18 | +74% | [-18%, +366%] | 5.34 vs 3.08 | 0.80 vs 0.79 | 1 vs 0 | 5 / 11 |
| Seasonal residual + EWMA | 18 | -3% | [-53%, +158%] | 2.98 vs 3.08 | 0.76 vs 0.79 | 0 vs 0 | 8 / 7 |
| Robust STL residual | 19 | +36% | [-31%, +211%] | 4.18 vs 3.08 | 0.72 vs 0.80 | 0 vs 0 | 6 / 9 |
| Isolation Forest | 19 | +89% | [+14%, +304%] | 5.82 vs 3.08 | 0.69 vs 0.80 | 0 vs 0 | 7 / 10 |
| Local outlier factor | 18 | +168% | [+23%, +648%] | 7.42 vs 2.77 | 0.72 vs 0.80 | 0 vs 0 | 5 / 11 |
| One-class SVM | 19 | +20% | [-48%, +233%] | 3.70 vs 3.08 | 0.73 vs 0.80 | 1 vs 0 | 9 / 8 |
| PCA reconstruction error | 19 | +45% | [-27%, +247%] | 4.48 vs 3.08 | 0.73 vs 0.80 | 0 vs 0 | 7 / 8 |
| Autoencoder | 19 | +25% | [-37%, +201%] | 3.85 vs 3.08 | 0.72 vs 0.80 | 0 vs 0 | 9 / 8 |
| Voting ensemble | 19 | +62% | [-14%, +256%] | 4.99 vs 3.08 | 0.78 vs 0.80 | 0 vs 0 | 5 / 12 |

## Alert-fatigue curves (alert budget, labels not used for thresholds)

Test-window event recall (mean over series, 95% CI) when the threshold keeps the calibration-window alert rate within the budget, detector alone, k=1.

Rule layer: none

| Method | 0.1/day | 0.25/day | 0.5/day | 1/day | 2/day | 4/day | 8/day |
|---|---|---|---|---|---|---|---|
| Global z-score (baseline) | 0.29 [0.17, 0.42] | 0.31 [0.20, 0.44] | 0.36 [0.23, 0.49] | 0.67 [0.57, 0.78] | 0.79 [0.68, 0.89] | 0.85 [0.74, 0.93] | 0.89 [0.79, 0.97] |
| Rolling median/MAD | 0.25 [0.14, 0.38] | 0.29 [0.17, 0.42] | 0.32 [0.20, 0.46] | 0.58 [0.47, 0.69] | 0.72 [0.59, 0.83] | 0.81 [0.71, 0.90] | 0.86 [0.76, 0.94] |
| Seasonal residual | 0.29 [0.17, 0.42] | 0.31 [0.20, 0.44] | 0.38 [0.25, 0.52] | 0.59 [0.48, 0.69] | 0.71 [0.61, 0.80] | 0.81 [0.73, 0.89] | 0.91 [0.83, 0.97] |
| EWMA control chart | 0.18 [0.10, 0.28] | 0.21 [0.13, 0.32] | 0.25 [0.16, 0.35] | 0.36 [0.24, 0.48] | 0.57 [0.46, 0.68] | 0.68 [0.58, 0.79] | 0.82 [0.72, 0.91] |
| Seasonal residual + EWMA | 0.29 [0.18, 0.42] | 0.32 [0.20, 0.44] | 0.37 [0.25, 0.50] | 0.54 [0.42, 0.66] | 0.65 [0.52, 0.76] | 0.82 [0.74, 0.89] | 0.89 [0.81, 0.95] |
| Robust STL residual | 0.30 [0.18, 0.43] | 0.32 [0.20, 0.44] | 0.35 [0.23, 0.48] | 0.55 [0.43, 0.67] | 0.68 [0.55, 0.79] | 0.82 [0.73, 0.91] | 0.89 [0.82, 0.95] |
| Isolation Forest | 0.23 [0.14, 0.34] | 0.25 [0.15, 0.36] | 0.28 [0.19, 0.39] | 0.38 [0.28, 0.48] | 0.52 [0.39, 0.65] | 0.73 [0.61, 0.85] | 0.86 [0.76, 0.94] |
| Local outlier factor | 0.26 [0.16, 0.38] | 0.26 [0.16, 0.38] | 0.26 [0.16, 0.38] | 0.44 [0.32, 0.57] | 0.54 [0.42, 0.67] | 0.65 [0.54, 0.77] | 0.75 [0.64, 0.86] |
| One-class SVM | 0.29 [0.17, 0.42] | 0.32 [0.20, 0.45] | 0.38 [0.25, 0.51] | 0.51 [0.37, 0.65] | 0.66 [0.53, 0.79] | 0.77 [0.66, 0.87] | 0.87 [0.77, 0.95] |
| PCA reconstruction error | 0.29 [0.19, 0.41] | 0.29 [0.19, 0.41] | 0.33 [0.22, 0.44] | 0.60 [0.48, 0.72] | 0.66 [0.54, 0.78] | 0.80 [0.69, 0.90] | 0.88 [0.81, 0.95] |
| Autoencoder | 0.29 [0.19, 0.39] | 0.30 [0.20, 0.40] | 0.34 [0.23, 0.45] | 0.55 [0.42, 0.68] | 0.66 [0.52, 0.78] | 0.81 [0.70, 0.91] | 0.86 [0.76, 0.95] |
| Voting ensemble | 0.22 [0.13, 0.33] | 0.25 [0.15, 0.36] | 0.29 [0.19, 0.41] | 0.53 [0.41, 0.66] | 0.65 [0.52, 0.77] | 0.78 [0.67, 0.89] | 0.88 [0.80, 0.96] |

Rule layer: all

| Method | 0.1/day | 0.25/day | 0.5/day | 1/day | 2/day | 4/day | 8/day |
|---|---|---|---|---|---|---|---|
| Global z-score (baseline) | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.87, 0.96] | 0.94 [0.90, 0.98] | 0.96 [0.92, 0.99] |
| Rolling median/MAD | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.87, 0.96] | 0.93 [0.89, 0.97] | 0.96 [0.92, 0.99] |
| Seasonal residual | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.87, 0.96] | 0.92 [0.88, 0.96] | 0.96 [0.92, 0.99] |
| EWMA control chart | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.86, 0.96] | 0.92 [0.88, 0.96] | 0.96 [0.92, 0.99] |
| Seasonal residual + EWMA | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.87, 0.96] | 0.93 [0.89, 0.97] | 0.96 [0.92, 0.99] |
| Robust STL residual | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.86, 0.96] | 0.93 [0.89, 0.97] | 0.94 [0.90, 0.98] |
| Isolation Forest | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.87, 0.97] | 0.93 [0.89, 0.97] | 0.95 [0.91, 0.99] |
| Local outlier factor | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.86, 0.96] | 0.92 [0.87, 0.96] | 0.95 [0.91, 0.98] |
| One-class SVM | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.86, 0.96] | 0.92 [0.88, 0.96] | 0.97 [0.93, 1.00] |
| PCA reconstruction error | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.86, 0.96] | 0.93 [0.89, 0.97] | 0.97 [0.93, 1.00] |
| Autoencoder | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.86, 0.96] | 0.94 [0.89, 0.98] | 0.97 [0.93, 1.00] |
| Voting ensemble | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.91 [0.85, 0.96] | 0.92 [0.86, 0.96] | 0.93 [0.88, 0.97] | 0.96 [0.93, 0.99] |

False alerts per day, precision and median time to detect at three budgets (detector alone, k=1):

| Method | 0.25/day: FA/day, precision, TTD min | 1/day: FA/day, precision, TTD min | 4/day: FA/day, precision, TTD min |
|---|---|---|---|
| Global z-score (baseline) | 0.31, 0.62, 10 | 1.73, 0.48, 0 | 5.70, 0.32, 0 |
| Rolling median/MAD | 0.27, 0.50, 10 | 0.82, 0.53, 5 | 4.25, 0.28, 0 |
| Seasonal residual | 0.29, 0.59, 12 | 0.88, 0.50, 5 | 3.41, 0.31, 0 |
| EWMA control chart | 0.28, 0.43, 26 | 0.96, 0.29, 18 | 4.10, 0.21, 5 |
| Seasonal residual + EWMA | 0.27, 0.60, 15 | 1.06, 0.43, 5 | 3.35, 0.29, 0 |
| Robust STL residual | 0.30, 0.57, 8 | 0.93, 0.47, 5 | 4.79, 0.27, 2 |
| Isolation Forest | 0.54, 0.49, 20 | 1.08, 0.44, 12 | 4.11, 0.26, 0 |
| Local outlier factor | 2.11, 0.40, 8 | 3.26, 0.35, 5 | 4.43, 0.28, 0 |
| One-class SVM | 1.07, 0.43, 5 | 1.88, 0.44, 5 | 3.51, 0.31, 0 |
| PCA reconstruction error | 0.53, 0.40, 10 | 1.36, 0.41, 10 | 4.59, 0.24, 1 |
| Autoencoder | 0.28, 0.43, 6 | 1.04, 0.44, 10 | 3.36, 0.26, 2 |
| Voting ensemble | 0.50, 0.41, 10 | 1.01, 0.41, 10 | 4.34, 0.22, 1 |

## Persistence rule (k consecutive flagged points), threshold re-calibrated per k

| Method | k | Target reached | Test FA/day | Test event recall | Median TTD, min |
|---|---|---|---|---|---|
| Global z-score (baseline) | 1 | 0.91 | 2.56 | 0.69 | 0 |
| Global z-score (baseline) | 2 | 0.73 | 4.83 | 0.67 | 2 |
| Global z-score (baseline) | 3 | 0.50 | 3.47 | 0.64 | 5 |
| Rolling median/MAD | 1 | 1.00 | 2.25 | 0.76 | 0 |
| Rolling median/MAD | 2 | 0.73 | 4.21 | 0.64 | 5 |
| Rolling median/MAD | 3 | 0.50 | 2.31 | 0.53 | 10 |
| Seasonal residual | 1 | 1.00 | 2.28 | 0.65 | 4 |
| Seasonal residual | 2 | 0.82 | 3.23 | 0.57 | 5 |
| Seasonal residual | 3 | 0.64 | 1.87 | 0.50 | 8 |
| EWMA control chart | 1 | 1.00 | 4.36 | 0.60 | 0 |
| EWMA control chart | 2 | 0.77 | 4.31 | 0.54 | 5 |
| EWMA control chart | 3 | 0.64 | 2.08 | 0.50 | 0 |
| Seasonal residual + EWMA | 1 | 1.00 | 1.71 | 0.66 | 0 |
| Seasonal residual + EWMA | 2 | 0.73 | 1.39 | 0.59 | 2 |
| Seasonal residual + EWMA | 3 | 0.64 | 1.39 | 0.57 | 0 |
| Robust STL residual | 1 | 1.00 | 2.45 | 0.68 | 4 |
| Robust STL residual | 2 | 0.82 | 4.30 | 0.62 | 5 |
| Robust STL residual | 3 | 0.55 | 2.95 | 0.55 | 5 |
| Isolation Forest | 1 | 1.00 | 3.86 | 0.50 | 2 |
| Isolation Forest | 2 | 1.00 | 2.90 | 0.52 | 5 |
| Isolation Forest | 3 | 0.59 | 2.48 | 0.46 | 10 |
| Local outlier factor | 1 | 0.91 | 5.54 | 0.63 | 0 |
| Local outlier factor | 2 | 0.86 | 4.51 | 0.59 | 5 |
| Local outlier factor | 3 | 0.73 | 3.70 | 0.53 | 0 |
| One-class SVM | 1 | 1.00 | 1.98 | 0.55 | 2 |
| One-class SVM | 2 | 0.95 | 1.85 | 0.59 | 6 |
| One-class SVM | 3 | 0.68 | 1.30 | 0.48 | 10 |
| PCA reconstruction error | 1 | 1.00 | 2.78 | 0.57 | 0 |
| PCA reconstruction error | 2 | 0.95 | 3.43 | 0.60 | 5 |
| PCA reconstruction error | 3 | 0.68 | 3.08 | 0.56 | 2 |
| Autoencoder | 1 | 1.00 | 2.43 | 0.60 | 6 |
| Autoencoder | 2 | 0.95 | 2.74 | 0.60 | 5 |
| Autoencoder | 3 | 0.68 | 2.59 | 0.52 | 2 |
| Voting ensemble | 1 | 1.00 | 2.62 | 0.58 | 2 |
| Voting ensemble | 2 | 1.00 | 3.72 | 0.63 | 4 |
| Voting ensemble | 3 | 0.68 | 2.76 | 0.55 | 0 |

## Rule-layer ablation (matched recall 0.65, k=1)

Test false alerts per day and event recall of each detector with each subset of the three rules. `all-X` removes rule X from the full layer.

| Method | Rules | Target reached | Test FA/day | Test event recall | Alerts |
|---|---|---|---|---|---|
| Global z-score (baseline) | none | 0.91 | 2.56 | 0.69 | 27.2 |
| Global z-score (baseline) | all | 1.00 | 7.44 | 0.89 | 71.9 |
| Global z-score (baseline) | missing | 0.91 | 2.56 | 0.69 | 27.2 |
| Global z-score (baseline) | range | 1.00 | 6.68 | 0.89 | 65.6 |
| Global z-score (baseline) | stuck | 0.91 | 3.45 | 0.70 | 34.6 |
| Global z-score (baseline) | all-missing | 1.00 | 7.44 | 0.89 | 71.9 |
| Global z-score (baseline) | all-range | 0.91 | 3.45 | 0.70 | 34.6 |
| Global z-score (baseline) | all-stuck | 1.00 | 6.68 | 0.89 | 65.6 |
| Rolling median/MAD | none | 1.00 | 2.25 | 0.76 | 25.7 |
| Rolling median/MAD | all | 1.00 | 7.45 | 0.90 | 71.9 |
| Rolling median/MAD | missing | 1.00 | 2.25 | 0.76 | 25.7 |
| Rolling median/MAD | range | 1.00 | 6.70 | 0.90 | 65.7 |
| Rolling median/MAD | stuck | 1.00 | 3.15 | 0.76 | 33.3 |
| Rolling median/MAD | all-missing | 1.00 | 7.45 | 0.90 | 71.9 |
| Rolling median/MAD | all-range | 1.00 | 3.15 | 0.76 | 33.3 |
| Rolling median/MAD | all-stuck | 1.00 | 6.70 | 0.90 | 65.7 |
| Seasonal residual | none | 1.00 | 2.28 | 0.65 | 24.5 |
| Seasonal residual | all | 1.00 | 7.54 | 0.89 | 72.5 |
| Seasonal residual | missing | 1.00 | 2.28 | 0.65 | 24.5 |
| Seasonal residual | range | 1.00 | 7.00 | 0.90 | 68.2 |
| Seasonal residual | stuck | 1.00 | 3.11 | 0.67 | 31.5 |
| Seasonal residual | all-missing | 1.00 | 7.54 | 0.89 | 72.5 |
| Seasonal residual | all-range | 1.00 | 3.11 | 0.67 | 31.5 |
| Seasonal residual | all-stuck | 1.00 | 7.00 | 0.90 | 68.2 |
| EWMA control chart | none | 1.00 | 4.36 | 0.60 | 43.2 |
| EWMA control chart | all | 1.00 | 7.08 | 0.90 | 69.5 |
| EWMA control chart | missing | 1.00 | 4.36 | 0.60 | 43.2 |
| EWMA control chart | range | 1.00 | 6.41 | 0.90 | 64.0 |
| EWMA control chart | stuck | 1.00 | 4.96 | 0.57 | 47.5 |
| EWMA control chart | all-missing | 1.00 | 7.08 | 0.90 | 69.5 |
| EWMA control chart | all-range | 1.00 | 4.96 | 0.57 | 47.5 |
| EWMA control chart | all-stuck | 1.00 | 6.41 | 0.90 | 64.0 |
| Seasonal residual + EWMA | none | 1.00 | 1.71 | 0.66 | 20.0 |
| Seasonal residual + EWMA | all | 1.00 | 7.30 | 0.90 | 70.6 |
| Seasonal residual + EWMA | missing | 1.00 | 1.71 | 0.66 | 20.0 |
| Seasonal residual + EWMA | range | 1.00 | 6.51 | 0.90 | 64.2 |
| Seasonal residual + EWMA | stuck | 1.00 | 2.50 | 0.67 | 26.6 |
| Seasonal residual + EWMA | all-missing | 1.00 | 7.30 | 0.90 | 70.6 |
| Seasonal residual + EWMA | all-range | 1.00 | 2.50 | 0.67 | 26.6 |
| Seasonal residual + EWMA | all-stuck | 1.00 | 6.51 | 0.90 | 64.2 |
| Robust STL residual | none | 1.00 | 2.45 | 0.68 | 31.1 |
| Robust STL residual | all | 1.00 | 7.50 | 0.90 | 72.5 |
| Robust STL residual | missing | 1.00 | 2.45 | 0.68 | 31.1 |
| Robust STL residual | range | 1.00 | 6.79 | 0.90 | 66.6 |
| Robust STL residual | stuck | 1.00 | 3.31 | 0.69 | 38.2 |
| Robust STL residual | all-missing | 1.00 | 7.50 | 0.90 | 72.5 |
| Robust STL residual | all-range | 1.00 | 3.31 | 0.69 | 38.2 |
| Robust STL residual | all-stuck | 1.00 | 6.79 | 0.90 | 66.6 |
| Isolation Forest | none | 1.00 | 3.86 | 0.50 | 39.6 |
| Isolation Forest | all | 1.00 | 8.09 | 0.90 | 78.0 |
| Isolation Forest | missing | 1.00 | 3.86 | 0.50 | 39.6 |
| Isolation Forest | range | 1.00 | 7.68 | 0.90 | 74.7 |
| Isolation Forest | stuck | 1.00 | 4.45 | 0.52 | 44.7 |
| Isolation Forest | all-missing | 1.00 | 8.09 | 0.90 | 78.0 |
| Isolation Forest | all-range | 1.00 | 4.45 | 0.52 | 44.7 |
| Isolation Forest | all-stuck | 1.00 | 7.68 | 0.90 | 74.7 |
| Local outlier factor | none | 0.91 | 5.54 | 0.63 | 56.0 |
| Local outlier factor | all | 1.00 | 7.38 | 0.91 | 71.7 |
| Local outlier factor | missing | 0.91 | 5.54 | 0.63 | 56.0 |
| Local outlier factor | range | 1.00 | 6.84 | 0.91 | 67.2 |
| Local outlier factor | stuck | 0.91 | 6.00 | 0.62 | 59.5 |
| Local outlier factor | all-missing | 1.00 | 7.38 | 0.91 | 71.7 |
| Local outlier factor | all-range | 0.91 | 6.00 | 0.62 | 59.5 |
| Local outlier factor | all-stuck | 1.00 | 6.84 | 0.91 | 67.2 |
| One-class SVM | none | 1.00 | 1.98 | 0.55 | 23.2 |
| One-class SVM | all | 1.00 | 7.39 | 0.90 | 71.3 |
| One-class SVM | missing | 1.00 | 1.98 | 0.55 | 23.2 |
| One-class SVM | range | 1.00 | 6.70 | 0.90 | 65.7 |
| One-class SVM | stuck | 1.00 | 2.76 | 0.55 | 29.8 |
| One-class SVM | all-missing | 1.00 | 7.39 | 0.90 | 71.3 |
| One-class SVM | all-range | 1.00 | 2.76 | 0.55 | 29.8 |
| One-class SVM | all-stuck | 1.00 | 6.70 | 0.90 | 65.7 |
| PCA reconstruction error | none | 1.00 | 2.78 | 0.57 | 29.0 |
| PCA reconstruction error | all | 1.00 | 7.24 | 0.90 | 70.1 |
| PCA reconstruction error | missing | 1.00 | 2.78 | 0.57 | 29.0 |
| PCA reconstruction error | range | 1.00 | 6.62 | 0.90 | 65.0 |
| PCA reconstruction error | stuck | 1.00 | 3.32 | 0.58 | 33.5 |
| PCA reconstruction error | all-missing | 1.00 | 7.24 | 0.90 | 70.1 |
| PCA reconstruction error | all-range | 1.00 | 3.32 | 0.58 | 33.5 |
| PCA reconstruction error | all-stuck | 1.00 | 6.62 | 0.90 | 65.0 |
| Autoencoder | none | 1.00 | 2.43 | 0.60 | 27.3 |
| Autoencoder | all | 1.00 | 7.29 | 0.91 | 71.1 |
| Autoencoder | missing | 1.00 | 2.43 | 0.60 | 27.3 |
| Autoencoder | range | 1.00 | 6.75 | 0.91 | 66.5 |
| Autoencoder | stuck | 1.00 | 3.13 | 0.61 | 33.1 |
| Autoencoder | all-missing | 1.00 | 7.29 | 0.91 | 71.1 |
| Autoencoder | all-range | 1.00 | 3.13 | 0.61 | 33.1 |
| Autoencoder | all-stuck | 1.00 | 6.75 | 0.91 | 66.5 |
| Voting ensemble | none | 1.00 | 2.62 | 0.58 | 29.3 |
| Voting ensemble | all | 1.00 | 7.54 | 0.90 | 73.4 |
| Voting ensemble | missing | 1.00 | 2.62 | 0.58 | 29.3 |
| Voting ensemble | range | 1.00 | 7.08 | 0.90 | 69.7 |
| Voting ensemble | stuck | 1.00 | 3.32 | 0.58 | 34.7 |
| Voting ensemble | all-missing | 1.00 | 7.54 | 0.90 | 73.4 |
| Voting ensemble | all-range | 1.00 | 3.32 | 0.58 | 34.7 |
| Voting ensemble | all-stuck | 1.00 | 7.08 | 0.90 | 69.7 |

## Best method chosen on calibration windows only

Candidates must reach the target on at least 90% of units; ranking by mean calibration false alerts per day (detector alone, k=1):

One-class SVM (1.23), Global z-score (baseline) (1.62), Seasonal residual (1.80), Seasonal residual + EWMA (1.81), Robust STL residual (2.12), Autoencoder (2.19), PCA reconstruction error (2.28), Voting ensemble (2.65), Rolling median/MAD (2.66), Isolation Forest (2.70), EWMA control chart (3.61), Local outlier factor (7.12)
