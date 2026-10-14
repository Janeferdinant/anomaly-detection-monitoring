# Evaluation run 2026-10-14

Status label: `Verified (run 2026-10-14, seeds 0, 1, 2, 3, 4)`. Numbers below were produced by `scripts/run_evaluation.py` on public data with injected, labelled anomalies. They are new numbers from an independent re-implementation. They are **not** the result reported for the earlier deployment on the CV, and they do not reproduce or confirm it.

## Development notes (full disclosure)

* Method development (detector parameters, calibration protocol, number of injected incidents per window) used development seeds 1000-1002. Seeds 0-4 were not used for tuning.
* The protocol and the primary comparison were fixed before the final runs; every table below is reported whatever its outcome.
* A first complete run on seeds 0-4 showed that stuck-value incidents were rarely caught by the rule layer: the stuck-run length was fitted as a high quantile of run lengths, which the injected stuck incidents in the calibration window inflated. The rule layer is shared by all detectors. It was changed to the median run length plus 2 (floor 4, needs at least 10 runs) and the whole evaluation was re-run. This log is from the re-run; the first run was discarded.

## Settings

* Command: `python scripts/run_evaluation.py --seeds 0 1 2 3 4`
* Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1
* Runtime: 7.6 minutes on CPU (AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, Windows)
* Data: UCI Individual Household Electric Power Consumption (CC BY 4.0), zip sha256 `9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff`
* Series: 12 windows of 120 days of hourly data (4 channels x 3 gap-free windows: global_active_power, voltage, global_intensity, global_reactive_power). Each window is split 60/60 days into a calibration window and a test window.
* Seeds (injection and model seeds): 0, 1, 2, 3, 4; units evaluated: 60 (series x seed)
* Injected incidents per window half: 3 each of spike, level shift, drift, stuck value and missing burst (15 per half).
* Alert-suppression rule: persistence of k consecutive flagged points, k in 1, 2, 3.
* Event recall = share of injected incidents with at least one alert inside them. False alert = an alert segment that overlaps no incident. FA/day = false alerts per day on the test window.
* Thresholds are chosen on the calibration window only (highest threshold whose calibration event recall >= target); the test window is never used for choosing them.
* Matched-recall comparisons use only the (series, seed) units where both the method and the baseline reached the target recall on the calibration window; the `Matched units` column shows how many that was.
* Percentage columns are the change in false alerts per day (negative = fewer false alerts than the global z-score). CIs are 95% percentile bootstrap intervals over series (10,000 resamples, seed 0). The per-series mean is unstable when the baseline has almost no false alerts; the pooled change is the more reliable column.

## Primary comparison: detector + rule layer, k=1, calibration recall target 0.80

| Target recall | Detector | Matched units | Test FA/day (method vs z-score) | Mean per-series change in FA/day, 95% CI | Pooled change, 95% CI | Series better / worse | Test recall (method vs z-score) |
|---|---|---|---|---|---|---|---|
| 0.80 | Rolling median/MAD | 60 / 60 | 0.20 vs 0.05 | +2499% [+671%, +4628%] | +310% [+112%, +1047%] | 0 / 8 | 0.80 vs 0.78 |
| 0.80 | Seasonal residual | 60 / 60 | 0.09 vs 0.05 | +778% [+97%, +1594%] | +78% [-44%, +442%] | 3 / 5 | 0.77 vs 0.78 |
| 0.80 | EWMA control chart | 60 / 60 | 0.05 vs 0.05 | +727% [+3%, +1676%] | -5% [-61%, +195%] | 4 / 4 | 0.75 vs 0.78 |
| 0.80 | Isolation Forest | 60 / 60 | 0.11 vs 0.05 | +1244% [+197%, +2509%] | +120% [-2%, +560%] | 3 / 4 | 0.79 vs 0.78 |
| 0.80 | Seasonal residual + EWMA | 60 / 60 | 0.07 vs 0.05 | +697% [+31%, +1551%] | +44% [-49%, +323%] | 3 / 5 | 0.77 vs 0.78 |
| 0.80 | Calibration-selected pipeline | 60 / 60 | 0.05 vs 0.05 | +716% [+28%, +1598%] | +5% [-53%, +228%] | 4 / 4 | 0.72 vs 0.78 |

## Detector alone (no rule layer), k=1

| Target recall | Detector | Matched units | Test FA/day (method vs z-score) | Mean per-series change in FA/day, 95% CI | Pooled change, 95% CI | Series better / worse | Test recall (method vs z-score) |
|---|---|---|---|---|---|---|---|
| 0.50 | Rolling median/MAD | 60 / 60 | 0.29 vs 0.09 | +934% [+451%, +1441%] | +205% [+72%, +604%] | 2 / 10 | 0.50 vs 0.46 |
| 0.50 | Seasonal residual | 60 / 60 | 0.17 vs 0.09 | +334% [+102%, +583%] | +86% [-1%, +336%] | 5 / 7 | 0.48 vs 0.46 |
| 0.50 | EWMA control chart | 60 / 60 | 0.13 vs 0.09 | +409% [+154%, +715%] | +42% [-24%, +250%] | 3 / 9 | 0.52 vs 0.46 |
| 0.50 | Isolation Forest | 60 / 60 | 0.18 vs 0.09 | +584% [+235%, +1009%] | +96% [-7%, +407%] | 4 / 8 | 0.50 vs 0.46 |
| 0.50 | Seasonal residual + EWMA | 60 / 60 | 0.14 vs 0.09 | +272% [+69%, +497%] | +51% [-15%, +241%] | 5 / 7 | 0.48 vs 0.46 |
| 0.65 | Rolling median/MAD | 49 / 60 | 1.26 vs 1.42 | +9% [-24%, +45%] | -11% [-31%, +17%] | 8 / 4 | 0.60 vs 0.64 |
| 0.65 | Seasonal residual | 50 / 60 | 0.59 vs 1.44 | -51% [-71%, -19%] | -59% [-70%, -40%] | 11 / 1 | 0.66 vs 0.64 |
| 0.65 | EWMA control chart | 50 / 60 | 0.30 vs 1.44 | -76% [-84%, -64%] | -79% [-84%, -72%] | 12 / 0 | 0.61 vs 0.64 |
| 0.65 | Isolation Forest | 50 / 60 | 0.49 vs 1.44 | -57% [-74%, -34%] | -66% [-75%, -51%] | 11 / 1 | 0.66 vs 0.64 |
| 0.65 | Seasonal residual + EWMA | 50 / 60 | 0.49 vs 1.44 | -59% [-75%, -37%] | -66% [-75%, -52%] | 11 / 1 | 0.65 vs 0.64 |
| 0.80 | Rolling median/MAD | 5 / 60 | 3.41 vs 2.99 | +18% [-19%, +53%] | +14% [-18%, +51%] | 1 / 3 | 0.64 vs 0.64 |
| 0.80 | Seasonal residual | 6 / 60 | 1.43 vs 2.98 | -57% [-78%, -37%] | -52% [-79%, -38%] | 4 / 0 | 0.73 vs 0.63 |
| 0.80 | EWMA control chart | 5 / 60 | 0.57 vs 2.96 | -84% [-95%, -74%] | -81% [-95%, -73%] | 4 / 0 | 0.65 vs 0.64 |
| 0.80 | Isolation Forest | 6 / 60 | 0.87 vs 2.98 | -69% [-77%, -58%] | -71% [-77%, -61%] | 4 / 0 | 0.72 vs 0.63 |
| 0.80 | Seasonal residual + EWMA | 6 / 60 | 1.13 vs 2.98 | -65% [-79%, -51%] | -62% [-80%, -48%] | 4 / 0 | 0.73 vs 0.63 |
| 0.90 | Rolling median/MAD | 0 / 60 | n/a | n/a | n/a | n/a | n/a |
| 0.90 | Seasonal residual | 0 / 60 | n/a | n/a | n/a | n/a | n/a |
| 0.90 | EWMA control chart | 0 / 60 | n/a | n/a | n/a | n/a | n/a |
| 0.90 | Isolation Forest | 0 / 60 | n/a | n/a | n/a | n/a | n/a |
| 0.90 | Seasonal residual + EWMA | 0 / 60 | n/a | n/a | n/a | n/a | n/a |

## Detector + rule layer, k=1, all recall targets

| Target recall | Detector | Matched units | Test FA/day (method vs z-score) | Mean per-series change in FA/day, 95% CI | Pooled change, 95% CI | Series better / worse | Test recall (method vs z-score) |
|---|---|---|---|---|---|---|---|
| 0.50 | Rolling median/MAD | 60 / 60 | 0.07 vs 0.02 | +238% [-46%, +621%] | +360% [+58%, +1567%] | 2 / 3 | 0.63 vs 0.56 |
| 0.50 | Seasonal residual | 60 / 60 | 0.03 vs 0.02 | +119% [-62%, +432%] | +62% [-64%, +406%] | 3 / 2 | 0.59 vs 0.56 |
| 0.50 | EWMA control chart | 60 / 60 | 0.01 vs 0.02 | -35% [-99%, +84%] | -48% [-95%, +77%] | 4 / 1 | 0.54 vs 0.56 |
| 0.50 | Isolation Forest | 60 / 60 | 0.04 vs 0.02 | +77% [-83%, +329%] | +136% [-19%, +725%] | 3 / 2 | 0.57 vs 0.56 |
| 0.50 | Seasonal residual + EWMA | 60 / 60 | 0.02 vs 0.02 | +64% [-80%, +312%] | +17% [-82%, +288%] | 3 / 1 | 0.58 vs 0.56 |
| 0.65 | Rolling median/MAD | 60 / 60 | 0.12 vs 0.03 | +241% [+6%, +595%] | +326% [+93%, +1213%] | 0 / 5 | 0.72 vs 0.67 |
| 0.65 | Seasonal residual | 60 / 60 | 0.05 vs 0.03 | +101% [-63%, +400%] | +61% [-56%, +388%] | 4 / 1 | 0.68 vs 0.67 |
| 0.65 | EWMA control chart | 60 / 60 | 0.02 vs 0.03 | -45% [-91%, +29%] | -40% [-80%, +79%] | 4 / 1 | 0.66 vs 0.67 |
| 0.65 | Isolation Forest | 60 / 60 | 0.06 vs 0.03 | +99% [-66%, +399%] | +128% [-19%, +590%] | 4 / 1 | 0.67 vs 0.67 |
| 0.65 | Seasonal residual + EWMA | 60 / 60 | 0.04 vs 0.03 | +62% [-66%, +298%] | +33% [-60%, +295%] | 4 / 1 | 0.69 vs 0.67 |
| 0.80 | Rolling median/MAD | 60 / 60 | 0.20 vs 0.05 | +2499% [+671%, +4628%] | +310% [+112%, +1047%] | 0 / 8 | 0.80 vs 0.78 |
| 0.80 | Seasonal residual | 60 / 60 | 0.09 vs 0.05 | +778% [+97%, +1594%] | +78% [-44%, +442%] | 3 / 5 | 0.77 vs 0.78 |
| 0.80 | EWMA control chart | 60 / 60 | 0.05 vs 0.05 | +727% [+3%, +1676%] | -5% [-61%, +195%] | 4 / 4 | 0.75 vs 0.78 |
| 0.80 | Isolation Forest | 60 / 60 | 0.11 vs 0.05 | +1244% [+197%, +2509%] | +120% [-2%, +560%] | 3 / 4 | 0.79 vs 0.78 |
| 0.80 | Seasonal residual + EWMA | 60 / 60 | 0.07 vs 0.05 | +697% [+31%, +1551%] | +44% [-49%, +323%] | 3 / 5 | 0.77 vs 0.78 |
| 0.90 | Rolling median/MAD | 60 / 60 | 0.29 vs 0.09 | +948% [+466%, +1452%] | +217% [+87%, +615%] | 0 / 12 | 0.90 vs 0.86 |
| 0.90 | Seasonal residual | 60 / 60 | 0.18 vs 0.09 | +357% [+117%, +617%] | +98% [+9%, +355%] | 5 / 7 | 0.87 vs 0.86 |
| 0.90 | EWMA control chart | 60 / 60 | 0.17 vs 0.09 | +603% [+297%, +931%] | +84% [+9%, +338%] | 3 / 9 | 0.88 vs 0.86 |
| 0.90 | Isolation Forest | 60 / 60 | 0.19 vs 0.09 | +610% [+253%, +1035%] | +102% [+1%, +416%] | 4 / 8 | 0.87 vs 0.86 |
| 0.90 | Seasonal residual + EWMA | 60 / 60 | 0.15 vs 0.09 | +285% [+78%, +514%] | +57% [-10%, +248%] | 5 / 7 | 0.86 vs 0.86 |

## Calibration-selected pipeline (detector and k chosen on the calibration window), rule layer on

| Target recall | Detector | Matched units | Test FA/day (method vs z-score) | Mean per-series change in FA/day, 95% CI | Pooled change, 95% CI | Series better / worse | Test recall (method vs z-score) |
|---|---|---|---|---|---|---|---|
| 0.50 | Calibration-selected pipeline | 60 / 60 | 0.06 vs 0.02 | +87% [-82%, +388%] | +243% [-18%, +1319%] | 3 / 1 | 0.61 vs 0.56 |
| 0.65 | Calibration-selected pipeline | 60 / 60 | 0.05 vs 0.03 | -25% [-83%, +78%] | +61% [-64%, +543%] | 4 / 1 | 0.67 vs 0.67 |
| 0.80 | Calibration-selected pipeline | 60 / 60 | 0.05 vs 0.05 | +716% [+28%, +1598%] | +5% [-53%, +228%] | 4 / 4 | 0.72 vs 0.78 |
| 0.90 | Calibration-selected pipeline | 60 / 60 | 0.12 vs 0.09 | +304% [+87%, +545%] | +25% [-35%, +207%] | 4 / 7 | 0.83 vs 0.86 |

* Target 0.50, configurations chosen: rolling_mad+k1 x36, seasonal+k1 x15, ewma+k1 x5, iforest+k1 x1, iforest+k2 x1, ewma+k2 x1, seasonal+k3 x1
* Target 0.65, configurations chosen: rolling_mad+k1 x22, seasonal+k1 x17, ewma+k1 x12, ewma+k3 x4, rolling_mad+k3 x1, seasonal+k2 x1, rolling_mad+k2 x1, combo+k2 x1, ewma+k2 x1
* Target 0.80, configurations chosen: rolling_mad+k1 x16, seasonal+k1 x15, ewma+k1 x9, ewma+k3 x6, rolling_mad+k3 x4, combo+k2 x3, seasonal+k2 x3, seasonal+k3 x2, combo+k1 x2
* Target 0.90, configurations chosen: seasonal+k1 x16, rolling_mad+k1 x11, ewma+k1 x10, seasonal+k2 x7, combo+k2 x5, iforest+k1 x4, combo+k1 x4, iforest+k2 x3

## Effect of the alert-suppression (persistence) rule at a fixed threshold

The threshold is the one calibrated for k=1 (target 0.80); only k changes, so this shows the raw trade-off. Detector alone, all units.

| Detector | k | Test event recall | False alerts/day | Change in false alerts vs k=1 |
|---|---|---|---|---|
| Global z-score (baseline) | 1 | 0.63 | 1.65 | - |
| Global z-score (baseline) | 2 | 0.53 | 0.85 | -49% |
| Global z-score (baseline) | 3 | 0.45 | 0.50 | -70% |
| Rolling median/MAD | 1 | 0.62 | 1.94 | - |
| Rolling median/MAD | 2 | 0.53 | 0.99 | -49% |
| Rolling median/MAD | 3 | 0.43 | 0.58 | -70% |
| Seasonal residual | 1 | 0.76 | 2.02 | - |
| Seasonal residual | 2 | 0.64 | 0.86 | -57% |
| Seasonal residual | 3 | 0.54 | 0.44 | -78% |
| EWMA control chart | 1 | 0.73 | 0.84 | - |
| EWMA control chart | 2 | 0.64 | 0.62 | -26% |
| EWMA control chart | 3 | 0.59 | 0.50 | -40% |
| Isolation Forest | 1 | 0.78 | 1.21 | - |
| Isolation Forest | 2 | 0.67 | 0.68 | -44% |
| Isolation Forest | 3 | 0.58 | 0.51 | -58% |
| Seasonal residual + EWMA | 1 | 0.75 | 1.49 | - |
| Seasonal residual + EWMA | 2 | 0.64 | 0.63 | -58% |
| Seasonal residual + EWMA | 3 | 0.55 | 0.39 | -74% |

## Persistence at matched recall (threshold re-calibrated for each k)

Detector alone, target 0.65. Share of units where the target was reachable and mean false alerts per day on those units.

| Detector | k | Target reached on calibration | Test FA/day (reached units) | Test event recall (reached units) |
|---|---|---|---|---|
| Global z-score (baseline) | 1 | 0.83 | 1.48 | 0.64 |
| Global z-score (baseline) | 2 | 0.42 | 1.04 | 0.51 |
| Global z-score (baseline) | 3 | 0.07 | 0.76 | 0.53 |
| Rolling median/MAD | 1 | 0.93 | 1.37 | 0.61 |
| Rolling median/MAD | 2 | 0.58 | 1.06 | 0.51 |
| Rolling median/MAD | 3 | 0.22 | 0.84 | 0.49 |
| Seasonal residual | 1 | 1.00 | 0.71 | 0.67 |
| Seasonal residual | 2 | 0.82 | 0.78 | 0.61 |
| Seasonal residual | 3 | 0.48 | 0.71 | 0.60 |
| EWMA control chart | 1 | 1.00 | 0.36 | 0.63 |
| EWMA control chart | 2 | 0.73 | 0.54 | 0.63 |
| EWMA control chart | 3 | 0.42 | 0.57 | 0.62 |
| Isolation Forest | 1 | 1.00 | 0.53 | 0.67 |
| Isolation Forest | 2 | 0.85 | 0.65 | 0.62 |
| Isolation Forest | 3 | 0.45 | 0.68 | 0.63 |
| Seasonal residual + EWMA | 1 | 1.00 | 0.59 | 0.66 |
| Seasonal residual + EWMA | 2 | 0.67 | 0.51 | 0.59 |
| Seasonal residual + EWMA | 3 | 0.45 | 0.63 | 0.59 |

## Ranking-quality check at matched test recall (threshold chosen on the test window; not deployable)

Uses test labels to pick the threshold, so it measures how well each score separates incidents from normal points, independent of threshold transfer. Only units where every detector could reach recall r are compared.

Detector alone:

| Test recall r | Detector | Common units | FA/day (method vs z-score) | Mean per-series change, 95% CI | Pooled change, 95% CI |
|---|---|---|---|---|---|
| 0.50 | Rolling median/MAD | 60 / 60 | 0.28 vs 0.14 | +348% [+79%, +707%] | +106% [+58%, +225%] |
| 0.50 | Seasonal residual | 60 / 60 | 0.11 vs 0.14 | +1% [-28%, +32%] | -16% [-33%, +15%] |
| 0.50 | EWMA control chart | 60 / 60 | 0.09 vs 0.14 | +56% [-29%, +156%] | -35% [-57%, +17%] |
| 0.50 | Isolation Forest | 60 / 60 | 0.14 vs 0.14 | +107% [+2%, +229%] | -1% [-27%, +57%] |
| 0.50 | Seasonal residual + EWMA | 60 / 60 | 0.10 vs 0.14 | -11% [-35%, +15%] | -26% [-41%, -1%] |
| 0.70 | Rolling median/MAD | 9 / 60 | 2.15 vs 2.09 | +18% [-14%, +50%] | +3% [-20%, +40%] |
| 0.70 | Seasonal residual | 9 / 60 | 0.59 vs 2.09 | -65% [-87%, -43%] | -72% [-85%, -54%] |
| 0.70 | EWMA control chart | 9 / 60 | 0.43 vs 2.09 | -75% [-85%, -64%] | -80% [-86%, -69%] |
| 0.70 | Isolation Forest | 9 / 60 | 0.53 vs 2.09 | -71% [-85%, -55%] | -75% [-85%, -63%] |
| 0.70 | Seasonal residual + EWMA | 9 / 60 | 0.48 vs 2.09 | -74% [-88%, -59%] | -77% [-88%, -65%] |

Detector + rule layer:

| Test recall r | Detector | Common units | FA/day (method vs z-score) | Mean per-series change, 95% CI | Pooled change, 95% CI |
|---|---|---|---|---|---|
| 0.50 | Rolling median/MAD | 60 / 60 | 0.02 vs 0.01 | +13% [-33%, +64%] | +227% [+72%, +2033%] |
| 0.50 | Seasonal residual | 60 / 60 | 0.01 vs 0.01 | +20% [-0%, +33%] | +85% [+17%, +700%] |
| 0.50 | EWMA control chart | 60 / 60 | 0.00 vs 0.01 | -71% [-100%, -45%] | -50% [-71%, +33%] |
| 0.50 | Isolation Forest | 60 / 60 | 0.03 vs 0.01 | +39% [+8%, +100%] | +288% [+95%, +2900%] |
| 0.50 | Seasonal residual + EWMA | 60 / 60 | 0.01 vs 0.01 | +18% [-17%, +36%] | +69% [+5%, +600%] |
| 0.70 | Rolling median/MAD | 60 / 60 | 0.12 vs 0.02 | +212% [+71%, +325%] | +419% [+256%, +1087%] |
| 0.70 | Seasonal residual | 60 / 60 | 0.04 vs 0.02 | +35% [-26%, +82%] | +69% [+44%, +144%] |
| 0.70 | EWMA control chart | 60 / 60 | 0.02 vs 0.02 | -35% [-66%, -4%] | -2% [-27%, +46%] |
| 0.70 | Isolation Forest | 60 / 60 | 0.05 vs 0.02 | +239% [+13%, +668%] | +136% [+58%, +512%] |
| 0.70 | Seasonal residual + EWMA | 60 / 60 | 0.03 vs 0.02 | +9% [-43%, +59%] | +36% [+7%, +121%] |
| 0.90 | Rolling median/MAD | 60 / 60 | 0.28 vs 0.13 | +342% [+81%, +686%] | +109% [+62%, +226%] |
| 0.90 | Seasonal residual | 60 / 60 | 0.13 vs 0.13 | +21% [-20%, +69%] | -3% [-25%, +33%] |
| 0.90 | EWMA control chart | 60 / 60 | 0.20 vs 0.13 | +162% [+53%, +284%] | +45% [+1%, +155%] |
| 0.90 | Isolation Forest | 60 / 60 | 0.17 vs 0.13 | +162% [+27%, +315%] | +26% [-6%, +97%] |
| 0.90 | Seasonal residual + EWMA | 60 / 60 | 0.12 vs 0.13 | +11% [-26%, +53%] | -13% [-32%, +19%] |

## Which incident types are caught (test window, k=1, target 0.80)

Detector alone:

| Detector | spike | level_shift | drift | stuck | missing |
|---|---|---|---|---|---|
| Seasonal residual + EWMA | 0.99 | 1.00 | 1.00 | 0.78 | 0.00 |
| EWMA control chart | 0.83 | 1.00 | 1.00 | 0.83 | 0.00 |
| Isolation Forest | 0.99 | 1.00 | 1.00 | 0.88 | 0.00 |
| Rolling median/MAD | 0.97 | 0.96 | 0.98 | 0.18 | 0.00 |
| Seasonal residual | 0.99 | 1.00 | 0.99 | 0.82 | 0.00 |
| Global z-score (baseline) | 0.98 | 0.93 | 0.97 | 0.25 | 0.00 |

Detector + rule layer:

| Detector | spike | level_shift | drift | stuck | missing |
|---|---|---|---|---|---|
| Seasonal residual + EWMA | 0.65 | 0.59 | 0.58 | 1.00 | 1.00 |
| EWMA control chart | 0.21 | 0.76 | 0.79 | 1.00 | 1.00 |
| Isolation Forest | 0.74 | 0.66 | 0.58 | 1.00 | 1.00 |
| Rolling median/MAD | 0.74 | 0.62 | 0.64 | 1.00 | 1.00 |
| Seasonal residual | 0.68 | 0.57 | 0.60 | 1.00 | 1.00 |
| Global z-score (baseline) | 0.65 | 0.50 | 0.73 | 1.00 | 1.00 |

## Point-adjusted vs event-level vs raw point scoring (same alerts, detector + rule layer, k=1, target 0.80)

| Detector | PA precision | PA recall | PA F1 | Raw point precision | Raw point recall | Point FPR | Event precision | Event recall | False alerts/day |
|---|---|---|---|---|---|---|---|---|---|
| Seasonal residual + EWMA | 0.96 | 0.71 | 0.79 | 0.94 | 0.41 | 0.0103 | 0.85 | 0.77 | 0.07 |
| EWMA control chart | 0.94 | 0.84 | 0.87 | 0.92 | 0.51 | 0.0148 | 0.87 | 0.75 | 0.05 |
| Isolation Forest | 0.92 | 0.71 | 0.78 | 0.88 | 0.40 | 0.0172 | 0.77 | 0.79 | 0.11 |
| Rolling median/MAD | 0.91 | 0.74 | 0.80 | 0.87 | 0.43 | 0.0186 | 0.65 | 0.80 | 0.20 |
| Seasonal residual | 0.97 | 0.71 | 0.79 | 0.94 | 0.36 | 0.0077 | 0.85 | 0.77 | 0.09 |
| Global z-score (baseline) | 0.98 | 0.76 | 0.84 | 0.97 | 0.32 | 0.0034 | 0.90 | 0.78 | 0.05 |

Point-adjusted (PA) scoring credits a whole incident when one point is flagged, so PA recall is always at least the raw point recall and PA precision at least the raw point precision.
