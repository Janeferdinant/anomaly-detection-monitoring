# Third-party notices

The original work in this repository (code, tests, documentation and the demo page) is
(c) Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0 (see `LICENSE`). The components below
are **not** covered by that license and keep their own terms.

## Dataset

### Individual Household Electric Power Consumption

| | |
|---|---|
| Authors | Georges Hebrail and Alice Berard |
| Source | UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption> |
| DOI | <https://doi.org/10.24432/C58K54> |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/> |
| Citation | Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C58K54> |

How it is used here:

* The original file is **downloaded at run time** by `scripts/download_data.py` into
  `data/raw/` (git-ignored) and verified against a SHA-256 checksum. The raw file is never
  committed to this repository.
* **Changes made to the data** (CC BY 4.0 requires saying so): minute readings are averaged
  to hourly means, four channels are kept (global active power, global reactive power,
  voltage, global intensity), 120-day windows without any missing hour are cut out, and
  labelled synthetic anomalies are injected by a seeded generator (`src/anomaly_monitoring/inject.py`).
* The interactive demo (`docs/index.html`) embeds a small number of these **modified**
  excerpts (rounded hourly values with injected anomalies, 60 days per sample series) as
  precomputed JSON so that it works offline. Those excerpts are an adaptation of the dataset
  above and are credited to its authors under CC BY 4.0. The `LICENSE` of this repository
  (CC BY-NC-ND 4.0) applies to the original code and text, not to that underlying data.
* The dataset is not endorsed by, and this project is not affiliated with, its authors.

### Server Machine Dataset (SMD)

| | |
|---|---|
| Authors | Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, Dan Pei (NetManAIOps, Tsinghua University) |
| Source | <https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset>, pinned to commit `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4` |
| License | MIT (the `ServerMachineDataset/LICENSE` file, copyright 2021 NetManAIOps-SMD, reproduced below) |
| Citation | Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W. & Pei, D. (2019). Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network. *KDD 2019*, 2828-2837. <https://doi.org/10.1145/3292500.3330672> |

28 server machines, 38 metrics each, one reading per minute, five weeks of unlabelled training data and five weeks of test data whose anomaly intervals were labelled by domain experts.

How it is used here:

* **Downloaded at run time** by `scripts/download_smd.py` into `data/raw/smd/` (git-ignored) from the pinned commit, and verified against the SHA-256 checksums in `src/anomaly_monitoring/smd_manifest.json`. Raw files are never committed.
* **Changes made to the data:** one-minute readings are averaged to five-minute bins (a bin is labelled anomalous if any of its minutes is), metrics that are constant in the training data are dropped, and the series are cut into fit, calibration and test windows.
* The interactive demo (`docs/index.html`) embeds rounded five-minute excerpts of a few machines (one metric each, the test window, plus precomputed detector scores) as JSON. Those excerpts are a modified adaptation of the dataset above and are credited to its authors.
* This project is not endorsed by, and not affiliated with, the dataset authors.

MIT License text of the dataset:

```
MIT License

Copyright (c) 2021 NetManAIOps-SMD

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Benchmarks checked and not used

| Dataset | License checked (2026-10-14) | Why not used |
|---|---|---|
| Numenta Anomaly Benchmark (NAB), <https://github.com/numenta/NAB> | `LICENSE.txt` is currently the MIT license (an earlier check found AGPL-3.0, so the terms changed) | Scope: SMD already gives 28 labelled multi-metric series; NAB is listed under future scope |
| SKAB, <https://github.com/waico/SKAB> | GPL-3.0 | Copyleft licence, and its short lab runs have no daily cycle for the seasonal methods; it could only be downloaded at run time |

## Python dependencies (installed from PyPI, not redistributed)

| Package | License |
|---|---|
| NumPy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| SciPy | BSD-3-Clause |
| statsmodels (robust STL) | BSD-3-Clause |
| pytest, ruff, mypy | MIT |

## Methods referenced

Isolation Forest (Liu, Ting and Zhou, 2008), local outlier factor (Breunig et al., 2000),
one-class SVM (Scholkopf et al., 2001), PCA reconstruction error, autoencoders, STL decomposition
(Cleveland et al., 1990), EWMA control charts (Roberts, 1959), robust statistics based on the
median absolute deviation, and point-adjusted anomaly scoring (Xu et al., 2018) are published techniques. They are re-implemented here from their
descriptions; no third-party source code is copied (the learned detectors use scikit-learn, STL uses statsmodels).
