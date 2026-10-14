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
  labelled synthetic anomalies are injected by a seeded generator (`anomaly_monitoring/inject.py`).
* The interactive demo (`docs/index.html`) embeds a small number of these **modified**
  excerpts (rounded hourly values with injected anomalies, 60 days per sample series) as
  precomputed JSON so that it works offline. Those excerpts are an adaptation of the dataset
  above and are credited to its authors under CC BY 4.0. The `LICENSE` of this repository
  (CC BY-NC-ND 4.0) applies to the original code and text, not to that underlying data.
* The dataset is not endorsed by, and this project is not affiliated with, its authors.

## Python dependencies (installed from PyPI, not redistributed)

| Package | License |
|---|---|
| NumPy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| pytest | MIT |

## Methods referenced

Isolation Forest (Liu, Ting and Zhou, 2008), EWMA control charts (Roberts, 1959), robust
statistics based on the median absolute deviation, and point-adjusted anomaly scoring
(Xu et al., 2018) are published techniques. They are re-implemented here from their
descriptions; no third-party source code is copied (Isolation Forest uses scikit-learn).
