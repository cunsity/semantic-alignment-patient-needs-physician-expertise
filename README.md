# Patient–Physician Fit in Online Healthcare Platforms

## Overview

This repository contains the complete reproducible analytical pipeline accompanying the manuscript:

> **Patient–Physician Fit in Online Healthcare Platforms: Measuring Symptom–Specialty Alignment with Large Language Model Embeddings**

The repository reproduces the statistical analyses reported in the manuscript, including semantic alignment computation, hierarchical logistic regression, diagnostic analyses, and Random Forest robustness analyses.

---

## Repository Structure

```text
github_repository/
│
├── README.md
├── LICENSE
├── .gitignore
├── data/                 # Local only (not included in repository)
├── notebooks/
├── scripts/
│   └── run_analysis.py
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── semantic_alignment.py
│   ├── models.py
│   ├── diagnostics.py
│   └── robustness.py
└── results/              # Generated automatically
```

---

## Reproducing the Study

Place the processed analytical dataset in:

```text
data/paper4_final_regression_dataset.csv
```

Then run:

```bash
python scripts/run_analysis.py
```

The script reproduces:

- Feature engineering
- Semantic alignment
- Hierarchical logistic regression
- Diagnostic analyses
- Random Forest robustness analyses
- Manuscript tables

---

## Data Availability

The original data were obtained from an online healthcare platform and cannot be redistributed due to licensing and privacy restrictions.

This repository therefore contains the complete analytical pipeline but **does not include the original dataset**.

Researchers with legitimate access to comparable data can reproduce the complete analysis using the provided scripts.

---

## Software

This repository was developed using **Python 3.13**.

Major packages include:

- pandas
- numpy
- scipy
- statsmodels
- scikit-learn
- sentence-transformers
- torch

---

## Citation

If you use this repository, please cite the associated manuscript.

---

## Author

**Isaac Owusu Asante**
