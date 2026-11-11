# Semantic alignment between patient-expressed needs and physician expertise on an online healthcare platform: Implications for patient navigation and satisfaction

**Associated manuscript:** *Semantic alignment between patient-expressed needs and physician expertise on an online healthcare platform: Implications for patient navigation and satisfaction*  
**Journal:** *Patient Education and Counseling*  
**Manuscript reference:** PEC-26-2254R1 (under revision)

## Overview

This repository provides analytical code and supporting documentation associated with a retrospective cross-sectional study of patient–physician textual correspondence on **WeDoctor**, a Chinese online healthcare platform. The study investigates whether semantic similarity between patient-expressed healthcare needs and physician expertise descriptions is associated with the likelihood of a **below-five-star rating** (1–4 stars versus 5 stars).

The primary measure uses multilingual sentence-transformer embeddings and cosine similarity; **TF-IDF cosine similarity** provides a lexical comparison. The study reports hierarchical logistic regression, physician-grouped Random Forest evaluation, sensitivity analyses, and a separate clinician assessment of symptom-only alignment.

**Interpretation:** Semantic alignment is a measure of textual correspondence, not a measure of clinical suitability, treatment effectiveness, actual physician-selection behavior, or a causal effect on satisfaction.

## Study at a glance

| Item | Description |
| --- | --- |
| Platform | WeDoctor (China) |
| Review period | May 2021–April 2026 |
| Original dataset | 725,920 de-identified reviews |
| Sampling | Outcome-stratified random sampling, followed by text-length exclusions |
| Final analytical sample | 46,674 reviews |
| Physicians | 5,910 |
| Standardized departments | 147 |
| Primary outcome | Rating below five stars (1–4 vs. 5) |
| Primary alignment | Cosine similarity of `paraphrase-multilingual-MiniLM-L12-v2` sentence embeddings |
| Lexical baseline | TF-IDF cosine similarity |
| Independent validation | 120 symptom–physician pairs rated by two clinicians |

The analytical sample includes 8,886 below-five-star reviews (19.04%) and 37,788 five-star reviews (80.96%). **These proportions are not the platform's natural rating distribution** because the sample was outcome-stratified. Consequently, model-based absolute probabilities and predictive metrics should not be interpreted as population-level estimates.

## Main reported findings

- Fully adjusted logistic regression: odds ratio **0.738** per standard-deviation increase in semantic alignment (95% CI **0.708–0.769**; *p* < .001).
- Physician-grouped Random Forest: ROC-AUC **0.762** and average precision **0.537** in the held-out test set; adding semantic alignment beyond structured controls and TF-IDF increased ROC-AUC by **0.038** and average precision by **0.074**.
- Clinician validation of **symptom-only** alignment: Spearman's ρ **0.561** for communication relevance, **0.588** for specialty relevance, and **0.581** for clinical appropriateness (all *p* < .001).

These figures summarize the manuscript findings; they are **not outputs verified by executing this repository during this README update**. Clinician validation concerns symptom-only alignment and should not be treated as clinical validation of the primary measure, which also includes retrospective review narratives.

## Repository structure

The existing project README identifies the following source-code layout. Refer to the actual checked-out repository for the definitive file inventory.

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt          # If present in your checkout
├── environment.yml           # If present in your checkout
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
├── data/                     # Local, restricted; not published
└── results/                  # Local/generated outputs, if applicable
```

The source files should be consulted to establish the exact supported steps, input schemas, and outputs. **Do not assume that every analysis described in the manuscript is implemented in the publicly available code.** In particular, the availability of code for clinician validation, alternative text constructions, bootstrap intervals, calibration, and manuscript figure generation should be checked separately.

## Using the code

The earlier repository documentation specifies **Python 3.13** and identifies the following packages: `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `sentence-transformers`, and `torch`.

Create a Python environment and install dependencies using the dependency file included in your checkout, if available:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

The previously documented entry point is:

```bash
python scripts/run_analysis.py
```

It expects a locally supplied analytical input file at:

```text
data/paper4_final_regression_dataset.csv
```

**Important:** The original data and this processed analytical file are **not provided** in the public repository. The command above is retained from the prior README; its execution and coverage of the revised manuscript's analyses have **not yet been independently verified**. Researchers must examine the scripts for the exact required columns, preprocessing assumptions, and supported outputs. A processed analytical file may not by itself allow the full text-embedding pipeline to be regenerated from raw texts.

## Analysis components described in the manuscript

The revised manuscript describes the following procedures:

1. **Text preparation:** Patient-side text combines reported symptoms and retrospective review narratives; physician-side text combines department and expertise/profile descriptions. Review tags are excluded from the primary text construction.
2. **Alignment measures:** Multilingual sentence embeddings and cosine similarity, with TF-IDF cosine similarity as a lexical baseline.
3. **Association models:** Four hierarchical logistic regression specifications, with physician-clustered standard errors.
4. **Predictive evaluation:** Physician-grouped training/test splitting, cross-validation, Random Forest models, discrimination and calibration metrics, and physician-level bootstrap confidence intervals.
5. **Robustness:** Alternative rating threshold (≤3 stars), physician and year fixed effects, nonlinear specifications, and alternative patient-text constructions.
6. **Independent clinician validation:** Two clinicians assess 120 symptom–physician pairs for communication relevance, specialty relevance, and clinical appropriateness.

This list describes **the manuscript's methods**, not a verified inventory of executable repository functions. The scripts and documentation are intended to support methodological inspection and reuse with appropriately authorized data; direct replication of every published result is not established here.

## Data availability and restrictions

**No original WeDoctor patient reviews or physician records are distributed in this repository.** No real de-identified numerical sample or synthetic demonstration dataset is included.

The original WeDoctor dataset cannot be publicly released because of **platform licensing and privacy restrictions**. Requests for access to de-identified research data may be directed to the corresponding author and will be considered subject to applicable permissions and data-protection requirements; access is **not guaranteed**.

The code and documentation can help researchers understand and implement the analytical workflow using appropriately authorized data. **The published empirical estimates cannot be independently replicated from this repository alone without the underlying data and any additional required inputs.**

## Study limitations relevant to reuse

- The primary patient-side measure includes **post-consultation review narratives**; it is not a verified prospective physician-selection measure.
- Historical physician profiles and the timing of symptom entry could not be confirmed.
- The five-star ceiling effect and outcome-stratified sampling limit interpretation of rating categories, sample proportions, predicted probabilities, and generalizability.
- Clinical appropriateness cannot be inferred from semantic similarity alone. Clinician assessments provide preliminary validation of **symptom-only** alignment, not proof of triage accuracy.
- Observational associations should not be interpreted causally.

## Citation

If you use the repository, please cite the associated manuscript once a final bibliographic citation is available:

> Asante, I. O., Chen, X., and Jiang, Y. *Semantic alignment between patient-expressed needs and physician expertise on an online healthcare platform: Implications for patient navigation and satisfaction*. Manuscript under revision at *Patient Education and Counseling*.

Please confirm the final author list and author order against the submitted title page before using this citation.

## Contact

**Isaac Owusu Asante** — corresponding author. Please use the contact details provided in the manuscript or the repository's GitHub contact options.

