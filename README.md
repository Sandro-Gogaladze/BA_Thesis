<img src="https://www.biubiu.ge/storage/images/modules/news/28/56.jpg" alt="ISET Banner" width="300" />

---

# Bachelor Thesis: Income Estimation Models Analysis

This repository contains the full analytical workflow and documentation for my Bachelor's Thesis.

**Title:** **"Income Estimation Models Analysis: Balancing Accuracy with Regulatory Requirements in the Georgian Banking Sector"**

**Author:** Sandro Gogaladze

**Supervisor:** Shota Natenadze

**Institution:** International School of Economics at Tbilisi State University (ISET)

**Year:** 2025

---

## Thesis Overview

Accurately assessing a borrower's creditworthiness is a core challenge for financial institutions, particularly in Georgia where many individuals operate within informal economic structures. Traditional income verification methods—such as document reviews and employer references—are often time-consuming, costly, and ineffective for borrowers with unstable or unrecorded earnings. This creates barriers to credit access and introduces inefficiencies in lending operations.

My thesis explores how Georgian banks can use automated income estimation models based on machine learning to address these issues. With the National Bank of Georgia (NBG) now permitting the use of data-driven techniques for income verification, this work evaluates model architectures that balance predictive accuracy with regulatory compliance. It compares the performance of several modeling strategies, offering practical insights for implementing regulation-aware income estimation in Georgia's financial sector.

The project evaluates and compares:

* A baseline XGBoost regression model
* Post-hoc quantile regression calibration
* Pre-hoc conservative objective functions:

  * Huber + Threshold penalty
  * Segment-Aware Huber + Threshold penalty

The study focuses on how machine learning models can balance accuracy with regulatory compliance, aligned with National Bank of Georgia (NBG) guidelines.

---

## Repository Structure

```
BA_THESIS/
├── data/
│   ├── raw/
│   │   └── income_data.xlsx                 # Synthetic dataset provided by BOG
│   └── processed/ (gitignored)
│       ├── train.csv                        # Preprocessed training data (generated)
│       └── test.csv                         # Preprocessed test data (generated)
├── docs/
│   ├── Bachelor Thesis - Sandro Gogaladze.docx  # Main thesis document
│   └── additional/                          # Supporting documents and references
├── IncomeEstimation/                        # Interactive codebase (see README)
│   ├── src/                                 # Core source code modules
│   ├── baseline/
│   │   └── xgboost/                         # Baseline XGBoost model
│   ├── posthoc/
│   │   └── quantile/                        # Post-hoc quantile calibration
│   ├── prehoc/                              # Pre-hoc regulation-aware models
│   │   ├── huber_threshold/                 # Huber loss + threshold penalty
│   │   └── segment_aware/                   # Segment-aware Huber + threshold
│   └── README.md                           # Detailed implementation guide
├── Notebooks/
│   └── notebook.ipynb                      # Complete research notebook
├── pyproject.toml                          # Project configuration and dependencies
├── Makefile                                # Main project workflow automation
├── .gitignore
└── README.md                               # This file
```

---

## How to Run the Analysis

This project can be executed in two ways, both producing identical results for consistency and reproducibility:

### Option 1: Interactive Notebook (For Visual Analysis)

The complete research analysis is available in the Jupyter `notebook.ipynb`.

This `notebook.ipynb` contains the full end-to-end analysis including:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training and evaluation for all four approaches
- Comprehensive visualizations and statistical analysis
- Research findings and regulatory compliance assessment

**Note:** The `notebook.ipynb` is primarily for visual purposes and research demonstration, though execution is possible.

### Option 2: Interactive Repository (Recommended for Hands-on Exploration)

For direct interaction with the models and pipeline components, use the structured repository:

```bash
cd IncomeEstimation/
make all  # Runs complete pipeline for all models
```

See the [IncomeEstimation README](IncomeEstimation/README.md) for detailed usage instructions and individual model documentation.

**Note:** Both approaches use identical preprocessing, training, and evaluation logic, ensuring consistent and reproducible results across different execution environments.

---

## Acknowledgements

* International School of Economics at TSU (ISET)

* Shota Natenadze – Supervisor, for the guidance throughout the thesis

* Bank of Georgia – for data provision

* National Bank of Georgia – for collaboration and sharing expertise

---

## License

This repository is for academic and non-commercial purposes only. Contact the author for reuse or distribution permissions.
