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
│   │   └── income_data.xlsx                 # Synthetic dataset provide by BOG
│   └── processed/
│       ├── train.csv                        # Created by notebook.ipynb
│       └── test.csv                         # Created by notebook.ipynb
├── docs/
│   └── Bachelor Thesis - Sandro Gogaladze.docx  # Thesis report
├── Notebooks/
│   └── notebook.ipynb                      # Full project in code
├── .gitignore
└── README.md                               
```

---

## Notebook and Thesis Document

The `Bachelor Thesis - Sandro Gogaladze.docx` file is the thesis report.

The `notebook.ipynb` contains the full end-to-end implementation of the thesis. It covers everything from exploratory data analysis and preprocessing to modeling, calibration, and evaluation. All figures and numerical findings in my thesis are derived directly from this source. The notebook is available upon request for verification and reproducibility.

---

## Acknowledgements

* International School of Economics at TSU (ISET)

* Shota Natenadze – Supervisor, for the guidance throughout the thesis

* Bank of Georgia – for data provision

* National Bank of Georgia – for collaboration and sharing expertise

---

## License

This repository is for academic and non-commercial purposes only. Contact the author for reuse or distribution permissions.
