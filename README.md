# Recession Prediction with Yield Curve PCA and Macroeconomic Variables

This repository contains a fully reproducible empirical machine learning pipeline for predicting U.S. recessions using the Treasury yield curve, principal component analysis (PCA), macroeconomic indicators, and modern classification models.
The project emphasizes **time-respecting validation**, **robust threshold selection**, and **scenario-based evaluation** (Global Financial Crisis vs. COVID-19).

The structure and workflow are designed to match best practices in applied ML and empirical macro-finance research.

---

## 1️⃣ Research Objective

The goal of this project is to evaluate whether information in the U.S. yield curve—summarized via PCA—and macroeconomic variables can predict NBER recessions at a fixed forecast horizon.

Key questions:

* How much predictive power is contained in yield curve principal components beyond simple spreads?
* Does combining yield curve information with macroeconomic variables improve performance?
* How stable are results across different validation schemes?
* Why do models that perform well during the GFC struggle during COVID?

---

## 2️⃣ Methodological Overview

### Target Variable

* Binary indicator of an NBER recession at horizon $t + h$
* Default horizon: **12 months ahead**

### Feature Blocks

* **Yield curve levels** (3m to 30y)
* **Yield spreads** (10y–3m, 10y–2y)
* **Yield curve PCA** (level, slope, curvature)
* **Macroeconomic variables**

  * Unemployment rate
  * Inflation (YoY)
  * Industrial production (YoY)
  * Consumer sentiment
  * Payroll employment (YoY)
* **Policy / credit indicators**
* **Regime dummies** (GFC, COVID, ZLB/QE)

### Models

* Logistic regression (elastic net)
* Random forest
* Gradient boosting
* XGBoost (optional)

### Validation Strategy

* Expanding-window cross-validation (for threshold tuning)
* Multiple holdout splits (sensitivity analysis)
* Scenario-based testing:

  * **GFC:** 2007–2009
  * **COVID:** 2019–2021

---

## 3️⃣ Repository Structure

```
.
├── data/                   # Raw and processed datasets
├── experiments/            # Experiment entry points
├── models/                 # Trained models
├── reports/                # LaTeX paper, tables, and figures
│   ├── figures/
│   ├── tables/
│   └── main.tex
├── src/                    # Core library code
│   ├── data/               # Data ingestion and construction
│   ├── features/           # PCA and feature engineering
│   ├── models/             # Model definitions
│   ├── evaluations/        # Metrics, thresholds, validation
│   ├── visualizations/     # Publication-quality figures
│   └── utils/              # Helpers (LaTeX export, misc)
├── environment.yml         # Conda environment (reproducible)
├── Makefile                # One-command replication
└── README.md
```

---

## 4️⃣ Reproducibility and Environment Setup

All experiments are fully reproducible using Conda.

### Step 1: Create the environment

```bash
conda env create -f environment.yml
conda activate ec48e-recession
```

### Step 2: Set FRED API key

```bash
export FREDAPI="YOUR_FRED_API_KEY"
```

---

## 5️⃣ Running the Full Pipeline

### Run all experiments

```bash
make run
```

This will:

* Download and process data from FRED
* Construct features and PCA representations
* Train all models
* Run holdout sensitivity and scenario tests
* Save results to `outputs/`

---

## 6️⃣ Generating Tables and Figures

### Export LaTeX tables

```bash
make tables
```

Tables are saved to:

```
reports/tables/
```

### Figures

All figures are saved as **publication-ready PDF files** to:

```
reports/figures/
```

---

## 7️⃣ Building the LaTeX Report

The final paper is fully automated.

```bash
make report
```

This compiles:

```
reports/main.tex
```

which pulls in:

* Tables generated from model outputs
* Figures produced by the pipeline
* Modular section files

---

## 8️⃣ Key Findings (Summary)

* Yield curve PCA components capture substantially more information than simple spreads.
* Combining yield PCA with macroeconomic variables improves predictive performance.
* Models perform well during the GFC but deteriorate during COVID, highlighting:

  * The limits of low-frequency macro predictors
  * The challenge of forecasting sudden, exogenous shocks
* Time-respecting validation materially affects reported performance.

---

## 9️⃣ Notes on COVID and Model Limitations

COVID represents a structural break driven by an exogenous shock.
Traditional yield curve and macro variables react with delay, especially at a 12-month forecast horizon.

Potential extensions:

* Mixed-frequency indicators (claims, financial stress indices)
* Shorter horizons (3–6 months)
* Real-time activity measures

---

## 🔟 Citation and Use

This repository is intended for academic and educational use.
If you use or adapt this code, please cite appropriately.

---

## 11️⃣ License

See `LICENSE` for details.