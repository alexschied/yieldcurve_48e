# yieldcurve_48e

Final Term Project for Financial Applications of Machine Learning (EC 48E, Fall Term 2025). 

# General Idea: Yield Curve Modeling via ML Methods

- Use the yield curve slope (long–short rate spread) as the primary predictor of U.S. recessions, following the empirical approach of Estrella & Mishkin (1996). 
- Compare the predictive power of the yield curve to alternative indicators.

# Outline (Based on [1] )

1. **Motivation: Why the Yield Curve Matters**
   - The slope reflects monetary policy stance and expectations of future real rates and inflation.
   - Prior evidence shows the yield curve outperforms many financial and macro indicators at forecasting recessions four quarters ahead.

2. **Data Collection**
   - Gather long- and short-term interest rates (e.g., 10Y Treasury minus 3M T-bill).
   - Compile comparison variables: stock returns, leading economic indicators, macro series.

3. **Label Construction**
   - Define recession quarters using NBER dating.
   - Create a binary recession indicator aligned with forecast horizons (1–6 quarters ahead).

4. **Feature Engineering**
   - Compute term spreads consistent with the paper.
   - Optionally reduce dimensionality of macroeconomic variables using PCA to create interpretable factors.

5. **Modeling Approach**
   - Train classification models (probability-of-recession framework) analogous to the paper’s probit specification.
   - Fit ML alternatives (logistic regression, tree models, ensemble methods) while keeping interpretability in mind.

6. **Forecast Evaluation**
   - Produce out-of-sample forecasts for multiple horizons.
   - Compare ML model performance to the baseline yield-curve model.
   - Evaluate metrics: calibration, ROC/AUC, and signal timing.

7. **Interpretation and Policy Insight**
   - Assess whether the yield curve still provides the strongest signal at longer horizons.
   - Discuss the economic meaning of the model outputs and how they relate to monetary policy dynamics.

# Literature 

 - [1] [General source](https://www.bis.org/publ/confp02n.pdf) (From Prof)

# Data
- [FRED](https://fred.stlouisfed.org/) (We can get a free API key)

# Structure
```text
.
├── data/
│   ├── raw/
│   └── processed/
│
├── models/                           # For trained models
│
├── notebooks/
│
├── reports/
│   └── figures/
│
├── src/                              # Our modules (data preperation, crossvalidation, ...)
│   ├── data/
│   └── models/
│
├── .gitignore                        # Standard Git ignores (venv, IDE files, etc.)
├── .gitkeep                          # Keep data parent folder
├── environment.yml                   # Conda environment definition (dependencies)
├── LICENSE
├── Makefile                          # Scripts for common commands (make data, make train, make report)
└── README.md                         # Project documentation and setup instructions
```
