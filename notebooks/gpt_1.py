import numpy as np
from pandas_datareader import data as web
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# ----------------------------------------------------
# 1. Download data from FRED
# ----------------------------------------------------
fred_series = {
    'T10Y2Y':   '10Y_2Y_Spread',      # 10y - 2y spread
    'UNRATE':   'UnemploymentRate',   # unemployment rate
    'CPIAUCSL': 'CPI',                # CPI (price level)
    'USREC':    'RecessionIndicator', # NBER recession dummy

    'GS1':      'Y1Y',
    'GS2':      'Y2Y',
    'GS5':      'Y5Y',
    'GS10':     'Y10Y'
}

start_date = '1960-01-01'
end_date   = '2025-12-31'

data = web.DataReader(list(fred_series.keys()), 'fred', start_date, end_date)
data.rename(columns=fred_series, inplace=True)

print("Raw from FRED (daily/mixed):", data.shape)

# ----------------------------------------------------
# 2. Convert daily/mixed to monthly (end of month)
# ----------------------------------------------------
data = data.resample('M').last()
print("After monthly resample:", data.shape)
print("Index min/max:", data.index.min(), "->", data.index.max())

# Drop rows with NaN in core series
base_cols = ['10Y_2Y_Spread', 'UnemploymentRate', 'CPI', 'RecessionIndicator']
data = data.dropna(subset=base_cols)
print("After dropping NaN in base columns:", data.shape)

# ----------------------------------------------------
# 3. Inflation and forward-looking recession target
# ----------------------------------------------------

# Year-on-year inflation rate (in %)
data['InflationRate'] = data['CPI'].pct_change(periods=12, fill_method=None) * 100
data = data.dropna(subset=['InflationRate'])
print("After InflationRate computation:", data.shape)

# Forward-looking 12-month recession indicator:
# RecessionNext12m = 1 if there is any recession in the next 12 months
data['RecessionNext12m'] = (
    data['RecessionIndicator']
    .rolling(window=12, min_periods=1)
    .max()
    .shift(-12)
)

# Drop the last 12 months (we do not know future USREC there)
data = data.dropna(subset=['RecessionNext12m'])
data['RecessionNext12m'] = data['RecessionNext12m'].astype(int)

print("After target creation, class counts:")
print(data['RecessionNext12m'].value_counts())

# ----------------------------------------------------
# 4. PCA on the yield curve (all maturities, one PCA)
# ----------------------------------------------------
# Use all columns whose names start with 'Y' as yield maturities
yield_cols = [c for c in data.columns if c.startswith('Y')]
print("Yield columns used for PCA:", yield_cols)

# Drop rows with NaNs in any yield maturity
data = data.dropna(subset=yield_cols)
print("After dropping NaN in yield columns:", data.shape)

# Train/test split index
data.index = pd.to_datetime(data.index)
train_mask = data.index < '2015-01-01'
test_mask  = data.index >= '2015-01-01'

Y = data[yield_cols].values
Y_train = Y[train_mask]

# Fit PCA only on the training sample (no look-ahead)
n_pcs = min(3, Y.shape[1])  # up to 3 components or fewer if yields < 3
pca = PCA(n_components=n_pcs)
pca.fit(Y_train)

# Transform the full period to get consistent factors
pcs_all = pca.transform(Y)
for i in range(n_pcs):
    data[f'PC{i+1}'] = pcs_all[:, i]

print("PCA explained variance ratios:", pca.explained_variance_ratio_)

# ----------------------------------------------------
# 5. Feature set + export to Excel
# ----------------------------------------------------
# Base features + PCA factors
feature_cols = ['10Y_2Y_Spread', 'UnemploymentRate', 'InflationRate'] + \
               [f'PC{i+1}' for i in range(n_pcs)]

print("\nNaN counts in final dataset (features + target):")
print(data[feature_cols + ['RecessionNext12m']].isna().sum())

print("Final dataset shape:", data.shape)

# Train/Test split
X_all = data[feature_cols].values
y_all = data['RecessionNext12m'].values

X_train, X_test = X_all[train_mask], X_all[test_mask]
y_train, y_test = y_all[train_mask], y_all[test_mask]

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train recession count:", y_train.sum(), "/", len(y_train))
print("Test recession count :", y_test.sum(), "/", len(y_test))

if X_train.shape[0] == 0 or X_test.shape[0] == 0:
    raise RuntimeError("Train or test set is empty. Check the date split or data pulling.")

# Excel export (dataset + train/test flag)
data['is_test'] = test_mask.astype(int)
output_excel = "yield_curve_ml_dataset.xlsx"
data.to_excel(output_excel, index=True)
print(f"\nExcel dataset written to: {output_excel}")

# ----------------------------------------------------
# 6. Models (Logit, RF, GB, MLP) + Time-series CV
# ----------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)

# Logistic regression with Elastic Net and class_weight='balanced'
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        solver='saga',
        penalty='elasticnet',
        max_iter=10000,
        class_weight='balanced'   # important for imbalanced data
    ))
])

param_grid = {
    'logreg__C': [0.01, 0.1, 1.0, 10.0],
    'logreg__l1_ratio': [0.0, 0.5, 1.0],
}

grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring='f1')
grid.fit(X_train, y_train)
best_logit = grid.best_estimator_
print("\nBest params (Logistic):", grid.best_params_)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    class_weight='balanced',  # also balance here
    random_state=0
)
rf.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=0
)
gb.fit(X_train, y_train)

# Neural Network (MLP)
mlp = MLPClassifier(
    hidden_layer_sizes=(20, 10),
    activation='relu',
    max_iter=1000,
    alpha=0.001,
    random_state=0
)
mlp.fit(X_train, y_train)

# ----------------------------------------------------
# 7. Evaluation function (with threshold)
# ----------------------------------------------------
def evaluate_model(name, y_true, y_proba, threshold=0.3):
    """
    Evaluate a binary classifier with a given decision threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_proba)

    print(f"\n=== {name} (threshold = {threshold:.2f}) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    return auc, f1, y_pred

# ----------------------------------------------------
# 8. Model evaluation + ROC curves
# ----------------------------------------------------
DECISION_THRESHOLD = 0.3  # <--- lower than 0.5 to avoid "all zeros" in imbalanced data

# Logistic
y_proba_logit = best_logit.predict_proba(X_test)[:, 1]
auc_logit, f1_logit, y_pred_logit = evaluate_model(
    "Logistic (Elastic Net)", y_test, y_proba_logit, threshold=DECISION_THRESHOLD
)

# Random Forest
y_proba_rf = rf.predict_proba(X_test)[:, 1]
auc_rf, f1_rf, y_pred_rf = evaluate_model(
    "Random Forest", y_test, y_proba_rf, threshold=DECISION_THRESHOLD
)

# Gradient Boosting
y_proba_gb = gb.predict_proba(X_test)[:, 1]
auc_gb, f1_gb, y_pred_gb = evaluate_model(
    "Gradient Boosting", y_test, y_proba_gb, threshold=DECISION_THRESHOLD
)

# MLP
y_proba_nn = mlp.predict_proba(X_test)[:, 1]
auc_nn, f1_nn, y_pred_nn = evaluate_model(
    "Neural Network (MLP)", y_test, y_proba_nn, threshold=DECISION_THRESHOLD
)

# --- ROC curves ---
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_proba_logit)
fpr_rf,    tpr_rf,    _ = roc_curve(y_test, y_proba_rf)
fpr_gb,    tpr_gb,    _ = roc_curve(y_test, y_proba_gb)
fpr_nn,    tpr_nn,    _ = roc_curve(y_test, y_proba_nn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_logit, tpr_logit, label=f"Logit (AUC={auc_logit:.2f})")
plt.plot(fpr_rf,    tpr_rf,    label=f"RF (AUC={auc_rf:.2f})")
plt.plot(fpr_gb,    tpr_gb,    label=f"GB (AUC={auc_gb:.2f})")
plt.plot(fpr_nn,    tpr_nn,    label=f"MLP (AUC={auc_nn:.2f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Test Set")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
