# Predicting silent customer churn in retail using temporal RFM trajectory patterns and explainable machine learning.

A complete machine-learning pipeline plus an interactive Streamlit dashboard for identifying customers who *look* active but are gradually disengaging — the highest-leverage segment for any retention programme.

---

## Why silent churn?

Silent churners are the most dangerous segment for a retailer because they still *look* active at a surface level. A standard churn model lumps them in with obviously lapsed customers, missing the window where a targeted campaign could bring them back. By modelling the *trajectory* of engagement (declining frequency, shrinking spend, lengthening gaps between purchases) rather than just a static snapshot, we can flag these customers before they disappear entirely.

This project tests whether **temporal RFM trajectory features** outperform **static RFM** snapshots at detecting silent churn.

---

## Headline results

A 2 × 3 factorial experiment (two feature sets × three model families) was run on the UCI Online Retail II dataset.

**Key findings:**
- Temporal features improve silent-churn F1 across every model family — XGB by 3×, RF by 30%, LR from 0 to 0.30.
- Logistic Regression with static features is **completely blind** to silent churners (0% recall). With temporal features it becomes the strongest minority-class detector (50% recall).
- **Temporal XGBoost (B1) is the recommended model** for production: best weighted F1, best accuracy, well-suited to tree-based SHAP explanations.

---

## What's in this repo

```
.
├── silent_churn_pipeline.ipynb   # End-to-end notebook: EDA → features → 6 experiments → SHAP → exports
├── app.py                        # Streamlit dashboard (zero compute at startup)
├── artifacts/
│   ├── models/                   # Trained .joblib estimators + scaler + feature manifest
│   ├── app_data/                 # Pre-scored features, monthly panel, app manifest
└── README.md
```

---

## Pipeline overview

The notebook runs end-to-end in 13 numbered sections:

1. **Environment setup** — fixed seeds, plot configuration, output directories.
2. **Data loading** — UCI Online Retail II from a local Excel/CSV source.
3. **Cleaning** — remove cancellations, refunds, missing IDs; filter to customers with ≥3 distinct invoices.
4. **Exploratory analysis** — transaction seasonality, customer-level RFM distributions.
5. **Static RFM features** — the literature-validated baseline (Recency, Frequency, Monetary).
6. **Temporal RFM features** — 9 trajectory features built from a monthly customer × month panel:
   - `freq_trend`, `monetary_trend`, `recency_trend` (last-3 vs first-3 means)
   - `consec_inactive_months`, `gap_mean`, `gap_std`, `freq_last3_vs_first3`
   - `monetary_cv`, `active_month_ratio`
7. **Labelling, split, SMOTE** — 3-class labels (Active / Regular Churn / Silent Churn), customer-level stratified 80/20 split, partial SMOTE on the train fold only.
8–9. **Six experiments** — A1–A3 on static features, B1–B3 on temporal features, each with 5-fold stratified CV and randomized hyperparameter search (`f1_weighted` scoring).
10. **Multi-class evaluation** — per-class metrics, confusion matrices, ROC curves, static-vs-temporal Δ table.
11. **SHAP explainability** — global feature importance (bar plot) and per-customer waterfall.
---

## The dashboard

The Streamlit app (`app.py`) is a presentation layer over pre-computed artefacts, it does **not** retrain or recompute anything at startup. Two pages:

### Risk Overview
Portfolio-level KPIs (cohort size, flagged customers, spend at risk), actual-vs-predicted distribution and confusion matrix, a top-50 silent-churn watchlist, and the global SHAP feature-importance bar chart.

### Customer Drill-Down
Pick any customer (sorted by silent-churn probability) to see:
- Risk badge (HIGH / MEDIUM / LOW) plus snapshot RFM and trend metrics
- Interactive 18-month trajectory (Plotly) showing monthly invoices, spend, and recency
- Per-customer SHAP waterfall explaining why the model made that prediction
- Plain-English recommended action keyed to the dominant SHAP driver

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `silent_churn_pipeline.ipynb` in Jupyter and run all cells.

This generates everything under `artifacts/` that the dashboard depends on.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

If `streamlit` isn't found, run via the active venv: `python -m streamlit run app.py`.

---

## License

MIT.
