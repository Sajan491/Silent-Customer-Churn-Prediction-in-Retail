"""
Silent-Churn Cockpit -- Streamlit demo for the silent-churn pipeline.

Two pages:
  1. Risk Overview        -- portfolio KPIs and a top-50 watchlist.
  2. Customer Drill-Down  -- per-customer card + trajectory + SHAP waterfall.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent
ARTIFACTS   = ROOT / "artifacts"
MODELS_DIR  = ARTIFACTS / "models"
APP_DATA    = ARTIFACTS / "app_data"

CLASS_NAMES = {0: "Active", 1: "Regular Churn", 2: "Silent Churn"}

# Friendly labels used everywhere instead of the raw column names
FRIENDLY = {
    "recency_days":            "Days since last purchase",
    "frequency":                "Total invoices",
    "monetary":                 "Total spend",
    "freq_trend":               "Frequency trend (last 3 vs first 3)",
    "monetary_trend":           "Spend trend (last 3 vs first 3)",
    "recency_trend":            "Recency trend (last 3 vs first 3)",
    "consec_inactive_months":   "Longest inactive streak (months)",
    "gap_mean":                 "Average gap between purchases (days)",
    "gap_std":                  "Variability of purchase gaps (days)",
    "freq_last3_vs_first3":     "Recent vs early activity ratio",
    "monetary_cv":              "Spend volatility (CV)",
    "active_month_ratio":       "Active months ratio",
}

# Plain-English action recommendations keyed by dominant SHAP feature.
ACTIONS = {
    "monetary_trend":         "Send a re-engagement discount tied to declining basket size.",
    "freq_trend":             "Trigger a personalised loyalty offer -- purchase frequency is dropping.",
    "recency_trend":          "Run a 'come back' email cadence to re-establish purchase rhythm.",
    "freq_last3_vs_first3":   "Promote loyalty perks; recent activity is far below early baseline.",
    "consec_inactive_months": "Sales / account-management call -- the customer has been silent.",
    "gap_mean":               "Send a 'we miss you' offer; their purchase cadence is too long.",
    "gap_std":                "Investigate volatility -- inconsistent purchase timing.",
    "monetary_cv":            "Look at order volatility -- feast-and-famine spend pattern.",
    "active_month_ratio":     "Re-introduce the brand -- customer was rarely active.",
    "monetary":               "High-value customer -- escalate to retention specialist.",
    "frequency":              "Tailor offer to total purchase volume.",
    "recency_days":           "Time-sensitive nudge -- last purchase was a long time ago.",
}

# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    return json.loads((APP_DATA / "manifest.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_csv(APP_DATA / "features.csv", index_col=0)


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    p = pd.read_csv(APP_DATA / "panel.csv")
    p["YearMonth"] = pd.to_datetime(p["YearMonth"])
    return p


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    """Load scaler, both temporal models, and their SHAP explainers."""
    manifest = load_manifest()
    cols     = json.loads((MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    scaler   = joblib.load(MODELS_DIR / "scaler.joblib")
    b1       = joblib.load(ROOT / manifest["models"]["b1"]["joblib"])
    b2       = joblib.load(ROOT / manifest["models"]["b2"]["joblib"])
    return {
        "manifest":  manifest,
        "cols":      cols,
        "scaler":    scaler,
        "b1":        b1,
        "b2":        b2,
        "explainer_b1": shap.TreeExplainer(b1),
        "explainer_b2": shap.TreeExplainer(b2),
    }


# ----------------------------------------------------------------------
# SHAP helpers (multi-class XGBoost vs RF behave differently)
# ----------------------------------------------------------------------
def get_class_shap(explainer, X: pd.DataFrame, class_idx: int = 2):
    """Return (shap_values_for_class, base_value_for_class) for an arbitrary
    multi-class tree explainer, regardless of which SHAP API shape it returns."""
    sv = explainer.shap_values(X)
    expected = explainer.expected_value
    if isinstance(sv, list):
        return np.asarray(sv[class_idx]), float(np.asarray(expected)[class_idx])
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[-1] == 3:
        return sv[..., class_idx], float(np.asarray(expected)[class_idx])
    if sv.ndim == 3 and sv.shape[0] == 3:
        return sv[class_idx], float(np.asarray(expected)[class_idx])
    return sv, float(np.asarray(expected).ravel()[0])


def render_waterfall(shap_row: np.ndarray, base: float, feature_values: pd.Series,
                     class_label: str, title: str) -> plt.Figure:
    """Render a single-customer SHAP waterfall as a matplotlib figure."""
    feat_names = [FRIENDLY.get(n, n) for n in feature_values.index]
    expl = shap.Explanation(
        values        = np.asarray(shap_row, dtype=float).ravel(),
        base_values   = float(base),
        data          = feature_values.values,
        feature_names = feat_names,
    )
    fig = plt.figure(figsize=(8.5, 4.6))
    shap.plots.waterfall(expl, show=False, max_display=10)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Action recommender
# ----------------------------------------------------------------------
def recommend_action(shap_row: np.ndarray, feature_names: list[str]) -> tuple[str, str]:
    """Return (top_feature_friendly, action_text) for the dominant POSITIVE
    silent-churn driver. If no positive driver exists the customer is unlikely
    to be silent-churn risk."""
    arr = np.asarray(shap_row, dtype=float).ravel()
    if not np.any(arr > 0):
        return ("--", "No silent-churn pressure -- customer looks healthy.")
    idx = int(np.argmax(arr))
    raw = feature_names[idx]
    return FRIENDLY.get(raw, raw), ACTIONS.get(raw, "Watch closely -- unusual trajectory.")


# ----------------------------------------------------------------------
# Page 1: Risk Overview
# ----------------------------------------------------------------------
def page_overview(features: pd.DataFrame, model_id: str) -> None:
    st.header("1. Portfolio Risk Overview")
    st.caption(
        "Rolled-up view across the 3,043 customers in the modelling cohort. "
        "Predictions come from the model selected in the sidebar."
    )

    prob_col = f"prob_silent_{model_id}"
    pred_col = f"pred_{model_id}"

    # KPI row
    n          = len(features)
    n_active   = int((features["label_3class"] == 0).sum())
    n_regular  = int((features["label_3class"] == 1).sum())
    n_silent   = int((features["label_3class"] == 2).sum())
    flagged    = int((features[pred_col] == 2).sum())
    spend_at_risk = float(features.loc[features[pred_col] == 2, "monetary"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers",            f"{n:,}")
    c2.metric("Actual silent churn",  f"{n_silent:,}", f"{n_silent / n:.1%}")
    c3.metric("Actual regular churn", f"{n_regular:,}", f"{n_regular / n:.1%}")
    c4.metric("Flagged silent churn", f"{flagged:,}",
              f"{flagged / n:.1%} of book")
    c5.metric("Spend at risk (silent)", f"GBP {spend_at_risk:,.0f}",
              "lifetime spend of flagged")

    st.divider()

    left, right = st.columns([1, 1])

    # Class distribution chart
    with left:
        st.subheader("Actual class distribution")
        counts = features["label_3class"].value_counts().sort_index()
        labels = [CLASS_NAMES[i] for i in counts.index]
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        bars = ax.bar(labels, counts.values, color=["#4c78a8", "#f58518", "#e45756"])
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v + max(counts.values) * 0.01,
                    f"{v:,}\n({v / counts.sum():.1%})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Customers")
        ax.set_ylim(0, max(counts.values) * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, clear_figure=True)

    # Predicted-vs-actual confusion
    with right:
        st.subheader("Predicted vs. actual (selected model)")
        cm = pd.crosstab(
            features["label_3class"].map(CLASS_NAMES),
            features[pred_col].map(CLASS_NAMES),
            rownames=["Actual"], colnames=["Predicted"], dropna=False,
        )
        cm = cm.reindex(index=list(CLASS_NAMES.values()),
                        columns=list(CLASS_NAMES.values()), fill_value=0)
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        im = ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_xticklabels(cm.columns, rotation=20, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(cm.index)
        for i in range(3):
            for j in range(3):
                v = cm.values[i, j]
                ax.text(j, i, f"{v}", ha="center", va="center",
                        color="white" if v > cm.values.max() / 2 else "black", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.divider()

    # Top-50 watchlist
    st.subheader("Top 50 silent-churn risks (by predicted probability)")
    watch = features.sort_values(prob_col, ascending=False).head(50).copy()
    watch_disp = pd.DataFrame({
        "Customer": watch.index.astype(str),
        "Country": watch["country"],
        "P(silent churn)": watch[prob_col].round(3),
        "Predicted": watch[pred_col].map(CLASS_NAMES),
        "Actual": watch["label_3class"].map(CLASS_NAMES),
        "Total spend": watch["monetary"].round(2),
        "Days since last buy": watch["recency_days"].astype(int),
        "Longest inactive (months)": watch["consec_inactive_months"].astype(int),
        "Spend trend": watch["monetary_trend"].round(2),
        "Frequency trend": watch["freq_trend"].round(2),
    })
    st.dataframe(watch_disp, use_container_width=True, hide_index=True, height=520)

    st.caption(
        "P(silent churn) is the model's predicted probability for class 2. "
        "Use the sidebar to switch between B1 (Temporal XGBoost) and B2 "
        "(Temporal Random Forest)."
    )


# ----------------------------------------------------------------------
# Page 2: Customer Drill-Down
# ----------------------------------------------------------------------
def page_drilldown(features: pd.DataFrame, panel: pd.DataFrame,
                   bundle: dict, model_id: str) -> None:
    st.header("2. Customer Drill-Down")
    st.caption(
        "Pick a customer to see their card, monthly trajectory, and the "
        "model's per-customer SHAP explanation."
    )

    prob_col = f"prob_silent_{model_id}"
    pred_col = f"pred_{model_id}"

    # ----- Selection controls ------------------------------------------------
    sel_col1, sel_col2, sel_col3, sel_col4 = st.columns([2, 1, 1, 1])
    with sel_col1:
        ids_sorted = list(features.sort_values(prob_col, ascending=False).index.astype(str))
        sel_id_str = st.selectbox(
            "Customer ID (sorted by silent-churn probability)",
            ids_sorted,
            key="cust_select",
        )

    def quick_pick(query):
        if not query.empty:
            st.session_state["cust_select"] = str(query.sample(1, random_state=None).index[0])
            st.rerun()

    with sel_col2:
        if st.button("Random TP\n(actual=2 & pred=2)"):
            quick_pick(features[(features["label_3class"] == 2) & (features[pred_col] == 2)])
    with sel_col3:
        if st.button("Random FN\n(actual=2 & pred!=2)"):
            quick_pick(features[(features["label_3class"] == 2) & (features[pred_col] != 2)])
    with sel_col4:
        if st.button("Random FP\n(actual!=2 & pred=2)"):
            quick_pick(features[(features["label_3class"] != 2) & (features[pred_col] == 2)])

    cust_id = int(sel_id_str)
    row = features.loc[cust_id]

    # ----- Customer card -----------------------------------------------------
    st.subheader(f"Customer {cust_id}")
    a, b, c, d, e = st.columns(5)
    a.metric("Country",            row["country"])
    b.metric("Total invoices",     int(row["frequency"]))
    c.metric("Total spend (GBP)",  f"{row['monetary']:,.0f}")
    d.metric("Days since last",    int(row["recency_days"]))
    e.metric("Longest gap (mo)",   int(row["consec_inactive_months"]))

    a, b, c, d, e = st.columns(5)
    a.metric("Spend trend",        f"{row['monetary_trend']:+.1f}")
    b.metric("Frequency trend",    f"{row['freq_trend']:+.2f}")
    c.metric("Recency trend",      f"{row['recency_trend']:+.1f}")
    d.metric("Active month ratio", f"{row['active_month_ratio']:.2f}")
    e.metric("Spend volatility",   f"{row['monetary_cv']:.2f}")

    st.markdown(
        f"**Actual label:** {CLASS_NAMES[int(row['label_3class'])]}  "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; **Model prediction:** "
        f"{CLASS_NAMES[int(row[pred_col])]}  "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; **P(silent)** = "
        f"{row[prob_col]:.3f}"
    )

    st.divider()

    # ----- Trajectory --------------------------------------------------------
    st.subheader("Monthly trajectory (18-month observation window)")
    cust_panel = panel[panel["Customer_ID"] == cust_id].sort_values("YearMonth")
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.0), sharex=True)
    axes[0].bar(cust_panel["YearMonth"], cust_panel["freq_t"], width=24, color="#4c78a8")
    axes[0].set_ylabel("Invoices / mo")
    axes[1].bar(cust_panel["YearMonth"], cust_panel["monetary_t"], width=24, color="#54a24b")
    axes[1].set_ylabel("Spend / mo (GBP)")
    axes[2].plot(cust_panel["YearMonth"], cust_panel["recency_t"], marker="o", color="#e45756")
    axes[2].set_ylabel("Recency (days)")
    axes[2].set_xlabel("Month")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.divider()

    # ----- SHAP waterfall ----------------------------------------------------
    st.subheader("Why did the model predict this?")
    explainer = bundle[f"explainer_{model_id}"]
    cols      = bundle["cols"]
    scaler    = bundle["scaler"]

    feat_row = features.loc[[cust_id], cols["all"]]
    scaled_row = pd.DataFrame(scaler.transform(feat_row),
                              columns=cols["all"], index=feat_row.index)
    X_temporal = scaled_row[cols["temporal"]]

    shap_vals, base = get_class_shap(explainer, X_temporal, class_idx=2)
    feat_for_display = feat_row[cols["temporal"]].iloc[0]

    fig = render_waterfall(
        shap_vals[0], base, feat_for_display,
        class_label="Silent churn",
        title=f"Drivers of silent-churn probability for customer {cust_id}",
    )
    st.pyplot(fig, clear_figure=True)

    top_feat, action = recommend_action(shap_vals[0], cols["temporal"])
    st.success(f"**Recommended action ({top_feat} dominates):** {action}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Silent Churn Cockpit",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Silent Churn Cockpit")
    st.caption(
        "Demo dashboard for *Predicting Silent Customer Churn in Retail Using "
        "Temporal RFM Trajectory Patterns and Explainable Machine Learning*."
    )

    if not (APP_DATA / "features.csv").exists():
        st.error(
            "Missing `artifacts/app_data/features.csv`. "
            "Run the notebook end-to-end first: "
            "`jupyter nbconvert --to notebook --execute --inplace silent_churn_pipeline.ipynb`."
        )
        st.stop()

    features = load_features()
    panel    = load_panel()
    bundle   = load_model_bundle()

    # Sidebar
    st.sidebar.header("Demo controls")

    page = st.sidebar.radio(
        "Page",
        ["1. Risk Overview", "2. Customer Drill-Down"],
        index=0,
    )

    model_label = st.sidebar.radio(
        "Model",
        ["B1 -- Temporal XGBoost", "B2 -- Temporal Random Forest"],
        index=0,
    )
    model_id = "b1" if model_label.startswith("B1") else "b2"

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Cohort:** {bundle['manifest']['n_customers']:,} customers  \n"
        f"**Held-out test set:** {bundle['manifest']['n_test']:,}  \n"
        f"**Best temporal model (F1w):** {bundle['manifest']['best_temporal_id']}"
    )

    if page.startswith("1"):
        page_overview(features, model_id)
    else:
        page_drilldown(features, panel, bundle, model_id)


if __name__ == "__main__":
    main()
