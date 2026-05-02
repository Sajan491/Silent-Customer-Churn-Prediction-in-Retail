"""
Silent-Churn Cockpit -- Streamlit demo for the silent-churn pipeline.

Two pages:
  1. Risk Overview        -- portfolio KPIs and a top-50 watchlist.
  2. Customer Drill-Down  -- per-customer card + trajectory + SHAP waterfall.

Uses the recommended Temporal XGBoost (B1) model throughout.

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
import plotly.graph_objects as go
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

# Consistent color palette
COLORS = {
    "active":   "#10b981",   # emerald
    "regular":  "#f59e0b",   # amber
    "silent":   "#ef4444",   # red
    "primary":  "#6366f1",   # indigo
    "muted":    "#64748b",   # slate
}
CLASS_COLORS = [COLORS["active"], COLORS["regular"], COLORS["silent"]]

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
# Custom CSS for a polished demo look
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Tighten top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* App title gradient */
    .app-title {
        background: linear-gradient(90deg, #6366f1 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #475569;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Section headers */
    h2, h3 {
        color: #1e293b;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    .risk-high   { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .risk-medium { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .risk-low    { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Hide deploy / hamburger noise during demo (optional) */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""


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
    """Load scaler, the recommended temporal model (B1), and its SHAP explainer."""
    manifest = load_manifest()
    cols     = json.loads((MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    scaler   = joblib.load(MODELS_DIR / "scaler.joblib")
    b1       = joblib.load(ROOT / manifest["models"]["b1"]["joblib"])
    return {
        "manifest":     manifest,
        "cols":         cols,
        "scaler":       scaler,
        "b1":           b1,
        "explainer_b1": shap.TreeExplainer(b1),
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


def render_local_bar(shap_row: np.ndarray, base: float, feature_values: pd.Series,
                     class_label: str, title: str) -> plt.Figure:
    """Render a single-customer SHAP bar plot as a matplotlib figure."""
    feat_names = [FRIENDLY.get(n, n) for n in feature_values.index]
    expl = shap.Explanation(
        values        = np.asarray(shap_row, dtype=float).ravel(),
        base_values   = float(base),
        data          = feature_values.values,
        feature_names = feat_names,
    )
    fig = plt.figure(figsize=(7.0, 3.0))
    shap.plots.bar(expl, show=False, max_display=6)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10, y=1.02)
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


def risk_tier(prob: float) -> tuple[str, str]:
    """Map probability to (tier_label, css_class)."""
    if prob >= 0.6:
        return "HIGH RISK", "risk-high"
    if prob >= 0.3:
        return "MEDIUM RISK", "risk-medium"
    return "LOW RISK", "risk-low"


# ----------------------------------------------------------------------
# Page 1: Risk Overview
# ----------------------------------------------------------------------
def page_overview(features: pd.DataFrame, model_id: str) -> None:
    st.header("Portfolio Risk Overview")
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
    c5.metric("Spend at risk", f"GBP {spend_at_risk:,.0f}",
              "lifetime spend of flagged")

    st.divider()

    # ---------- Row 1: class distribution + confusion matrix ----------
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Actual class distribution")
        counts = features["label_3class"].value_counts().sort_index()
        labels = [CLASS_NAMES[i] for i in counts.index]
        colors = [CLASS_COLORS[i] for i in counts.index]
        pcts = [v / counts.sum() * 100 for v in counts.values]

        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=counts.values,
            marker_color=colors,
            text=[f"{v:,}<br>({p:.1f}%)" for v, p in zip(counts.values, pcts)],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<extra></extra>",
        )])
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Customers", xaxis_title="",
            plot_bgcolor="white", showlegend=False,
            yaxis=dict(gridcolor="#e2e8f0"),
        )
        fig.update_yaxes(range=[0, max(counts.values) * 1.18])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Predicted vs. actual")
        cm = pd.crosstab(
            features["label_3class"].map(CLASS_NAMES),
            features[pred_col].map(CLASS_NAMES),
            rownames=["Actual"], colnames=["Predicted"], dropna=False,
        )
        cm = cm.reindex(index=list(CLASS_NAMES.values()),
                        columns=list(CLASS_NAMES.values()), fill_value=0)

        fig = go.Figure(data=go.Heatmap(
            z=cm.values,
            x=list(cm.columns),
            y=list(cm.index),
            colorscale="Blues",
            text=cm.values,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=True,
        ))
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Predicted", yaxis_title="Actual",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------- Top 50 watchlist ----------
    st.subheader("Top 50 silent-churn risks")
    st.caption("Sorted by predicted probability. These are your priority outreach targets.")
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
    st.dataframe(
        watch_disp,
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "P(silent churn)": st.column_config.ProgressColumn(
                "P(silent churn)",
                help="Model-predicted probability of silent churn",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Total spend": st.column_config.NumberColumn(
                "Total spend (GBP)", format="GBP %.2f"
            ),
        },
    )

    st.caption(
        "P(silent churn) is the predicted probability for class 2 from the "
        "Temporal XGBoost (B1) model."
    )

    st.divider()
    st.subheader("Global Feature Importance (SHAP)")
    st.caption("Which features drive the silent-churn predictions across the customer portfolio.")
    bar_path = ROOT / "images" / "shap_summary_bar.png"
    if bar_path.exists():
        col_a, _ = st.columns([2, 1])
        with col_a:
            st.image(str(bar_path), use_container_width=True)


# ----------------------------------------------------------------------
# Page 2: Customer Drill-Down
# ----------------------------------------------------------------------
def page_drilldown(features: pd.DataFrame, panel: pd.DataFrame,
                   bundle: dict, model_id: str) -> None:
    st.header("Customer Drill-Down")
    st.caption(
        "Pick a customer to see their card, monthly trajectory, and the "
        "model's per-customer SHAP explanation."
    )

    prob_col = f"prob_silent_{model_id}"
    pred_col = f"pred_{model_id}"

    # ----- Selection controls ------------------------------------------------
    ids_sorted = list(features.sort_values(prob_col, ascending=False).index.astype(str))

    sel_col, filter_col = st.columns([3, 1])
    with sel_col:
        sel_id_str = st.selectbox(
            "Customer ID (sorted by silent-churn probability)",
            ids_sorted,
            key="cust_select",
        )
    with filter_col:
        st.markdown("&nbsp;")  # vertical alignment
        st.caption(f"{len(ids_sorted):,} customers in cohort")

    cust_id = int(sel_id_str)
    row = features.loc[cust_id]
    prob = float(row[prob_col])
    tier_label, tier_class = risk_tier(prob)

    # ----- Customer card -----------------------------------------------------
    actual_label  = CLASS_NAMES[int(row["label_3class"])]
    pred_label    = CLASS_NAMES[int(row[pred_col])]

    st.markdown(
        f"### Customer {cust_id} "
        f"<span class='risk-badge {tier_class}'>{tier_label}</span>",
        unsafe_allow_html=True,
    )

    # Snapshot row
    a, b, c, d, e = st.columns(5)
    a.metric("Country",            row["country"])
    b.metric("Total invoices",     int(row["frequency"]))
    c.metric("Total spend (GBP)",  f"{row['monetary']:,.0f}")
    d.metric("Days since last",    int(row["recency_days"]))
    e.metric("Longest gap (mo)",   int(row["consec_inactive_months"]))

    # Trend row
    a, b, c, d, e = st.columns(5)
    a.metric("Spend trend",        f"{row['monetary_trend']:+.1f}")
    b.metric("Frequency trend",    f"{row['freq_trend']:+.2f}")
    c.metric("Recency trend",      f"{row['recency_trend']:+.1f}")
    d.metric("Active month ratio", f"{row['active_month_ratio']:.2f}")
    e.metric("Spend volatility",   f"{row['monetary_cv']:.2f}")

    # Prediction summary panel
    pcol1, pcol2 = st.columns([1, 1])
    with pcol1:
        st.markdown(
            f"**Actual label:** {actual_label}  \n"
            f"**Model prediction:** {pred_label}"
        )
    with pcol2:
        st.markdown("**Probability of Silent Churn:**")
        st.progress(prob, text=f"{prob:.1%}")

    st.divider()

    # ----- Trajectory (Plotly, interactive) ---------------------------------
    st.subheader("Monthly trajectory (18-month observation window)")
    st.caption("Hover any bar or point to see the exact monthly value.")
    cust_panel = panel[panel["Customer_ID"] == cust_id].sort_values("YearMonth")

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Invoices per month", "Spend per month (GBP)", "Recency (days since previous buy)"),
    )
    fig.add_trace(
        go.Bar(x=cust_panel["YearMonth"], y=cust_panel["freq_t"],
               marker_color=COLORS["primary"], name="Invoices",
               hovertemplate="%{x|%b %Y}<br>%{y} invoices<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=cust_panel["YearMonth"], y=cust_panel["monetary_t"],
               marker_color=COLORS["active"], name="Spend",
               hovertemplate="%{x|%b %Y}<br>GBP %{y:,.0f}<extra></extra>"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=cust_panel["YearMonth"], y=cust_panel["recency_t"],
                   mode="lines+markers", line=dict(color=COLORS["silent"], width=2),
                   marker=dict(size=8), name="Recency",
                   hovertemplate="%{x|%b %Y}<br>%{y:.0f} days<extra></extra>"),
        row=3, col=1,
    )
    fig.update_layout(
        height=520, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#e2e8f0", row=3, col=1)
    fig.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----- SHAP waterfall ----------------------------------------------------
    st.subheader("Why did the model predict this?")
    st.caption("Each bar shows how much a feature pushed the silent-churn probability up or down.")
    explainer = bundle[f"explainer_{model_id}"]
    cols      = bundle["cols"]
    scaler    = bundle["scaler"]

    feat_row = features.loc[[cust_id], cols["all"]]
    scaled_row = pd.DataFrame(scaler.transform(feat_row),
                              columns=cols["all"], index=feat_row.index)
    X_temporal = scaled_row[cols["temporal"]]

    shap_vals, base = get_class_shap(explainer, X_temporal, class_idx=2)
    feat_for_display = feat_row[cols["temporal"]].iloc[0]

    fig = render_local_bar(
        shap_vals[0], base, feat_for_display,
        class_label="Silent churn",
        title=f"Drivers of silent-churn probability for customer {cust_id}",
    )
    shap_left, shap_right = st.columns([2, 1])
    with shap_left:
        st.pyplot(fig, clear_figure=True)

    top_feat, action = recommend_action(shap_vals[0], cols["temporal"])
    st.success(f"**Recommended action ({top_feat} dominates):** {action}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Silent Churn Cockpit",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Branded header
    st.markdown('<div class="app-title">Silent Churn Cockpit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Predicting Silent Customer Churn in Retail '
        'using Temporal RFM Trajectory Patterns and Explainable Machine Learning</div>',
        unsafe_allow_html=True,
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

    # ---------- Sidebar ----------
    st.sidebar.markdown("### Demo Controls")

    page = st.sidebar.radio(
        "Page",
        ["Risk Overview", "Customer Drill-Down"],
        index=0,
    )

    # The app uses a single recommended model: Temporal XGBoost (B1).
    model_id = "b1"

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model")
    st.sidebar.markdown(
        "**Temporal XGBoost (B1)**  \n"
        "<span style='color:#64748b; font-size:0.85rem;'>"
        "Best weighted F1 among all six experiments. "
        "Trained on 9 temporal trajectory features."
        "</span>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Cohort")
    s1, s2 = st.sidebar.columns(2)
    s1.metric("Customers", f"{bundle['manifest']['n_customers']:,}")
    s2.metric("Test set", f"{bundle['manifest']['n_test']:,}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 0.8rem; color: #64748b;'>"
        "Built with Streamlit • Model: XGBoost • "
        "Explanations: SHAP"
        "</div>",
        unsafe_allow_html=True,
    )

    if page == "Risk Overview":
        page_overview(features, model_id)
    else:
        page_drilldown(features, panel, bundle, model_id)


if __name__ == "__main__":
    main()