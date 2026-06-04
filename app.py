import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Discus Performance Predictor",
    page_icon="🥏",
    layout="wide",
)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE  = '#378ADD'
CORAL = '#D85A30'
AMBER = '#BA7517'
TEAL  = '#1D9E75'
GRAY  = '#888780'

# ── Feature definitions ───────────────────────────────────────────────────────
QUANTITATIVE = ['Bench', 'Squat', 'Clean', 'Snatch']
ESTIMATES    = ['Speed', 'Vertical', 'Shape']
TRAVEL       = ['Days_away', 'Trip_meet', 'Overseas']
TARGET       = 'Result_normalized'
CATEGORICAL  = ['Trip_meet', 'Overseas']

FEATURE_LABELS = {
    'Bench':     'Bench Press (lbs)',
    'Squat':     'Squat (kg)',
    'Clean':     'Power Clean (kg)',
    'Snatch':    'Power Snatch (kg)',
    'Weight':    'Bodyweight (lbs)',
    'Speed':     'Speed (1–10)',
    'Vertical':  'Vertical Jump (in)',
    'Shape':     'Shape / Readiness (1–10)',
    'Days_away': 'Days Away from Home',
    'Trip_meet': 'Meet # in Trip',
    'Overseas':  'Overseas',
}

KG_FEATURES = {'Squat', 'Clean', 'Snatch'}

# (min, max, step)
SLIDER_CFG = {
    'Bench':     (380, 620, 5),
    'Squat':     (220, 360, 5),
    'Clean':     (140, 220, 1),
    'Snatch':    (110, 170, 1),
    'Weight':    (240, 305, 1),
    'Speed':     (1, 10, 1),
    'Vertical':  (28.0, 42.0, 0.5),
    'Shape':     (1, 10, 1),
    'Days_away': (0, 35, 1),
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/yeets.csv').iloc[:, :17]
    df.rename(columns={'Power Clean': 'Clean', 'Power Snatch': 'Snatch'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df['Year'] = df['Date'].dt.year
    df.sort_values(by='Date', inplace=True, ignore_index=True)
    return df

# ── Model helpers ─────────────────────────────────────────────────────────────
def prepare_X(df, features):
    cols = [f for f in features if f in df.columns]
    X = df[cols].copy()
    cats = [c for c in CATEGORICAL if c in cols]
    if cats:
        X = pd.get_dummies(X, columns=cats, drop_first=True)
    return X.astype(float)

@st.cache_data
def train_model(features_tuple):
    df = load_data()
    X = prepare_X(df, list(features_tuple))
    y = df[TARGET].astype(float)
    n, p = X.shape
    mdl = LinearRegression(fit_intercept=True)
    mdl.fit(X, y)
    pred = mdl.predict(X)
    # Precompute components needed for exact OLS prediction intervals
    X_aug   = np.column_stack([np.ones(n), X.values])
    XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
    s_resid = np.sqrt(np.sum((y.values - pred) ** 2) / (n - p - 1))
    return mdl, X.columns.tolist(), pred, y.values, XtX_inv, s_resid, n

def compute_interval(mdl, XtX_inv, s_resid, n_train, x_row_df, confidence):
    """Return (y_hat, lower, upper) OLS prediction interval for one row."""
    y_hat = mdl.predict(x_row_df)[0]
    x_vec = np.concatenate([[1.0], x_row_df.values[0]])
    leverage = float(x_vec @ XtX_inv @ x_vec)
    se = s_resid * np.sqrt(1.0 + leverage)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n_train - x_row_df.shape[1] - 1)
    margin = t_crit * se
    return y_hat, y_hat - margin, y_hat + margin

def build_row(input_vals: dict, train_cols: list) -> pd.DataFrame:
    """Convert UI input values to a model-ready single-row DataFrame."""
    row = {}
    for col in train_cols:
        if col in input_vals:
            row[col] = float(input_vals[col])
        elif col.startswith('Trip_meet_'):
            level = int(col.split('_')[-1])
            row[col] = 1.0 if input_vals.get('Trip_meet') == level else 0.0
        elif col == 'Overseas_1':
            row[col] = float(input_vals.get('Overseas', 0))
        else:
            row[col] = 0.0
    return pd.DataFrame([row])

# ── Bootstrap ─────────────────────────────────────────────────────────────────
df = load_data()
latest = df.iloc[-1]

# ── Sidebar – feature group toggles ──────────────────────────────────────────
st.sidebar.title("⚙️ Model Configuration")
st.sidebar.markdown("Toggle which feature groups to include:")

use_strength  = st.sidebar.toggle("Strength Lifts", value=True,
    help="Bench Press, Squat, Power Clean, Power Snatch")
use_bw        = st.sidebar.toggle("Bodyweight", value=False,
    help="Athlete bodyweight in lbs")
use_estimates = st.sidebar.toggle("Athleticism Estimates", value=False,
    help="Speed, Vertical Jump, and Shape/Readiness (subjective 1–10 scores)")
use_travel    = st.sidebar.toggle("Travel Logistics", value=True,
    help="Days away from home, meet number within the trip, overseas flag")

st.sidebar.divider()
confidence = st.sidebar.select_slider(
    "Prediction interval",
    options=[0.80, 0.90, 0.95],
    value=0.90,
    format_func=lambda x: f"{int(x * 100)}%",
    help="How confident should the interval be? Wider = more certain the true throw lands inside.",
)

features = []
if use_strength:  features += QUANTITATIVE
if use_bw:        features += ['Weight']
if use_estimates: features += ESTIMATES
if use_travel:    features += TRAVEL

if not features:
    st.warning("Enable at least one feature group in the sidebar to build the model.")
    st.stop()

# ── Train ─────────────────────────────────────────────────────────────────────
model, train_cols, pred_arr, y_arr, XtX_inv, s_resid, n_train = train_model(tuple(features))

r2   = r2_score(y_arr, pred_arr)
mae  = mean_absolute_error(y_arr, pred_arr)
rmse = np.sqrt(mean_squared_error(y_arr, pred_arr))

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🥏 Discus Performance Predictor")
st.caption(
    f"OLS model · **{len(df)} competition results** (2022–2026) · "
    f"**{len(train_cols)} features** · "
    f"R² = {r2:.3f} · MAE = {mae:.3f} m"
)

c1, c2, c3 = st.columns(3)
c1.metric("R²",   f"{r2:.3f}",    help="Variance in throw distance explained by the model (1.0 = perfect)")
c2.metric("MAE",  f"{mae:.3f} m", help="Average prediction error")
c3.metric("RMSE", f"{rmse:.3f} m", help="Root mean squared error; penalises large misses more than MAE")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_perf = st.tabs(["🎯  Predict My Throw", "📊  Model Performance"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Enter Your Current Stats")

    hdr_left, hdr_right = st.columns([4, 1])
    hdr_left.caption("Pre-filled from your most recent competition — adjust any value to update the prediction instantly.")

    continuous_feats = [f for f in features if f not in CATEGORICAL]
    cat_feats         = [f for f in features if f in CATEGORICAL]

    def latest_default(feat):
        lo, hi, step = SLIDER_CFG.get(feat, (0, 100, 1))
        val = float(latest[feat])
        val = float(min(max(val, lo), hi))
        return val if isinstance(step, float) else int(val)

    # Reset button — clears all slider session state keys back to latest values
    if hdr_right.button("↺  Reset to latest", use_container_width=True):
        for feat in continuous_feats:
            st.session_state[f"sl_{feat}"] = latest_default(feat)
        for feat in cat_feats:
            if feat == 'Trip_meet':
                st.session_state["cat_trip_meet"] = int(latest['Trip_meet'])
            elif feat == 'Overseas':
                st.session_state["cat_overseas"] = 'Overseas' if int(latest['Overseas']) == 1 else 'Domestic'
        st.rerun()

    input_vals = {}

    # Continuous sliders — up to 4 per row
    n_cols = min(4, len(continuous_feats))
    slider_cols = st.columns(n_cols)
    for i, feat in enumerate(continuous_feats):
        col = slider_cols[i % n_cols]
        lo, hi, step = SLIDER_CFG.get(feat, (0, 100, 1))
        default = latest_default(feat)
        if isinstance(step, float):
            input_vals[feat] = col.slider(
                FEATURE_LABELS.get(feat, feat),
                min_value=float(lo), max_value=float(hi),
                value=default, step=step,
                key=f"sl_{feat}",
            )
        else:
            input_vals[feat] = col.slider(
                FEATURE_LABELS.get(feat, feat),
                min_value=int(lo), max_value=int(hi),
                value=int(default), step=int(step),
                key=f"sl_{feat}",
            )

    # Categorical inputs
    if cat_feats:
        cat_cols = st.columns(len(cat_feats))
        for i, feat in enumerate(cat_feats):
            if feat == 'Trip_meet':
                input_vals[feat] = cat_cols[i].selectbox(
                    FEATURE_LABELS['Trip_meet'],
                    options=[1, 2, 3, 4],
                    index=min(int(latest['Trip_meet']) - 1, 3),
                    help="Which meet number is this within the current trip?",
                    key="cat_trip_meet",
                )
            elif feat == 'Overseas':
                sel = cat_cols[i].radio(
                    FEATURE_LABELS['Overseas'],
                    options=['Domestic', 'Overseas'],
                    index=int(latest['Overseas']),
                    horizontal=True,
                    key="cat_overseas",
                )
                input_vals[feat] = 1 if sel == 'Overseas' else 0

    # ── Prediction display ────────────────────────────────────────────────────
    pred_row = build_row(input_vals, train_cols)
    prediction, pi_lo, pi_hi = compute_interval(
        model, XtX_inv, s_resid, n_train, pred_row, confidence
    )
    pb  = df[TARGET].max()
    med = df[TARGET].median()

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        st.markdown("#### Predicted Distance")
        color = TEAL if prediction >= med else CORAL
        st.markdown(
            f"<div style='font-size:3.4rem; font-weight:700; color:{color}; "
            f"line-height:1.1'>{prediction:.2f} m</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**{int(confidence*100)}% interval:** {pi_lo:.2f} – {pi_hi:.2f} m",
        )
        gap = prediction - pb
        st.markdown(f"**PB:** {pb:.2f} m &nbsp;·&nbsp; **Gap to PB:** {gap:+.2f} m")
        st.caption(f"Historical median: {med:.2f} m")

    with right:
        fig_hist = go.Figure()
        fig_hist.add_vrect(
            x0=pi_lo, x1=pi_hi,
            fillcolor=TEAL, opacity=0.12,
            layer='below', line_width=0,
        )
        fig_hist.add_trace(go.Histogram(
            x=df[TARGET], nbinsx=14,
            marker_color=BLUE, opacity=0.55,
            hovertemplate='%{x:.1f} m: %{y} meets',
            name='Historical results',
        ))
        fig_hist.add_vline(
            x=prediction, line_color=TEAL, line_width=2.5,
            annotation_text=f"Predicted: {prediction:.2f} m",
            annotation_position="top right",
        )
        fig_hist.add_vline(
            x=med, line_color=GRAY, line_width=1.5, line_dash='dash',
            annotation_text="Median", annotation_position="top left",
        )
        fig_hist.update_layout(
            title=f"Where this prediction falls — shaded band = {int(confidence*100)}% interval",
            xaxis_title="Distance (m)", yaxis_title="Number of meets",
            showlegend=False, height=260,
            margin=dict(t=40, b=30, l=10, r=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── What-If ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("What-If Simulation")
    st.caption(
        "Hold everything at the values above. Sweep one stat across a range "
        "to see how much it moves the predicted throw."
    )

    wi_left, wi_right = st.columns([1, 3])
    sweep_options = [f for f in features if f != 'Overseas']

    with wi_left:
        vary_feat = st.selectbox(
            "Vary this feature",
            options=sweep_options,
            format_func=lambda f: FEATURE_LABELS.get(f, f),
        )
        cur_val = input_vals[vary_feat]

        # Actual value from the last competition (fixed reference)
        last_val = float(latest[vary_feat])

        unit_label = (
            'kg'   if vary_feat in KG_FEATURES else
            'lbs'  if vary_feat in ('Bench', 'Weight') else
            'in'   if vary_feat == 'Vertical' else
            'days' if vary_feat == 'Days_away' else
            '1–10'
        )

        if vary_feat == 'Trip_meet':
            sweep = np.array([1, 2, 3, 4])
            st.caption("Sweeping all 4 trip-meet positions")

        elif vary_feat == 'Vertical':
            rng = st.slider(f"Sweep range ({unit_label})", 28.0, 42.0,
                            (max(28.0, last_val - 3.0), min(42.0, last_val + 3.0)),
                            step=0.5)
            sweep = np.arange(rng[0], rng[1] + 0.5, 0.5)

        elif vary_feat in ('Speed', 'Shape'):
            rng = st.slider(f"Sweep range ({unit_label})", 1, 10,
                            (max(1, int(last_val) - 3), min(10, int(last_val) + 3)))
            sweep = np.arange(rng[0], rng[1] + 1, dtype=float)

        elif vary_feat == 'Days_away':
            rng = st.slider(f"Sweep range ({unit_label})", 0, 35,
                            (max(0, int(last_val) - 10), min(35, int(last_val) + 10)))
            sweep = np.arange(rng[0], rng[1] + 1, dtype=float)

        elif vary_feat == 'Weight':
            lv_i = int(last_val)
            rng = st.slider(f"Sweep range ({unit_label})", lv_i - 20, lv_i + 20,
                            (lv_i - 10, lv_i + 10), step=1)
            sweep = np.arange(rng[0], rng[1] + 1, dtype=float)

        else:  # strength lifts
            lv_i = int(last_val)
            rng = st.slider(f"Sweep range ({unit_label})", lv_i - 60, lv_i + 60,
                            (lv_i - 25, lv_i + 25), step=5)
            sweep = np.arange(rng[0], rng[1] + 1, 2, dtype=float)

    with wi_right:
        sweep_results = [
            compute_interval(model, XtX_inv, s_resid, n_train,
                             build_row({**input_vals, vary_feat: v}, train_cols),
                             confidence)
            for v in sweep
        ]
        sweep_preds = np.array([r[0] for r in sweep_results])
        sweep_lower = np.array([r[1] for r in sweep_results])
        sweep_upper = np.array([r[2] for r in sweep_results])

        # Prediction at the adjusted (slider) value
        adj_val  = float(cur_val)
        adj_pred, adj_lo, adj_hi = compute_interval(
            model, XtX_inv, s_resid, n_train,
            build_row({**input_vals, vary_feat: adj_val}, train_cols), confidence
        )

        # Prediction at the last competition value (holding other sliders as-is)
        last_pred, _, _ = compute_interval(
            model, XtX_inv, s_resid, n_train,
            build_row({**input_vals, vary_feat: last_val}, train_cols), confidence
        )

        fig_wi = go.Figure()
        # Interval band — drawn first so line sits on top
        fig_wi.add_trace(go.Scatter(
            x=sweep, y=sweep_upper,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ))
        fig_wi.add_trace(go.Scatter(
            x=sweep, y=sweep_lower,
            mode='lines', line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(55, 138, 221, 0.15)',
            name=f'{int(confidence*100)}% prediction interval',
            hoverinfo='skip',
        ))
        fig_wi.add_trace(go.Scatter(
            x=sweep, y=sweep_preds,
            mode='lines+markers',
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=6, color=BLUE),
            name='Predicted distance',
            hovertemplate=(
                f"{FEATURE_LABELS.get(vary_feat, vary_feat)}: %{{x}}<br>"
                f"Predicted: %{{y:.2f}} m<br>"
                f"{int(confidence*100)}% interval: "
                + f"{sweep_lower[0]:.1f}–{sweep_upper[0]:.1f} m"
                if len(sweep_lower) else ""
            ),
        ))
        # Last competition reference — circle on the curve
        fig_wi.add_trace(go.Scatter(
            x=[last_val], y=[last_pred],
            mode='markers',
            marker=dict(size=14, color=CORAL, symbol='circle',
                        line=dict(color='white', width=1.5)),
            name=f'Last competition ({latest["Meet"]})',
            hovertemplate=f"Last comp: %{{x}}<br>Predicted: %{{y:.2f}} m",
        ))
        # Adjusted value — star
        fig_wi.add_trace(go.Scatter(
            x=[adj_val], y=[adj_pred],
            mode='markers',
            marker=dict(size=16, color=TEAL, symbol='star',
                        line=dict(color='white', width=1.5)),
            name='Your adjusted projection',
            hovertemplate="Adjusted: %{y:.2f} m",
        ))
        fig_wi.add_hline(
            y=pb, line_color=AMBER, line_dash='dot', line_width=1.5,
            annotation_text=f"PB  {pb:.2f} m",
            annotation_position="bottom right",
        )
        fig_wi.update_layout(
            title=f"Predicted distance vs {FEATURE_LABELS.get(vary_feat, vary_feat)}",
            xaxis_title=FEATURE_LABELS.get(vary_feat, vary_feat),
            yaxis_title="Predicted Distance (m)",
            height=380,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
            margin=dict(t=70, b=40, l=10, r=20),
        )
        st.plotly_chart(fig_wi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    left2, right2 = st.columns(2)

    with left2:
        lim_lo = min(y_arr.min(), pred_arr.min()) - 0.5
        lim_hi = max(y_arr.max(), pred_arr.max()) + 0.5

        fig_scatter = go.Figure()
        for yr in sorted(df['Year'].unique()):
            mask = df['Year'].values == yr
            fig_scatter.add_trace(go.Scatter(
                x=y_arr[mask], y=pred_arr[mask],
                mode='markers',
                name=str(yr),
                marker=dict(size=10, opacity=0.85),
                text=df.loc[mask, 'Meet'].values,
                hovertemplate='<b>%{text}</b><br>Actual: %{x:.2f} m<br>Predicted: %{y:.2f} m',
            ))
        fig_scatter.add_trace(go.Scatter(
            x=[lim_lo, lim_hi], y=[lim_lo, lim_hi],
            mode='lines',
            line=dict(color=CORAL, dash='dash', width=1.5),
            name='Perfect prediction',
        ))
        fig_scatter.update_layout(
            title='Predicted vs Actual — by Year',
            xaxis_title='Actual (m)', yaxis_title='Predicted (m)',
            height=420,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with right2:
        coef_df = pd.DataFrame({'feature': train_cols, 'coef': model.coef_})
        coef_df = coef_df.iloc[coef_df['coef'].abs().argsort()]

        fig_coef = go.Figure(go.Bar(
            x=coef_df['coef'], y=coef_df['feature'],
            orientation='h',
            marker_color=[TEAL if v >= 0 else CORAL for v in coef_df['coef']],
            text=coef_df['coef'].round(3),
            textposition='outside',
            hovertemplate='%{y}: %{x:.4f} m per unit',
        ))
        fig_coef.update_layout(
            title='Feature Coefficients (sorted by magnitude)',
            xaxis_title='Effect on predicted distance (m per unit)',
            height=420,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    # Residuals over time with rolling mean
    residuals = y_arr - pred_arr
    res_series = pd.Series(residuals, index=pd.to_datetime(df['Date'])).sort_index()
    win = max(5, len(residuals) // 5)
    rolling = res_series.rolling(win, center=True).mean().dropna()

    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(
        x=df['Date'], y=residuals,
        mode='markers',
        marker=dict(
            color=[BLUE if r >= 0 else CORAL for r in residuals],
            size=10, opacity=0.8,
        ),
        text=df['Meet'],
        hovertemplate='<b>%{text}</b><br>Actual − Predicted: %{y:.2f} m',
        showlegend=False,
    ))
    fig_resid.add_trace(go.Scatter(
        x=rolling.index, y=rolling.values,
        mode='lines',
        line=dict(color=AMBER, width=2.2),
        name=f'{win}-meet rolling mean',
    ))
    fig_resid.add_hline(y=0, line_color=GRAY, line_dash='dash', line_width=1)
    fig_resid.update_layout(
        title='Residuals Over Time  (blue = model undershot, red = model overshot)',
        xaxis_title='Date', yaxis_title='Actual − Predicted (m)',
        height=320,
        legend=dict(orientation='h'),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_resid, use_container_width=True)
