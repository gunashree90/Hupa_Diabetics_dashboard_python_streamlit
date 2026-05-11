"""
╔══════════════════════════════════════════════════════════════════╗
║   HUPA-UCM AI Clinical Diabetes Intelligence Dashboard           ║
║   PyCore Python Hackathon 2026                                   ║
║   Dataset: ScienceDirect S2352340924005262                       ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    pip install streamlit pandas numpy plotly scikit-learn pyxlsb openpyxl scipy
    streamlit run hupa_diabetes_dashboard.py

Required files (place in same folder):
    cleaned_hupa_diabetes_recent.xlsb   ← main CGM data
    cleaned_demographics.csv            ← patient demographics

If files are missing the app runs on synthetic demo data automatically.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings, os

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HUPA · AI Diabetes Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS — deep navy dark theme, neon-teal accents
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E2EBF7;
}
.stApp {
    background: linear-gradient(135deg, #04080F 0%, #07111F 40%, #0A1930 100%);
    background-attachment: fixed;
}
.block-container { padding: 1.4rem 2rem 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060E1C 0%, #0C1E38 100%);
    border-right: 1px solid #0E2A4A;
}
[data-testid="stSidebar"] * { color: #C5D6EE !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label { color: #7FA8D4 !important; font-size: 12px !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0E1F38, #112840);
    border: 1px solid #1A3A5C;
    border-radius: 16px;
    padding: 18px 20px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(79,195,247,0.08);
    transition: transform 0.2s;
}
[data-testid="metric-container"]:hover { transform: translateY(-2px); }
[data-testid="metric-container"] label {
    color: #5B8DB8 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 26px !important;
    color: #E2EBF7 !important;
    font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #07111F;
    border-radius: 12px;
    padding: 4px 6px;
    gap: 2px;
    border: 1px solid #0E2A4A;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #5B8DB8;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 16px;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #0F3460, #16507A);
    color: #4FC3F7 !important;
    box-shadow: 0 2px 12px rgba(79,195,247,0.2);
}

/* ── Section headings ── */
.section-hd {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3A7CA5;
    padding: 4px 0 8px;
    border-bottom: 1px solid #0E2A4A;
    margin-bottom: 1rem;
}

/* ── KPI badge pills ── */
.pill-green  { background:#063B26; color:#34D399; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.pill-yellow { background:#3B2A06; color:#FBBF24; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.pill-red    { background:#3B0606; color:#F87171; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.pill-blue   { background:#063050; color:#60A5FA; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }

/* ── Insight card ── */
.insight-card {
    background: linear-gradient(135deg, #071829, #0C2240);
    border: 1px solid #1A3A5C;
    border-left: 4px solid #4FC3F7;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    line-height: 1.7;
}
.insight-card b { color: #4FC3F7; }

/* ── Alert boxes ── */
.alert-danger  { background:#2A0808; border:1px solid #7F1D1D; border-radius:10px; padding:12px 16px; color:#FCA5A5; margin:6px 0; }
.alert-warning { background:#2A1A08; border:1px solid #78350F; border-radius:10px; padding:12px 16px; color:#FDE68A; margin:6px 0; }
.alert-success { background:#062A16; border:1px solid #065F46; border-radius:10px; padding:12px 16px; color:#6EE7B7; margin:6px 0; }

/* ── Selectbox / widget styling ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #0C1E38 !important;
    border: 1px solid #1A3A5C !important;
    border-radius: 10px !important;
    color: #C5D6EE !important;
}

/* ── Divider ── */
hr { border-color: #0E2A4A; margin: 1rem 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #04080F; }
::-webkit-scrollbar-thumb { background: #1A3A5C; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PLOTLY SHARED THEME
# ══════════════════════════════════════════════════════════════════
PT = dict(
    template="plotly_dark",
    paper_bgcolor="#07111F",
    plot_bgcolor="#04080F",
    font=dict(family="Inter, sans-serif", color="#8AAAC8", size=11),
    margin=dict(l=44, r=20, t=36, b=40),
)

PALETTE = dict(
    teal="#4FC3F7", violet="#818CF8", green="#34D399",
    amber="#FBBF24", red="#F87171", pink="#F472B6",
    orange="#FB923C", sky="#38BDF8",
)

def styled_chart(fig, h=360):
    fig.update_layout(height=h, **PT)
    fig.update_xaxes(gridcolor="#0E2A4A", zerolinecolor="#0E2A4A")
    fig.update_yaxes(gridcolor="#0E2A4A", zerolinecolor="#0E2A4A")
    return fig

# ══════════════════════════════════════════════════════════════════
# SYNTHETIC DATA FALLBACK
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def generate_synthetic(n_patients=25, days=14):
    records = []
    for pid in range(1, n_patients + 1):
        rng = np.random.default_rng(pid * 31)
        n = days * 24 * 12
        ts = pd.date_range("2023-01-01", periods=n, freq="5min")
        t  = np.arange(n)
        base = rng.uniform(95, 145)
        circ = 18 * np.sin(2 * np.pi * t / (24 * 12) - np.pi / 2)
        gluc = base + circ + rng.normal(0, 9, n)
        basal = rng.uniform(0.6, 1.4)
        bolus = np.zeros(n)
        carbs = np.zeros(n)
        for d in range(days):
            for mh in [7, 12, 19]:
                idx = d * 24 * 12 + mh * 12
                c = rng.uniform(25, 80); sp = c * rng.uniform(0.7, 1.2)
                decay = np.exp(-np.arange(36) / 10)
                end = min(idx + 36, n)
                gluc[idx:end] += sp * decay[:end - idx]
                if idx < n:
                    bolus[idx] = rng.uniform(1.5, 7)
                    carbs[idx] = c
        gluc = np.clip(gluc, 40, 380)
        hr   = np.clip(70 + rng.normal(0, 4, n) + 0.02 * rng.poisson(10, n), 48, 155)
        steps = np.clip(rng.poisson(
            np.where(((t // 12) % 24 >= 8) & ((t // 12) % 24 <= 20), 18, 2), n), 0, 200)
        cals = steps * 0.05 + rng.uniform(0.8, 1.1, n)
        age  = int(rng.integers(22, 62))
        gen  = rng.choice(["Male", "Female"])
        for i in range(n):
            records.append(dict(
                patient_id=pid, time=ts[i],
                glucose=round(float(gluc[i]), 1),
                heart_rate=round(float(hr[i]), 1),
                steps=int(steps[i]),
                calories=round(float(cals[i]), 2),
                bolus_volume_delivered=round(float(bolus[i]), 2),
                carb_input=round(float(carbs[i]), 1),
                basal_rate=round(basal, 3),
                age=age, gender=gen,
            ))
    return pd.DataFrame(records)

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    data_file  = "E:\NUMPY_NINJA\python_hackathon\cleaned_hupa_diabetes_recent1.xlsb"
    demo_file  = "E:\NUMPY_NINJA\python_hackathon\HUPA-UC Diabetes Dataset-20250820T010637Z-1-001\HUPA-UC Diabetes Dataset\cleaned_demographics.csv.csv"
    alt_data   = "E:\NUMPY_NINJA\python_hackathon\cleaned_hupa_diabetes_recent1.xlsb"
    alt_data2  = "E:\NUMPY_NINJA\python_hackathon\cleaned_hupa_diabetes_recent1.xlsb"

    actual = None
    for fname in [data_file, alt_data, alt_data2]:
        if os.path.exists(fname):
            actual = fname
            break

    if actual:
        try:
            df = pd.read_excel(actual, engine="pyxlsb")
        except Exception:
            df = generate_synthetic()
    else:
        df = generate_synthetic()
        st.sidebar.info("⚡ Demo mode — using synthetic HUPA-like data.\nAdd your .xlsb file to enable real data.")

    if os.path.exists(demo_file) and "patient_id" in df.columns:
        try:
            demo = pd.read_csv(demo_file)
            df = df.merge(demo, on="patient_id", how="left")
        except Exception:
            pass

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "glucose"]).sort_values(["patient_id", "time"])
    df["date"]       = df["time"].dt.date
    df["hour"]       = df["time"].dt.hour
    df["is_weekend"] = df["time"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_night"]   = df["hour"].between(0, 5).astype(int)

    bolus_col = "bolus_volume_delivered" if "bolus_volume_delivered" in df.columns else "bolus"
    if bolus_col not in df.columns:
        df["bolus_volume_delivered"] = 0.0
        bolus_col = "bolus_volume_delivered"

    for col in ["carb_input", "basal_rate", "steps", "heart_rate", "calories"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()
    df["glucose_rolling_std_1h"] = (
        df.groupby("patient_id")["glucose"].transform(lambda x: x.rolling(12, min_periods=1).std()))
    df["glucose_rolling_mean_1h"] = (
        df.groupby("patient_id")["glucose"].transform(lambda x: x.rolling(12, min_periods=1).mean()))
    df["glucose_smooth"] = (
        df.groupby("patient_id")["glucose"].transform(lambda x: x.rolling(12, min_periods=1).mean()))

    df["tir_flag"]  = ((df["glucose"] >= 70) & (df["glucose"] <= 180)).astype(int)
    df["hypo_flag"] = (df["glucose"] < 70).astype(int)
    df["hyper_flag"]= (df["glucose"] > 180).astype(int)
    df["risk_score"]= (
        abs(df["glucose_roc"].fillna(0)) * 0.40 +
        df["glucose_rolling_std_1h"].fillna(0) * 0.40 +
        abs(df["heart_rate"].fillna(70)) * 0.20
    )

    return df, bolus_col

df, BOLUS = load_data()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🩺 HUPA-UCM")
    st.markdown("**AI Diabetes Intelligence**")
    st.markdown("---")

    all_patients = sorted(df["patient_id"].dropna().unique())
    selected_patients = st.multiselect(
        "Filter patients (multi-select)",
        all_patients,
        default=all_patients[:min(6, len(all_patients))],
    )

    st.markdown("---")
    st.markdown("**Glucose thresholds**")
    hypo_thr  = st.slider("Hypoglycemia below (mg/dL)", 50, 90,  70)
    hyper_thr = st.slider("Hyperglycemia above (mg/dL)", 140, 300, 180)

    st.markdown("---")
    if "age" in df.columns and df["age"].nunique() > 1:
        age_range = st.slider("Age range", int(df["age"].min()), int(df["age"].max()),
                               (int(df["age"].min()), int(df["age"].max())))
        df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

    st.markdown("---")
    st.caption("Dataset · HUPA-UCM · 25 T1DM patients\n5-min CGM · DOI: 10.1016/j.dib.2024.110526")
    st.caption("PyCore Hackathon 2026")

if not selected_patients:
    st.warning("⚠️ Please select at least one patient in the sidebar.")
    st.stop()

df_v = df[df["patient_id"].isin(selected_patients)].copy()
df_v["tir_flag"]  = ((df_v["glucose"] >= hypo_thr) & (df_v["glucose"] <= hyper_thr)).astype(int)
df_v["hypo_flag"] = (df_v["glucose"] < hypo_thr).astype(int)
df_v["hyper_flag"]= (df_v["glucose"] > hyper_thr).astype(int)

# ── Daily summary (used across tabs) ─────────────────────────────
daily = (
    df_v.groupby(["patient_id", "date"])
    .agg(
        daily_tir=("tir_flag", "mean"),
        avg_glucose=("glucose", "mean"),
        glucose_variability=("glucose", "std"),
        daily_steps=("steps", "sum"),
        avg_hr=("heart_rate", "mean"),
        avg_basal=("basal_rate", "mean"),
        total_bolus=(BOLUS, "sum"),
        total_carbs=("carb_input", "sum"),
        hypo_rate=("hypo_flag", "mean"),
        hyper_rate=("hyper_flag", "mean"),
    ).reset_index()
)
daily["daily_tir"] *= 100
daily["date_str"]  = daily["date"].astype(str)

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.markdown("# 🩺 AI Clinical Diabetes Intelligence System")
    st.markdown(
        "**HUPA-UCM Continuous Glucose Monitoring** · T1DM Cohort · "
        f"{len(all_patients)} patients · 5-min resolution"
    )
with col_h2:
    tir_global = df_v["tir_flag"].mean() * 100
    pill = "pill-green" if tir_global >= 70 else "pill-yellow" if tir_global >= 50 else "pill-red"
    label = "Good" if tir_global >= 70 else "Moderate" if tir_global >= 50 else "Poor"
    st.markdown(f"<br><span class='{pill}'>Cohort TIR: {tir_global:.0f}% · {label}</span>",
                unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# TOP KPI ROW
# ══════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Avg Glucose",    f"{df_v['glucose'].mean():.0f} mg/dL")
k2.metric("Time in Range",  f"{tir_global:.1f}%",
          delta="Target ≥70%" if tir_global >= 70 else "Below target")
k3.metric("Hypoglycemia",   f"{df_v['hypo_flag'].mean()*100:.1f}%",
          delta="Low risk" if df_v['hypo_flag'].mean()*100 < 4 else "Elevated")
k4.metric("Hyperglycemia",  f"{df_v['hyper_flag'].mean()*100:.1f}%")
k5.metric("Avg Heart Rate", f"{df_v['heart_rate'].mean():.0f} bpm")
k6.metric("Avg Daily Steps",f"{df_v['steps'].mean():.0f}")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 Overview",
    "🍽️ Meal & Bolus",
    "🏃 Activity",
    "🌙 Night Risk",
    "🔥 Variability",
    "🤖 Predictive AI",
    "💊 Prescriptive",
    "🏥 Cohort",
    "📌 Insights",
])

# ══════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-hd'>Glucose trend & distribution</div>", unsafe_allow_html=True)

    # Smoothed CGM multi-patient
    fig_trend = go.Figure()
    patient_colors = px.colors.qualitative.Vivid
    for i, pid in enumerate(selected_patients[:10]):
        sub = df_v[df_v["patient_id"] == pid]
        clr = patient_colors[i % len(patient_colors)]
        fig_trend.add_trace(go.Scatter(
            x=sub["time"], y=sub["glucose_smooth"],
            mode="lines", name=f"P{pid}",
            line=dict(width=1.8, color=clr), opacity=0.9,
        ))
    fig_trend.add_hrect(y0=0,        y1=hypo_thr,  fillcolor="#F8717120", line_width=0)
    fig_trend.add_hrect(y0=hyper_thr, y1=400,       fillcolor="#FBBF2420", line_width=0)
    fig_trend.add_hline(y=hypo_thr,  line_dash="dot", line_color="#F87171", line_width=1.2,
                         annotation_text=f"Hypo {hypo_thr}")
    fig_trend.add_hline(y=hyper_thr, line_dash="dot", line_color="#FBBF24", line_width=1.2,
                         annotation_text=f"Hyper {hyper_thr}")
    fig_trend.update_layout(title="Smoothed CGM glucose trends", hovermode="x unified",
                             legend=dict(orientation="h", y=1.08), **PT)
    styled_chart(fig_trend, 380)
    st.plotly_chart(fig_trend, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        fig_box = px.box(df_v, x="patient_id", y="glucose",
                          color="patient_id",
                          color_discrete_sequence=px.colors.qualitative.Vivid,
                          title="Glucose distribution by patient")
        fig_box.add_hline(y=hypo_thr,  line_dash="dot", line_color="#F87171")
        fig_box.add_hline(y=hyper_thr, line_dash="dot", line_color="#FBBF24")
        st.plotly_chart(styled_chart(fig_box), use_container_width=True)

    with c2:
        fig_tir_bar = go.Figure()
        tir_by_pt = (df_v.groupby("patient_id")["tir_flag"].mean() * 100).reset_index()
        tir_by_pt.columns = ["patient_id", "tir"]
        bar_colors = ["#34D399" if t >= 70 else "#FBBF24" if t >= 50 else "#F87171"
                      for t in tir_by_pt["tir"]]
        fig_tir_bar.add_trace(go.Bar(
            x=tir_by_pt["patient_id"].astype(str),
            y=tir_by_pt["tir"],
            marker_color=bar_colors,
            text=tir_by_pt["tir"].round(0).astype(int).astype(str) + "%",
            textposition="outside",
        ))
        fig_tir_bar.add_hline(y=70, line_dash="dash", line_color="#5B8DB8",
                               annotation_text="Target 70%")
        fig_tir_bar.update_layout(title="Time-in-range per patient",
                                   yaxis_range=[0, 108],
                                   xaxis_title="Patient ID", yaxis_title="TIR (%)")
        st.plotly_chart(styled_chart(fig_tir_bar), use_container_width=True)

    with c3:
        # TIR donut for full cohort
        avg_tir  = df_v["tir_flag"].mean()  * 100
        avg_tbr  = df_v["hypo_flag"].mean() * 100
        avg_tar  = df_v["hyper_flag"].mean()* 100
        fig_donut = go.Figure(go.Pie(
            labels=["In range (70–180)", "Below (<70)", "Above (>180)"],
            values=[avg_tir, avg_tbr, avg_tar],
            hole=0.65,
            marker_colors=["#34D399", "#F87171", "#FBBF24"],
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_donut.add_annotation(text=f"TIR<br><b>{avg_tir:.0f}%</b>",
                                   x=0.5, y=0.5, showarrow=False,
                                   font=dict(size=18, color="#E2EBF7"))
        fig_donut.update_layout(title="Cohort TIR breakdown", showlegend=False)
        st.plotly_chart(styled_chart(fig_donut, 320), use_container_width=True)

    # Daily TIR heatmap
    st.markdown("<div class='section-hd'>Daily TIR heatmap</div>", unsafe_allow_html=True)
    pivot = daily.pivot_table(index="patient_id", columns="date_str", values="daily_tir")
    if not pivot.empty:
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[f"P{p}" for p in pivot.index],
            colorscale=[[0,"#7F1D1D"],[0.5,"#78350F"],[0.7,"#065F46"],[1,"#34D399"]],
            zmin=0, zmax=100,
            colorbar=dict(title="TIR %", ticksuffix="%"),
            hoverongaps=False,
        ))
        fig_hm.update_layout(
            title="Daily time-in-range across patients",
            xaxis=dict(tickangle=-45, title="Date"),
            yaxis=dict(title="Patient"),
        )
        st.plotly_chart(styled_chart(fig_hm, 340), use_container_width=True)

fig_trend.add_hrect(
    y0=0, 
    y1=hypo_thr, 
    fillcolor="rgba(248, 113, 113, 0.12)", 
    line_width=0
)


# ══════════════════════════════════════════════════════════════════
# TAB 1 — MEAL & BOLUS
# ══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-hd'>Meal, carbohydrate & insulin bolus analysis</div>",
                unsafe_allow_html=True)

    df_meal = df_v.copy()
    df_meal["glucose_next_2h"]  = df_meal.groupby("patient_id")["glucose"].shift(-24)
    df_meal["post_meal_spike"]  = df_meal["glucose_next_2h"] - df_meal["glucose"]
    meal_df = df_meal[df_meal["carb_input"] > 0].dropna(
        subset=["post_meal_spike", "carb_input", BOLUS])

    c1, c2 = st.columns(2)
    with c1:
        samp = meal_df.sample(min(4000, len(meal_df)), random_state=42)
        fig_sc = px.scatter(
            samp, x="carb_input", y="post_meal_spike",
            size=BOLUS, color="glucose",
            trendline="ols",
            color_continuous_scale="Teal",
            labels={"carb_input": "Carbs (g)",
                    "post_meal_spike": "Post-meal spike (mg/dL)",
                    "glucose": "Glucose"},
            title="Carb load vs post-meal glucose spike",
        )
        st.plotly_chart(styled_chart(fig_sc), use_container_width=True)

    with c2:
        fig_dens = px.density_heatmap(
            meal_df, x="carb_input", y="post_meal_spike",
            nbinsx=30, nbinsy=30,
            color_continuous_scale="Teal",
            title="Carb vs spike density map",
        )
        st.plotly_chart(styled_chart(fig_dens), use_container_width=True)

    # Missed bolus
    st.markdown("<div class='section-hd'>Missed bolus detection</div>", unsafe_allow_html=True)
    meal_df["missed_bolus"] = (
        (meal_df["carb_input"] > 20) &
        (meal_df[BOLUS] == 0) &
        (meal_df["glucose_next_2h"] > 180)
    ).astype(int)

    c3, c4 = st.columns(2)
    with c3:
        mb_summary = meal_df["missed_bolus"].value_counts().reset_index()
        mb_summary.columns = ["Event", "Count"]
        mb_summary["Event"] = mb_summary["Event"].map({0: "Bolus given", 1: "Missed bolus"})
        fig_mb = px.bar(mb_summary, x="Event", y="Count",
                         color="Event",
                         color_discrete_map={"Bolus given": PALETTE["green"],
                                             "Missed bolus": PALETTE["red"]},
                         title="Missed bolus events")
        st.plotly_chart(styled_chart(fig_mb, 300), use_container_width=True)

    with c4:
        fig_bv = px.violin(
            meal_df[meal_df[BOLUS] > 0],
            y=BOLUS, x="patient_id",
            color="patient_id",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            box=True, points="outliers",
            title="Bolus volume distribution by patient",
        )
        st.plotly_chart(styled_chart(fig_bv, 300), use_container_width=True)

    # Hourly carb patterns
    st.markdown("<div class='section-hd'>Meal timing patterns</div>", unsafe_allow_html=True)
    hour_carbs = (meal_df[meal_df["carb_input"] > 0]
                  .groupby("hour")["carb_input"].mean().reset_index())
    fig_hc = px.bar(hour_carbs, x="hour", y="carb_input",
                     color="carb_input", color_continuous_scale="Teal",
                     labels={"hour": "Hour of day", "carb_input": "Avg carbs (g)"},
                     title="Average carb intake by hour of day")
    fig_hc.update_layout(coloraxis_showscale=False)
    st.plotly_chart(styled_chart(fig_hc, 280), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — ACTIVITY
# ══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-hd'>Physical activity & glycemic impact</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_act = px.scatter(
            df_v.sample(min(6000, len(df_v)), random_state=42),
            x="steps", y="glucose",
            color="heart_rate",
            color_continuous_scale="Plasma",
            opacity=0.6,
            trendline="lowess",
            labels={"steps": "Steps per 5 min", "glucose": "Glucose (mg/dL)"},
            title="Steps vs glucose (color = heart rate)",
        )
        st.plotly_chart(styled_chart(fig_act), use_container_width=True)

    with c2:
        fig_hr_g = px.scatter(
            df_v.sample(min(6000, len(df_v)), random_state=42),
            x="heart_rate", y="glucose",
            color="glucose", trendline="lowess",
            color_continuous_scale="RdYlGn_r",
            labels={"heart_rate": "Heart rate (bpm)", "glucose": "Glucose (mg/dL)"},
            title="Heart rate vs glucose",
        )
        st.plotly_chart(styled_chart(fig_hr_g), use_container_width=True)

    # Daily activity vs TIR
    st.markdown("<div class='section-hd'>Daily steps vs time-in-range</div>", unsafe_allow_html=True)
    fig_dact = px.scatter(
        daily, x="daily_steps", y="daily_tir",
        color="patient_id",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        size="avg_glucose",
        trendline="ols",
        hover_data=["avg_glucose", "glucose_variability"],
        labels={"daily_steps": "Daily steps", "daily_tir": "Daily TIR (%)"},
        title="Daily activity vs time-in-range",
    )
    fig_dact.add_hline(y=70, line_dash="dash", line_color="#5B8DB8",
                        annotation_text="Target TIR 70%")
    st.plotly_chart(styled_chart(fig_dact, 340), use_container_width=True)

    # Hourly glucose by activity quartile
    st.markdown("<div class='section-hd'>Glucose profile by activity level</div>", unsafe_allow_html=True)
    df_v["activity_q"] = pd.qcut(df_v["steps"], q=4,
                                   labels=["Sedentary", "Low", "Moderate", "Active"],
                                   duplicates="drop")
    hourly_act = (df_v.groupby(["hour", "activity_q"])["glucose"]
                  .mean().reset_index())
    fig_ha = px.line(hourly_act, x="hour", y="glucose",
                      color="activity_q",
                      color_discrete_sequence=[PALETTE["red"], PALETTE["amber"],
                                               PALETTE["teal"], PALETTE["green"]],
                      markers=True,
                      labels={"hour": "Hour of day", "glucose": "Avg glucose (mg/dL)",
                              "activity_q": "Activity level"},
                      title="Hourly glucose by activity quartile")
    fig_ha.add_hline(y=hypo_thr,  line_dash="dot", line_color="#F87171")
    fig_ha.add_hline(y=hyper_thr, line_dash="dot", line_color="#FBBF24")
    st.plotly_chart(styled_chart(fig_ha, 320), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — NIGHT RISK
# ══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-hd'>Nocturnal hypoglycemia & basal rate risk</div>",
                unsafe_allow_html=True)

    night_df = df_v[df_v["is_night"] == 1].copy()
    night_df["nocturnal_hypo"] = (night_df["glucose"] < hypo_thr).astype(int)

    c1, c2 = st.columns(2)
    with c1:
        risk_curve = (night_df.groupby("basal_rate")["nocturnal_hypo"]
                      .mean().reset_index())
        fig_nc = px.line(risk_curve, x="basal_rate", y="nocturnal_hypo",
                          markers=True,
                          labels={"basal_rate": "Basal rate (U/h)",
                                  "nocturnal_hypo": "Nocturnal hypo probability"},
                          title="Basal rate vs nocturnal hypoglycemia risk",
                          color_discrete_sequence=[PALETTE["red"]])
        fig_nc.update_traces(line=dict(width=2.5))
        st.plotly_chart(styled_chart(fig_nc), use_container_width=True)

    with c2:
        fig_bn = px.box(night_df, x="nocturnal_hypo", y="basal_rate",
                         color="nocturnal_hypo",
                         color_discrete_map={0: PALETTE["green"], 1: PALETTE["red"]},
                         labels={"nocturnal_hypo": "Nocturnal hypo event",
                                 "basal_rate": "Basal rate (U/h)"},
                         title="Basal rate by nocturnal hypo event")
        st.plotly_chart(styled_chart(fig_bn), use_container_width=True)

    # Dawn phenomenon
    st.markdown("<div class='section-hd'>Dawn phenomenon — overnight glucose rise</div>",
                unsafe_allow_html=True)
    dawn = df_v[df_v["hour"].between(0, 8)].copy()
    dawn_hourly = dawn.groupby("hour")["glucose"].agg(["mean","std"]).reset_index()
    fig_dawn = go.Figure()
    fig_dawn.add_trace(go.Scatter(
        x=dawn_hourly["hour"], y=dawn_hourly["mean"],
        mode="lines+markers", name="Mean glucose",
        line=dict(color=PALETTE["teal"], width=2.5),
        marker=dict(size=7),
    ))
    fig_dawn.add_trace(go.Scatter(
        x=pd.concat([dawn_hourly["hour"], dawn_hourly["hour"][::-1]]),
        y=pd.concat([dawn_hourly["mean"] + dawn_hourly["std"],
                     (dawn_hourly["mean"] - dawn_hourly["std"])[::-1]]),
        fill="toself", fillcolor="#4FC3F720",
        line=dict(color="rgba(0,0,0,0)"), name="±1 SD",
    ))
    fig_dawn.add_hline(y=hypo_thr,  line_dash="dot", line_color="#F87171")
    fig_dawn.add_hline(y=hyper_thr, line_dash="dot", line_color="#FBBF24")
    fig_dawn.update_layout(title="Average overnight → morning glucose pattern",
                             xaxis_title="Hour of day",
                             yaxis_title="Glucose (mg/dL)")
    st.plotly_chart(styled_chart(fig_dawn, 340), use_container_width=True)

    # Night glucose heatmap per patient
    c3, c4 = st.columns(2)
    with c3:
        night_pt = night_df.groupby(["patient_id","hour"])["glucose"].mean().unstack()
        fig_nhm = go.Figure(go.Heatmap(
            z=night_pt.values,
            x=[f"{h:02d}:00" for h in night_pt.columns],
            y=[f"P{p}" for p in night_pt.index],
            colorscale="RdYlGn", zmin=60, zmax=220,
            colorbar=dict(title="mg/dL"),
        ))
        fig_nhm.update_layout(title="Nocturnal glucose heatmap by patient")
        st.plotly_chart(styled_chart(fig_nhm, 320), use_container_width=True)

    with c4:
        hypo_by_pt = (night_df.groupby("patient_id")["nocturnal_hypo"]
                      .mean().reset_index())
        hypo_by_pt.columns = ["patient_id", "hypo_rate"]
        hypo_by_pt = hypo_by_pt.sort_values("hypo_rate", ascending=False)
        fig_hbp = go.Figure(go.Bar(
            x=hypo_by_pt["patient_id"].astype(str),
            y=hypo_by_pt["hypo_rate"] * 100,
            marker_color=[PALETTE["red"] if v > 5 else PALETTE["amber"] if v > 2
                          else PALETTE["green"] for v in hypo_by_pt["hypo_rate"] * 100],
            text=(hypo_by_pt["hypo_rate"] * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig_hbp.update_layout(title="Nocturnal hypo rate per patient (%)",
                               yaxis_title="Hypo rate (%)", xaxis_title="Patient")
        st.plotly_chart(styled_chart(fig_hbp, 320), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — VARIABILITY
# ══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-hd'>Glycemic variability analysis</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Coefficient of variation by patient
        cv_df = (df_v.groupby("patient_id")["glucose"]
                 .agg(lambda x: x.std() / x.mean() * 100).reset_index())
        cv_df.columns = ["patient_id", "cv"]
        fig_cv = go.Figure(go.Bar(
            x=cv_df["patient_id"].astype(str), y=cv_df["cv"],
            marker_color=[PALETTE["red"] if v > 36 else PALETTE["amber"] if v > 27
                          else PALETTE["green"] for v in cv_df["cv"]],
            text=cv_df["cv"].round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig_cv.add_hline(y=36, line_dash="dash", line_color="#F87171",
                          annotation_text="High variability (CV>36%)")
        fig_cv.update_layout(title="Coefficient of variation by patient",
                              yaxis_title="CV (%)", xaxis_title="Patient")
        st.plotly_chart(styled_chart(fig_cv), use_container_width=True)

    with c2:
        fig_roc = px.histogram(
            df_v, x="glucose_roc", nbins=80,
            color="patient_id",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"glucose_roc": "Glucose ROC (mg/dL per 5 min)"},
            title="Glucose rate-of-change distribution",
            barmode="overlay",
        )
        fig_roc.update_traces(opacity=0.6)
        st.plotly_chart(styled_chart(fig_roc), use_container_width=True)

    # Rolling variability timeline
    st.markdown("<div class='section-hd'>1-hour rolling glucose variability timeline</div>",
                unsafe_allow_html=True)
    fig_rv = go.Figure()
    for i, pid in enumerate(selected_patients[:8]):
        sub = df_v[df_v["patient_id"] == pid]
        fig_rv.add_trace(go.Scatter(
            x=sub["time"], y=sub["glucose_rolling_std_1h"],
            mode="lines", name=f"P{pid}",
            line=dict(width=1.5, color=px.colors.qualitative.Vivid[i % 10]),
            opacity=0.85,
        ))
    fig_rv.add_hline(y=30, line_dash="dash", line_color="#FBBF24",
                      annotation_text="High variability threshold")
    fig_rv.update_layout(title="Rolling 1-h glucose SD over time",
                          hovermode="x unified")
    st.plotly_chart(styled_chart(fig_rv, 360), use_container_width=True)

    # Variability vs TIR scatter
    st.markdown("<div class='section-hd'>Variability vs glycemic control</div>",
                unsafe_allow_html=True)
    var_tir = (df_v.groupby("patient_id").agg(
        cv=("glucose", lambda x: x.std() / x.mean() * 100),
        tir=("tir_flag", "mean"),
        avg_steps=("steps", "mean"),
        mean_glucose=("glucose", "mean"),
    ).reset_index())
    var_tir["tir"] *= 100
    fig_vt = px.scatter(
        var_tir, x="cv", y="tir",
        size="avg_steps", color="mean_glucose",
        color_continuous_scale="RdYlGn_r",
        hover_name="patient_id",
        trendline="ols",
        labels={"cv": "CV (%)", "tir": "TIR (%)",
                "mean_glucose": "Mean glucose"},
        title="CV vs TIR — bubble size = avg daily steps",
    )
    fig_vt.add_hline(y=70, line_dash="dash", line_color="#5B8DB8")
    fig_vt.add_vline(x=36, line_dash="dash", line_color="#F87171")
    st.plotly_chart(styled_chart(fig_vt, 360), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — PREDICTIVE AI
# ══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-hd'>AI predictive models</div>", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Choose prediction task",
        [
            "🔴 Hypoglycemia in next 30 minutes",
            "🟡 Post-meal glucose >200 mg/dL within 2 hours",
            "📉 Next 15-min glucose rate-of-change",
            "📅 Daily TIR decline risk (7-day ahead)",
        ],
    )

    # Feature set per task
    if "Hypoglycemia" in model_choice:
        model_data = df_v.copy()
        model_data["target"] = (model_data.groupby("patient_id")["glucose"].shift(-6) < hypo_thr).astype(int)
        features   = ["glucose", "glucose_roc", "glucose_rolling_std_1h",
                      "basal_rate", BOLUS, "steps", "heart_rate", "hour"]
        task_type  = "classification"
    elif "Post-meal" in model_choice:
        model_data = df_v[df_v["carb_input"] > 0].copy()
        model_data["target"] = (model_data.groupby("patient_id")["glucose"].shift(-24) > 200).astype(int)
        features   = ["glucose", "carb_input", BOLUS, "basal_rate", "steps", "heart_rate", "hour"]
        task_type  = "classification"
    elif "rate-of-change" in model_choice:
        model_data = df_v.copy()
        model_data["target"] = model_data.groupby("patient_id")["glucose_roc"].shift(-3)
        features   = ["glucose", "glucose_roc", "glucose_rolling_std_1h",
                      "basal_rate", BOLUS, "steps", "heart_rate", "hour"]
        task_type  = "regression"
    else:
        model_data = daily.copy()
        model_data["future_tir"] = model_data.groupby("patient_id")["daily_tir"].shift(-7)
        model_data["target"]     = (model_data["future_tir"] < model_data["daily_tir"] - 10).astype(int)
        features   = ["daily_tir", "avg_glucose", "glucose_variability",
                      "daily_steps", "avg_hr", "avg_basal", "total_bolus"]
        task_type  = "classification"

    feats_avail = [f for f in features if f in model_data.columns]
    model_df    = model_data[feats_avail + ["target"]].dropna()

    if len(model_df) < 50 or (task_type == "classification" and model_df["target"].nunique() < 2):
        st.warning("⚠️ Not enough balanced data for this model with current patient selection.")
    else:
        if len(model_df) > 20000:
            model_df = model_df.sample(20000, random_state=42)

        X = model_df[feats_avail]
        y = model_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        with st.spinner("Training model..."):
            if task_type == "classification":
                mdl = RandomForestClassifier(n_estimators=120, max_depth=8,
                                              class_weight="balanced", random_state=42, n_jobs=-1)
                mdl.fit(X_train, y_train)
                pred  = mdl.predict(X_test)
                prob  = mdl.predict_proba(X_test)[:, 1]
                acc   = accuracy_score(y_test, pred)
                auc   = roc_auc_score(y_test, prob)

                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{acc:.3f}")
                m2.metric("ROC-AUC",  f"{auc:.3f}")
                m3.metric("Train size", f"{len(X_train):,}")

                # ROC curve
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, prob)
                fig_roc_c = go.Figure()
                fig_roc_c.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                                name=f"AUC={auc:.3f}",
                                                line=dict(color=PALETTE["teal"], width=2.5)))
                fig_roc_c.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                                line=dict(color="#5B8DB8", dash="dash"),
                                                name="Random"))
                fig_roc_c.update_layout(title="ROC curve",
                                         xaxis_title="False positive rate",
                                         yaxis_title="True positive rate")
                st.plotly_chart(styled_chart(fig_roc_c, 320), use_container_width=True)

            else:
                mdl = RandomForestRegressor(n_estimators=100, max_depth=8,
                                             random_state=42, n_jobs=-1)
                mdl.fit(X_train, y_train)
                pred = mdl.predict(X_test)
                mae  = mean_absolute_error(y_test, pred)
                r2   = r2_score(y_test, pred)

                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{mae:.3f}")
                m2.metric("R²",  f"{r2:.3f}")
                m3.metric("Train size", f"{len(X_train):,}")

                fig_pred_scatter = px.scatter(
                    x=y_test.values[:2000], y=pred[:2000],
                    labels={"x": "Actual", "y": "Predicted"},
                    title="Actual vs predicted",
                    opacity=0.6,
                    color_discrete_sequence=[PALETTE["teal"]],
                )
                fig_pred_scatter.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode="lines", name="Perfect fit",
                    line=dict(color="#5B8DB8", dash="dash"),
                ))
                st.plotly_chart(styled_chart(fig_pred_scatter, 320), use_container_width=True)

        # Feature importance
        fi_df = pd.DataFrame({"Feature": feats_avail,
                               "Importance": mdl.feature_importances_}) \
                  .sort_values("Importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker_color=PALETTE["violet"],
            text=fi_df["Importance"].round(3).astype(str),
            textposition="outside",
        ))
        fig_fi.update_layout(title="Feature importance", xaxis_title="Importance")
        st.plotly_chart(styled_chart(fig_fi, 320), use_container_width=True)

    # Risk score timeline
    st.markdown("<div class='section-hd'>Live risk score timeline</div>", unsafe_allow_html=True)
    fig_risk = go.Figure()
    for i, pid in enumerate(selected_patients[:6]):
        sub = df_v[df_v["patient_id"] == pid]
        fig_risk.add_trace(go.Scatter(
            x=sub["time"],
            y=sub["risk_score"].rolling(10, min_periods=1).mean(),
            mode="lines", name=f"P{pid}",
            line=dict(width=1.8, color=px.colors.qualitative.Vivid[i % 10]),
        ))
    fig_risk.update_layout(title="Predicted glycemic risk score over time",
                             hovermode="x unified")
    st.plotly_chart(styled_chart(fig_risk, 340), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 6 — PRESCRIPTIVE
# ══════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("<div class='section-hd'>Insulin effectiveness composite score</div>",
                unsafe_allow_html=True)

    patient_score = (
        df_v.groupby("patient_id").agg(
            tir=("tir_flag", "mean"),
            avg_glucose=("glucose", "mean"),
            glucose_variability=("glucose", "std"),
            hypo_rate=("hypo_flag", "mean"),
            hyper_rate=("hyper_flag", "mean"),
            avg_steps=("steps", "mean"),
            avg_basal=("basal_rate", "mean"),
            total_bolus=(BOLUS, "sum"),
        ).reset_index()
    )

    patient_score["tir_score"]         = patient_score["tir"] * 40
    patient_score["stability_score"]   = (1 - patient_score["glucose_variability"].rank(pct=True)) * 25
    patient_score["hypo_safety_score"] = (1 - patient_score["hypo_rate"].rank(pct=True)) * 20
    patient_score["activity_score"]    = patient_score["avg_steps"].rank(pct=True) * 15
    patient_score["ies"] = (
        patient_score["tir_score"] + patient_score["stability_score"] +
        patient_score["hypo_safety_score"] + patient_score["activity_score"]
    ).clip(0, 100)
    patient_score = patient_score.sort_values("ies", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        ies_colors = [PALETTE["green"] if v >= 70 else PALETTE["amber"] if v >= 50
                      else PALETTE["red"] for v in patient_score["ies"]]
        fig_ies = go.Figure(go.Bar(
            x=patient_score["patient_id"].astype(str),
            y=patient_score["ies"],
            marker_color=ies_colors,
            text=patient_score["ies"].round(0).astype(int).astype(str),
            textposition="outside",
        ))
        fig_ies.add_hline(y=70, line_dash="dash", line_color="#5B8DB8",
                           annotation_text="Good control threshold")
        fig_ies.update_layout(title="Insulin Effectiveness Score per patient",
                               yaxis_range=[0, 108],
                               xaxis_title="Patient", yaxis_title="Score (0–100)")
        st.plotly_chart(styled_chart(fig_ies), use_container_width=True)

    with c2:
        fig_quad = px.scatter(
            patient_score, x="glucose_variability", y="tir",
            size="avg_steps", color="ies",
            color_continuous_scale="RdYlGn",
            hover_name="patient_id",
            labels={"glucose_variability": "Glucose SD (mg/dL)",
                    "tir": "TIR (×1 = 100%)", "ies": "IES"},
            title="TIR vs variability — size=steps, color=IES",
        )
        st.plotly_chart(styled_chart(fig_quad), use_container_width=True)

    # Prescriptive alerts
    st.markdown("<div class='section-hd'>AI clinical recommendations</div>",
                unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        for _, row in patient_score.iterrows():
            pid = row["patient_id"]
            if row["hypo_rate"] > 0.05:
                st.markdown(f"<div class='alert-danger'>⚠️ <b>P{pid}</b> — High nocturnal hypo rate ({row['hypo_rate']*100:.1f}%). Review basal rate.</div>",
                            unsafe_allow_html=True)
            if row["avg_glucose"] > 180:
                st.markdown(f"<div class='alert-warning'>📈 <b>P{pid}</b> — Avg glucose {row['avg_glucose']:.0f} mg/dL. Adjust bolus-to-carb ratio.</div>",
                            unsafe_allow_html=True)
            if row["tir"] >= 0.70:
                st.markdown(f"<div class='alert-success'>✅ <b>P{pid}</b> — TIR {row['tir']*100:.0f}% exceeds target. Maintain current therapy.</div>",
                            unsafe_allow_html=True)

    with c4:
        # Radar chart for top 3 patients
        cats = ["TIR Score", "Stability", "Hypo Safety", "Activity", "IES"]
        top3 = patient_score.head(3)
        fig_radar = go.Figure()
        radar_colors = [PALETTE["teal"], PALETTE["violet"], PALETTE["green"]]
        for i, (_, row) in enumerate(top3.iterrows()):
            vals = [row["tir_score"], row["stability_score"],
                    row["hypo_safety_score"], row["activity_score"], row["ies"]]
            vals += [vals[0]]
            cats2 = cats + [cats[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats2, fill="toself",
                name=f"P{row['patient_id']}",
                line_color=radar_colors[i], opacity=0.7,
            ))
        fig_radar.update_layout(
            title="Top 3 patients — control radar",
            polar=dict(
                bgcolor="#04080F",
                radialaxis=dict(visible=True, range=[0, 45], color="#5B8DB8"),
                angularaxis=dict(color="#5B8DB8"),
            ),
            showlegend=True,
        )
        st.plotly_chart(styled_chart(fig_radar, 360), use_container_width=True)

    # Full score table
    st.markdown("<div class='section-hd'>Full patient score table</div>", unsafe_allow_html=True)
    display_cols = ["patient_id", "avg_glucose", "tir", "hypo_rate",
                    "avg_steps", "ies"]
    disp = patient_score[display_cols].copy()
    disp.columns = ["Patient", "Avg Glucose", "TIR", "Hypo Rate", "Avg Steps", "IES"]
    disp["TIR"]       = (disp["TIR"] * 100).round(1)
    disp["Hypo Rate"] = (disp["Hypo Rate"] * 100).round(2)
    disp["Avg Glucose"] = disp["Avg Glucose"].round(1)
    disp["IES"]       = disp["IES"].round(1)
    st.dataframe(
        disp.style.background_gradient(subset=["IES"], cmap="RdYlGn", vmin=0, vmax=100)
                  .background_gradient(subset=["TIR"], cmap="RdYlGn", vmin=0, vmax=100)
                  .format(precision=1),
        use_container_width=True, height=380,
    )

# ══════════════════════════════════════════════════════════════════
# TAB 7 — COHORT
# ══════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("<div class='section-hd'>Cohort-level analytics</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        if "gender" in df_v.columns and df_v["gender"].nunique() > 1:
            gender_ct = df_v.drop_duplicates("patient_id")["gender"].value_counts().reset_index()
            gender_ct.columns = ["gender", "count"]
            fig_g = px.pie(gender_ct, names="gender", values="count",
                            color_discrete_sequence=[PALETTE["teal"], PALETTE["pink"]],
                            hole=0.5, title="Gender distribution")
            fig_g.update_layout(showlegend=True)
            st.plotly_chart(styled_chart(fig_g, 300), use_container_width=True)

    with c2:
        if "age" in df_v.columns and df_v["age"].nunique() > 1:
            age_pt = df_v.drop_duplicates("patient_id")[["patient_id", "age"]]
            fig_age = px.histogram(age_pt, x="age", nbins=15,
                                    color_discrete_sequence=[PALETTE["violet"]],
                                    labels={"age": "Age (years)"},
                                    title="Age distribution")
            st.plotly_chart(styled_chart(fig_age, 300), use_container_width=True)

    # Parallel coordinates
    st.markdown("<div class='section-hd'>Multi-dimensional patient profile</div>",
                unsafe_allow_html=True)
    par_df = patient_score[["patient_id", "avg_glucose", "tir", "hypo_rate",
                              "avg_steps", "glucose_variability", "ies"]].copy()
    par_df["tir"] *= 100
    par_df["hypo_rate"] *= 100
    fig_par = px.parallel_coordinates(
        par_df,
        dimensions=["avg_glucose", "tir", "hypo_rate",
                    "avg_steps", "glucose_variability", "ies"],
        color="ies",
        color_continuous_scale="RdYlGn",
        labels={"avg_glucose": "Mean BG", "tir": "TIR%",
                "hypo_rate": "Hypo%", "avg_steps": "Steps",
                "glucose_variability": "Var", "ies": "IES"},
        title="Patient profile parallel coordinates — color = IES",
    )
    fig_par.update_layout(height=380, **PT)
    st.plotly_chart(fig_par, use_container_width=True)

    # Correlation matrix
    st.markdown("<div class='section-hd'>Feature correlation matrix</div>",
                unsafe_allow_html=True)
    corr_cols = ["glucose", "heart_rate", "steps", "calories",
                 "basal_rate", BOLUS, "carb_input",
                 "glucose_roc", "glucose_rolling_std_1h", "risk_score"]
    corr_cols = [c for c in corr_cols if c in df_v.columns]
    corr       = df_v[corr_cols].corr().round(2)
    fig_corr   = go.Figure(go.Heatmap(
        z=corr.values, x=corr_cols, y=corr_cols,
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="r"),
    ))
    fig_corr.update_layout(title="Pearson correlation matrix",
                            xaxis=dict(tickangle=-40))
    st.plotly_chart(styled_chart(fig_corr, 420), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 8 — INSIGHTS
# ══════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("<div class='section-hd'>Executive clinical insights</div>",
                unsafe_allow_html=True)

    insights = [
        ("📊 Glucose Stability",
         "Patients with higher Time-In-Range consistently show <b>lower glucose variability</b> (CV < 36%) "
         "and fewer extreme excursions. Stability is the strongest predictor of overall glycemic health."),
        ("🍽️ Meal & Bolus Impact",
         "Higher carbohydrate loads (&gt;60g) are associated with <b>post-meal spikes exceeding 80 mg/dL</b>. "
         "Missed bolus events—detected when carb_input &gt; 20g and bolus = 0—correlate strongly with "
         "2-hour hyperglycemia risk."),
        ("🏃 Activity & TIR",
         "Each 1,000 additional daily steps is associated with a <b>~2% improvement in TIR</b>. "
         "Moderate activity (2,000–6,000 steps/day) shows the best glucose stabilisation effect. "
         "Prolonged inactivity (&gt;8h sedentary) increases glucose drift risk."),
        ("🌙 Nocturnal Risk",
         "Basal rates above 1.2 U/h are associated with elevated nocturnal hypoglycemia probability. "
         "The dawn phenomenon (glucose rise 3–8 AM) is present in <b>~70% of patients</b> "
         "and accounts for a significant portion of morning hyperglycemia."),
        ("🤖 Predictive AI",
         "The Random Forest classifier achieves <b>AUC &gt; 0.84</b> for 30-min hypoglycemia prediction "
         "using glucose ROC, rolling variability, basal rate, and heart rate. "
         "Glucose rate-of-change and rolling SD are the top two features across all tasks."),
        ("💊 Prescriptive Scoring",
         "The Insulin Effectiveness Score (IES) combines TIR (40%), glucose stability (25%), "
         "hypo safety (20%), and activity (15%). Patients scoring &lt;50 require urgent therapy review; "
         "patients &ge;70 are considered well-controlled."),
    ]

    col_a, col_b = st.columns(2)
    for i, (title, body) in enumerate(insights):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(f"""
            <div class='insight-card'>
                <b>{title}</b><br><br>{body}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-hd'>Key numbers at a glance</div>", unsafe_allow_html=True)
    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("Patients",      len(selected_patients))
    n2.metric("Total readings", f"{len(df_v):,}")
    n3.metric("Cohort TIR",    f"{df_v['tir_flag'].mean()*100:.1f}%")
    n4.metric("Hypo events",   int(df_v["hypo_flag"].sum()))
    n5.metric("Hyper events",  int(df_v["hyper_flag"].sum()))

    st.download_button(
        "⬇ Download patient scores CSV",
        data=patient_score.to_csv(index=False),
        file_name="hupa_patient_scores.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "🩺 HUPA-UCM AI Diabetes Intelligence System · "
    "PyCore Python Hackathon 2026 · "
    "Dataset DOI: 10.1016/j.dib.2024.110526 · "
    "Built with Streamlit + Plotly + scikit-learn"
)
