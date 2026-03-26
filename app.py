import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os, warnings, json
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="G-Net CDSS — HIV Youth ART",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background: #0d1117; }
.stApp { background: #0d1117; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

/* ── Cards ── */
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 16px 20px; margin: 6px 0;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;
}
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 600; }
.green { color: #3fb950; } .amber { color: #d29922; } .blue { color: #58a6ff; }
.red   { color: #f85149; } .muted { color: #8b949e; } .orange { color: #e3894b; }

/* ── Section heads ── */
.section-head {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #58a6ff;
    letter-spacing: 2px; text-transform: uppercase;
    border-bottom: 1px solid #21262d; padding-bottom: 6px; margin: 22px 0 14px 0;
}

/* ── CATE bars ── */
.cate-bar-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px 24px; margin: 10px 0; }
.cate-bar-label { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #c9d1d9; margin-bottom: 8px; display: flex; justify-content: space-between; }
.cate-bar-track { background: #21262d; border-radius: 4px; height: 18px; width: 100%; overflow: hidden; margin-bottom: 14px; }
.cate-bar-fill  { height: 100%; border-radius: 4px; }

/* ── Recommendation ── */
.recommendation-box {
    background: linear-gradient(135deg, #1a2f1a 0%, #162116 100%);
    border: 1px solid #3fb950; border-radius: 10px; padding: 20px 24px; margin: 12px 0;
}
.rec-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #3fb950; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.rec-name  { font-family: 'IBM Plex Mono', monospace; font-size: 24px; font-weight: 600; color: #3fb950; }

/* ── Safety box ── */
.safety-box {
    background: #1a1a2e; border: 1px solid #58a6ff; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #8b949e;
}
.safety-pass { color: #3fb950; } .safety-warn { color: #d29922; }

/* ── Header ── */
.header-strip { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 0 10px 0; margin-bottom: 24px; }
.header-title { font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600; color: #58a6ff; }
.header-sub   { font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: #8b949e; margin-top: 2px; }
.badge { display: inline-block; background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 2px 8px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8b949e; margin-right: 6px; }

/* ── Risk factor tiles ── */
.risk-tile {
    background: #161b22; border-radius: 8px; padding: 12px 16px;
    border-left: 3px solid #444; margin-bottom: 8px; font-size: 13px;
}
.risk-tile.high   { border-left-color: #f85149; }
.risk-tile.medium { border-left-color: #d29922; }
.risk-tile.low    { border-left-color: #3fb950; }
.risk-tile .rtitle { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.risk-tile .rval   { font-size: 15px; color: #c9d1d9; font-weight: 500; margin-top: 2px; }

/* ── Questionnaire section card ── */
.q-section {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 18px 20px; margin: 10px 0;
}
.q-section-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #58a6ff;
    font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}

/* ── Adherence gauge ── */
.gauge-wrap { text-align: center; padding: 10px; }
.gauge-ring { display: inline-block; width: 90px; height: 90px; border-radius: 50%; position: relative; }

/* ── Narrative box ── */
.narrative-box {
    background: #0f1924; border: 1px solid #1d3350; border-radius: 10px;
    padding: 18px 22px; margin: 12px 0; line-height: 1.8;
    font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; color: #8b949e;
}
.narrative-box b { color: #c9d1d9; }
.narrative-box .highlight { color: #58a6ff; font-weight: 600; }
.narrative-box .concern   { color: #f85149; font-weight: 600; }
.narrative-box .positive  { color: #3fb950; font-weight: 600; }

/* ── VL category badge ── */
.vl-badge { display: inline-block; border-radius: 20px; padding: 4px sss14px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; }
.vl-suppressed { background: #1a3a1a; color: #3fb950; border: 1px solid #3fb950; }
.vl-unsuppressed { background: #3a1a1a; color: #f85149; border: 1px solid #f85149; }
</style>
""", unsafe_allow_html=True)

# ── Load model bundle ──────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    path = os.path.join(os.path.dirname(__file__), 'gnet_bundle.joblib')
    return joblib.load(path)

bundle        = load_bundle()
outcome_model = bundle['outcome_model']
prop_models   = bundle['prop_models']
scaler        = bundle['scaler']
FEATURE_COLS  = bundle['feature_cols']

TREATMENTS    = bundle['treatments']
TREAT_LABELS  = bundle['treat_labels']
gnet_ate      = bundle['gnet_ate']
MODEL_AUC     = bundle['model_auc']
MODEL_ACC     = bundle['model_acc']
MODEL_BAL     = bundle['model_bal']

INT_NAMES = {'enrolled_in_OTZ':'OTZ','enrolled_in_DSD':'DSD','has_YAPS_support':'YAPS'}
INT_FULL  = {
    'OTZ':  'Optimised Treatment for Adolescents (OTZ)',
    'DSD':  'Differentiated Service Delivery (DSD)',
    'YAPS': 'Youth Adherence & Psychosocial Support (YAPS)'
}
INT_DESC = {
    'OTZ':  'A peer-led adherence support programme for adolescents. Includes group sessions, peer mentors, and clinic integration.',
    'DSD':  'Flexible appointment scheduling and multi-month dispensing to reduce clinic burden, esp. for patients far from clinic.',
    'YAPS': 'Psychosocial counselling and adherence coaching. Particularly effective for patients with depression, stigma, or low social support.'
}

# ── Helper functions ───────────────────────────────────────────────────────
def adherence_level(pct):
    if pct >= 95: return "Optimal", "green"
    if pct >= 85: return "Adequate", "amber"
    if pct >= 70: return "Suboptimal", "orange"
    return "Poor", "red"

def phq9_severity(score):
    if score <= 4:  return "Minimal / None", "green"
    if score <= 9:  return "Mild", "amber"
    if score <= 14: return "Moderate", "orange"
    if score <= 19: return "Moderately Severe", "red"
    return "Severe", "red"

def stigma_level(score):
    if score <= 10: return "Low", "green"
    if score <= 22: return "Moderate", "amber"
    return "High", "red"

def social_support_level(score):
    if score >= 15: return "Strong", "green"
    if score >= 8:  return "Moderate", "amber"
    return "Low", "red"

def moh_safety_check(age, phq9, years_on_art, adherence):
    flags = []
    if age < 10 or age > 24:
        flags.append(("WARN", f"Age {age} outside youth cohort range (10–24)"))
    if phq9 >= 15:
        flags.append(("WARN", "Severe depression (PHQ-9 ≥ 15) — mental health referral recommended before ART intervention"))
    if years_on_art < 0.5:
        flags.append(("INFO", "Patient on ART < 6 months — early CATE estimates may have wider uncertainty"))
    if adherence < 50:
        flags.append(("WARN", "Critically low adherence (<50%) — intensive support urgently indicated"))
    return flags

def run_inference(patient_dict):
    row = {c: 0.0 for c in FEATURE_COLS}
    for k, v in patient_dict.items():
        if k in row:
            row[k] = float(v)
    x = np.array([list(row.values())], dtype=float)
    results = {}
    for t in TREATMENTS:
        label = TREAT_LABELS[t]
        x1 = x.copy(); x1[0, FEATURE_COLS.index(t)] = 1.0
        x0 = x.copy(); x0[0, FEATURE_COLS.index(t)] = 0.0
        p1 = outcome_model.predict_proba(scaler.transform(x1))[0, 1]
        p0 = outcome_model.predict_proba(scaler.transform(x0))[0, 1]
        results[label] = {'p1': p1, 'p0': p0, 'cate': p1 - p0}
    base_prob = outcome_model.predict_proba(scaler.transform(x))[0, 1]
    return results, base_prob

def compute_risk_score(phq9, adherence, stigma, food_ins, tx_int, social, missed_d, vl_trend, haz_drink, oi_last):
    """Compute a simple composite risk score 0–100 for display."""
    score = 0
    score += min(phq9 / 27, 1.0) * 20
    score += max(0, (100 - adherence) / 100) * 25
    score += min(stigma / 40, 1.0) * 12
    score += (1 if food_ins else 0) * 10
    score += (1 if tx_int else 0) * 12
    score += max(0, (10 - social) / 10) * 8
    score += min(missed_d / 15, 1.0) * 8
    score += (5 if vl_trend == 'Increasing' else 0)
    score += (1 if haz_drink else 0) * 5
    score += min(oi_last / 5, 1.0) * 5
    return min(int(score), 100)

def risk_color(score):
    if score < 30: return "#3fb950", "Low"
    if score < 55: return "#d29922", "Moderate"
    if score < 75: return "#e3894b", "High"
    return "#f85149", "Critical"

def narrative(patient, base_prob, results, best, flags):
    adh_lbl, _ = adherence_level(patient['adherence_self_report'])
    phq_lbl, _ = phq9_severity(patient['PHQ9_score'])
    stig_lbl, _ = stigma_level(patient['stigma_score'])
    best_cate_pct = results[best]['cate'] * 100
    base_pct = base_prob * 100
    supp_class = "positive" if base_prob >= 0.5 else "concern"
    parts = []
    parts.append(f"This <b>{'female' if patient['sex']==1 else 'male'} patient aged {int(patient['age'])}</b>, "
                 f"on ART for <b>{patient['years_on_ART']:.1f} years</b> "
                 f"({'DTG-based' if patient.get('ART_regimen_EFV-based',0)==0 else 'EFV-based'}), "
                 f"has a baseline predicted viral suppression probability of "
                 f"<span class='{supp_class}'>{base_pct:.1f}%</span>.")
    # Adherence narrative
    if adh_lbl in ("Poor","Suboptimal"):
        parts.append(f"<span class='concern'>Adherence is {adh_lbl.lower()} at {patient['adherence_self_report']:.0f}%</span> "
                     f"with a mean of {patient['mean_missed_doses_30d']:.1f} missed doses in the past 30 days — "
                     "this is a primary driver of suppression risk and warrants urgent counselling.")
    else:
        parts.append(f"Adherence is <span class='positive'>{adh_lbl.lower()} ({patient['adherence_self_report']:.0f}%)</span>, "
                     "which is a protective factor for viral suppression.")
    # Psychosocial
    if patient['PHQ9_score'] >= 10:
        parts.append(f"<span class='concern'>Depression severity is {phq_lbl} (PHQ-9 = {patient['PHQ9_score']})</span>, "
                     "which is significantly associated with reduced adherence and suppression rates in youth.")
    if patient['stigma_score'] > 20:
        parts.append(f"HIV-related stigma is <span class='concern'>{stig_lbl.lower()}</span> (score {patient['stigma_score']:.0f}/40), "
                     "which may be limiting clinic attendance and disclosure to caregivers.")
    # Social support
    soc_lbl, _ = social_support_level(patient['social_support_score'])
    if soc_lbl == "Low":
        parts.append("Social support is <span class='concern'>low</span> — consider linking the patient with peer support networks.")
    # Contextual
    ctx_flags = []
    if patient.get('food_insecurity'): ctx_flags.append("food insecurity")
    if patient.get('treatment_interruption'): ctx_flags.append("prior treatment interruption")
    if patient.get('hazardous_drinking'): ctx_flags.append("hazardous alcohol use")
    if ctx_flags:
        parts.append(f"Additional barriers identified: <span class='concern'>{', '.join(ctx_flags)}</span>.")
    # Recommendation
    parts.append(f"Causal inference estimates that enrolling this patient in <span class='highlight'>{best} — {INT_FULL[best]}</span> "
                 f"would increase suppression probability by <span class='positive'>+{best_cate_pct:.1f} percentage points</span>.")
    return " ".join(parts)

# ─────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="header-title">🧬 G-Net CDSS</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">HIV Youth ART · Makerere University · 2026</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-head">Model performance</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="metric-card"><div class="metric-label">ROC-AUC</div><div class="metric-value green">{MODEL_AUC:.4f}</div></div>
    <div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value blue">{MODEL_ACC*100:.1f}%</div></div>
    <div class="metric-card"><div class="metric-label">Balanced Accuracy</div><div class="metric-value blue">{MODEL_BAL*100:.1f}%</div></div>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="section-head">Population ATEs (G-Net)</div>', unsafe_allow_html=True)
    for label, v in gnet_ate.items():
        color = "green" if v['ATE'] > 0 else "red"
        sign  = "+" if v['ATE'] > 0 else ""
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">{label} population effect</div>
            <div class="metric-value {color}">{sign}{v["ATE"]*100:.2f}%</div>
        </div>''', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('''<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#484f58;line-height:1.6">
    G-Net · sklearn MLP · G-Computation<br>
    Training cohort: 20,890 youth · Central Uganda<br>
    MoH Uganda ART Guidelines 2020
    </div>''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="header-strip">
  <div class="header-title">G-Net Causal Inference Engine</div>
  <div class="header-sub">
    Individualised CATE estimation · G-Computation counterfactuals · MoH 2020 safety filter
    <span class="badge">G-Net MLP</span>
    <span class="badge">Offline edge</span>
    <span class="badge">Central Uganda Cohort</span>
  </div>
</div>
''', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬  Patient Assessment", "📊  Population ATEs", "📋  Feature Reference"])

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — PATIENT ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_results = st.columns([1.15, 0.85], gap="large")

    with col_form:

        # ── SECTION A: Demographics ──────────────────────────────────────
        st.markdown('''<div class="q-section-title" style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#58a6ff;font-weight:600;margin:18px 0 10px 0;">
        🧑 SECTION A — Demographics & ART History</div>''', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 10, 24, 17,
                help="Youth cohort eligible: 10–24 years. Enter patient's current age.")
            sex = st.selectbox("Biological sex", ["Male","Female"],
                help="Recorded biological sex. Used as a covariate in the causal model.")
        with c2:
            years_art = st.number_input("Years on ART", 0.0, 20.0, 2.5, step=0.1,
                help="Total duration the patient has been on antiretroviral therapy.")
            art_reg = st.selectbox("ART regimen", ["DTG-based","EFV-based"],
                help="Current first-line ART regimen. DTG (dolutegravir) is preferred per MoH 2020.")
        with c3:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 45.0, 21.0, step=0.1,
                help="Body Mass Index. Low BMI may indicate poor nutrition or disease progression.")
            dist_clin = st.number_input("Distance to clinic (km)", 0.0, 100.0, 6.0, step=0.5,
                help="One-way travel distance to the ART clinic. Affects adherence via DSD eligibility.")

        # ── SECTION B: Clinical Markers ──────────────────────────────────
        st.markdown('''<div class="q-section-title" style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#58a6ff;font-weight:600;margin:18px 0 10px 0;">
        🩺 SECTION B — Clinical & Laboratory Markers</div>''', unsafe_allow_html=True)

        c4, c5, c6 = st.columns(3)
        with c4:
            cd4_base = st.number_input("Baseline CD4 (cells/μL)", 0, 2000, 350,
                help="CD4 count at ART initiation. Reference for immune trajectory.")
        with c5:
            cd4_curr = st.number_input("Current CD4 (cells/μL)", 0, 2000, 500,
                help="Most recent CD4 count. CD4 ≥ 500 is associated with stable immune function.")
        with c6:
            oi_last = st.number_input("Opportunistic infections (last 12m)", 0, 20, 0,
                help="Number of opportunistic infection episodes in the past 12 months.")

        c7, c8 = st.columns(2)
        with c7:
            vl_trend = st.selectbox("Viral load trend", ["Stable","Declining","Increasing"],
                help="Direction of viral load over recent measurements. 'Increasing' is a red flag.")
        with c8:
            st.markdown(f"""<div style="padding-top:8px">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">VL trend interpretation</div>
            <div style="font-size:13px;color:#c9d1d9">
            {"⬆ Increasing viral load — urgent review" if vl_trend=="Increasing" else
             "⬇ Declining — treatment working" if vl_trend=="Declining" else
             "➡ Stable — monitor routinely"}
            </div></div>""", unsafe_allow_html=True)

        # ── SECTION C: Adherence ─────────────────────────────────────────
        st.markdown('''<div class="q-section-title" style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#58a6ff;font-weight:600;margin:18px 0 10px 0;">
        💊 SECTION C — Adherence Assessment</div>''', unsafe_allow_html=True)

        st.markdown("""<div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:#8b949e;margin-bottom:10px;padding:10px 14px;background:#161b22;border-radius:6px;border-left:3px solid #58a6ff">
        <b style="color:#c9d1d9">Ask the patient:</b><br>
        "In the past month, how often did you take your HIV medicines as prescribed?"<br>
        "How many times did you miss a dose in the past 30 days?"
        </div>""", unsafe_allow_html=True)

        c9, c10 = st.columns(2)
        with c9:
            adherence = st.slider("Self-reported adherence (%)", 0.0, 100.0, 72.0, step=1.0,
                help="Patient's self-reported percentage of doses taken as prescribed in the last month. ≥95% = optimal.")
            adh_lbl, adh_col = adherence_level(adherence)
            st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:12px;margin-top:-6px;margin-bottom:8px">
            Adherence level: <span class="{adh_col}" style="color:{'#3fb950' if adh_col=='green' else '#d29922' if adh_col=='amber' else '#e3894b' if adh_col=='orange' else '#f85149'}">{adh_lbl}</span>
            </div>""", unsafe_allow_html=True)
        with c10:
            missed_d = st.slider("Mean missed doses / 30 days", 0.0, 15.0, 2.5, step=0.1,
                help="Average number of ART doses missed per month (clinician or pill count estimate).")

        tx_int = st.checkbox("History of treatment interruption",
            help="Has the patient ever stopped ART for ≥1 month? Prior interruptions significantly increase suppression failure risk.")

        # ── SECTION D: Psychosocial ──────────────────────────────────────
        st.markdown('''<div class="q-section-title" style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#58a6ff;font-weight:600;margin:18px 0 10px 0;">
        🧠 SECTION D — Psychosocial & Mental Health</div>''', unsafe_allow_html=True)

        st.markdown("""<div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:#8b949e;margin-bottom:10px;padding:10px 14px;background:#161b22;border-radius:6px;border-left:3px solid #58a6ff">
        <b style="color:#c9d1d9">Validated screening tools:</b> PHQ-9 for depression · Berger HIV Stigma Scale · Oslo Social Support Scale (OSS-3) · AUDIT-C for alcohol
        </div>""", unsafe_allow_html=True)

        c11, c12, c13 = st.columns(3)
        with c11:
            phq9 = st.slider("PHQ-9 depression score", 0, 27, 7,
                help="Patient Health Questionnaire-9. Scores: 0–4 minimal, 5–9 mild, 10–14 moderate, 15–19 moderately severe, 20–27 severe.")
            phq_lbl, phq_col = phq9_severity(phq9)
            st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{'#3fb950' if phq_col=='green' else '#d29922' if phq_col=='amber' else '#e3894b' if phq_col=='orange' else '#f85149'};">{phq_lbl}</div>""", unsafe_allow_html=True)
        with c12:
            stigma = st.slider("HIV stigma score (0–40)", 0.0, 40.0, 15.0, step=0.5,
                help="Berger HIV Stigma Scale (condensed). Higher scores = greater perceived/enacted stigma. Threshold for high stigma: >22.")
            stig_lbl, stig_col = stigma_level(stigma)
            st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{'#3fb950' if stig_col=='green' else '#d29922' if stig_col=='amber' else '#f85149'};">{stig_lbl} stigma</div>""", unsafe_allow_html=True)
        with c13:
            social = st.slider("Social support score (0–20)", 0.0, 20.0, 10.0, step=0.5,
                help="Oslo Social Support Scale (OSS-3, adapted). Higher = stronger support network. <8 = low, 8–14 = moderate, ≥15 = strong.")
            soc_lbl, soc_col = social_support_level(social)
            st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{'#3fb950' if soc_col=='green' else '#d29922' if soc_col=='amber' else '#f85149'};">{soc_lbl} support</div>""", unsafe_allow_html=True)

        c14, c15 = st.columns(2)
        with c14:
            audit_c = st.slider("AUDIT-C alcohol score (0–12)", 0, 12, 2,
                help="Alcohol Use Disorders Identification Test-Concise. Score ≥3 (women) or ≥4 (men) = positive screen for hazardous drinking.")
        with c15:
            haz_drink = st.checkbox("Hazardous drinking confirmed",
                help="Clinician-confirmed hazardous alcohol use (AUDIT-C positive or clinical assessment).")

        # ── SECTION E: Social Context ────────────────────────────────────
        st.markdown('''<div class="q-section-title" style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#58a6ff;font-weight:600;margin:18px 0 10px 0;">
        🏘 SECTION E — Social & Economic Context</div>''', unsafe_allow_html=True)

        st.markdown("""<div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:#8b949e;margin-bottom:10px;padding:10px 14px;background:#161b22;border-radius:6px;border-left:3px solid #58a6ff">
        <b style="color:#c9d1d9">Ask the patient / caregiver:</b><br>
        "In the past month, did you worry about having enough food?" | "Who helps you remember to take your medicines?"
        </div>""", unsafe_allow_html=True)

        ce1, ce2 = st.columns(2)
        with ce1:
            food_ins = st.checkbox("Food insecure (household)",
                help="Patient/household experiences food insecurity. Associated with treatment interruptions and poor adherence.")
            caregiver = st.selectbox("Caregiver support level", ["Low","Medium","High"],
                help="Level of active support from a caregiver (parent, guardian, or trusted adult) for ART adherence.")
        with ce2:
            st.markdown("""<div style="padding-top: 6px; font-size:13px; color:#8b949e; line-height:1.7">
            <b style="color:#c9d1d9">Caregiver support guide:</b><br>
            <b style="color:#c9d1d9">High:</b> Reminds patient daily, attends clinic<br>
            <b style="color:#c9d1d9">Medium:</b> Occasionally reminds, aware of HIV<br>
            <b style="color:#c9d1d9">Low:</b> Unaware or uninvolved
            </div>""", unsafe_allow_html=True)

        # ── SECTION F: Programme Enrolment ───────────────────────────────
        st.markdown('''<div class="q-section-title" style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#58a6ff;font-weight:600;margin:18px 0 10px 0;">
        🏥 SECTION F — Current Programme Enrolment</div>''', unsafe_allow_html=True)

        st.markdown("""<div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:#8b949e;margin-bottom:10px;padding:10px 14px;background:#161b22;border-radius:6px;border-left:3px solid #58a6ff">
        Tick the interventions this patient is <b style="color:#c9d1d9">currently enrolled in</b>. The model will compute counterfactual CATE for each — showing what would happen if you added or changed enrolment.
        </div>""", unsafe_allow_html=True)

        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            in_otz  = st.checkbox("Currently in OTZ", value=False,
                help="Optimised Treatment for Adolescents — peer-led adolescent adherence club.")
        with cf2:
            in_dsd  = st.checkbox("Currently in DSD", value=True,
                help="Differentiated Service Delivery — multi-month dispensing / reduced clinic visits.")
        with cf3:
            in_yaps = st.checkbox("Currently has YAPS", value=False,
                help="Youth Adherence & Psychosocial Support — counselling and peer mentoring.")

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run G-Computation Inference", type="primary", use_container_width=True)

    # ── RESULTS PANEL ────────────────────────────────────────────────────
    with col_results:
        st.markdown('<div class="section-head">Causal inference output</div>', unsafe_allow_html=True)

        if run_btn:
            patient = {
                'age':                      age,
                'sex':                      1 if sex == 'Female' else 0,
                'years_on_ART':             years_art,
                'baseline_CD4':             cd4_base,
                'current_CD4':              cd4_curr,
                'BMI':                      bmi,
                'distance_to_clinic_km':    dist_clin,
                'PHQ9_score':               phq9,
                'stigma_score':             stigma,
                'social_support_score':     social,
                'alcohol_AUDITC_score':     audit_c,
                'adherence_self_report':    adherence,
                'mean_missed_doses_30d':    missed_d,
                'caregiver_support':        {'Low':0,'Medium':1,'High':2}[caregiver],
                'food_insecurity':          int(food_ins),
                'treatment_interruption':   int(tx_int),
                'hazardous_drinking':       int(haz_drink),
                'enrolled_in_OTZ':         int(in_otz),
                'enrolled_in_DSD':         int(in_dsd),
                'has_YAPS_support':        int(in_yaps),
                'opportunistic_infections_last12m': oi_last,
                'ART_regimen_EFV-based':    1 if art_reg == 'EFV-based' else 0,
                'vl_trend_direction_Increasing': 1 if vl_trend == 'Increasing' else 0,
                'vl_trend_direction_Stable':     1 if vl_trend == 'Stable'    else 0,
            }

            results, base_prob = run_inference(patient)
            flags = moh_safety_check(age, phq9, years_art, adherence)
            best = max(results, key=lambda k: results[k]['cate'])
            best_cate = results[best]['cate'] * 100

            # ── Composite risk score ──
            risk_score = compute_risk_score(phq9, adherence, stigma, food_ins, tx_int, social, missed_d, vl_trend, haz_drink, oi_last)
            risk_hex, risk_lbl = risk_color(risk_score)

            # ── Baseline suppression ──
            base_color = "green" if base_prob >= 0.5 else "red"
            supp_label = "LIKELY SUPPRESSED" if base_prob >= 0.5 else "SUPPRESSION AT RISK"
            st.markdown(f'''
            <div class="metric-card" style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div class="metric-label">Baseline suppression probability</div>
                    <div class="metric-value {base_color}">{base_prob*100:.1f}%</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;margin-top:4px">
                    P(suppressed | current profile, no change)
                    </div>
                </div>
                <div style="text-align:right">
                    <div class="metric-label">Composite risk</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;color:{risk_hex}">{risk_score}/100</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{risk_hex}">{risk_lbl}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # ── Risk factor breakdown ──
            st.markdown('<div class="section-head">Key risk factor summary</div>', unsafe_allow_html=True)

            def risk_tile(label, value, level):
                col = {"high":"#f85149","medium":"#d29922","low":"#3fb950"}.get(level,"#8b949e")
                return f'''<div class="risk-tile {level}"><div class="rtitle">{label}</div><div class="rval" style="color:{col}">{value}</div></div>'''

            adh_lvl_key = "high" if adherence < 70 else ("medium" if adherence < 85 else "low")
            phq_lvl_key = "high" if phq9 >= 15 else ("medium" if phq9 >= 10 else "low")
            stig_lvl_key = "high" if stigma > 22 else ("medium" if stigma > 10 else "low")
            soc_lvl_key = "high" if social < 8 else ("medium" if social < 14 else "low")
            cd4_trend_key = "high" if cd4_curr < cd4_base * 0.7 else ("medium" if cd4_curr < cd4_base else "low")
            vl_lvl_key = "high" if vl_trend == "Increasing" else ("medium" if vl_trend == "Stable" else "low")

            tiles_html = ""
            tiles_html += risk_tile("Adherence (self-report)", f"{adherence:.0f}% — {adherence_level(adherence)[0]}", adh_lvl_key)
            tiles_html += risk_tile("Depression (PHQ-9)", f"{phq9}/27 — {phq9_severity(phq9)[0]}", phq_lvl_key)
            tiles_html += risk_tile("HIV Stigma", f"{stigma:.0f}/40 — {stigma_level(stigma)[0]}", stig_lvl_key)
            tiles_html += risk_tile("Social Support", f"{social:.0f}/20 — {social_support_level(social)[0]}", soc_lvl_key)
            tiles_html += risk_tile("CD4 trajectory", f"Baseline {cd4_base} → Current {cd4_curr} cells/μL", cd4_trend_key)
            tiles_html += risk_tile("Viral load trend", vl_trend, vl_lvl_key)
            if food_ins:
                tiles_html += risk_tile("Food insecurity", "YES — household food insecure", "high")
            if tx_int:
                tiles_html += risk_tile("Treatment interruption history", "YES — prior interruption on record", "high")
            if haz_drink:
                tiles_html += risk_tile("Hazardous drinking", "YES — confirmed", "medium")
            st.markdown(tiles_html, unsafe_allow_html=True)

            # ── CATE bars ──
            st.markdown('<div class="section-head">Counterfactual CATE estimates</div>', unsafe_allow_html=True)
            bars_html = '<div class="cate-bar-wrap">'
            for label, v in results.items():
                cate_pct = v['cate'] * 100
                p1_pct   = v['p1']  * 100
                p0_pct   = v['p0']  * 100
                bar_w    = min(abs(cate_pct) * 4, 100)
                bar_col  = "#3fb950" if cate_pct > 0 else "#f85149"
                sign     = "+" if cate_pct > 0 else ""
                star     = " ★" if label == best and cate_pct > 0 else ""
                bars_html += f'''
                <div class="cate-bar-label">
                  <span>{label}{star}</span>
                  <span style="color:{bar_col};font-weight:600">{sign}{cate_pct:.2f}%</span>
                </div>
                <div class="cate-bar-track">
                  <div class="cate-bar-fill" style="width:{bar_w}%;background:{bar_col}"></div>
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;margin-bottom:14px">
                  P(supp|do({label})=1) = {p1_pct:.1f}%&nbsp;·&nbsp;P(supp|do({label})=0) = {p0_pct:.1f}%
                </div>'''
            bars_html += '</div>'
            st.markdown(bars_html, unsafe_allow_html=True)

            # ── Recommendation ──
            if best_cate > 0:
                st.markdown(f'''
                <div class="recommendation-box">
                  <div class="rec-label">▲ Recommended intervention</div>
                  <div class="rec-name">{best}</div>
                  <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:#8b949e;margin-top:6px">
                    {INT_FULL[best]}<br>
                    Expected suppression gain: <b style="color:#3fb950">+{best_cate:.2f}%</b><br><br>
                    <span style="color:#c9d1d9;font-size:12px">{INT_DESC[best]}</span>
                  </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="safety-box" style="border-color:#d29922">
                  <span class="safety-warn">⚠ No positive CATE detected.</span><br>
                  All interventions show neutral or negative predicted effect for this profile. Clinical review recommended.
                </div>
                ''', unsafe_allow_html=True)

            # ── Clinical narrative ──
            st.markdown('<div class="section-head">Clinical narrative summary</div>', unsafe_allow_html=True)
            narr = narrative(patient, base_prob, results, best, flags)
            st.markdown(f'<div class="narrative-box">{narr}</div>', unsafe_allow_html=True)

            # ── Safety flags ──
            if flags:
                st.markdown('<div class="section-head">MoH 2020 safety filter</div>', unsafe_allow_html=True)
                for level, msg in flags:
                    icon  = "⚠" if level == "WARN" else "ℹ"
                    color = "safety-warn" if level == "WARN" else "blue"
                    st.markdown(f'<div class="safety-box"><span class="{color}">{icon} {msg}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="safety-box">
                  <span class="safety-pass">✓ MoH 2020 ART safety check passed — no contraindications flagged</span>
                </div>
                ''', unsafe_allow_html=True)

            # ── Raw JSON ──
            with st.expander("Raw inference output (JSON)"):
                out = {
                    'base_suppression_prob': round(float(base_prob), 4),
                    'composite_risk_score': risk_score,
                    'risk_level': risk_lbl,
                    'counterfactuals': {
                        k: {
                            'p1_treated':   round(float(v['p1']), 4),
                            'p0_untreated': round(float(v['p0']), 4),
                            'CATE':         round(float(v['cate']), 4)
                        } for k, v in results.items()
                    },
                    'recommended': best,
                    'safety_flags': [f[1] for f in flags]
                }
                st.code(json.dumps(out, indent=2), language='json')

        else:
            st.markdown('''
            <div class="safety-box" style="margin-top:40px;text-align:center;padding:40px 24px">
                <div style="font-size:32px;margin-bottom:12px">🧬</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#8b949e">
                Complete the patient assessment form<br>then click<br>
                <span style="color:#58a6ff">▶ Run G-Computation Inference</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — Population ATEs
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">Population-level average treatment effects (G-Computation, test set)</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (label, v) in enumerate(gnet_ate.items()):
        ate_pct = v['ATE'] * 100
        color   = "green" if ate_pct > 0 else "red"
        sign    = "+" if ate_pct > 0 else ""
        with cols[i]:
            st.markdown(f'''
            <div class="metric-card" style="text-align:center;padding:28px 20px">
                <div class="metric-label" style="text-align:center">{label}</div>
                <div class="metric-value {color}" style="font-size:40px">{sign}{ate_pct:.2f}%</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8b949e;margin-top:8px">
                    Average Treatment Effect<br>
                    E[Y|do(A=1)] = {v["E_Y1"]*100:.1f}%<br>
                    E[Y|do(A=0)] = {v["E_Y0"]*100:.1f}%
                </div>
            </div>''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-head">Interpretation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:14px;color:#8b949e;line-height:1.8;max-width:700px">
    <b style="color:#c9d1d9">OTZ (+4.02%):</b> Enrolling a patient in OTZ is estimated to increase viral suppression probability by 4.02 pp on average. Aligns with Uganda MoH evidence for adolescent peer-led adherence clubs.<br><br>
    <b style="color:#c9d1d9">DSD (+3.47%):</b> Differentiated service delivery shows a consistent positive causal effect, particularly for patients with high distance-to-clinic scores.<br><br>
    <b style="color:#c9d1d9">YAPS (+0.24%):</b> YAPS shows a small population-level ATE, but individual CATE may be substantially higher for specific subgroups (high PHQ-9, high stigma, low social support).<br><br>
    <b style="color:#58a6ff">Note:</b> These are causal estimates from G-Computation, not correlations. DAG-based backdoor adjustment controls for confounding from PHQ-9, CD4, adherence, and other factors.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Model training summary</div>', unsafe_allow_html=True)
    summary = {
        'Architecture':     'G-Net (Sequential G-Computation MLP)',
        'Outcome model':    'MLPClassifier (128→64→32, ReLU, early stopping)',
        'Propensity model': 'MLPClassifier (64→32) per intervention',
        'Training cohort':  '20,890 HIV-positive youth, Central Uganda ART clinics',
        'Feature count':    str(len(FEATURE_COLS)),
        'Train/Test split': '80/20 stratified',
        'ROC-AUC':          f'{MODEL_AUC:.4f}',
        'Accuracy':         f'{MODEL_ACC*100:.2f}%',
        'Balanced Accuracy':f'{MODEL_BAL*100:.2f}%',
        'Deployment format':'joblib bundle (sklearn) · ONNX export ready',
        'Inference mode':   'Offline edge — no cloud dependency',
    }
    for k, v in summary.items():
        st.markdown(f'''
        <div style="display:flex;border-bottom:1px solid #21262d;padding:8px 0;font-size:13px">
            <div style="font-family:'IBM Plex Mono',monospace;color:#8b949e;width:220px;flex-shrink:0">{k}</div>
            <div style="font-family:'IBM Plex Sans',sans-serif;color:#c9d1d9">{v}</div>
        </div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — Feature Reference
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">Feature descriptions & clinical reference</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:#8b949e;margin-bottom:16px">
    All features used by the G-Net model, with clinical interpretation guidance for healthcare workers.
    </div>""", unsafe_allow_html=True)

    features_ref = [
        ("Demographics & ART", [
            ("age", "10–24", "Patient age in years. Youth cohort only."),
            ("sex", "0=Male, 1=Female", "Biological sex. Covariate for immune response and adherence patterns."),
            ("years_on_ART", "0–20", "Duration on ART in years. Longer duration → better immune reconstitution."),
            ("ART regimen", "DTG/EFV", "DTG-based is preferred per MoH 2020. EFV is legacy."),
            ("BMI", "10–45 kg/m²", "Nutritional status marker. Low BMI associated with disease progression."),
            ("distance_to_clinic_km", "0–100 km", "Barrier to care. Key predictor for DSD benefit."),
        ]),
        ("Clinical & Laboratory", [
            ("baseline_CD4", "cells/μL", "CD4 count at ART initiation. Sets immune trajectory baseline."),
            ("current_CD4", "cells/μL", "Latest CD4 count. ≥500 = stable immune function."),
            ("opportunistic_infections_last12m", "Count", "Recent OIs indicate immunosuppression or treatment failure."),
            ("vl_trend_direction", "Stable/Declining/Increasing", "Direction of viral load over recent measurements. Increasing = urgent concern."),
        ]),
        ("Adherence", [
            ("adherence_self_report", "0–100%", "Self-reported % of doses taken. ≥95% = optimal. <70% = poor."),
            ("mean_missed_doses_30d", "0–15", "Mean missed ART doses per month. Key predictor of virological failure."),
            ("treatment_interruption", "Yes/No", "Any prior ART gap ≥1 month. Increases resistance risk."),
        ]),
        ("Psychosocial & Mental Health", [
            ("PHQ9_score", "0–27", "Depression: 0–4=minimal, 5–9=mild, 10–14=moderate, ≥15=severe. Depression strongly predicts non-adherence."),
            ("stigma_score", "0–40", "HIV-related stigma (Berger scale). >22 = high stigma. Reduces clinic attendance."),
            ("social_support_score", "0–20", "OSS-3 adapted. <8=low, 8–14=moderate, ≥15=strong."),
            ("alcohol_AUDITC_score", "0–12", "AUDIT-C screen. ≥3 (women) or ≥4 (men) = positive for hazardous drinking."),
            ("hazardous_drinking", "Yes/No", "Clinician-confirmed hazardous alcohol use."),
        ]),
        ("Social & Economic Context", [
            ("food_insecurity", "Yes/No", "Household-level food insecurity. Associated with missed doses and clinic non-attendance."),
            ("caregiver_support", "0=Low, 1=Med, 2=High", "Active caregiver involvement in ART adherence. Protective factor especially for adolescents."),
        ]),
        ("Programme Enrolment (Interventions)", [
            ("enrolled_in_OTZ", "Yes/No", "Optimised Treatment for Adolescents — peer-led adherence club. ATE +4.02%."),
            ("enrolled_in_DSD", "Yes/No", "Differentiated Service Delivery — flexible dispensing. ATE +3.47%."),
            ("has_YAPS_support", "Yes/No", "Youth Adherence & Psychosocial Support — counselling. ATE +0.24% (higher in high-risk subgroups)."),
        ]),
    ]

    for section_name, rows in features_ref:
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin:20px 0 8px 0;border-bottom:1px solid #21262d;padding-bottom:4px">{section_name}</div>', unsafe_allow_html=True)
        table_html = '''<table style="width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Sans',sans-serif">
        <thead><tr>
          <th style="text-align:left;padding:8px 12px;color:#8b949e;font-family:'IBM Plex Mono',monospace;font-size:11px;border-bottom:1px solid #21262d;width:22%">Feature</th>
          <th style="text-align:left;padding:8px 12px;color:#8b949e;font-family:'IBM Plex Mono',monospace;font-size:11px;border-bottom:1px solid #21262d;width:20%">Range / Values</th>
          <th style="text-align:left;padding:8px 12px;color:#8b949e;font-family:'IBM Plex Mono',monospace;font-size:11px;border-bottom:1px solid #21262d">Clinical interpretation</th>
        </tr></thead><tbody>'''
        for fname, frange, fdesc in rows:
            table_html += f'''<tr>
              <td style="padding:8px 12px;color:#c9d1d9;font-family:'IBM Plex Mono',monospace;border-bottom:1px solid #161b22">{fname}</td>
              <td style="padding:8px 12px;color:#58a6ff;border-bottom:1px solid #161b22">{frange}</td>
              <td style="padding:8px 12px;color:#8b949e;border-bottom:1px solid #161b22;line-height:1.5">{fdesc}</td>
            </tr>'''
        table_html += '</tbody></table>'
        st.markdown(table_html, unsafe_allow_html=True)
