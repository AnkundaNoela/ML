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

.section-head {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #58a6ff;
    letter-spacing: 2px; text-transform: uppercase;
    border-bottom: 1px solid #21262d; padding-bottom: 6px; margin: 22px 0 14px 0;
}
.cate-bar-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px 24px; margin: 10px 0; }
.cate-bar-label { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #c9d1d9; margin-bottom: 8px; display: flex; justify-content: space-between; }
.cate-bar-track { background: #21262d; border-radius: 4px; height: 18px; width: 100%; overflow: hidden; margin-bottom: 14px; }
.cate-bar-fill  { height: 100%; border-radius: 4px; }

.recommendation-box {
    background: linear-gradient(135deg, #1a2f1a 0%, #162116 100%);
    border: 1px solid #3fb950; border-radius: 10px; padding: 20px 24px; margin: 12px 0;
}
.rec-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #3fb950; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.rec-name  { font-family: 'IBM Plex Mono', monospace; font-size: 24px; font-weight: 600; color: #3fb950; }

.safety-box {
    background: #1a1a2e; border: 1px solid #58a6ff; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #8b949e;
}
.safety-pass { color: #3fb950; } .safety-warn { color: #d29922; }

.header-strip { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 0 10px 0; margin-bottom: 24px; }
.header-title { font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600; color: #58a6ff; }
.header-sub   { font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: #8b949e; margin-top: 2px; }
.badge { display: inline-block; background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 2px 8px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8b949e; margin-right: 6px; }

.risk-tile {
    background: #161b22; border-radius: 8px; padding: 12px 16px;
    border-left: 3px solid #444; margin-bottom: 8px; font-size: 13px;
}
.risk-tile.high   { border-left-color: #f85149; }
.risk-tile.medium { border-left-color: #d29922; }
.risk-tile.low    { border-left-color: #3fb950; }
.risk-tile .rtitle { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.risk-tile .rval   { font-size: 15px; color: #c9d1d9; font-weight: 500; margin-top: 2px; }

.narrative-box {
    background: #0f1924; border: 1px solid #1d3350; border-radius: 10px;
    padding: 18px 22px; margin: 12px 0; line-height: 1.8;
    font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; color: #8b949e;
}
.narrative-box b { color: #c9d1d9; }
.narrative-box .highlight { color: #58a6ff; font-weight: 600; }
.narrative-box .concern   { color: #f85149; font-weight: 600; }
.narrative-box .positive  { color: #3fb950; font-weight: 600; }

.ask-box {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: #8b949e;
    margin-bottom: 10px; padding: 10px 14px; background: #161b22;
    border-radius: 6px; border-left: 3px solid #58a6ff;
}

/* ── Input / widget label visibility fix ─────────────────────────────── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stRadio"] label,
div[class*="stNumberInput"] label,
div[class*="stSelectbox"] label,
div[class*="stSlider"] label,
div[class*="stCheckbox"] label,
.stCheckbox span,
p, label {
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* Slider labels (min/max tick marks) */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #8b949e !important;
}

/* Help icon tooltip colour */
button[data-testid="stTooltipHoverTarget"] svg {
    fill: #58a6ff !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model bundle ──────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    path = os.path.join(os.path.dirname(__file__), 'gnet_bundle.joblib')
    return joblib.load(path)

bundle        = load_bundle()
outcome_model = bundle['gnet_outcome_model']
prop_models   = bundle['prop_models']
scaler        = bundle['scaler']
FEATURE_COLS  = bundle['feature_cols']       # 60 features
TREATMENTS    = bundle['treatment_cols']
TREAT_LABELS  = bundle['treat_labels']
gnet_ate      = bundle['gnet_ate']

# ── Derive model performance from stored validation scores ─────────────────
MODEL_BEST_VAL = getattr(outcome_model, 'best_validation_score_', 0.0) or 0.0

INT_FULL = {
    'OTZ':  'Optimised Treatment for Adolescents (OTZ)',
    'DSD':  'Differentiated Service Delivery (DSD)',
    'YAPS': 'Youth Adherence & Psychosocial Support (YAPS)'
}
INT_DESC = {
    'OTZ':  'Peer-led adherence support programme for adolescents. Includes group sessions, peer mentors, and clinic integration.',
    'DSD':  'Flexible appointment scheduling and multi-month dispensing to reduce clinic burden — especially for patients far from clinic.',
    'YAPS': 'Psychosocial counselling and adherence coaching — most effective for patients with depression, stigma, or low social support.'
}

DISTRICTS  = ['Kampala', 'Luwero', 'Masaka', 'Mityana', 'Mpigi', 'Mukono', 'Wakiso']
SCHOOLING  = ['Primary', 'Secondary', 'Tertiary', 'None']
EMPLOYMENT = ['Student', 'Unemployed', 'Employed']

# ── Helper functions ───────────────────────────────────────────────────────
def adherence_level(pct):
    if pct >= 95: return "Optimal", "green"
    if pct >= 85: return "Adequate", "amber"
    if pct >= 70: return "Suboptimal", "orange"
    return "Poor", "red"

def phq9_severity(score):
    if score <= 4:  return "Minimal", "green"
    if score <= 9:  return "Mild", "amber"
    if score <= 14: return "Moderate", "orange"
    if score <= 19: return "Moderately Severe", "red"
    return "Severe", "red"

def depression_category(score):
    """Return (Moderate, ModeratelySevere) one-hot encoding values."""
    if score <= 9:  return 0, 0
    if score <= 14: return 1, 0
    if score <= 19: return 0, 1
    return 0, 0

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
        flags.append(("WARN", "Severe depression (PHQ-9 >= 15) — mental health referral recommended before ART intervention"))
    if years_on_art < 0.5:
        flags.append(("INFO", "Patient on ART < 6 months — early CATE estimates may have wider uncertainty"))
    if adherence < 50:
        flags.append(("WARN", "Critically low adherence (<50%) — intensive support urgently indicated"))
    return flags

def run_inference(patient_dict):
    """Build a 60-feature vector and run G-Computation counterfactuals."""
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

def compute_risk_score(phq9, adherence, stigma, food_ins, tx_int, social,
                       missed_d, vl_trend, haz_drink, oi_last, missed_appts, refill_gap):
    score = 0
    score += min(phq9 / 27, 1.0) * 18
    score += max(0, (100 - adherence) / 100) * 22
    score += min(stigma / 40, 1.0) * 10
    score += (1 if food_ins else 0) * 8
    score += (1 if tx_int else 0) * 10
    score += max(0, (10 - social) / 10) * 7
    score += min(missed_d / 15, 1.0) * 8
    score += (5 if vl_trend == 'Increasing' else 0)
    score += (1 if haz_drink else 0) * 4
    score += min(oi_last / 5, 1.0) * 4
    score += min(missed_appts / 12, 1.0) * 4
    score += min(refill_gap / 60, 1.0) * 4
    return min(int(score), 100)

def risk_color(score):
    if score < 30: return "#3fb950", "Low"
    if score < 55: return "#d29922", "Moderate"
    if score < 75: return "#e3894b", "High"
    return "#f85149", "Critical"

def narrative(patient, base_prob, results, best):
    adh_lbl, _ = adherence_level(patient['adherence_self_report'])
    phq_lbl, _ = phq9_severity(patient['PHQ9_score'])
    stig_lbl, _ = stigma_level(patient['stigma_score'])
    best_cate_pct = results[best]['cate'] * 100
    base_pct = base_prob * 100
    supp_class = "positive" if base_prob >= 0.5 else "concern"
    parts = []
    sex_str = 'female' if patient['sex'] == 1 else 'male'
    parts.append(f"This <b>{sex_str} patient aged {int(patient['age'])}</b>, "
                 f"on ART for <b>{patient['years_on_ART']:.1f} years</b>, "
                 f"has a baseline predicted viral suppression probability of "
                 f"<span class='{supp_class}'>{base_pct:.1f}%</span>.")
    if adh_lbl in ("Poor", "Suboptimal"):
        parts.append(f"<span class='concern'>Adherence is {adh_lbl.lower()} at {patient['adherence_self_report']:.0f}%</span> "
                     f"with {patient['mean_missed_doses_30d']:.1f} missed doses/30 days — "
                     "a primary driver requiring urgent counselling.")
    else:
        parts.append(f"Adherence is <span class='positive'>{adh_lbl.lower()} ({patient['adherence_self_report']:.0f}%)</span>, "
                     "a protective factor for viral suppression.")
    if patient['PHQ9_score'] >= 10:
        parts.append(f"<span class='concern'>Depression is {phq_lbl} (PHQ-9 = {patient['PHQ9_score']})</span>, "
                     "strongly associated with reduced adherence in youth.")
    if patient['stigma_score'] > 20:
        parts.append(f"HIV-related stigma is <span class='concern'>{stig_lbl.lower()}</span> ({patient['stigma_score']:.0f}/40), "
                     "possibly limiting clinic attendance and disclosure.")
    soc_lbl, _ = social_support_level(patient['social_support_score'])
    if soc_lbl == "Low":
        parts.append("Social support is <span class='concern'>low</span> — consider linking to peer support.")
    ctx = []
    if patient.get('food_insecurity'):         ctx.append("food insecurity")
    if patient.get('treatment_interruption'):  ctx.append("prior treatment interruption")
    if patient.get('hazardous_drinking'):      ctx.append("hazardous alcohol use")
    if patient.get('drug_resistance_detected'): ctx.append("detected drug resistance")
    if ctx:
        parts.append(f"Additional barriers: <span class='concern'>{', '.join(ctx)}</span>.")
    if patient.get('HIV_status_disclosed') == 0:
        parts.append("HIV status is <span class='concern'>not disclosed</span> — associated with higher stigma and lower adherence.")
    parts.append(f"Causal inference estimates that enrolling in <span class='highlight'>{best} — {INT_FULL[best]}</span> "
                 f"would increase suppression probability by <span class='positive'>+{best_cate_pct:.1f} pp</span>.")
    return " ".join(parts)

# ─────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="header-title">🧬 G-Net CDSS</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">HIV Youth ART · Makerere University · 2026</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-head">Model info</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">Architecture</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#c9d1d9;margin-top:4px">G-Net MLP (128→64→32)</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Best validation score</div>
        <div class="metric-value green">{MODEL_BEST_VAL:.4f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Features</div>
        <div class="metric-value blue">{len(FEATURE_COLS)}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Training iterations</div>
        <div class="metric-value blue">{getattr(outcome_model, "n_iter_", "N/A")}</div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="section-head">Population ATEs (G-Net)</div>', unsafe_allow_html=True)
    for label, v in gnet_ate.items():
        color = "green" if v['ATE'] > 0 else "red"
        sign  = "+" if v['ATE'] > 0 else ""
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">{label} ATE</div>
            <div class="metric-value {color}">{sign}{v["ATE"]*100:.2f}%</div>
        </div>''', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('''<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#484f58;line-height:1.6">
    G-Net · sklearn MLP · G-Computation<br>
    Cohort: 20,890 youth · Central Uganda<br>
    MoH Uganda ART Guidelines 2020
    </div>''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="header-strip">
  <div class="header-title">G-Net Causal Inference Engine</div>
  <div class="header-sub">
    Individualised CATE estimation · G-Computation counterfactuals 
    <div>Central Uganda Cohort</div>
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
        st.markdown('<div class="section-head">A — Demographics & ART History</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age       = st.number_input("Age (years)", 10, 24, 17)
            sex       = st.selectbox("Biological sex", ["Male", "Female"])
        with c2:
            years_art = st.number_input("Years on ART", 0.0, 20.0, 2.5, step=0.1)
            art_reg   = st.selectbox("ART regimen", ["DTG-based", "EFV-based", "PI-based"],
                help="DTG = dolutegravir (preferred, MoH 2020). EFV = efavirenz. PI = protease inhibitor.")
        with c3:
            bmi       = st.number_input("BMI (kg/m²)", 10.0, 45.0, 21.0, step=0.1)
            dist_clin = st.number_input("Distance to clinic (km)", 0.0, 100.0, 6.0, step=0.5)

        c4, c5 = st.columns(2)
        with c4:
            district  = st.selectbox("District", DISTRICTS)
            schooling = st.selectbox("Schooling status", SCHOOLING)
        with c5:
            employment = st.selectbox("Employment status", EMPLOYMENT)
            boarding   = st.checkbox("Boarding school resident",
                help="Lives at a boarding school — affects caregiver access and clinic patterns.")

        # ── SECTION B: Clinical & Lab ────────────────────────────────────
        st.markdown('<div class="section-head">B — Clinical & Laboratory Markers</div>', unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        with c6:
            cd4_base = st.number_input("Baseline CD4 (cells/μL)", 0, 2000, 350)
        with c7:
            cd4_curr = st.number_input("Current CD4 (cells/μL)", 0, 2000, 500)
        with c8:
            oi_last  = st.number_input("Opportunistic infections (last 12m)", 0, 20, 0)

        c9, c10, c11 = st.columns(3)
        with c9:
            drug_res_tested   = st.checkbox("Drug resistance tested")
        with c10:
            drug_res_detected = st.checkbox("Drug resistance detected",
                help="Only applicable if resistance testing was done.")
        with c11:
            ever_switched = st.checkbox("Ever switched ART regimen")

        c12, c13, c14 = st.columns(3)
        with c12:
            dtg_transition = st.checkbox("DTG transition made",
                help="Patient transitioned to dolutegravir-based regimen.")
        with c13:
            supp_improving = st.checkbox("Suppression improving",
                help="Recent VL trending towards suppression.")
        with c14:
            vl_trend = st.selectbox("Viral load trend", ["Stable", "Declining", "Increasing"])

        # ── SECTION C: Adherence ─────────────────────────────────────────
        st.markdown('<div class="section-head">C — Adherence Assessment</div>', unsafe_allow_html=True)
        st.markdown("""<div class="ask-box"><b style="color:#c9d1d9">Ask the patient:</b>
        "In the past month, how often did you take your HIV medicines as prescribed?"
        "How many doses did you miss in the past 30 days?"</div>""", unsafe_allow_html=True)

        col_map = {'green':'#3fb950','amber':'#d29922','orange':'#e3894b','red':'#f85149'}

        c15, c16 = st.columns(2)
        with c15:
            adherence = st.slider("Self-reported adherence (%)", 0.0, 100.0, 72.0, step=1.0,
                help="% of doses taken as prescribed in the past month. >=95% = optimal.")
            adh_lbl, adh_col = adherence_level(adherence)
            st.markdown(f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:12px;color:{col_map[adh_col]};margin-top:-4px\">{adh_lbl}</div>", unsafe_allow_html=True)
        with c16:
            missed_d = st.slider("Mean missed doses / 30 days", 0.0, 15.0, 2.5, step=0.1)

        c17, c18, c19 = st.columns(3)
        with c17:
            appt_adh_pct = st.slider("Appointment adherence (%)", 0.0, 100.0, 80.0, step=1.0,
                help="% of scheduled appointments attended.")
        with c18:
            missed_appts = st.number_input("Missed appointments (last 12m)", 0, 24, 2)
        with c19:
            refill_gap   = st.number_input("Pharmacy refill gap (days)", 0, 180, 10,
                help="Mean days overdue between prescription refills.")

        c20, c21, c22 = st.columns(3)
        with c20:
            tx_int = st.checkbox("Treatment interruption history",
                help="Any prior ART gap >= 1 month.")
        with c21:
            mmd    = st.checkbox("Multi-month dispensing",
                help="Currently on 3+ month supply per visit.")
        with c22:
            mean_refill_months = st.number_input("Mean refill supply (months)", 1.0, 6.0, 1.0, step=0.5)

        c23, c24, c25 = st.columns(3)
        with c23:
            total_visits    = st.number_input("Total clinic visits", 0, 200, 12)
        with c24:
            followup_months = st.number_input("Follow-up duration (months)", 0, 240, 30)
        with c25:
            appts_kept      = st.number_input("Appointments kept (total)", 0, 200, 10)

        # ── SECTION D: Psychosocial ──────────────────────────────────────
        st.markdown('<div class="section-head">D — Psychosocial & Mental Health</div>', unsafe_allow_html=True)
        st.markdown("""<div class="ask-box"><b style="color:#c9d1d9">Validated tools:</b>
        PHQ-9 (depression) · Berger HIV Stigma Scale · Oslo Social Support (OSS-3) · AUDIT-C (alcohol)</div>""", unsafe_allow_html=True)

        c26, c27, c28 = st.columns(3)
        with c26:
            phq9   = st.slider("PHQ-9 depression score", 0, 27, 7,
                help="0-4: minimal | 5-9: mild | 10-14: moderate | 15-19: mod. severe | 20-27: severe")
            phq_lbl, phq_col = phq9_severity(phq9)
            st.markdown(f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:11px;color:{col_map[phq_col]}\">{phq_lbl}</div>", unsafe_allow_html=True)
        with c27:
            stigma = st.slider("HIV stigma score (0–40)", 0.0, 40.0, 15.0, step=0.5,
                help="Berger HIV Stigma Scale. >22 = high stigma.")
            stig_lbl, stig_col = stigma_level(stigma)
            st.markdown(f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:11px;color:{col_map[stig_col]}\">{stig_lbl} stigma</div>", unsafe_allow_html=True)
        with c28:
            social = st.slider("Social support (0–20)", 0.0, 20.0, 10.0, step=0.5,
                help="OSS-3 adapted. <8: low | 8-14: moderate | >=15: strong.")
            soc_lbl, soc_col = social_support_level(social)
            st.markdown(f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:11px;color:{col_map[soc_col]}\">{soc_lbl} support</div>", unsafe_allow_html=True)

        c29, c30 = st.columns(2)
        with c29:
            audit_c   = st.slider("AUDIT-C alcohol score (0–12)", 0, 12, 2,
                help=">=3 (women) or >=4 (men) = positive screen for hazardous drinking.")
        with c30:
            haz_drink = st.checkbox("Hazardous drinking confirmed")

        c31, c32, c33 = st.columns(3)
        with c31:
            mean_dep_visits    = st.number_input("Depression screening visits", 0, 100, 5)
        with c32:
            alcohol_use_visits = st.number_input("Alcohol use assessment visits", 0, 100, 2)
        with c33:
            mean_adh_visits    = st.number_input("Adherence counselling visits", 0, 100, 4)

        # ── SECTION E: IAC & Peer Support ───────────────────────────────
        st.markdown('<div class="section-head">E — IAC & Peer Support</div>', unsafe_allow_html=True)
        c34, c35, c36 = st.columns(3)
        with c34:
            iac_sessions  = st.number_input("IAC sessions received", 0, 20, 0,
                help="Intensive Adherence Counselling sessions received.")
        with c35:
            iac_completed = st.checkbox("IAC course completed",
                help="Full IAC course completed (typically 3-5 sessions).")
        with c36:
            peer_sessions = st.number_input("Peer support sessions", 0, 50, 0)

        # ── SECTION F: Social Context ────────────────────────────────────
        st.markdown('<div class="section-head">F — Social & Economic Context</div>', unsafe_allow_html=True)
        st.markdown("""<div class="ask-box"><b style="color:#c9d1d9">Ask the patient / caregiver:</b>
        "In the past month, did you worry about having enough food?" &nbsp;|&nbsp;
        "Has anyone else been told about your HIV status with your permission?"</div>""", unsafe_allow_html=True)

        c37, c38 = st.columns(2)
        with c37:
            food_ins  = st.checkbox("Food insecure (household)")
            disclosed = st.checkbox("HIV status disclosed",
                help="Patient has disclosed HIV status to at least one trusted person.")
        with c38:
            caregiver = st.selectbox("Caregiver support level", ["Low", "Medium", "High"],
                help="High: daily reminder + attends clinic. Medium: occasional. Low: unaware/uninvolved.")

        # ── SECTION G: Programme Enrolment ───────────────────────────────
        st.markdown('<div class="section-head">G — Current Programme Enrolment</div>', unsafe_allow_html=True)
        st.markdown("""<div class="ask-box">Tick interventions the patient is <b style="color:#c9d1d9">currently enrolled in</b>.
        The model computes counterfactual benefit of adding/removing each.</div>""", unsafe_allow_html=True)

        cg1, cg2, cg3 = st.columns(3)
        with cg1:
            in_otz  = st.checkbox("Currently in OTZ", value=False)
        with cg2:
            in_dsd  = st.checkbox("Currently in DSD", value=True)
        with cg3:
            in_yaps = st.checkbox("Currently has YAPS", value=False)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶Generate Clinical Recommendation", type="primary", use_container_width=True)

    # ── RESULTS PANEL ────────────────────────────────────────────────────
    with col_results:
        st.markdown('<div class="section-head">Causal inference output</div>', unsafe_allow_html=True)

        if run_btn:
            dep_mod, dep_mod_sev = depression_category(phq9)

            patient = {
                'age':                              age,
                'sex':                              1 if sex == 'Female' else 0,
                'boarding_school':                  int(boarding),
                'years_on_ART':                     years_art,
                'baseline_CD4':                     cd4_base,
                'current_CD4':                      cd4_curr,
                'BMI':                              bmi,
                'opportunistic_infections_last12m': oi_last,
                'drug_resistance_tested':           int(drug_res_tested),
                'drug_resistance_detected':         int(drug_res_detected),
                'food_insecurity':                  int(food_ins),
                'distance_to_clinic_km':            dist_clin,
                'caregiver_support':                {'Low':0,'Medium':1,'High':2}[caregiver],
                'HIV_status_disclosed':             int(disclosed),
                'stigma_score':                     stigma,
                'social_support_score':             social,
                'PHQ9_score':                       phq9,
                'alcohol_AUDITC_score':             audit_c,
                'hazardous_drinking':               int(haz_drink),
                'treatment_interruption':           int(tx_int),
                'missed_appointments_last12m':      missed_appts,
                'pharmacy_refill_gap_days':         refill_gap,
                'multi_month_dispensing':           int(mmd),
                'enrolled_in_OTZ':                 int(in_otz),
                'enrolled_in_DSD':                 int(in_dsd),
                'has_YAPS_support':                int(in_yaps),
                'adherence_self_report':            adherence,
                'total_clinic_visits':              total_visits,
                'followup_months':                  followup_months,
                'iac_sessions_received':            iac_sessions,
                'iac_completed':                    int(iac_completed),
                'peer_support_sessions':            peer_sessions,
                'appointments_kept':                appts_kept,
                'appointment_adherence_pct':        appt_adh_pct,
                'mean_missed_doses_30d':            missed_d,
                'mean_refill_months':               mean_refill_months,
                'mean_adherence_visits':            mean_adh_visits,
                'mean_depression_visits':           mean_dep_visits,
                'alcohol_use_visits':               alcohol_use_visits,
                'ever_switched_regimen':            int(ever_switched),
                'dtg_transition_made':              int(dtg_transition),
                'suppression_improving':            int(supp_improving),
                'district_Kampala':                 1 if district == 'Kampala' else 0,
                'district_Luwero':                  1 if district == 'Luwero'  else 0,
                'district_Masaka':                  1 if district == 'Masaka'  else 0,
                'district_Mityana':                 1 if district == 'Mityana' else 0,
                'district_Mpigi':                   1 if district == 'Mpigi'   else 0,
                'district_Mukono':                  1 if district == 'Mukono'  else 0,
                'district_Wakiso':                  1 if district == 'Wakiso'  else 0,
                'schooling_status_Primary':         1 if schooling == 'Primary'   else 0,
                'schooling_status_Secondary':       1 if schooling == 'Secondary' else 0,
                'schooling_status_Tertiary':        1 if schooling == 'Tertiary'  else 0,
                'ART_regimen_EFV-based':            1 if art_reg == 'EFV-based' else 0,
                'ART_regimen_PI-based':             1 if art_reg == 'PI-based'  else 0,
                'employment_status_Student':        1 if employment == 'Student'    else 0,
                'employment_status_Unemployed':     1 if employment == 'Unemployed' else 0,
                'vl_trend_direction_Increasing':    1 if vl_trend == 'Increasing' else 0,
                'vl_trend_direction_Stable':        1 if vl_trend == 'Stable'     else 0,
                'depression_category_Moderate':             dep_mod,
                'depression_category_Moderately Severe':    dep_mod_sev,
            }

            results, base_prob = run_inference(patient)
            flags     = moh_safety_check(age, phq9, years_art, adherence)
            best      = max(results, key=lambda k: results[k]['cate'])
            best_cate = results[best]['cate'] * 100

            risk_score          = compute_risk_score(phq9, adherence, stigma, food_ins, tx_int,
                                                     social, missed_d, vl_trend, haz_drink,
                                                     oi_last, missed_appts, refill_gap)
            risk_hex, risk_lbl  = risk_color(risk_score)

            # ── Baseline suppression card ──
            base_color = "green" if base_prob >= 0.5 else "red"
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
            </div>''', unsafe_allow_html=True)

            # ── Risk factor tiles ──
            st.markdown('<div class="section-head">Key risk factors</div>', unsafe_allow_html=True)

            def rt(label, value, level):
                c = {"high":"#f85149","medium":"#d29922","low":"#3fb950"}.get(level,"#8b949e")
                return f'<div class="risk-tile {level}"><div class="rtitle">{label}</div><div class="rval" style="color:{c}">{value}</div></div>'

            tiles = ""
            tiles += rt("Adherence", f"{adherence:.0f}% — {adherence_level(adherence)[0]}",
                        "high" if adherence < 70 else "medium" if adherence < 85 else "low")
            tiles += rt("PHQ-9 Depression", f"{phq9}/27 — {phq9_severity(phq9)[0]}",
                        "high" if phq9 >= 15 else "medium" if phq9 >= 10 else "low")
            tiles += rt("HIV Stigma", f"{stigma:.0f}/40 — {stigma_level(stigma)[0]}",
                        "high" if stigma > 22 else "medium" if stigma > 10 else "low")
            tiles += rt("Social Support", f"{social:.0f}/20 — {social_support_level(social)[0]}",
                        "high" if social < 8 else "medium" if social < 14 else "low")
            tiles += rt("Missed appointments (12m)", str(missed_appts),
                        "high" if missed_appts > 4 else "medium" if missed_appts > 1 else "low")
            tiles += rt("VL trend", vl_trend,
                        "high" if vl_trend == "Increasing" else "medium" if vl_trend == "Stable" else "low")
            if food_ins:              tiles += rt("Food insecurity", "YES", "high")
            if tx_int:                tiles += rt("Treatment interruption history", "YES", "high")
            if drug_res_detected:     tiles += rt("Drug resistance", "DETECTED", "high")
            if haz_drink:             tiles += rt("Hazardous drinking", "YES", "medium")
            if not disclosed:         tiles += rt("HIV status disclosed", "NO", "medium")
            st.markdown(tiles, unsafe_allow_html=True)

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
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="safety-box" style="border-color:#d29922">
                  <span class="safety-warn">No positive CATE detected for any intervention.</span><br>
                  All interventions show neutral or negative predicted effect. Clinical review recommended.
                </div>''', unsafe_allow_html=True)

            # ── Clinical narrative ──
            st.markdown('<div class="section-head">Clinical narrative</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="narrative-box">{narrative(patient, base_prob, results, best)}</div>', unsafe_allow_html=True)

            # ── Safety flags ──
            if flags:
                st.markdown('<div class="section-head">MoH 2020 safety filter</div>', unsafe_allow_html=True)
                for level, msg in flags:
                    icon  = "⚠" if level == "WARN" else "ℹ"
                    color = "safety-warn" if level == "WARN" else "blue"
                    st.markdown(f'<div class="safety-box"><span class="{color}">{icon} {msg}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('''<div class="safety-box">
                  <span class="safety-pass">✓ MoH 2020 ART safety check passed — no contraindications flagged</span>
                </div>''', unsafe_allow_html=True)

            

        else:
            st.markdown('''
            <div class="safety-box" style="margin-top:40px;text-align:center;padding:40px 24px">
                <div style="font-size:32px;margin-bottom:12px">🧬</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#8b949e">
                Complete all sections of the patient assessment<br>then click<br>
                <span style="color:#58a6ff">▶ Generate Clinical Recommendation </span>
                </div>
            </div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — Population ATEs
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">Population-level ATEs (G-Computation, test set)</div>', unsafe_allow_html=True)
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
    <b style="color:#c9d1d9">OTZ (+4.02%):</b> Enrolling in OTZ increases suppression probability by ~4 pp on average. Aligns with Uganda MoH evidence for adolescent peer-led adherence clubs.<br><br>
    <b style="color:#c9d1d9">DSD (+3.47%):</b> Differentiated service delivery shows consistent positive effect, especially for patients with long distances to clinic.<br><br>
    <b style="color:#c9d1d9">YAPS (+0.24%):</b> Small population-level ATE — individual CATE may be substantially higher for patients with high PHQ-9, stigma, or low social support.<br><br>
    <b style="color:#58a6ff">Note:</b> These are causal estimates from G-Computation, not correlations. DAG-based backdoor adjustment controls for confounding.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Model architecture summary</div>', unsafe_allow_html=True)
    summary = {
        'Architecture':         'G-Net (Sequential G-Computation MLP)',
        'Outcome model':        f'MLPClassifier {outcome_model.hidden_layer_sizes}, ReLU, early stopping',
        'Propensity models':    'MLPClassifier (64x32) per intervention (OTZ, DSD, YAPS)',
        'Training cohort':      '20,890 HIV-positive youth, Central Uganda ART clinics',
        'Feature count':        str(len(FEATURE_COLS)),
        'Train/Test split':     '80/20 stratified',
        'Best validation score':f'{MODEL_BEST_VAL:.4f}',
        'Training iterations':  str(getattr(outcome_model, "n_iter_", "N/A")),
        'Deployment format':    'joblib bundle (sklearn)',
        
    }
    for k, v in summary.items():
        st.markdown(f'''
        <div style="display:flex;border-bottom:1px solid #21262d;padding:8px 0;font-size:13px">
            <div style="font-family:'IBM Plex Mono',monospace;color:#8b949e;width:240px;flex-shrink:0">{k}</div>
            <div style="font-family:'IBM Plex Sans',sans-serif;color:#c9d1d9">{v}</div>
        </div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — Feature Reference
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">All 60 model features — clinical reference</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:#8b949e;margin-bottom:16px">
    Complete feature list used by the G-Net outcome model. [OHE] = one-hot encoded; reference category shown in parentheses.
    </div>""", unsafe_allow_html=True)

    features_ref = [
        ("Demographics & ART", [
            ("age", "10–24", "Patient age. Youth cohort only."),
            ("sex", "0=Male, 1=Female", "Biological sex."),
            ("boarding_school", "0/1", "Living at boarding school — affects caregiver access."),
            ("years_on_ART", "0–20 yrs", "Duration on antiretroviral therapy."),
            ("BMI", "10–45 kg/m²", "Nutritional status. Low BMI = disease progression risk."),
            ("distance_to_clinic_km", "0–100 km", "Barrier to care. Key predictor for DSD benefit."),
            ("district [OHE]", "Kampala / Luwero / Masaka / Mityana / Mpigi / Mukono / Wakiso", "Home district within Central Uganda."),
            ("schooling_status [OHE]", "Primary / Secondary / Tertiary (ref=None)", "Current or highest schooling level."),
            ("employment_status [OHE]", "Student / Unemployed (ref=Employed)", "Employment status."),
        ]),
        ("Clinical & Laboratory", [
            ("baseline_CD4", "cells/µL", "CD4 at ART initiation — sets trajectory baseline."),
            ("current_CD4", "cells/µL", "Latest CD4 count. >=500 = stable immune function."),
            ("opportunistic_infections_last12m", "Count", "Recent OIs indicate immunosuppression or treatment failure."),
            ("drug_resistance_tested", "0/1", "Whether drug resistance testing has been performed."),
            ("drug_resistance_detected", "0/1", "Confirmed drug resistance — major suppression barrier."),
            ("ever_switched_regimen", "0/1", "Prior ART regimen change (failure, toxicity, or upgrade)."),
            ("dtg_transition_made", "0/1", "Transitioned to dolutegravir-based regimen."),
            ("suppression_improving", "0/1", "Recent VL trending towards suppression."),
            ("ART_regimen [OHE]", "EFV-based / PI-based (ref=DTG-based)", "Current ART regimen."),
            ("vl_trend_direction [OHE]", "Increasing / Stable (ref=Declining)", "Direction of viral load over recent measurements."),
        ]),
        ("Adherence", [
            ("adherence_self_report", "0–100%", "Self-reported dose adherence. >=95% = optimal."),
            ("mean_missed_doses_30d", "0–15", "Mean missed ART doses per month."),
            ("treatment_interruption", "0/1", "Any prior ART gap >= 1 month."),
            ("missed_appointments_last12m", "Count", "Clinic appointments missed in past 12 months."),
            ("pharmacy_refill_gap_days", "Days", "Mean days overdue between refills."),
            ("multi_month_dispensing", "0/1", "On multi-month (3m+) dispensing schedule."),
            ("total_clinic_visits", "Count", "Cumulative ART clinic visits."),
            ("followup_months", "Months", "Total follow-up duration since ART start."),
            ("appointments_kept", "Count", "Total scheduled appointments attended."),
            ("appointment_adherence_pct", "0–100%", "% of scheduled appointments attended."),
            ("mean_refill_months", "1–6 months", "Average supply dispensed per refill."),
            ("mean_adherence_visits", "Count", "Visits with adherence counselling recorded."),
        ]),
        ("Psychosocial & Mental Health", [
            ("PHQ9_score", "0–27", "Depression: 0-4 minimal, 5-9 mild, 10-14 moderate, >=15 severe."),
            ("depression_category [OHE]", "Moderate / Moderately Severe (ref=Minimal/Mild)", "Derived from PHQ-9 score."),
            ("stigma_score", "0–40", "Berger HIV Stigma Scale. >22 = high stigma."),
            ("social_support_score", "0–20", "OSS-3 adapted. <8=low, 8-14=moderate, >=15=strong."),
            ("alcohol_AUDITC_score", "0–12", "AUDIT-C. >=3 (women) or >=4 (men) = hazardous drinking."),
            ("hazardous_drinking", "0/1", "Clinician-confirmed hazardous alcohol use."),
            ("mean_depression_visits", "Count", "Visits with depression screening/management."),
            ("alcohol_use_visits", "Count", "Visits with alcohol use assessed."),
        ]),
        ("IAC & Peer Support", [
            ("iac_sessions_received", "Count", "Intensive Adherence Counselling sessions received."),
            ("iac_completed", "0/1", "Full IAC course completed (typically 3-5 sessions)."),
            ("peer_support_sessions", "Count", "Peer mentor/peer support sessions attended."),
        ]),
        ("Social & Economic Context", [
            ("food_insecurity", "0/1", "Household food insecurity — associated with missed doses."),
            ("HIV_status_disclosed", "0/1", "Patient has disclosed HIV status to a trusted person."),
            ("caregiver_support", "0=Low, 1=Med, 2=High", "Active caregiver involvement in ART adherence."),
        ]),
        ("Programme Enrolment (Interventions)", [
            ("enrolled_in_OTZ", "0/1", "Optimised Treatment for Adolescents — Population ATE: +4.02%."),
            ("enrolled_in_DSD", "0/1", "Differentiated Service Delivery — Population ATE: +3.47%."),
            ("has_YAPS_support", "0/1", "Youth Adherence & Psychosocial Support — Population ATE: +0.24%."),
        ]),
    ]

    for section_name, rows in features_ref:
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#58a6ff;text-transform:uppercase;letter-spacing:1px;margin:20px 0 8px 0;border-bottom:1px solid #21262d;padding-bottom:4px">{section_name}</div>', unsafe_allow_html=True)
        table_html = """<table style="width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Sans',sans-serif">
        <thead><tr>
          <th style="text-align:left;padding:8px 12px;color:#8b949e;font-family:'IBM Plex Mono',monospace;font-size:11px;border-bottom:1px solid #21262d;width:22%">Feature</th>
          <th style="text-align:left;padding:8px 12px;color:#8b949e;font-family:'IBM Plex Mono',monospace;font-size:11px;border-bottom:1px solid #21262d;width:20%">Range / Values</th>
          <th style="text-align:left;padding:8px 12px;color:#8b949e;font-family:'IBM Plex Mono',monospace;font-size:11px;border-bottom:1px solid #21262d">Clinical interpretation</th>
        </tr></thead><tbody>"""
        for fname, frange, fdesc in rows:
            table_html += f"""<tr>
              <td style="padding:8px 12px;color:#c9d1d9;font-family:'IBM Plex Mono',monospace;border-bottom:1px solid #161b22">{fname}</td>
              <td style="padding:8px 12px;color:#58a6ff;border-bottom:1px solid #161b22">{frange}</td>
              <td style="padding:8px 12px;color:#8b949e;border-bottom:1px solid #161b22;line-height:1.5">{fdesc}</td>
            </tr>"""
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
