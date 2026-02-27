import streamlit as st
import pandas as pd
import numpy as np
import os, time, random, base64
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI as _OpenAI
    _openai_available = True
    _ai_client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
except ImportError:
    _openai_available = False
    _ai_client = None

# ============================================================
#  LOGO / BRANDING
# ============================================================
@st.cache_data
def get_base64_img(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
    return ""
LOGO_B64 = get_base64_img("Logo.jpeg")

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Ayushman Bharat – Fraud Intelligence",
    page_icon="Logo.jpeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  GLOBAL CSS  — Clean Healthcare White Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset ── */
html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }
.stApp { background:#F3F4F6 !important; color:#1F2937 !important; }

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility:hidden; height:0; display:none; }
[data-testid="stSidebarNav"] { display:none; }
[data-testid="stHeader"] { display:none; }
[data-testid="stDecoration"] { display:none; }
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background:#FFFFFF !important;
    border-right:1px solid #E5E7EB;
    box-shadow:2px 0 12px rgba(22,163,74,.07);
    width:240px !important;
}
[data-testid="stSidebar"] > div { padding:0 !important; }

/* ── Sidebar nav buttons ── */
.stButton > button {
    background:transparent !important;
    color:#4B5563 !important;
    border:none !important;
    border-radius:10px !important;
    font-size:.9rem !important;
    font-weight:500 !important;
    text-align:left !important;
    padding:10px 16px !important;
    width:100% !important;
    transition:all .15s !important;
}
.stButton > button:hover {
    background:#DCFCE7 !important;
    color:#16A34A !important;
}
.nav-active > .stButton > button {
    background:#DCFCE7 !important;
    color:#16A34A !important;
    font-weight:700 !important;
}

/* ── Main metric cards ── */
.kpi-card {
    background:#fff;
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:22px 24px;
    box-shadow:0 2px 12px rgba(22,163,74,.06);
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    height:130px;
}
.kpi-icon {
    width:44px; height:44px;
    border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    font-size:1.3rem;
}
.kpi-label { font-size:.9rem; font-weight:500; color:#1F2937; margin-bottom:6px; }
.kpi-value { font-size:2.2rem; font-weight:800; color:#14532D; line-height:1; }
.kpi-delta-up   { font-size:.75rem; font-weight:700; color:#16A34A; margin-left:6px; }
.kpi-delta-warn { font-size:.75rem; font-weight:700; color:#DC2626; margin-left:6px; }
.kpi-sub   { font-size:.73rem; color:#1F2937; margin-top:4px; }

/* ── Section card ── */
.section-card {
    background:#fff;
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:22px 24px;
    box-shadow:0 2px 12px rgba(22,163,74,.05);
    margin-bottom:18px;
}
.section-title {
    font-size:1.2rem; font-weight:700; color:#14532D;
    display:flex; align-items:center; gap:8px;
    margin-bottom:16px;
}
.section-link {
    font-size:.8rem; font-weight:600; color:#16A34A;
    cursor:pointer; text-decoration:none;
}

/* ── Timeline ── */
.timeline-item {
    display:flex; gap:14px; padding:12px 0;
    border-bottom:1px solid #DCFCE7;
}
.timeline-dot {
    width:36px; height:36px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:.9rem; flex-shrink:0;
    margin-top:2px;
}
.timeline-title { font-size:.88rem; font-weight:700; color:#14532D; }
.timeline-time  { font-size:.72rem; font-weight:600; color:#1F2937; margin-left:8px; }
.timeline-desc  { font-size:.8rem; color:#4B5563; margin-top:3px; line-height:1.5; }

/* ── Feature Cards ── */
.feat-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 24px;
    padding: 28px; box-shadow: 0 4px 24px rgba(0,0,0,0.02);
    margin-bottom: 24px; transition: all 0.3s ease;
}
.feat-card:hover { transform: translateY(-5px); }
.feat-icon { font-size: 2rem; margin-bottom: 12px; color: #16A34A; }
.feat-title { font-size: 1.25rem; font-weight: 800; color: #14532D; margin-bottom: 8px; }
.feat-desc { font-size: 1.1rem; color: #1F2937; line-height: 1.6; }

/* ── Flagged claims table ── */
.claims-table { width:100%; border-collapse:collapse; font-size:.83rem; }
.claims-table th {
    color:#1F2937; font-weight:700; font-size:.7rem;
    text-transform:uppercase; letter-spacing:.6px;
    padding:8px 12px; border-bottom:2px solid #DCFCE7;
    text-align:left;
}
.claims-table td { padding:12px 12px; border-bottom:1px solid #F9FAFB; }
.claims-table tr:hover td { background:#F0FDF4; }
.claim-id   { font-weight:700; color:#16A34A; }

/* ── Status Tags ── */
.tag-high   { background:#FEE2E2; color:#DC2626; border:1px solid #FECACA; border-radius:20px; padding:3px 10px; font-size:.7rem; font-weight:700; }
.tag-inv    { background:#FEF3C7; color:#D97706; border:1px solid #FDE68A; border-radius:20px; padding:3px 10px; font-size:.7rem; font-weight:700; }
.tag-safe   { background:#DCFCE7; color:#16A34A; border:1px solid #BBF7D0; border-radius:20px; padding:3px 10px; font-size:.7rem; font-weight:700; }

/* ── Input Fields & Labels ── */
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label {
    color: #14532D !important; font-weight: 700 !important; font-size: 0.95rem !important;
    margin-bottom: 8px !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #16A34A !important;
    box-shadow: 0 0 0 4px rgba(22, 163, 74, 0.1) !important;
    background-color: #FAFAFA !important;
}

/* ── Floating Chatbot ── */
/* ── Chat Visibility & Clean UI ── */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 16px !important;
    padding: 18px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
}
[data-testid="stChatMessage"] p, 
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #1F2937 !important;
    font-size: 1.15rem !important;
    line-height: 1.6 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: #111827 !important;
    font-weight: 500 !important;
}
/* Force white text for user inputs in chat because of dark background */
[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* ── Upload Zone ── */
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: 2px dashed #D1D5DB !important;
    border-radius: 24px !important;
    transition: all 0.3s ease !important;
    padding: 60px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #16A34A !important;
    background: #F0FDF4 !important;
}
/* Force absolute black for all upload section text (instructions & sub-text) */
[data-testid="stFileUploaderDropzone"] *:not(button):not(button *) {
    color: #000000 !important;
    opacity: 1 !important;
}

/* Browse Files Button Fix - Keep text white */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] button div {
    color: white !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color: #16A34A !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(22, 163, 74, 0.2) !important;
}

/* ── Metrics & Cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03) !important;
    transition: transform 0.3s ease !important;
}
[data-testid="metric-container"]:hover { transform: translateY(-4px); }

.section-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 24px;
    padding: 28px; box-shadow: 0 4px 24px rgba(0,0,0,0.02);
    margin-bottom: 24px; transition: all 0.3s ease;
}
.section-card:hover { box-shadow: 0 12px 40px rgba(22,163,74,0.08); transform: translateY(-2px); }

/* ── Premium Landing ── */
.hero-section {
    background: linear-gradient(135deg, #14532D 0%, #16A34A 100%);
    color: white; padding: 60px 50px; border-radius: 32px;
    margin-bottom: 40px; position: relative; overflow: hidden;
    box-shadow: 0 20px 40px rgba(20, 83, 45, 0.2);
}
.hero-section::after {
    content: ''; position: absolute; right: -40px; bottom: -40px;
    width: 300px; height: 300px; background: url('""" + LOGO_B64 + """') no-repeat center center;
    background-size: contain; opacity: 0.07; transform: rotate(-20deg);
}

/* ── Auth Container ── */
.auth-container {
    max-width: 480px; margin: 0 auto;
}

/* ── Primary Button Overhaul ── */
.stButton button[kind="primary"],
div[data-testid="stBaseButton-primary"] button {
    background-color: #16A34A !important;
    background: #16A34A !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    height: 48px !important;
    width: 100% !important;
    font-weight: 700 !important;
    margin-top: 10px !important;
}
.stButton button[kind="primary"]:hover {
    background-color: #14532D !important;
    box-shadow: 0 4px 12px rgba(22, 163, 74, 0.2) !important;
}

/* ── Secondary Button (Forgot Password) ── */
.stButton button[kind="secondary"] {
    color: white !important;
    background: #1F2937 !important;
    border-radius: 10px !important;
    border: none !important;
}

/* ── Tabs Underline ── */
button[data-baseweb="tab"] { color: #1F2937 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #16A34A !important; border-bottom-color: #16A34A !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #16A34A !important;
}
</style>
""", unsafe_allow_html=True)

# ── Supabase (graceful if not configured yet) ───────────────
try:
    import supabase_db as sb
    _supabase_ready = sb.is_configured()
except Exception:
    _supabase_ready = False
    sb = None

# ── Global Session Cache ─────────────────────────────────────
@st.cache_resource
def get_session_cache():
    return {}

_global_sessions = get_session_cache()

# ============================================================
#  SESSION STATE INITIALIZATION
# ============================================================
for k, v in [("page","Account"),("contamination",.12),("n_estimators",200),
              ("df",None),("cost_col",None),("chat_open",False),("chat_history",[]),
              ("user",None),("uid",None),("auth_mode","login")]:
    if k not in st.session_state: st.session_state[k] = v

# --- SESSION RECOVERY LOGIC ---
if st.session_state.user is None:
    # 1. Try URL parameters first
    sid = st.query_params.get("s")
    if sid and sid in _global_sessions:
        cached = _global_sessions[sid]
        st.session_state.user = cached.get("user")
        st.session_state.uid = cached.get("uid")
        # Don't overwrite the page if it was explicitly set, but default to Dashboard if recovering
        if st.session_state.page == "Account":
            st.session_state.page = "Home"
        st.rerun()
    
    # 2. As a last resort, check Supabase directly (might work if client stayed alive)
    elif _supabase_ready:
        try:
            usr = sb.get_current_user()
            if usr:
                st.session_state.user = usr
                st.session_state.uid = usr.id
        except:
            pass

# --- VALIDATE PAGE ---
if st.session_state.user is None:
    st.session_state.page = "Account"
elif st.session_state.page not in ["Home", "Report", "Account", "Settings", "Chat"]:
    st.session_state.page = "Home"

# ============================================================
#  PIPELINE (CACHED FOR SPEED)
# ============================================================
@st.cache_data(show_spinner=False)
def run_pipeline(df, contamination, n_estimators):
    cost_col = ("Final_Billed_Amount" if "Final_Billed_Amount" in df.columns else
                "TreatmentCost"       if "TreatmentCost"       in df.columns else None)
    for col in ["Admission_Timestamp","Discharge_Timestamp","PreAuth_Request_Date","PreAuth_Approval_Date"]:
        if col in df.columns: df[col] = pd.to_datetime(df[col], errors="coerce")
    df["LOS"] = ((df["Discharge_Timestamp"]-df["Admission_Timestamp"]).dt.days
                 if "Discharge_Timestamp" in df.columns and "Admission_Timestamp" in df.columns else 1)
    df["PreAuth_Delay"] = ((df["PreAuth_Approval_Date"]-df["PreAuth_Request_Date"]).dt.days
                           if "PreAuth_Approval_Date" in df.columns else 0)
    df["Cost_to_Package"] = (df[cost_col]/df["Base_Package_Rate"].replace(0,1)
                             if cost_col and "Base_Package_Rate" in df.columns else 1.0)
    if "Hospital_PIN" in df.columns and cost_col:
        df["Hospital_Avg_Cost"] = df["Hospital_PIN"].map(df.groupby("Hospital_PIN")[cost_col].mean())
    else:
        df["Hospital_Avg_Cost"] = df[cost_col] if cost_col else 0
    if "PatientID" in df.columns and "TransactionID" in df.columns:
        df["Patient_Claim_Count"] = df["PatientID"].map(df.groupby("PatientID")["TransactionID"].count())
    else:
        df["Patient_Claim_Count"] = 1

    df["Rule_Fraud"] = 0
    if "Base_Package_Rate" in df.columns: df.loc[df["Base_Package_Rate"]==0,"Rule_Fraud"] = 1
    df.loc[df["Cost_to_Package"]>2.5,"Rule_Fraud"] = 1
    df.loc[df["LOS"]<=0,"Rule_Fraud"] = 1
    if "Gender" in df.columns and "Primary_Diagnosis" in df.columns:
        df.loc[(df["Gender"]=="Male")&(df["Primary_Diagnosis"]=="Maternity Care"),"Rule_Fraud"] = 1

    fcols = [c for c in [cost_col,"LOS","Cost_to_Package","PreAuth_Delay","Hospital_Avg_Cost","Patient_Claim_Count","Age"] if c and c in df.columns]
    iso = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
    df["ML_Anomaly"] = iso.fit_predict(df[fcols].fillna(0))

    df["Risk_Score"] = (df["Rule_Fraud"]*.5 + (df["ML_Anomaly"]==-1).astype(int)*.4 + (df["Cost_to_Package"]>2).astype(int)*.1).round(2)
    df["Risk score per claim"] = df["Risk_Score"]
    df["Fraud_Flag"]  = (df["Risk_Score"]>.5).astype(int)
    df["Suspicion_Score"] = (df["Risk_Score"]*100).round(0).astype(int)

    def classify(r):
        if "Base_Package_Rate" in r and r["Base_Package_Rate"]==0: return "Ghost Billing"
        elif r.get("Cost_to_Package",1)>2.5: return "Up-coding"
        elif r.get("Patient_Claim_Count",1)>2: return "Identity Misuse"
        elif r.get("LOS",1)<=0: return "Fake Admission"
        else: return "Anomalous Pattern"

    df["Fraud_Type"] = df.apply(classify, axis=1)

    def smart_just(row):
        if row["Fraud_Flag"]!=1: return ""
        cost  = float(row[cost_col]) if cost_col and cost_col in row.index else 0
        age   = row.get("Age","?"); ftype = row.get("Fraud_Type","Anomalous Pattern")
        hosp  = row.get("Hospital_PIN",row.get("HospitalID","?"))
        los   = row.get("LOS","?"); ctp = float(row.get("Cost_to_Package",1))
        diag  = row.get("Primary_Diagnosis","Unknown"); pid = row.get("PatientID","?")
        risk  = float(row.get("Risk_Score",0))
        if ftype=="Ghost Billing":
            return (f"GHOST BILLING DETECTED — Hospital {hosp} submitted ₹{cost:,.0f} for patient {pid} "
                    f"(Age: {age}, Dx: {diag}), but the base package rate is ₹0 — no legitimate procedure "
                    f"was registered. Under AB guidelines, a zero package rate triggers immediate claim rejection "
                    f"and potential hospital de-empanelment. Freeze payment and schedule field verification.")
        elif ftype=="Up-coding":
            return (f"UP-CODING FRAUD DETECTED — Billed ₹{cost:,.0f} at hospital {hosp} is {ctp:.1f}x "
                    f"the approved package rate for '{diag}'. Ayushman Bharat caps claims at 2.5x base rate; "
                    f"this breach indicates deliberate billing for a higher-complexity procedure than was performed. "
                    f"Risk Score: {risk:.2f}/1.00. Request procedure records & discharge summary for cross-verification.")
        elif ftype=="Fake Admission":
            return (f"FAKE ADMISSION — Patient {pid} (Age: {age}) at hospital {hosp} has LOS of {los} days "
                    f"(zero or negative) yet ₹{cost:,.0f} was billed for '{diag}'. Records appear fabricated to "
                    f"trigger reimbursement without actual treatment. Verify admission/discharge records directly "
                    f"with the hospital and cross-check with the patient.")
        elif ftype=="Identity Misuse":
            return (f"IDENTITY MISUSE — PatientID {pid} (Age: {age}) appears across multiple claims, suggesting "
                    f"the same identity is being reused to generate repeated reimbursements. The current claim of "
                    f"₹{cost:,.0f} for '{diag}' at hospital {hosp} is part of a serial billing pattern. "
                    f"Audit all claims under {pid}, verify Aadhaar linkage, and check for simultaneous admissions.")
        else:
            return (f"ANOMALOUS PATTERN — ML model flagged patient {pid} (Age: {age}) at hospital {hosp} "
                    f"with risk score {risk:.2f}/1.00. Billed ₹{cost:,.0f}, LOS {los} days for '{diag}' deviates "
                    f"significantly from peer benchmarks. Possible service bundling, unnecessary procedures, or "
                    f"inflated consumables. Request itemised billing and clinical notes for review.")

    df["AI_Justification"] = df.apply(smart_just, axis=1)
    return df, cost_col


# ============================================================
#  HELPERS
# ============================================================
def status_tag(score):
    if score>=70: return "<span class='tag-high'>High Risk</span>"
    elif score>=40: return "<span class='tag-inv'>Investigating</span>"
    else: return "<span class='tag-safe'>Safe</span>"

def score_bar_html(score):
    color = "#E74C3C" if score>=70 else "#F39C12" if score>=40 else "#27ae60"
    return (f"<div class='score-bar-wrap'>"
            f"<div style='flex:1; background:#F0F5FF; border-radius:3px; height:6px;'>"
            f"<div class='score-bar' style='width:{score}%; background:{color};'></div></div>"
            f"<span class='score-text' style='color:{color};'>{score}</span></div>")

def fmt_crore(amount):
    if amount>=10_000_000: return f"₹{amount/10_000_000:.1f} Cr"
    elif amount>=100_000:  return f"₹{amount/100_000:.1f} L"
    else:                  return f"₹{amount:,.0f}"

# ── Generate synthetic daily fraud trend data ─────────────────
@st.cache_data
def make_trend_data(n_days=30, seed=42):
    rng  = np.random.default_rng(seed)
    base = datetime.today() - timedelta(days=n_days)
    dates  = [base + timedelta(days=i) for i in range(n_days)]
    flagged = rng.integers(2, 12, size=n_days).tolist()
    total   = rng.integers(15, 40, size=n_days).tolist()
    return pd.DataFrame({"Date":dates,"Flagged":flagged,"Total":total})

# ── Generate synthetic audit timeline ─────────────────────────
def make_audit_log(fraud_df):
    events = [
        {"icon":"✅","color":"#27ae60","bg":"#EAFAF1","title":"Audit Completed",
         "desc":"Investigator verified high-risk claim — payment hold confirmed.","ago":"2H AGO","tag":None},
        {"icon":"🚩","color":"#E74C3C","bg":"#FDECEA","title":"New Flag Raised",
         "desc":"System flagged suspicious billing pattern at Regional Medical Center.","ago":"5H AGO","tag":None},
        {"icon":"📋","color":"#007BFF","bg":"#EBF4FF","title":"Batch Analysis Complete",
         "desc":f"Pipeline processed {len(fraud_df)} fraud cases from latest upload.","ago":"8H AGO","tag":None},
        {"icon":"🔒","color":"#8E44AD","bg":"#F5EEF8","title":"Payment Freeze Applied",
         "desc":"Treasury notified — 3 high-risk hospital payments suspended pending audit.","ago":"1D AGO","tag":None},
        {"icon":"📊","color":"#F39C12","bg":"#FFF8E1","title":"Weekly Report Generated",
         "desc":"Fraud rate this week: 13.3%. Up 2.1% from last week.","ago":"2D AGO","tag":None},
    ]
    # Inject real flagged patient at top if available
    if not fraud_df.empty:
        pid  = fraud_df.iloc[0].get("PatientID","CLM-001")
        hosp = fraud_df.iloc[0].get("Hospital_PIN","Unknown")
        events[0]["title"] = f"Audit Completed: {pid}"
        events[0]["desc"]  = f"Investigator verified high-risk claim for {pid} at hospital {hosp}."
        if len(fraud_df)>1:
            pid2  = fraud_df.iloc[1].get("PatientID","CLM-002")
            hosp2 = fraud_df.iloc[1].get("Hospital_PIN","Unknown")
            events[1]["desc"] = f"System flagged suspicious billing pattern for {pid2} at hospital {hosp2}."
    return events


# ============================================================
#  AUTO-LOAD DATA  (Supabase cloud-first, then local CSV)
# ============================================================
CSV_PATH = "ayushman_claims.csv"

if st.session_state.df is None:
    # 1️⃣  Try Supabase
    if _supabase_ready and st.session_state.uid:
        try:
            with st.spinner("☁️ Loading your claims from Supabase..."):
                cloud_df = sb.fetch_data_from_supabase(user_id=st.session_state.uid) # Filter by user
            if not cloud_df.empty:
                st.session_state.df, st.session_state.cost_col = run_pipeline(
                    cloud_df,
                    st.session_state.contamination,
                    st.session_state.n_estimators
                )
        except Exception as _sb_err:
            st.warning(f"⚠️ Supabase load failed: {_sb_err}. Falling back to local CSV.")

    # 2️⃣  Fall back to local CSV
    if st.session_state.df is None and os.path.exists(CSV_PATH):
        st.session_state.df, st.session_state.cost_col = run_pipeline(
            pd.read_csv(CSV_PATH),
            st.session_state.contamination,
            st.session_state.n_estimators
        )


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style='padding:22px 20px 16px; border-bottom:1px solid #E5E7EB;'>
        <div style='display:flex;align-items:center;gap:12px;'>
            <img src='{LOGO_B64}' style='width:38px;height:38px;border-radius:8px;object-fit:cover;'>
            <div>
                <div style='font-size:.95rem;font-weight:800;color:#14532D;'>Ayushman Bharat</div>
                <div style='font-size:.65rem;font-weight:600;color:#16A34A;letter-spacing:.5px;'>PM-JAY SYSTEM</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:12px 12px 4px;'>", unsafe_allow_html=True)

    if st.session_state.user:
        pages = {"Home": "Home", "Fraud Audit Report": "Report",
                 "AI Assistant": "Chat", "Account": "Account", "Settings": "Settings"}
        for label, key in pages.items():
            is_active = st.session_state.page == key
            btn_style = ("background:#DCFCE7;color:#16A34A;font-weight:700;" if is_active
                         else "background:transparent;color:#4B5563;font-weight:500;")
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
    else:
        st.info("🔒 Please Login to access the system.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Data source status chips ──────────────────────
    sb_status  = ("Live: Supabase" if _supabase_ready else "Local: CSV")
    sb_color   = "#27ae60" if _supabase_ready else "#F39C12"
    df_loaded  = st.session_state.df is not None
    n_rows     = len(st.session_state.df) if df_loaded else 0
    n_fraud    = int(st.session_state.df["Fraud_Flag"].sum()) if df_loaded and "Fraud_Flag" in st.session_state.df.columns else 0
    st.markdown(f"""
    <div style='margin:8px 12px;padding:10px 14px;background:#F9FAFB;
                border:1px solid #E5E7EB;border-radius:10px;'>
        <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
            <span style='font-size:.68rem;color:#1F2937;font-weight:600;'>DATA SOURCE</span>
            <span style='font-size:.68rem;font-weight:700;color:{sb_color};'>{sb_status}</span>
        </div>
        <div style='font-size:.72rem;color:#4B5563;'>{'<b>'+str(n_rows)+"</b> claims &nbsp;|&nbsp; <b style='color:#DC2626;'>"+str(n_fraud)+"</b> flagged" if df_loaded else 'No data loaded'}</div>
    </div>
    """, unsafe_allow_html=True)

    # User Profile at Bottom
    user = st.session_state.user
    user_display = user.email.split("@")[0].title() if user else "Guest Investigator"
    user_role = "Chief Auditor" if user else "Limited Access"
    user_initial = user_display[0].upper() if user else "G"

    st.markdown(f"""
    <div style='position:absolute;bottom:0;left:0;right:0;padding:16px 20px;
                border-top:1px solid #E5E7EB;display:flex;align-items:center;gap:12px;
                background:#fff;'>
        <div style='width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#16A34A,#22C55E);
                    display:flex;align-items:center;justify-content:center;color:#fff;font-size:.9rem;font-weight:800;'>{user_initial}</div>
        <div>
            <div style='font-size:.85rem;font-weight:700;color:#14532D;'>{user_display}</div>
            <div style='font-size:.7rem;color:#1F2937;'>{user_role}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Navigation & Routing Logic Updated ──

# ── Legacy Floating Logic Removed ──


# ── Navigation & Routing Logic Updated ──

# ============================================================
#  PAGE: FRAUD AUDIT REPORT
# ============================================================
if st.session_state.page == "Report":
    st.markdown("""
    <div style='padding:6px 0 18px;'>
      <span style='font-size:1.8rem;font-weight:800;color:#14532D;'>Fraud Audit Report</span>
      <span style='color:#DC2626;font-size:1.8rem;'> — Critical Investigation Queue</span>
    </div>""", unsafe_allow_html=True)

    if not _supabase_ready:
        st.error("❌ Live database required for Audit Reports.")
        st.stop()

    with st.spinner("📥 Fetching your critical cases from Cloud..."):
        frauds_df = sb.fetch_detected_frauds(user_id=st.session_state.uid)

    if frauds_df.empty:
        st.info("✅ No critical frauds pending review in the cloud database.")
    else:
        # ── Premium Forensic Summary ──
        total_amt = frauds_df.get("Final_Billed_Amount", pd.Series([0])).sum()
        avg_risk = frauds_df.get("Risk_Score", pd.Series([0])).mean() * 100
        
        summary_html = f"""
        <div style='background: white; border: 1px solid #E5E7EB; border-radius: 20px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.02);'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
                <div style='font-size: 1.4rem; font-weight: 800; color: #14532D;'>Forensic Audit Summary</div>
                <div style='background: #DCFCE7; color: #16A34A; padding: 5px 15px; border-radius: 20px; font-size: 0.85rem; font-weight: 800;'>SECURE CLOUD DATA</div>
            </div>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;'>
                <div style='background: #F9FAFB; padding: 15px; border-radius: 12px; border: 1px solid #F3F4F6;'>
                    <div style='font-size: 0.85rem; color: #6B7280; font-weight: 700; text-transform: uppercase;'>Investigation Queue</div>
                    <div style='font-size: 2.2rem; font-weight: 800; color: #111827;'>{len(frauds_df)} <span style='font-size: 1rem; color: #6B7280;'>Cases</span></div>
                </div>
                <div style='background: #FFF7ED; padding: 15px; border-radius: 12px; border: 1px solid #FFEDD5;'>
                    <div style='font-size: 0.85rem; color: #9A3412; font-weight: 700; text-transform: uppercase;'>Pipeline Value</div>
                    <div style='font-size: 2.2rem; font-weight: 800; color: #9A3412;'>{fmt_crore(total_amt)}</div>
                </div>
                <div style='background: #FEF2F2; padding: 15px; border-radius: 12px; border: 1px solid #FEE2E2;'>
                    <div style='font-size: 0.85rem; color: #991B1B; font-weight: 700; text-transform: uppercase;'>Avg Risk Level</div>
                    <div style='font-size: 2.2rem; font-weight: 800; color: #991B1B;'>{int(avg_risk)}%</div>
                </div>
            </div>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)
        
        # ── Category Distribution (Split View) ──
        c1, c2 = st.columns([1.5, 1])
        with c1:
            if "Fraud_Type" in frauds_df.columns:
                chart_data = frauds_df["Fraud_Type"].value_counts().reset_index()
                chart_data.columns = ["Category", "Counts"]
                st.markdown("<div style='font-size:0.95rem; font-weight:700; color:#4B5563; margin-bottom:12px; letter-spacing:0.5px;'>CATEGORY DISTRIBUTION</div>", unsafe_allow_html=True)
                st.bar_chart(chart_data.set_index("Category"), color="#16A34A", use_container_width=True)
        
        with c2:
            st.markdown("<div style='font-size:0.95rem; font-weight:700; color:#4B5563; margin-bottom:12px; letter-spacing:0.5px;'>RECENT ALERTS</div>", unsafe_allow_html=True)
            for _, row in frauds_df.head(3).iterrows():
                pid = row.get("PatientID", "—")
                ft = row.get("Fraud_Type", "—")
                st.markdown(f"""
                <div style='background:white; border:1px solid #F3F4F6; padding:12px 18px; border-radius:10px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <div style='font-size:1.1rem; font-weight:700; color:#111827;'>{pid}</div>
                        <div style='font-size:0.85rem; color:#6B7280;'>{ft}</div>
                    </div>
                    <div style='width:8px; height:8px; border-radius:50%; background:#DC2626;'></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"<p style='color:#14532D; font-weight:700; font-size:1.25rem;'>Listing {len(frauds_df)} priority investigations:</p>", unsafe_allow_html=True)
        
        # Display Cards in a 3-column grid
        for i in range(0, len(frauds_df), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(frauds_df):
                    row = frauds_df.iloc[i + j]
                    pid    = row.get("PatientID", "—")
                    age    = row.get("Age", "—")
                    diag   = row.get("Primary_Diagnosis", "—")
                    amt    = row.get("Final_Billed_Amount", 0)
                    risk   = row.get("Risk_Score", 0.0)
                    ftype  = row.get("Fraud_Type", "Anomalous")
                    ai     = row.get("AI_Justification", "Risk pattern detected. Manual verification required.")
                    
                    # Sanitize strings for HTML inclusion
                    s_pid  = str(pid).strip().replace("\n", " ")
                    s_diag = str(diag).strip().replace("\n", " ")
                    s_ai   = str(ai).strip().replace("\n", " ")

                    card_html = (
                        f"<div style=\"font-family:sans-serif; background:#fff; border:1px solid #E5E7EB; border-left:5px solid #DC2626; border-radius:12px; padding:20px; box-shadow: 0 4px 12px rgba(22,163,74, 0.05); height: 420px; overflow: hidden;\">"
                        f"<div style=\"display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:15px;\">"
                        f"<div><div style=\"font-size:.75rem; color:#DC2626; font-weight:700; text-transform:uppercase;\">Patient ID</div>"
                        f"<div style=\"font-size:1.1rem; font-weight:800; color:#14532D;\">{s_pid}</div></div>"
                        f"<div style=\"text-align:right;\"><div style=\"background:#DC2626; color:#fff; font-size:.7rem; font-weight:800; padding:2px 10px; border-radius:12px;\">{int(risk*100)}% RISK</div></div></div>"
                        f"<div style=\"background:#F9FAFB; border-radius:8px; padding:12px; border:1px solid #F3F4F6; margin-bottom:15px;\">"
                        f"<div style=\"margin-bottom:8px;\"><div style=\"font-size:.7rem; font-weight:700; color:#6B7280;\">DIAGNOSIS</div><div style=\"font-size:.85rem; font-weight:700; color:#14532D; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;\">{s_diag}</div></div>"
                        f"<div style=\"display:flex; justify-content:space-between;\">"
                        f"<div><div style=\"font-size:.7rem; font-weight:700; color:#6B7280;\">AGE</div><div style=\"font-size:.9rem; font-weight:700; color:#14532D;\">{age} yrs</div></div>"
                        f"<div><div style=\"font-size:.7rem; font-weight:700; color:#6B7280; text-align:right;\">AMOUNT</div><div style=\"font-size:.9rem; font-weight:700; color:#DC2626;\">&#8377;{amt:,.0f}</div></div></div></div>"
                        f"<div><div style=\"font-size:.75rem; color:#16A34A; font-weight:800; text-transform:uppercase; margin-bottom:5px;\">&#129302; AI Justification</div>"
                        f"<div style=\"font-size:.85rem; color:#1F2937; line-height:1.5; background:#DCFCE7; padding:10px; border-radius:8px; border-left:3px solid #16A34A; height: 160px; overflow-y: auto;\">{s_ai}</div></div>"
                        f"</div>"
                    )
                    with cols[j]:
                        st.components.v1.html(card_html, height=440)
        
        # --- Single Master Download Button at the Bottom ---
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.subheader("📥 Export Audit Data")
        st.info("Download the consolidated list of detected frauds in CSV format for further investigation.")
        
        # Prepare CSV data for export (Strip Emojis for better Excel compatibility)
        import re
        export_df = frauds_df.copy()
        
        # Remove emojis from all string columns
        for col in export_df.select_dtypes(include=['object']).columns:
            export_df[col] = export_df[col].apply(lambda x: re.sub(r'[^\x00-\x7F\u0900-\u097F\u20B9]', '', str(x)) if pd.notnull(x) else x)
        
        csv_data = export_df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            label="📊 DOWNLOAD AUDIT REPORT (CSV)",
            data=csv_data,
            file_name=f"Fraud_Audit_Export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
        st.markdown("<br><br>", unsafe_allow_html=True)


# ============================================================
#  PAGE: HOME (Formerly Upload Center)
# ============================================================
elif st.session_state.page == "Home":
    # ── 1. ABOUT US (Hero) ──────────────────────────────────
    st.markdown("""
    <div class='hero-section'>
        <div style='font-size:2.6rem; font-weight:800; margin-bottom:12px; letter-spacing:-1px;'>Ayushman Bharat Fraud Intelligence</div>
        <div style='font-size:1.35rem; opacity:1.0; max-width:850px; line-height:1.7; font-weight:400;'>
            Deploying state-of-the-art <b>Deep Intelligence</b> to safeguard India's healthcare future. 
            Our platform serves as a "Digital Auditor," monitoring clinical claims across the 
            PM-JAY network to eliminate leakage, prevent procedure up-coding, and protect 
            taxpayer funds for the millions who depend on it.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. FEATURES ─────────────────────────────────────────
    st.markdown("<div style='margin-bottom:25px;'><span style='font-size:1.2rem; font-weight:800;'>Key Capabilities</span></div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class='feat-card'>
            <div class='feat-icon'>1</div>
            <div class='feat-title'>Anomalous Detection</div>
            <div class='feat-desc'>Proprietary <i>Isolation Forest</i> models identify outlier billing behavior by analyzing LOS, cost ratios, and clinical triggers.</div>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class='feat-card'>
            <div class='feat-icon'>2</div>
            <div class='feat-title'>Clinical AI Reasoning</div>
            <div class='feat-desc'>Large Language Models (GPT-4o) provide forensic justifications for every flag, translating complex data into actionable audit trails.</div>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class='feat-card'>
            <div class='feat-icon'>3</div>
            <div class='feat-title'>Unified Investigations</div>
            <div class='feat-desc'>Centralized repository for detected frauds, enabling investigators to track, audit, and finalize cases with cloud persistence.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── 3. ANALYZE NOW ─────────────────────────────────────
    st.markdown("""
    <div style='margin-bottom:25px; margin-top:10px;'>
      <span style='font-size:1.4rem; font-weight:800; color:#000000;'>🚀 Analyze Now</span>
      <span style='color:#000000; font-size:1.25rem; font-weight:500;'> — Secure Claim Forensic Analysis</span>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")

    if uploaded:
        raw = pd.read_csv(uploaded)
        pb  = st.progress(0, text="Auditing Data...")
        # Removed artificial delays for instant forensic analysis
        result_df, cc = run_pipeline(raw.copy(), st.session_state.contamination, st.session_state.n_estimators)
        fraud_up = result_df[result_df["Fraud_Flag"]==1]
        susp_up  = fraud_up[cc].sum() if cc else 0

        if _supabase_ready:
            uid = st.session_state.user.id if st.session_state.user else None
            time.sleep(.2); pb.progress(90, "☁️ Syncing all claims & saving detected frauds...")
            upload_res = sb.insert_new_rows_only(raw, conflict_col="PatientID", user_id=uid)
            sb.save_fraud_results_to_supabase(result_df, user_id=uid)
            
            # 🔥 STEP 1 & 2: EXTRACT ONLY FRAUDS & UPSERT TO detected_frauds
            fraud_only = result_df[result_df["Fraud_Flag"] == 1].copy()
            if not fraud_only.empty:
                # Map columns to match detected_frauds table schema
                if cc and cc != "Final_Billed_Amount":
                    fraud_only["Final_Billed_Amount"] = fraud_only[cc]
                if "Primary_Diagnosis" not in fraud_only.columns and "Disease" in fraud_only.columns:
                    fraud_only["Primary_Diagnosis"] = fraud_only["Disease"]
                
                sb.upsert_detected_frauds(fraud_only, user_id=uid)
            
            # Log session
            log_uid = st.session_state.user.id if st.session_state.user else "guest"
            
            sb.log_upload_session(
                uid=uid,
                filename=uploaded.name,
                total_rows=len(result_df),
                new_rows=upload_res.get("new", 0),
                skipped_rows=upload_res.get("skipped", 0),
                fraud_detected=len(fraud_up),
                suspicious_amt=susp_up
            )
            
            sb.upsert_audit_log(
                uid=uid,
                action="Batch Analysis",
                description=f"Processed {len(result_df)} rows from {uploaded.name}. Detected {len(fraud_up)} frauds.",
                amount=susp_up
            )

        time.sleep(.2); pb.progress(100, "✅ Complete!")
        time.sleep(.3); pb.empty()

        # Update global session state with the full database if connected
        refreshed = False
        if _supabase_ready and st.session_state.uid:
            try:
                cloud_df = sb.fetch_data_from_supabase(user_id=st.session_state.uid) # Fetch user data
                if not cloud_df.empty:
                    st.session_state.df, st.session_state.cost_col = run_pipeline(
                        cloud_df, st.session_state.contamination, st.session_state.n_estimators)
                    refreshed = True
            except:
                pass
        
        # Fallback to just the uploaded file if cloud fetch fails or Supabase isn't ready
        if not refreshed:
            st.session_state.df = result_df
            st.session_state.cost_col = cc
            
        fraud_up = result_df[result_df["Fraud_Flag"]==1]
        susp_up  = fraud_up[cc].sum() if cc else 0

        m1,m2,m3 = st.columns(3)
        m1.metric("📋 Total Rows",   f"{len(result_df):,}")
        m2.metric("🚨 Fraud Found",  f"{len(fraud_up):,}", delta=f"{len(fraud_up)/len(result_df)*100:.1f}%", delta_color="inverse")
        m3.metric("💰 Suspicious",   fmt_crore(susp_up))

        st.success("✅ Pipeline complete! AI analysis generated for all flagged cases.")
        st.markdown("---")

        # Fraud type chips
        if "Fraud_Type" in fraud_up.columns and not fraud_up.empty:
            st.markdown("<div class='section-card'><div class='section-title'>🔎 Fraud Type Summary</div>", unsafe_allow_html=True)
            tc   = fraud_up["Fraud_Type"].value_counts()
            cols = st.columns(len(tc))
            clrs = {"Ghost Billing":"#DC2626","Up-coding":"#D97706","Fake Admission":"#EA580C","Identity Misuse":"#7C3AED","Anomalous Pattern":"#16A34A"}
            for col_w,(ft,cnt) in zip(cols,tc.items()):
                c=clrs.get(ft,"#16A34A")
                col_w.markdown(f"<div style='background:{c}18;border:1px solid {c};border-radius:12px;padding:12px;text-align:center;'><div style='font-size:1.4rem;font-weight:800;color:{c};'>{cnt}</div><div style='font-size:.72rem;font-weight:600;color:{c};'>{ft}</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # AI cards
        st.markdown("<div class='section-card'><div class='section-title'>🤖 AI Analysis — Flagged Cases</div>", unsafe_allow_html=True)
        if fraud_up.empty:
            st.success("✅ No fraud detected.")
        else:
            id_u = "PatientID" if "PatientID" in fraud_up.columns else fraud_up.columns[0]
            tc2  = {"Ghost Billing":"#DC2626","Up-coding":"#D97706","Fake Admission":"#EA580C","Identity Misuse":"#7C3AED","Anomalous Pattern":"#16A34A"}
            for _,fr in fraud_up.iterrows():
                pid2  = fr.get(id_u,"—"); age2=fr.get("Age","?"); ft2=fr.get("Fraud_Type","—")
                rs2   = float(fr.get("Risk_Score",0)); cv2=float(fr.get(cc,0)) if cc else 0
                ai2   = str(fr.get("AI_Justification","")).strip(); cl2=tc2.get(ft2,"#16A34A")
                sc2   = int(fr.get("Suspicion_Score",int(rs2*100)))
                if not ai2: ai2=f"₹{cv2:,.0f} flagged — {ft2}. Risk: {rs2:.2f}. Manual review needed."
                st.markdown(f"""
                <div class='section-card' style='margin-bottom:10px;'>
                  <div style='display:flex;justify-content:space-between;margin-bottom:10px;'>
                    <div><span style='font-weight:700;color:#16A34A;font-size:.95rem;'>{pid2}</span>
                      &nbsp;<span style='background:{cl2}18;color:{cl2};border:1px solid {cl2};border-radius:20px;padding:2px 8px;font-size:.7rem;font-weight:700;'>{ft2}</span>
                    </div>
                    <div style='font-size:.75rem;color:#1F2937;'>Age:{age2} | ₹{cv2:,.0f} | Risk:{fr.get("Risk score per claim",0.0):.2f}</div>
                  </div>
                  <div style='background:#DCFCE7;border-left:3px solid #16A34A;border-radius:8px;padding:12px 14px;'>
                    <div style='font-size:.68rem;font-weight:700;color:#14532D;text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px;'>Forensic AI Analysis</div>
                    <div style='font-size:.87rem;color:#1F2937;line-height:1.65;'>{ai2}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        btn_col1, btn_col2 = st.columns([1,1])
        with btn_col1:
            if st.button("→ Go to Home", use_container_width=True):
                st.session_state.page = "Home"; st.rerun()
        with btn_col2:
            st.empty()
        st.markdown("""
        <div class='section-card' style='text-align:center;padding:40px;'>
          <div style='font-size:3rem;'>☁️</div>
          <div style='font-weight:700;color:#000000;font-size:1rem;margin:10px 0 6px;'>Drag &amp; Drop a Claims CSV</div>
          <div style='font-size:.82rem;color:#000000;'>Required columns: <code>PatientID</code>, <code>Final_Billed_Amount</code>, <code>Age</code></div>
        </div>""", unsafe_allow_html=True)

        # ── Supabase quick actions ───────────────────────────
        if _supabase_ready:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("☁️ **Supabase Quick Actions**")
            qa1, qa2 = st.columns(2)

            with qa1:
                if st.button("↺ Refresh my Supabase Data", use_container_width=True):
                    with st.spinner("Fetching your latest data from Supabase..."):
                        try:
                            cloud_df = sb.fetch_data_from_supabase(user_id=st.session_state.uid) 
                            if not cloud_df.empty:
                                st.session_state.df, st.session_state.cost_col = run_pipeline(
                                    cloud_df,
                                    st.session_state.contamination,
                                    st.session_state.n_estimators
                                )
                                log_uid = st.session_state.user.id if st.session_state.user else "guest"
                                sb.upsert_audit_log(log_uid, "Data Sync", f"Refreshed {len(cloud_df)} rows from Supabase.")
                                st.success(f"✅ Loaded {len(cloud_df):,} rows from Supabase.")
                                st.session_state.page = "Home"; st.rerun()
                            else:
                                st.warning("⚠️ Claims table is empty in Supabase.")
                        except Exception as e:
                            st.error(f"❌ {e}")

            with qa2:
                if st.button("⇳ Sync Local CSV → Supabase", use_container_width=True):
                    if os.path.exists(CSV_PATH):
                        with st.spinner(f"Uploading '{CSV_PATH}' to Supabase..."):
                            uid = st.session_state.user.id if st.session_state.user else None
                            ok, msg = sb.sync_local_csv_to_supabase(CSV_PATH, user_id=uid)
                        (st.success if ok else st.error)(msg)
                    else:
                        st.error(f"❌ File not found: {CSV_PATH}")

            st.markdown("</div>", unsafe_allow_html=True)


#  PAGE: ACCOUNT
# ============================================================
elif st.session_state.page == "Account":
    
    if st.session_state.user is None:
        # ── LOGIN / REGISTER / FORGOT PASSWORD FLOW ─────────
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown(f"""
                <div style='text-align:center; margin-bottom:20px; padding: 24px; background: white; border-radius: 28px; border: 1px solid #E5E7EB; box-shadow: 0 8px 24px rgba(0,0,0,0.04);'>
                    <img src='{LOGO_B64}' style='width:56px; height:56px; border-radius:12px; margin-bottom:12px;'>
                    <div style='font-size:1.6rem; font-weight:800; color:#14532D;'>Welcome Back</div>
                    <div style='font-size:0.88rem; color:#1F2937; margin-top:2px;'>Access your forensic audit dashboard</div>
                </div>
            """, unsafe_allow_html=True)

            mode = st.session_state.get("auth_mode", "login")

            if mode == "reset":
                st.markdown("### Reset Password")
                st.info("Enter your email address and we'll send you a recovery link.")
                with st.form("reset_form"):
                    reset_email = st.text_input("Email Address")
                    reset_submit = st.form_submit_button("Send Recovery Link", use_container_width=True, type="primary")
                    if reset_submit:
                        if reset_email:
                            try:
                                sb.send_password_reset_email(reset_email)
                                st.success("Recovery link sent! Check your inbox.")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                        else:
                            st.error("Please enter email.")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("← Back to Login", use_container_width=True):
                    st.session_state.auth_mode = "login"; st.rerun()
            
            else:
                tabs = st.tabs(["Login", "Register"])
                
                with tabs[0]:
                    with st.form("login_form"):
                        email = st.text_input("Email Address")
                        password = st.text_input("Password", type="password")
                        submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                        if submit:
                            if email and password:
                                try:
                                    res = sb.sign_in(email, password)
                                    if res.user:
                                        st.session_state.user = res.user
                                        st.session_state.uid = res.user.id
                                        import uuid
                                        sid = str(uuid.uuid4())[:8]
                                        st.query_params["s"] = sid
                                        get_session_cache()[sid] = {"user": res.user, "uid": res.user.id}
                                        st.session_state.page = "Home"
                                        sb.upsert_audit_log(res.user.id, "Login", f"User {email} logged in.")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Login failed: {e}")
                            else:
                                st.error("Fields required.")
                    
                    st.markdown("<div style='text-align:center; padding-top:12px;'>", unsafe_allow_html=True)
                    if st.button("Forgot Password?", key="forgot_link", help="Click to reset your password"):
                        st.session_state.auth_mode = "reset"; st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                with tabs[1]:
                    with st.form("register_form"):
                        reg_email = st.text_input("Email Address")
                        reg_pass = st.text_input("Password", type="password")
                        reg_submit = st.form_submit_button("Create Investigator Account", use_container_width=True, type="primary")
                        if reg_submit:
                            if reg_email and reg_pass:
                                try:
                                    res = sb.sign_up(reg_email, reg_pass)
                                    st.success("Account created! Verify your email and login.")
                                    inner_uid = res.user.id if res and res.user else "new"
                                    sb.upsert_audit_log(inner_uid, "Registration", f"New account: {reg_email}")
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                            else:
                                st.error("Fields required.")
            
    else:
        # ── LOGGED IN PROFILE & LOGS ────────────────────────
        c1, c2 = st.columns([1, 2])
        user = st.session_state.user
        uid = user.id
        email = user.email
        
        with c1:
            # ── Unified Profile Card ──
            st.markdown(f"""<div class='section-card' style='text-align:center; padding: 32px 24px; margin-bottom: 20px;'>
<div style='width:90px; height:90px; border-radius:50%; background:linear-gradient(135deg,#16A34A,#22C55E);
display:flex; align-items:center; justify-content:center; font-size:2.5rem;
margin:0 auto 16px; color:#fff; box-shadow: 0 8px 24px rgba(22,163,74,0.15);'>👨‍⚕️</div>
<div style='font-size:1.4rem; font-weight:800; color:#14532D; margin-bottom:4px;'>{email.split('@')[0].title()}</div>
<div style='font-size:0.85rem; color:#4B5563; font-weight:500; margin-bottom:12px;'>ID: <code style='color:#16A34A;'>{uid[:8]}</code></div>
<div style='display:inline-block; background:#DCFCE7; color:#16A34A; border:1px solid #BBF7D0; border-radius:30px; padding:4px 14px; font-size:0.75rem; font-weight:700;'>● Active Session</div>
</div>""", unsafe_allow_html=True)
            
            if st.button("🔓 Logout Investigator", use_container_width=True, type="primary"):
                sb.upsert_audit_log(uid, "Logout", "Investigator session ended.")
                sid = st.query_params.get("s")
                if sid in get_session_cache(): del get_session_cache()[sid]
                st.query_params.clear()
                sb.sign_out()
                st.session_state.user = None
                st.session_state.df = None
                st.session_state.page = "Account"
                st.rerun()

        with c2:
            # ── Combined Information & Activity Card ──
            st.markdown(f"""<div class='section-card' style='padding: 30px;'>
<div style='font-size:1.1rem; font-weight:800; color:#14532D; margin-bottom:24px; display:flex; align-items:center; gap:10px;'>
<span style='background:#DCFCE7; width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center;'>👤</span>
Investigator Profile & Activity
</div>
<div style='display:grid; grid-template-columns: 1fr 1fr; gap:24px; margin-bottom:32px; background:#F9FAFB; padding:20px; border-radius:16px; border:1px solid #E5E7EB;'>
<div>
<div style='font-size:0.75rem; color:#1F2937; font-weight:700; text-transform:uppercase; opacity:0.6;'>Organization</div>
<div style='font-size:0.95rem; font-weight:700; color:#14532D;'>National Health Authority</div>
</div>
<div>
<div style='font-size:0.75rem; color:#1F2937; font-weight:700; text-transform:uppercase; opacity:0.6;'>Designation</div>
<div style='font-size:0.95rem; font-weight:700; color:#14532D;'>Lead Audit Investigator</div>
</div>
<div>
<div style='font-size:0.75rem; color:#1F2937; font-weight:700; text-transform:uppercase; opacity:0.6;'>Email Address</div>
<div style='font-size:0.95rem; font-weight:700; color:#14532D;'>{email}</div>
</div>
<div>
<div style='font-size:0.75rem; color:#1F2937; font-weight:700; text-transform:uppercase; opacity:0.6;'>Verification</div>
<div style='font-size:0.95rem; font-weight:700; color:#16A34A;'>Verified Auditor ●</div>
</div>
</div>
<div style='font-size:0.95rem; font-weight:800; color:#14532D; margin-bottom:16px;'>Recent Audit Actions</div>""", unsafe_allow_html=True)
            
            logs = sb.fetch_audit_log(uid, limit=8)
            if not logs:
                st.info("No recent audit activity recorded.")
            else:
                for l in logs:
                    st.markdown(f"""
                    <div style='padding:12px 0; border-bottom:1px solid #F3F4F6; display:flex; justify-content:space-between; align-items:center;'>
                        <div>
                            <div style='font-size:0.85rem; font-weight:700; color:#1F2937;'>{l['action']}</div>
                            <div style='font-size:0.78rem; color:#4B5563;'>{l['description']}</div>
                        </div>
                        <div style='font-size:0.68rem; color:#1F2937; font-weight:700; background:#DCFCE7; padding:2px 10px; border-radius:12px;'>{l['ago']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)


#  PAGE: SETTINGS
# ============================================================
elif st.session_state.page == "Settings":
    st.markdown("<div style='padding:6px 0 18px;'><span style='font-size:1.4rem;font-weight:800;color:#14532D;'>Settings</span><span style='color:#1F2937;font-size:1.4rem;'> — Detection Configuration</span></div>", unsafe_allow_html=True)

    # ── Parameters Dashboard ──
    st.markdown(f"""<div style='background:white; border:1px solid #E5E7EB; border-radius:20px; padding:30px; margin-bottom:25px; box-shadow: 0 4px 15px rgba(0,0,0,0.03);'>
<div style='display:grid; grid-template-columns: 1fr 1fr; gap:40px;'>
<div>
<div style='font-size:1.2rem; font-weight:800; color:#14532D; margin-bottom:8px; display:flex; align-items:center; gap:8px;'>🎯 Contamination Rate</div>
<div style='font-size:1rem; color:#4B5563; margin-bottom:20px;'>Defines the expected proportion of anomalous claims in your dataset. High rates increase sensitivity but may flag more false positives.</div>
</div>
<div>
<div style='font-size:1.2rem; font-weight:800; color:#14532D; margin-bottom:8px; display:flex; align-items:center; gap:8px;'>🌲 Algorithm Complexity (Trees)</div>
<div style='font-size:1rem; color:#4B5563; margin-bottom:20px;'>The number of Isolation Trees built for the forest. Higher values (200+) provide more stable results but increase processing time.</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        nc = st.slider("Detection Sensitivity (Contamination)", 0.01, 0.30, float(st.session_state.contamination), 0.01, format="%.2f", help="Adjust this based on expected fraud density.")
        lbl = "🟢 Conservative" if nc < 0.08 else "🟡 Balanced" if nc < 0.16 else "🔴 Aggressive"
        st.markdown(f"<div style='background:#F9FAFB; padding:10px 15px; border-radius:8px; font-size:0.95rem; color:#1F2937;'>Current Mode: <b>{lbl}</b></div>", unsafe_allow_html=True)
    with s2:
        ne = st.slider("Analytical Depth (n_estimators)", 50, 500, int(st.session_state.n_estimators), 50, help="Higher trees equal better precision on complex fraud.")
        st.markdown(f"<div style='background:#F9FAFB; padding:10px 15px; border-radius:8px; font-size:0.95rem; color:#1F2937;'>Tree Depth: <b>{ne} Iterations</b></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Apply & Re-run Pipeline", type="primary"):
        if st.session_state.df is None:
            st.warning("No data loaded.")
        else:
            st.session_state.contamination=nc; st.session_state.n_estimators=ne
            raw2 = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else st.session_state.df.copy()
            with st.spinner("Re-running..."):
                st.session_state.df, st.session_state.cost_col = run_pipeline(raw2, nc, ne)
                uid = st.session_state.user.id if st.session_state.user else "guest"
                sb.upsert_audit_log(uid, "Pipeline Configuration", f"Updated sensitivity to {nc} and trees to {ne}.")
            st.success(f"✅ Done! {st.session_state.df['Fraud_Flag'].sum()} fraud cases detected.")
            time.sleep(1); st.session_state.page="Home"; st.rerun()

    # ── Technical Reference Guide ──
    st.markdown(f"""<div class='section-card' style='margin-top:20px; padding:25px;'>
<div style='font-size:1.2rem; font-weight:800; color:#14532D; margin-bottom:18px; display:flex; align-items:center; gap:10px;'>
<span style='background:#DCFCE7; width:28px; height:28px; border-radius:6px; display:flex; align-items:center; justify-content:center;'>🔬</span>
Detection Parameter Guide
</div>
<table style='width:100%; border-collapse:collapse; font-size:1rem;'>
<tr style='background:#F9FAFB; border-bottom:2px solid #E5E7EB;'>
<th style='padding:12px; text-align:left; color:#1F2937; font-weight:800;'>Parameter</th>
<th style='padding:12px; text-align:left; color:#1F2937; font-weight:800;'>Effect of LOW Setting</th>
<th style='padding:12px; text-align:left; color:#1F2937; font-weight:800;'>Effect of HIGH Setting</th>
</tr>
<tr style='border-bottom:1px solid #F3F4F6;'>
<td style='padding:15px 12px; color:#14532D; font-weight:700;'>Contamination</td>
<td style='padding:15px 12px; color:#4B5563;'>Fewer flags, highly verified cases. Reduces overhead.</td>
<td style='padding:15px 12px; color:#4B5563;'>Aggressive flagging, catches subtle anomalies. Increase audits.</td>
</tr>
<tr style='border-bottom:1px solid #F3F4F6;'>
<td style='padding:15px 12px; color:#14532D; font-weight:700;'>Depth (Trees)</td>
<td style='padding:15px 12px; color:#4B5563;'>Fast execution, but flags may be less statistically stable.</td>
<td style='padding:15px 12px; color:#4B5563;'>Robust results with minimal variance. Slower re-runs.</td>
</tr>
</table>
<div style='margin-top:20px; padding:12px; background:#F0FDF4; border-radius:10px; border:1px solid #DCFCE7; display:inline-flex; align-items:center; gap:10px;'>
<span style='font-size:1.2rem;'>💡</span>
<span style='font-size:1rem; color:#14532D; font-weight:600;'>Recommended for NHA Audit:</span>
<code style='background:#14532D; color:white; padding:2px 8px; border-radius:4px;'>Contamination: 0.12</code>
<code style='background:#14532D; color:white; padding:2px 8px; border-radius:4px;'>Trees: 200</code>
</div>
</div>""", unsafe_allow_html=True)

# ── PAGE: AI ASSISTANT (CHAT)
# ============================================================
elif st.session_state.page == "Chat":
    st.markdown("""
    <div style='padding:6px 0 18px;'>
      <span style='font-size:1.8rem;font-weight:800;color:#14532D;'>Digital Assistant</span>
      <span style='color:#196a39;font-size:1.8rem;'> — AI Forensic Auditor</span>
    </div>""", unsafe_allow_html=True)
    
    # ── Chat Canvas ──
    st.markdown("""
    <div style='background:#fff; border:1px solid #E5E7EB; border-radius:16px; padding:25px; margin-bottom:20px; box-shadow:0 4px 12px rgba(0,0,0,0.02);'>
        <div style='display:flex; align-items:center; gap:15px; border-bottom:1px solid #F3F4F6; padding-bottom:15px;'>
            <div style='width:45px; height:45px; border-radius:12px; background:#DCFCE7; display:flex; align-items:center; justify-content:center; font-size:1.4rem;'>�</div>
            <div>
                <div style='font-size:1.3rem; font-weight:800; color:#14532D;'>MedShield Intelligence</div>
                <div style='font-size:0.85rem; color:#16A34A; font-weight:700; letter-spacing:0.5px;'>SECURE FORENSIC PROTOCOL • ACTIVE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Message Container
    chat_container = st.container()
    
    with chat_container:
        # Render History
        for role, msg in st.session_state.chat_history:
            c1, c2 = st.columns([1, 1])
            if role == "You":
                with c2: # Right Side
                    with st.chat_message("user"):
                        st.markdown(f"<div style='font-size:1.1rem; color:#1F2937; background:#F3F4F6; padding:12px; border-radius:12px; border-right:4px solid #4B5563;'>{msg}</div>", unsafe_allow_html=True)
            else:
                with c1: # Left Side
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(f"<div style='font-size:1.1rem; color:#1F2937; background:#DCFCE7; padding:12px; border-radius:12px; border-left:4px solid #16A34A;'>{msg}</div>", unsafe_allow_html=True)

    # Dedicated Bottom Input
    user_input = st.chat_input("Query the forensic database...")

    if user_input:
        st.session_state.chat_history.append(("You", user_input))
        
        # Manually render the user message immediately so it's visible while AI thinks
        with chat_container:
            c1, c2 = st.columns([1, 1])
            with c2:
                with st.chat_message("user"):
                    st.markdown(f"<div style='font-size:1.1rem; color:#1F2937; background:#F3F4F6; padding:12px; border-radius:12px; border-right:4px solid #4B5563;'>{user_input}</div>", unsafe_allow_html=True)
            
            # Show "Thinking" status locally
            with st.spinner("🧠 MedShield AI is analyzing forensic patterns..."):
                # ── Fetch Live Context from Supabase ──
                frauds_df = sb.fetch_detected_frauds(user_id=st.session_state.uid)
                
                # Prepare contextual data for the AI
                if not frauds_df.empty:
                    top_cases = frauds_df.head(5).to_dict(orient="records")
                    fraud_types = frauds_df["Fraud_Type"].value_counts().to_dict()
                    total_value = fmt_crore(frauds_df["Final_Billed_Amount"].sum())
                    
                    context = (
                        f"You are the MedShield AI Forensic Auditor. The current audit queue has {len(frauds_df)} cases "
                        f"with a total suspicious value of {total_value}. "
                        f"Common patterns: {fraud_types}. "
                        f"Top cases: {top_cases}. "
                        "Answer accurately and clinically."
                    )
                else:
                    context = "You are the MedShield AI Forensic Auditor. No fraud cases in queue."

                # ── Generate Response via OpenAI ──
                if _ai_client:
                    try:
                        messages = [{"role": "system", "content": context}]
                        for role, msg in st.session_state.chat_history[-6:]:
                            r = "user" if role == "You" else "assistant"
                            messages.append({"role": r, "content": msg})
                        
                        resp = _ai_client.chat.completions.create(
                            model="gpt-4",
                            messages=messages,
                            temperature=0.7
                        )
                        reply = resp.choices[0].message.content
                    except Exception as e:
                        reply = f"System error: {str(e)}."
                else:
                    responses = {
                        "ghost": "Ghost Billing is when a hospital bills for a patient who was never admitted.",
                        "upcode": "Up-coding involves inflating reimbursement rates.",
                        "data": f"Currently, I see {len(frauds_df)} critical cases in the cloud database."
                    }
                    msg_lower = user_input.lower()
                    reply = next((v for k, v in responses.items() if k in msg_lower),
                                 f"Forensic DB indicates {len(frauds_df)} investigations pending. Configure OpenAI or use specific keywords.")

                st.session_state.chat_history.append(("Bot", reply))
                st.rerun()
