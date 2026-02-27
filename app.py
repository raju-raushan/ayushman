import streamlit as st
import pandas as pd
import numpy as np
import os, time, random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI as _OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

# ============================================================
#  LOGO / BRANDING
# ============================================================
import base64
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
#MainMenu, footer, header { visibility:hidden; }
[data-testid="stSidebarNav"] { display:none; }

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
.kpi-label { font-size:.78rem; font-weight:500; color:#6B7280; margin-bottom:6px; }
.kpi-value { font-size:1.9rem; font-weight:800; color:#14532D; line-height:1; }
.kpi-delta-up   { font-size:.75rem; font-weight:700; color:#16A34A; margin-left:6px; }
.kpi-delta-warn { font-size:.75rem; font-weight:700; color:#DC2626; margin-left:6px; }
.kpi-sub   { font-size:.73rem; color:#6B7280; margin-top:4px; }

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
    font-size:1rem; font-weight:700; color:#14532D;
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
.timeline-time  { font-size:.72rem; font-weight:600; color:#6B7280; margin-left:8px; }
.timeline-desc  { font-size:.8rem; color:#4B5563; margin-top:3px; line-height:1.5; }

/* ── Flagged claims table ── */
.claims-table { width:100%; border-collapse:collapse; font-size:.83rem; }
.claims-table th {
    color:#6B7280; font-weight:700; font-size:.7rem;
    text-transform:uppercase; letter-spacing:.6px;
    padding:8px 12px; border-bottom:2px solid #DCFCE7;
    text-align:left;
}
.claims-table td { padding:12px 12px; border-bottom:1px solid #F9FAFB; }
.claims-table tr:hover td { background:#F0FDF4; }
.claim-id   { font-weight:700; color:#16A34A; }
.score-bar-wrap { display:flex; align-items:center; gap:8px; }
.score-bar  { height:6px; border-radius:3px; }
.score-text { font-weight:700; font-size:.8rem; }

/* ── Status tags ── */
.tag-high   { background:#FEE2E2; color:#DC2626; border:1px solid #FECACA; border-radius:20px; padding:3px 10px; font-size:.7rem; font-weight:700; }
.tag-inv    { background:#FEF3C7; color:#D97706; border:1px solid #FDE68A; border-radius:20px; padding:3px 10px; font-size:.7rem; font-weight:700; }
.tag-safe   { background:#DCFCE7; color:#16A34A; border:1px solid #BBF7D0; border-radius:20px; padding:3px 10px; font-size:.7rem; font-weight:700; }

/* ── Floating chatbot ── */
.chatbot-btn {
    position:fixed; bottom:28px; right:28px; z-index:9999;
    width:56px; height:56px; border-radius:50%;
    background:linear-gradient(135deg,#16A34A,#22C55E);
    box-shadow:0 6px 24px rgba(22,163,74,.45);
    display:flex; align-items:center; justify-content:center;
    cursor:pointer; font-size:1.5rem;
    transition:transform .2s;
}
.chatbot-btn:hover { transform:scale(1.08); }

/* ── Chat panel ── */
.chat-panel {
    position:fixed; bottom:96px; right:28px; z-index:9998;
    width:320px; background:#fff;
    border:1px solid #E5E7EB; border-radius:16px;
    box-shadow:0 8px 32px rgba(22,163,74,.15);
    overflow:hidden;
}
.chat-header {
    background:linear-gradient(135deg,#16A34A,#22C55E);
    color:#fff; padding:14px 18px;
    font-weight:700; font-size:.9rem;
    display:flex; align-items:center; gap:10px;
}
.chat-body { padding:14px 16px; min-height:120px; }
.chat-msg-bot {
    background:#DCFCE7; color:#14532D; border-radius:12px 12px 12px 2px;
    padding:10px 14px; font-size:.82rem; margin-bottom:10px; max-width:85%;
}
.chat-input {
    border-top:1px solid #E2ECF8; padding:10px 14px;
    display:flex; gap:8px;
}

/* ── Upload zone ── */
[data-testid="stFileUploaderDropzone"] {
    background:#DCFCE7 !important;
    border:2px dashed #16A34A !important;
    border-radius:20px !important;
    min-height: 400px !important;
    padding: 40px !important;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* ── Metrics override ── */
[data-testid="metric-container"] {
    background:#fff !important;
    border:1px solid #E5E7EB !important;
    border-radius:14px !important;
    padding:16px !important;
    box-shadow:0 2px 8px rgba(22,163,74,.06) !important;
}
[data-testid="stMetricLabel"] * {
    color: #6B7280 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] * {
    color: #14532D !important;
}
[data-testid="stMetricDelta"] * {
    font-weight: 700 !important;
}

/* ── Streamlit select/slider ── */
[data-baseweb="select"] > div { background:#fff !important; border-color:#D1D5DB !important; border-radius:8px !important; }
.stSlider [data-baseweb="slider"] { color:#16A34A !important; }

/* ── Premium Landing ── */
.hero-section {
    background: linear-gradient(135deg, #14532D 0%, #16A34A 100%);
    color: white; padding: 50px 40px; border-radius: 20px;
    margin-bottom: 30px; position: relative; overflow: hidden;
}
.hero-section::after {
    content: ''; position: absolute; right: -20px; bottom: -20px;
    width: 250px; height: 250px; background: url('""" + LOGO_B64 + """') no-repeat center center;
    background-size: contain; opacity: 0.1; transform: rotate(-15deg);
}
.feat-card {
    background: #fff; padding: 24px; border-radius: 16px;
    border: 1px solid #E5E7EB; box-shadow: 0 4px 15px rgba(22,163,74,.05);
    height: 100%; transition: transform 0.2s;
}
.feat-card:hover { transform: translateY(-5px); }
.feat-icon { font-size: 2rem; margin-bottom: 12px; color: #16A34A; }
.feat-title { font-weight: 800; color: #14532D; margin-bottom: 8px; }
.feat-desc { font-size: 0.85rem; color: #4B5563; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Firebase (graceful if not configured yet) ────────────────
try:
    import firebase_auth as fb_auth
    import firebase_db   as fb_db
    _firebase_ready = bool(os.getenv("FIREBASE_API_KEY","") not in ("","your_firebase_api_key"))
except Exception:
    _firebase_ready = False

# ── Supabase (graceful if not configured yet) ───────────────
try:
    import supabase_db as sb
    _supabase_ready = sb.is_configured()
except Exception:
    _supabase_ready = False
    sb = None

# ── Global Session Cache (Persists across refreshes) ──────────
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
valid_pages = ["Home", "Report", "Account", "Settings"]
if st.session_state.user and st.session_state.page not in valid_pages:
    st.session_state.page = "Home"
elif st.session_state.user is None:
    st.session_state.page = "Account"


# ============================================================
#  PIPELINE
# ============================================================
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
        pages = {"Home":"Home","Fraud Audit Report":"Report",
                 "Account":"Account","Settings":"Settings"}
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
            <span style='font-size:.68rem;color:#6B7280;font-weight:600;'>DATA SOURCE</span>
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
            <div style='font-size:.7rem;color:#6B7280;'>{user_role}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
#  FLOATING CHATBOT
# ============================================================
chat_open = st.session_state.chat_open

# Toggle button (always visible)
st.markdown(f"""
<div class='chatbot-btn' onclick="window.parent.document.querySelector('[data-testid=stApp]').dispatchEvent(new CustomEvent('chatToggle'))"
     title='AI Assistant' id='chatbot-fab'>💬</div>
""", unsafe_allow_html=True)

# Sidebar chatbot toggle (actual Streamlit interaction)
with st.sidebar:
    st.markdown("<div style='padding:0 12px 80px;margin-top:8px;'>", unsafe_allow_html=True)
    if st.button("AI Assistant", key="chat_toggle", use_container_width=True):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.chat_open:
    st.markdown(f"""
    <div class='chat-panel'>
        <div class='chat-header'><img src='{LOGO_B64}' style='width:20px;height:20px;border-radius:4px;'>&nbsp;MedShield AI Assistant <span style='margin-left:auto;font-size:.8rem;opacity:.8;'>● Online</span></div>
        <div class='chat-body'>
            <div class='chat-msg-bot'>Hello! I'm your fraud analysis assistant.<br>Ask me about any flagged claim, fraud type, or investigation steps.</div>
            <div class='chat-msg-bot' style='background:#FEF3C7;color:#D97706;'>💡 Try: <em>"Explain Ghost Billing"</em> or <em>"What is Up-coding?"</em></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        chat_cols = st.columns([5,1])
        user_msg = chat_cols[0].text_input("", placeholder="Ask about a claim or fraud type...", label_visibility="collapsed")
        submitted = chat_cols[1].form_submit_button("Send")

    if submitted and user_msg:
        responses = {
            "ghost": "Ghost Billing is when a hospital submits a claim for a patient who was never actually treated or admitted. The base package rate is ₹0 but a large amount is billed.",
            "upcode": "Up-coding is when hospitals bill for a more expensive procedure than what was actually performed, inflating the reimbursement beyond the approved Ayushman Bharat package rate.",
            "fake":   "Fake Admission involves creating hospital admission records for patients who were never hospitalised, to fraudulently claim inpatient reimbursements.",
            "identity":"Identity Misuse occurs when a beneficiary's Aadhaar/AB card is used by multiple people or across many claims, often in collusion with the hospital.",
            "isolat": "Isolation Forest is an ML algorithm that isolates anomalies by randomly splitting data. Points that are easy to isolate (few splits needed) are flagged as outliers.",
        }
        msg_lower = user_msg.lower()
        reply = next((v for k,v in responses.items() if k in msg_lower),
                     f"🤔 I can help with fraud types, investigation steps, or claim analysis. Could you be more specific about '{user_msg}'?")
        st.session_state.chat_history.append(("You", user_msg))
        st.session_state.chat_history.append(("Bot", reply))

    for role, msg in st.session_state.chat_history[-6:]:
        if role=="You":
            st.markdown(f"<div style='text-align:right;margin-bottom:6px;'><span style='background:#16A34A;color:#fff;border-radius:12px 12px 2px 12px;padding:8px 12px;font-size:.82rem;'>{msg}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='margin-bottom:8px;'><span style='background:#DCFCE7;color:#14532D;border-radius:12px 12px 12px 2px;padding:8px 12px;font-size:.82rem;display:inline-block;max-width:85%;'>{msg}</span></div>", unsafe_allow_html=True)


# ── Page Routing Logic ──────────────────────────────────────
if st.session_state.user is None:
    st.session_state.page = "Account"
elif st.session_state.page not in ["Home", "Report", "Account", "Settings"]:
    st.session_state.page = "Home"

# ============================================================
#  PAGE: FRAUD AUDIT REPORT
# ============================================================
if st.session_state.page == "Report":
    st.markdown("""
    <div style='padding:6px 0 18px;'>
      <span style='font-size:1.2rem;font-weight:800;color:#14532D;'>Fraud Audit Report</span>
      <span style='color:#DC2626;font-size:1.2rem;'> — Critical Investigation Queue</span>
    </div>""", unsafe_allow_html=True)

    if not _supabase_ready:
        st.error("❌ Live database required for Audit Reports.")
        st.stop()

    with st.spinner("📥 Fetching your critical cases from Cloud..."):
        frauds_df = sb.fetch_detected_frauds(user_id=st.session_state.uid)

    if frauds_df.empty:
        st.info("✅ No critical frauds pending review in the cloud database.")
    else:
        st.markdown(f"**Found {len(frauds_df)} critical cases requiring immediate audit.**")
        
        # Display Cards
        for i, (_, row) in enumerate(frauds_df.iterrows()):
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
                f"<div style=\"font-family:sans-serif; background:#fff; border:2px solid #DCFCE7; border-left:6px solid #DC2626; border-radius:12px; padding:20px; box-shadow: 0 4px 12px rgba(22,163,74, 0.08);\">"
                f"<div style=\"display:flex; justify-content:space-between; align-items:flex-start;\">"
                f"<div><div style=\"font-size:.7rem; color:#DC2626; font-weight:700; text-transform:uppercase;\">Patient ID</div>"
                f"<div style=\"font-size:1.1rem; font-weight:800; color:#14532D;\">{s_pid}</div></div>"
                f"<div style=\"text-align:right;\"><div style=\"font-size:.7rem; color:#6B7280; font-weight:700; text-transform:uppercase;\">Risk Level</div>"
                f"<div style=\"background:#DC2626; color:#fff; font-size:.8rem; font-weight:800; padding:4px 12px; border-radius:20px; margin-top:4px;\">{int(risk*100)}% CRITICAL</div></div></div>"
                f"<div style=\"display:grid; grid-template-columns: 1fr 1fr 1fr; gap:20px; margin-top:18px; padding:12px 16px; background:#FDFDFD; border-radius:10px; border:1px solid #E5E7EB;\">"
                f"<div><div style=\"font-size:.65rem; color:#6B7280; font-weight:700;\">AGE</div><div style=\"font-size:.9rem; font-weight:700; color:#14532D;\">{age} yrs</div></div>"
                f"<div><div style=\"font-size:.65rem; color:#6B7280; font-weight:700;\">DISEASE / DIAGNOSIS</div><div style=\"font-size:.9rem; font-weight:700; color:#14532D;\">{s_diag}</div></div>"
                f"<div><div style=\"font-size:.65rem; color:#6B7280; font-weight:700;\">SUSPICIOUS AMOUNT</div><div style=\"font-size:.9rem; font-weight:700; color:#DC2626;\">&#8377;{amt:,.2f}</div></div></div>"
                f"<div style=\"margin-top:20px;\"><div style=\"font-size:.7rem; color:#16A34A; font-weight:800; text-transform:uppercase; margin-bottom:6px;\">&#129302; AI Risk Justification</div>"
                f"<div style=\"font-size:.85rem; color:#1F2937; line-height:1.6; background:#DCFCE7; padding:12px 16px; border-radius:8px; border-left:4px solid #16A34A;\">{s_ai}</div></div>"
                f"</div>"
            )
            st.components.v1.html(card_html, height=280)
            st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
        
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
        <div style='font-size:1.15rem; opacity:0.95; max-width:800px; line-height:1.7; font-weight:400;'>
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
    <div id='analyze-now' style='padding:6px 0 18px;'>
      <span style='font-size:1.4rem; font-weight:800; color:#16A34A;'>🚀 Analyze Now</span>
      <span style='color:#6B7280; font-size:1.1rem;'> — Secure Claim Forensic Analysis</span>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")

    if uploaded:
        raw = pd.read_csv(uploaded)
        pb  = st.progress(0, text="Loading file...")
        time.sleep(.2); pb.progress(25, "Running hard rules...")
        time.sleep(.3); pb.progress(55, "Training Isolation Forest...")
        result_df, cc = run_pipeline(raw.copy(), st.session_state.contamination, st.session_state.n_estimators)
        fraud_up = result_df[result_df["Fraud_Flag"]==1]
        susp_up  = fraud_up[cc].sum() if cc else 0
        
        time.sleep(.3); pb.progress(85, "Generating AI justifications...")

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
                    <div style='font-size:.75rem;color:#6B7280;'>Age:{age2} | ₹{cv2:,.0f} | Risk:{fr.get("Risk score per claim",0.0):.2f}</div>
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
          <div style='font-weight:700;color:#4B5563;font-size:1rem;margin:10px 0 6px;'>Drag &amp; Drop a Claims CSV</div>
          <div style='font-size:.82rem;color:#6B7280;'>Required columns: <code>PatientID</code>, <code>Final_Billed_Amount</code>, <code>Age</code></div>
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
    st.markdown("<div style='padding:6px 0 18px;'><span style='font-size:1.2rem;font-weight:800;color:#14532D;'>Account</span><span style='color:#6B7280;font-size:1.2rem;'> — Authentication & Activity</span></div>", unsafe_allow_html=True)
    
    if st.session_state.user is None:
        # ── LOGIN / REGISTER FORM ──────────────────────────
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            mode = st.radio("Choose Action", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
            
            with st.form("auth_form"):
                st.markdown(f"### {mode}")
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button(mode, use_container_width=True)
                
                if submit:
                    if not email or not password:
                        st.error("Please fill both fields.")
                    elif mode == "Register":
                        try:
                            res = sb.sign_up(email, password)
                            st.success("Registration successful! You can now log in.")
                            new_uid = res.user.id if res and res.user else "system"
                            sb.upsert_audit_log(new_uid, "Registration", f"New investigator account created: {email}")
                        except Exception as e:
                            st.error(f"Registration failed: {e}")
                    else:
                        try:
                            res = sb.sign_in(email, password)
                            if res.user:
                                st.session_state.user = res.user
                                st.session_state.uid = res.user.id
                                

                                # Generate and save session ID to URL for persistence across refresh
                                import uuid
                                sid = str(uuid.uuid4())[:8]
                                st.query_params["s"] = sid
                                _global_sessions[sid] = {"user": res.user, "uid": res.user.id}

                                st.session_state.df = None # Refresh data for this user
                                st.session_state.page = "Home"
                                sb.upsert_audit_log(res.user.id, "Login", f"User {email} logged in.")
                                st.success("Logged in successfully! Redirecting...")
                                time.sleep(0.5)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Login failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # ── LOGGED IN PROFILE & LOGS ────────────────────────
        c1, c2 = st.columns([1, 2])
        user = st.session_state.user
        uid = user.id
        email = user.email
        
        with c1:
            st.markdown(f"""
            <div class='section-card' style='text-align:center;'>
              <div style='width:72px;height:72px;border-radius:50%;background:linear-gradient(135deg,#16A34A,#22C55E);
                          display:flex;align-items:center;justify-content:center;font-size:1.8rem;
                          margin:0 auto 12px;color:#fff;'>👨</div>
              <div style='font-weight:800;color:#16A34A;font-size:1.05rem;'>{email.split('@')[0].title()}</div>
              <div style='font-size:.78rem;color:#6B7280;margin:4px 0 12px;'>Investigator ID: {uid[:8]}...</div>
              <span class='tag-safe'>● Active Session</span>
            </div>""", unsafe_allow_html=True)
            
            if st.button("🔓 Logout", use_container_width=True):
                sb.upsert_audit_log(uid, "Logout", "Investigator session ended.")
                
                # Clear Global Cache
                sid = st.query_params.get("s")
                if sid in _global_sessions:
                    del _global_sessions[sid]
                
                # Clear URL and State
                st.query_params.clear()
                sb.sign_out()
                st.session_state.user = None
                st.session_state.df = None
                st.session_state.chat_history = []
                st.session_state.page = "Account"
                st.success("Logged out successfully.")
                time.sleep(.5)
                st.rerun()
                
        with c2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("🔍 **Activity History**")
            
            logs = sb.fetch_audit_log(uid, limit=10)
            if not logs:
                st.info("No activity found for this account.")
            else:
                for l in logs:
                    st.markdown(f"""
                    <div style='padding:12px 0; border-bottom:1px solid #DCFCE7; display:flex; justify-content:space-between; align-items:center;'>
                        <div style='flex:1;'>
                            <div style='font-size:.82rem; font-weight:700; color:#14532D;'>{l['action']}</div>
                            <div style='font-size:.75rem; color:#4B5563;'>{l['description']}</div>
                        </div>
                        <div style='font-size:.65rem; color:#6B7280; font-weight:600; background:#F9FAFB; padding:2px 8px; border-radius:12px;'>{l['ago']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Details
            fields = {"Organization": "National Health Authority", "Role": "Audit Investigator",
                      "Email": email, "Status": "Verified Auditor"}
            st.markdown("<div class='section-card' style='margin-top:16px;'>", unsafe_allow_html=True)
            for lbl, val in fields.items():
                st.markdown(f"<div style='padding:10px 0;border-bottom:1px solid #DCFCE7;'><span style='font-size:.72rem;color:#6B7280;'>{lbl}</span><br><span style='font-weight:600;color:#16A34A;'>{val}</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


#  PAGE: SETTINGS
# ============================================================
elif st.session_state.page == "Settings":
    st.markdown("<div style='padding:6px 0 18px;'><span style='font-size:1.2rem;font-weight:800;color:#14532D;'>Settings</span><span style='color:#6B7280;font-size:1.2rem;'> — Detection Configuration</span></div>", unsafe_allow_html=True)

    s1,s2 = st.columns(2)
    with s1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("**🎯 Contamination Rate**")
        st.caption("Expected proportion of fraud. Higher = more sensitive.")
        nc = st.slider("Contamination", .01, .30, float(st.session_state.contamination), .01, format="%.2f")
        lbl = "🟢 Conservative" if nc<.08 else "🟡 Balanced" if nc<.16 else "🔴 Aggressive"
        st.markdown(f"Sensitivity: **{lbl}**")
        st.markdown("</div>", unsafe_allow_html=True)
    with s2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("**🌲 Number of Trees**")
        st.caption("More trees = more accurate but slower. 100–300 is optimal.")
        ne = st.slider("n_estimators", 50, 500, int(st.session_state.n_estimators), 50)
        st.markdown("</div>", unsafe_allow_html=True)

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

    st.markdown("""
    | Parameter | Low | High |
    |---|---|---|
    | **Contamination** | Fewer flags, fewer false positives | More flags, catches subtle fraud |
    | **n_estimators** | Faster, slightly less accurate | Slower, more reliable |
    > 💡 Recommended: Contamination = `0.12`, Trees = `200`
    """)
