import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API SETUP
# ============================================================
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None

# ============================================================
# LOAD DATA
# ============================================================
print("📂 Loading dataset...")
df = pd.read_csv("ayushman_claims.csv")
print(f"✅ Loaded {len(df)} rows\n")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("⚙️ Creating advanced fraud signals...")

df["Admission_Timestamp"] = pd.to_datetime(df["Admission_Timestamp"], errors="coerce")
df["Discharge_Timestamp"] = pd.to_datetime(df["Discharge_Timestamp"], errors="coerce")
df["PreAuth_Request_Date"] = pd.to_datetime(df["PreAuth_Request_Date"], errors="coerce")
df["PreAuth_Approval_Date"] = pd.to_datetime(df["PreAuth_Approval_Date"], errors="coerce")

df["LOS"] = (df["Discharge_Timestamp"] - df["Admission_Timestamp"]).dt.days
df["PreAuth_Delay"] = (df["PreAuth_Approval_Date"] - df["PreAuth_Request_Date"]).dt.days
df["Cost_to_Package"] = df["Final_Billed_Amount"] / df["Base_Package_Rate"].replace(0,1)

hospital_avg = df.groupby("Hospital_PIN")["Final_Billed_Amount"].mean()
df["Hospital_Avg_Cost"] = df["Hospital_PIN"].map(hospital_avg)

patient_claims = df.groupby("PatientID")["TransactionID"].count()
df["Patient_Claim_Count"] = df["PatientID"].map(patient_claims)

# ============================================================
# PHASE 1 — RULE ENGINE
# ============================================================
print("🧠 Applying fraud rules...")
df["Rule_Fraud"] = 0

df.loc[df["Base_Package_Rate"] == 0, "Rule_Fraud"] = 1
df.loc[df["Cost_to_Package"] > 2.5, "Rule_Fraud"] = 1
df.loc[df["LOS"] <= 0, "Rule_Fraud"] = 1

multi_hospital = df.groupby("PatientID")["Hospital_PIN"].nunique()
suspicious_ids = multi_hospital[multi_hospital > 1].index
df.loc[df["PatientID"].isin(suspicious_ids), "Rule_Fraud"] = 1

# ============================================================
# PHASE 2 — MACHINE LEARNING
# ============================================================
print("🤖 Running ML anomaly detection...")

features = df[[
    "Final_Billed_Amount",
    "Cost_to_Package",
    "LOS",
    "PreAuth_Delay",
    "Hospital_Avg_Cost",
    "Patient_Claim_Count",
    "Age"
]].fillna(0)

model = IsolationForest(
    contamination=0.12,
    n_estimators=200,
    random_state=42
)

df["ML_Anomaly"] = model.fit_predict(features)

# ============================================================
# PHASE 3 — RISK SCORE FUSION
# ============================================================
df["Risk_Score"] = (
    df["Rule_Fraud"] * 0.5 +
    (df["ML_Anomaly"] == -1).astype(int) * 0.4 +
    (df["Cost_to_Package"] > 2).astype(int) * 0.1
)

df["Fraud_Flag"] = (df["Risk_Score"] > 0.5).astype(int)

# ============================================================
# FRAUD TYPE CLASSIFICATION
# ============================================================
def classify(row):
    if row["Base_Package_Rate"] == 0:
        return "Ghost Billing"
    elif row["Cost_to_Package"] > 2.5:
        return "Upcoding"
    elif row["Patient_Claim_Count"] > 2:
        return "Identity Misuse"
    elif row["LOS"] <= 0:
        return "Fake Admission"
    else:
        return "Anomalous Pattern"

df["Fraud_Type"] = df.apply(classify, axis=1)

# ============================================================
# PHASE 4 — AI EXPLANATION AGENT
# ============================================================
def ai_explain(row):
    if not client:
        return "AI disabled (no API key)"

    prompt = f"""
    You are auditing an Ayushman Bharat claim.

    Age: {row['Age']}
    Diagnosis: {row['Primary_Diagnosis']}
    Cost: ₹{row['Final_Billed_Amount']}
    Package Rate: {row['Base_Package_Rate']}
    Length of Stay: {row['LOS']} days
    Fraud Type: {row['Fraud_Type']}
    Don't use emojis.
    Don't include hospital/patient IDs.
    Just give the core reason in one simple sentence.
    Explain briefly why this claim looks suspicious.
    """

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a government health insurance fraud auditor."},
                {"role":"user","content":prompt}
            ],
            max_tokens=80,
            temperature=0.3
        )
        return r.choices[0].message.content.strip()
    except:
        return "AI explanation unavailable"

fraud_cases = df[df["Fraud_Flag"] == 1].copy()

print("🧾 Generating AI explanations...")
fraud_cases["AI_Justification"] = fraud_cases.apply(ai_explain, axis=1)

# ============================================================
# OUTPUT
# ============================================================
cols = [
    "TransactionID","PatientID","Hospital_PIN",
    "Fraud_Type","Final_Billed_Amount",
    "Risk_Score","AI_Justification"
]

print("\n🚨 FINAL FRAUD REPORT\n")
print(fraud_cases[cols].to_string(index=False))

print("\n📊 SUMMARY")
print(f"Total Claims: {len(df)}")
print(f"Frauds Detected: {len(fraud_cases)}")
print("Method: Hybrid Rules + Behavioral ML + AI Explanation")