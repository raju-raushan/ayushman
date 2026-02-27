# 🛡️ Ayushman Bharat Fraud Detection Dashboard

**AI-Powered Fraud Intelligence Platform for PM-JAY**

![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)

---

## 📌 Overview

The **Ayushman Bharat Fraud Detection Dashboard** is an AI-driven investigative platform built for the **Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (PM-JAY)** ecosystem.

It empowers healthcare investigators to:

* Upload claims data securely
* Detect suspicious claims using Machine Learning
* Understand fraud reasoning via Generative AI
* Maintain cloud-based audit tracking
* Monitor fraud KPIs in real-time

This solution was built as a **Hackathon/Demo project** for strengthening fraud analytics in India’s public healthcare insurance framework.

---

# ✨ Core Features

### 🔐 1. Authentication & Audit Logging (Firebase)

* Secure Login / Signup / Password Reset
* Role-based access (investigators only)
* Real-time audit logs stored in Firestore
* Tracks:

  * Logins
  * File uploads
  * Fraud detection events
  * Claim drill-down actions

---

### ☁️ 2. Cloud Database Sync (Supabase)

* Drag-and-drop CSV upload
* Intelligent duplicate skipping
* Real-time sync to Supabase
* Live dashboard statistics
* Upload session tracking:

  * Total rows
  * New rows
  * Skipped rows
  * Fraud count
  * Suspicious amount

---

### 🤖 3. Machine Learning (Isolation Forest)

An advanced anomaly detection pipeline using:

* `IsolationForest`
* Feature engineering:

  * Age
  * Final billed amount
  * Cost-to-package ratio
  * Length of stay
  * Delays
* Detects patterns like:

  * Ghost Billing
  * Up-coding
  * Fake Admissions
  * Inflated Claims

---

### 🧠 4. Generative AI Risk Explanation

Integrated with the **OpenAI API**

For every flagged claim:

* Generates plain-English justification
* Explains anomaly reasoning
* Converts ML scores into investigator-friendly explanations

Example:

> “Claim flagged due to unusually high billing ratio compared to package rate and demographic mismatch.”

---

### 📊 5. Interactive Healthcare-Themed UI

Built using **Streamlit**

Includes:

* KPI Cards
* Fraud Trend Charts
* Suspicious Amount Tracking
* Fraud Type Chips
* Patient Drill-Down View
* Clean White/Blue Healthcare Theme

---

# 🛠️ Technology Stack

| Layer            | Technology                      |
| ---------------- | ------------------------------- |
| Frontend         | Streamlit + Custom HTML/CSS     |
| Backend          | Python                          |
| Data Processing  | Pandas                          |
| ML Model         | Scikit-Learn (Isolation Forest) |
| Authentication   | Pyrebase4 (Firebase)            |
| Realtime Logging | Firebase Admin SDK              |
| Cloud DB         | Supabase                        |
| Generative AI    | OpenAI API                      |

---

# 🚀 Getting Started

---

## 1️⃣ Requirements

* Python **3.10+**
* pip
* Supabase Project
* Firebase Project
* OpenAI API Key

Install dependencies:

```bash
pip install streamlit pandas numpy scikit-learn python-dotenv openai pyrebase4 firebase-admin supabase-py
```

---

## 2️⃣ Environment Configuration

Create a `.env` file in the root directory:

```ini
# OpenAI API Key
OPENAI_API_KEY="your-openai-api-key"

# Supabase
SUPABASE_URL="your-supabase-url"
SUPABASE_ANON_KEY="your-supabase-anon-key"

# Firebase credentials handled via JSON file
```

---

## 3️⃣ Supabase SQL Setup

Run this inside your Supabase SQL Editor:

```sql
-- Claims Table
create table if not exists claims (
    "PatientID"           text primary key,
    "Age"                 int,
    "Final_Billed_Amount" float8,
    "Fraud_Flag"          int     default 0,
    "Risk_Score"          float8  default 0,
    "Fraud_Type"          text    default '',
    "AI_Justification"    text    default '',
    "uploaded_at"         timestamptz default now()
);

-- Upload Sessions Log
create table if not exists upload_sessions (
    id             bigserial primary key,
    uid            text,
    filename       text,
    total_rows     int  default 0,
    new_rows       int  default 0,
    skipped_rows   int  default 0,
    fraud_detected int default 0,
    suspicious_amt float8 default 0,
    uploaded_at    timestamptz default now()
);

-- Supabase Audit Log (Optional Fallback)
create table if not exists audit_log (
    id          bigserial primary key,
    uid         text,
    action      text,
    description text,
    patient_id  text    default '',
    fraud_type  text    default '',
    amount      float8  default 0,
    created_at  timestamptz default now()
);
```

---

## 4️⃣Run the Application

```bash
streamlit run app.py
```

Access locally:

```
http://localhost:8501
```

---

# Project Structure

```
├── app.py
├── firebase_auth.py
├── firebase_db.py
├── supabase_db.py
├── ayushman_claims.csv
├── .env
└── README.md
```

### File Descriptions

| File                  | Purpose                      |
| --------------------- | ---------------------------- |
| `app.py`              | Core dashboard + ML pipeline |
| `firebase_auth.py`    | Login / Signup / Reset       |
| `firebase_db.py`      | Audit logging via Firebase   |
| `supabase_db.py`      | Cloud sync + deduplication   |
| `ayushman_claims.csv` | Local test dataset           |

---

# ⚙️ System Workflow

---

## High-Level Flowchart (CLI Format)

```bash
+--------------------+
| Investigator Login |
+---------+----------+
          |
          v
+----------------------+
| Upload Claims CSV    |
+---------+------------+
          |
          v
+----------------------+
| Hard Rule Validation |
+---------+------------+
          |
          v
+----------------------+
| Isolation Forest ML  |
+---------+------------+
          |
          v
+----------------------+
| Fraud Detected?      |
+----+-----------+-----+
     |           |
    Yes          No
     |           |
     v           v
+------------------------+     +-------------------+
| Send to OpenAI API     |     | Save as Clean     |
| Generate Justification |     | Record            |
+-----------+------------+     +-------------------+
            |
            v
+--------------------------+
| Save to Supabase Cloud   |
+-----------+--------------+
            |
            v
+--------------------------+
| Update Dashboard KPIs    |
+--------------------------+
```

---

# Use Case Diagram (CLI UML Representation)

```bash
Actor: Investigator

Use Cases:
------------------------------------------
(1) Login
(2) Upload Claims CSV
(3) View Fraud Dashboard
(4) Drill Down into Patient Claim
(5) View AI Justification
(6) Track Upload Sessions
(7) View Audit Logs
------------------------------------------

                +----------------------+
                | Fraud Detection App  |
                +----------------------+
                   ^   ^   ^   ^   ^
                   |   |   |   |   |
                   |   |   |   |   |
             +--------------------------------+
             | Investigator (Authenticated)   |
             +--------------------------------+
```

---

# System Architecture Diagram

                ┌────────────────────────────┐
                │   Claim Data Sources       │
                │  • Hospital Portal         │
                │  • CSV Upload              │
                │  • Govt DB / Mock API      │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │     Data Ingestion Layer   │
                │  • Validation              │
                │  • Schema Mapping          │
                │  • Duplicate Check         │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │ Fraud Detection Engine     │
                │  • Rule-based checks       │
                │  • ML anomaly model        │
                │  • Risk scoring logic      │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │        Database Layer      │
                │  • Claims                  │
                │  • Fraud flags             │
                │  • Hospital risk profiles  │
                └─────────────┬──────────────┘
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
     ┌──────────────────────┐   ┌──────────────────────┐
     │    Backend API Layer │   │     AI Assistant     │
     │  • Data retrieval    │   │  • Fraud explanation │
     │  • Report generation │   │  • Query interface   │
     │  • Dashboard queries │   │  • Report summaries  │
     └─────────────┬────────┘   └─────────────┬────────┘
                   │                          │
                   ▼                          ▼
           ┌────────────────────────────────────────┐
           │             Web Dashboard              │
           │  • Alerts & Risk Scores                │
           │  • Fraud Trends                        │
           │  • Claim Details                       │
           │  • Investigation Workflow              │
           └────────────────────────────────────────┘

# Detailed Processing Pipeline

### Step 1: Authentication

* Firebase verifies user credentials.
* Unauthorized access blocked.

### Step 2: Upload Center

* CSV parsed via Pandas.
* Duplicate PatientID skipped.

### Step 3: Hard Rules Engine

Examples:

* Negative length of stay
* Zero package rate
* Invalid demographic logic

### Step 4: ML Anomaly Detection

* `IsolationForest`
* Produces:

  * Fraud_Flag
  * Risk_Score
  * Fraud_Type

### Step 5: Generative AI Explanation

* OpenAI API generates:

  * Plain English explanation
  * Context-aware justification

### Step 6: Cloud Sync

* Save analyzed claims
* Update dashboard metrics
* Log audit activity



# Dashboard Metrics

* Total Claims
* Fraudulent Claims
* Fraud Percentage
* Suspicious Amount (₹)
* Trend Over Time
* Fraud Category Distribution



# Security Design

* Firebase Auth
* Role-based access
* Server-side ML validation
* Cloud audit logs
* Duplicate prevention logic


# Sample Fraud Scenarios Detected

| Fraud Type       | Example                                |
| ---------------- | -------------------------------------- |
| Ghost Billing    | No admission record but billing exists |
| Upcoding         | Higher-cost procedure claimed          |
| Fake Admission   | Demographic mismatch                   |
| Inflated Billing | Cost ratio unusually high              |



# Future Enhancements

* Hospital Risk Scoring
* Geo-Fraud Heatmaps
* Graph-Based Fraud Networks
* Model Retraining Pipeline
* Role-based Admin Panel
* Automated Investigation Report PDF Export



# Conclusion

The **Ayushman Bharat Fraud Detection Dashboard** is a scalable, AI-powered fraud monitoring system that combines:

* Machine Learning
* Generative AI
* Secure Authentication
* Cloud Data Sync
* Real-Time Analytics

It bridges the gap between raw anomaly detection and investigator-ready intelligence by translating risk signals into human-understandable explanations.
