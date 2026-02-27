"""
supabase_db.py — Supabase integration for Ayushman Bharat Fraud Detection
─────────────────────────────────────────────────────────────────────────
Key functions:
  insert_new_rows_only(df, ...)      → Insert fresh rows, SKIP duplicates
  fetch_data_from_supabase()         → pd.DataFrame (all claims)
  get_db_stats()                     → cumulative counts + last-updated date
  log_upload_session(...)            → record each upload in upload_sessions
  get_upload_history(uid, limit)     → list of past uploads with date + counts
  upsert_audit_log(...)              → write one audit event
  fetch_audit_log(uid, limit)        → read audit events

Required Supabase SQL (run once in SQL Editor):
─────────────────────────────────────────────────
  -- 1. Main claims table
  create table if not exists claims (
      "PatientID"           text primary key,
      "Age"                 int,
      "Final_Billed_Amount" float8,
      "Fraud_Flag"          int     default 0,
      "Risk_Score"          float8  default 0,
      "Fraud_Type"          text    default '',
      "AI_Justification"    text    default '',
      "Risk score per claim" float8  default 0,
      "user_id"             uuid    references auth.users,
      "uploaded_at"         timestamptz default now()
  );

  -- Enable RLS (Allow all authenticated users to read for Global Dashboard)
  alter table claims enable row level security;
  create policy "Everyone can view claims" on claims for select using (true);
  create policy "Users can insert their own" on claims for insert with check (auth.uid() = user_id);

  -- 2. Upload sessions log
  create table if not exists upload_sessions (
      id            bigserial primary key,
      uid           text,
      user_id       uuid default auth.uid(),
      filename      text,
      total_rows    int  default 0,
      new_rows      int  default 0,
      skipped_rows  int  default 0,
      fraud_detected int default 0,
      suspicious_amt float8 default 0,
      uploaded_at   timestamptz default now()
  );

  alter table upload_sessions enable row level security;
  create policy "Everyone can view sessions" on upload_sessions for select using (true);
  create policy "Users can insert their own sessions" on upload_sessions for insert with check (auth.uid() = user_id);

  -- 3. Audit log
  create table if not exists audit_log (
      id          bigserial primary key,
      uid         text,
      user_id     uuid default auth.uid(),
      action      text,
      description text,
      patient_id  text,
      fraud_type  text,
      amount      float8,
      created_at  timestamptz default now()
  );

  alter table audit_log enable row level security;
  create policy "Users can only see their own activity" on audit_log for all using (auth.uid() = user_id);

  -- 4. Detected frauds table (Extracted after analysis)
  create table if not exists detected_frauds (
      "PatientID"           text primary key,
      "Age"                 int,
      "Primary_Diagnosis"   text    default '',
      "Final_Billed_Amount" float8,
      "Fraud_Type"          text    default 'Anomalous',
      "AI_Justification"    text    default '',
      "Risk_Score"          float8  default 0,
      "user_id"             uuid    references auth.users,
      "detected_at"         timestamptz default now()
  );

  alter table detected_frauds enable row level security;
  create policy "Authenticated can read frauds" on detected_frauds for select using (auth.role() = 'authenticated');
  create policy "Users can upsert frauds" on detected_frauds for insert with check (auth.uid() = user_id);
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "")
_client = None


# ══════════════════════════════════════════════════════════════
#  CLIENT
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def init_supabase():
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_URL or SUPABASE_URL == "your_supabase_url":
        return None
    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return _client
    except Exception as e:
        print(f"[Supabase] Connection error: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════
def sign_up(email, password):
    client = init_supabase()
    return client.auth.sign_up({"email": email, "password": password})

def sign_in(email, password):
    client = init_supabase()
    return client.auth.sign_in_with_password({"email": email, "password": password})

def sign_out():
    client = init_supabase()
    return client.auth.sign_out()

def send_password_reset_email(email):
    client = init_supabase()
    return client.auth.reset_password_for_email(email)

def get_current_user():
    client = init_supabase()
    try:
        res = client.auth.get_user()
        return res.user if res else None
    except:
        return None

def recover_session(access_token, refresh_token):
    """Attempt to restore a session using tokens."""
    client = init_supabase()
    try:
        res = client.auth.set_session(access_token, refresh_token)
        return res.user if res else None
    except:
        return None


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY
                and SUPABASE_URL not in ("", "your_supabase_url"))


# ══════════════════════════════════════════════════════════════
#  FETCH ALL CLAIMS  (paginated)
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner=False)
def fetch_data_from_supabase(table: str = "claims",
                             page_size: int = 1000,
                             user_id: str = None) -> pd.DataFrame:
    """
    Fetch all rows from the claims table using cursor-based pagination.
    Returns a pd.DataFrame (empty if table is empty).
    """
    client   = init_supabase()
    all_rows = []
    start    = 0
    
    # Try fetching with user filter if requested
    try:
        while True:
            query = client.table(table).select("*").range(start, start + page_size - 1)
            if user_id:
                query = query.eq("user_id", user_id)
            resp  = query.execute()
            batch = resp.data or []
            all_rows.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
    except Exception as e:
        # If user_id column is missing, fall back to global fetch
        if "user_id" in str(e).lower() and user_id:
            all_rows = []
            start = 0
            while True:
                query = client.table(table).select("*").range(start, start + page_size - 1)
                resp  = query.execute()
                batch = resp.data or []
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                start += page_size
        else:
            raise e

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  INSERT NEW ROWS ONLY  (skip duplicates)
# ══════════════════════════════════════════════════════════════
def insert_new_rows_only(df: pd.DataFrame,
                         table: str = "claims",
                         conflict_col: str = "PatientID",
                         chunk_size: int = 200,
                         user_id: str = None) -> dict:
    """
    Insert rows from `df` that don't exist yet in the Supabase table.
    Rows whose `conflict_col` value already exists are SKIPPED (not updated).

    Strategy:
      1. Pull all existing primary-key values from Supabase.
      2. Split incoming df into NEW (not in DB) and SKIPPED (already in DB).
      3. Insert NEW rows in chunks.

    Returns a dict:
      {
        "total":   int,   # rows in the uploaded file
        "new":     int,   # rows actually inserted
        "skipped": int,   # duplicates skipped
        "error":   str | None
      }
    """
    result = {"total": len(df), "new": 0, "skipped": 0, "error": None}

    try:
        client = init_supabase()

        # ── Step 1: Fetch existing keys ───────────────────────
        existing_keys = set()
        page, ps = 0, 1000
        while True:
            query = client.table(table).select(conflict_col).range(page, page + ps - 1)
            if user_id:
                query = query.eq("user_id", user_id)
            resp  = query.execute()
            batch = resp.data or []
            existing_keys.update(row[conflict_col] for row in batch if conflict_col in row)
            if len(batch) < ps:
                break
            page += ps

        # ── Step 2: Separate new vs duplicate rows ────────────
        if conflict_col in df.columns:
            mask        = ~df[conflict_col].astype(str).isin(existing_keys)
            new_df      = df[mask].copy()
            result["skipped"] = int((~mask).sum())
        else:
            # No conflict column — insert everything
            new_df = df.copy()

        if user_id:
            new_df["user_id"] = user_id

        result["new"] = len(new_df)

        if new_df.empty:
            return result   # nothing to insert

        # ── Step 3: Insert in chunks ──────────────────────────
        records  = _clean_df_for_json(new_df).to_dict(orient="records")
        n_chunks = math.ceil(len(records) / chunk_size)
        for i in range(n_chunks):
            chunk = records[i * chunk_size : (i + 1) * chunk_size]
            client.table(table).insert(chunk).execute()

    except RuntimeError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = str(e)[:400]

    return result


# ══════════════════════════════════════════════════════════════
#  CUMULATIVE DATABASE STATS
# ══════════════════════════════════════════════════════════════
def get_db_stats(table: str = "claims",
                 sessions_table: str = "upload_sessions",
                 cost_col: str = None,
                 user_id: str = None) -> dict:
    """
    Return aggregate stats from Supabase for the dashboard KPI cards.
    """
    default = {"total_claims": 0, "total_fraud": 0,
               "suspicious_amt": 0.0, "last_updated": None,
               "total_uploads": 0, "error": None}
    try:
        client = init_supabase()

        # 1. Total claims count
        q1 = client.table(table).select("*", count="exact")
        if user_id: q1 = q1.eq("user_id", user_id)
        resp   = q1.limit(0).execute()
        total  = resp.count or 0

        # 2. Fraud count
        q2 = client.table(table).select("*", count="exact").eq("Fraud_Flag", 1)
        if user_id: q2 = q2.eq("user_id", user_id)
        f_resp  = q2.limit(0).execute()
        fraud   = f_resp.count or 0

        # 3. Suspicious amount
        amt = 0.0
        if not cost_col:
            # Try to find a valid cost column
            try:
                sample = client.table(table).select("*").limit(1).execute()
                if sample.data:
                    row = sample.data[0]
                    if "Final_Billed_Amount" in row: cost_col = "Final_Billed_Amount"
                    elif "TreatmentCost" in row:      cost_col = "TreatmentCost"
            except:
                pass

        if cost_col:
            try:
                q3 = client.table(table).select(cost_col).eq("Fraud_Flag", 1)
                if user_id: q3 = q3.eq("user_id", user_id)
                a_resp = q3.execute()
                amt    = sum(float(r.get(cost_col) or 0) for r in (a_resp.data or []))
            except Exception as e:
                print(f"[Supabase] Error summing cost: {e}")

        # 4. Last upload date from upload_sessions
        last_dt = None
        uploads = 0
        try:
            q4 = client.table(sessions_table).select("uploaded_at", count="exact").order("uploaded_at", desc=True)
            if user_id: q4 = q4.eq("user_id", user_id)
            s_resp  = q4.limit(1).execute()
            uploads = s_resp.count or 0
            rows    = s_resp.data or []
            if rows:
                raw_ts = rows[0].get("uploaded_at","")
                if raw_ts:
                    try:
                        dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                        last_dt = dt.strftime("%d %b %Y")
                    except:
                        last_dt = raw_ts[:10]
        except Exception:
            pass

        return {
            "total_claims":   int(total),
            "total_fraud":    int(fraud),
            "suspicious_amt": float(amt),
            "last_updated":   last_dt or "Never",
            "total_uploads":  int(uploads),
            "error":          None,
        }
    except Exception as e:
        default["error"] = str(e)[:300]
        return default
def get_user_activity_stats(uid):
    """
    Fetch total claims and fraud detections for a specific user ID from upload_sessions.
    """
    try:
        client = init_supabase()
        response = client.table("upload_sessions") \
            .select("total_rows, fraud_detected") \
            .or_(f"user_id.eq.{uid},uid.eq.{uid}") \
            .execute()
        
        data = response.data
        if not data:
            return 0, 0
            
        total_claims = sum(int(row.get('total_rows', 0)) for row in data)
        total_frauds = sum(int(row.get('fraud_detected', 0)) for row in data)
        
        return total_claims, total_frauds
    except Exception as e:
        print(f"[Supabase] get_user_activity_stats error: {e}")
        return 0, 0



def get_trend_data(sessions_table: str = "upload_sessions", n_days: int = 30, user_id: str = None) -> pd.DataFrame:
    """
    Fetch real daily analysis stats from upload_sessions.
    Returns a DataFrame with columns: Date, Flagged, Total
    """
    try:
        client = init_supabase()
        q = client.table(sessions_table).select("uploaded_at, fraud_detected, total_rows").order("uploaded_at", desc=True)
        if user_id:
            q = q.eq("user_id", user_id)
        resp = q.limit(100).execute()
        rows = resp.data or []
        
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["uploaded_at"]).dt.date
        
        daily = df.groupby("Date").agg({
            "fraud_detected": "sum",
            "total_rows": "sum"
        }).reset_index()
        
        daily.columns = ["Date", "Flagged", "Total"]
        return daily.sort_values("Date")
    except Exception as e:
        print(f"[Supabase] get_trend_data error: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  UPLOAD SESSION LOG
# ══════════════════════════════════════════════════════════════
def log_upload_session(uid: str, filename: str,
                       total_rows: int, new_rows: int,
                       skipped_rows: int, fraud_detected: int,
                       suspicious_amt: float = 0.0) -> bool:
    """
    Write one row to `upload_sessions` to record upload history.
    """
    try:
        client = init_supabase()
        client.table("upload_sessions").insert({
            "uid":            uid,
            "user_id":        uid if len(str(uid or "")) > 20 else None, # Assume UUID if long
            "filename":       filename,
            "total_rows":     int(total_rows),
            "new_rows":       int(new_rows),
            "skipped_rows":   int(skipped_rows),
            "fraud_detected": int(fraud_detected),
            "suspicious_amt": float(suspicious_amt),
            "uploaded_at":    datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as e:
        print(f"[Supabase] log_upload_session error: {e}")
        return False


def get_upload_history(uid: str = None, limit: int = 10) -> list[dict]:
    """
    Retrieve recent upload sessions (newest first).
    If uid is None, returns sessions for ALL users (admin view).
    Returns list of dicts with keys: filename, total_rows, new_rows,
    skipped_rows, fraud_detected, suspicious_amt, uploaded_at, date_label.
    """
    try:
        client = init_supabase()
        q = (client.table("upload_sessions")
                   .select("*")
                   .order("uploaded_at", desc=True)
                   .limit(limit))
        if uid:
            q = q.eq("uid", uid)
        resp = q.execute()
        rows = resp.data or []
        result = []
        for row in rows:
            ts_raw = row.get("uploaded_at", "")
            try:
                ts       = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                date_lbl = ts.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                date_lbl = ts_raw[:16]
            result.append({**row, "date_label": date_lbl})
        return result
    except Exception as e:
        print(f"[Supabase] get_upload_history error: {e}")
        return []


# ══════════════════════════════════════════════════════════════
#  AUDIT LOG
# ══════════════════════════════════════════════════════════════
def upsert_audit_log(uid: str, action: str, description: str,
                     patient_id: str = "", fraud_type: str = "",
                     amount: float = 0.0) -> bool:
    try:
        client = init_supabase()
        client.table("audit_log").insert({
            "uid":         uid,
            "user_id":     uid if len(str(uid or "")) > 20 else None,
            "action":      action,
            "description": description,
            "patient_id":  patient_id or "",
            "fraud_type":  fraud_type  or "",
            "amount":      float(amount or 0),
            "created_at":  datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as e:
        print(f"[Supabase] audit_log error: {e}")
        return False


def fetch_audit_log(uid: str, limit: int = 20) -> list[dict]:
    try:
        client   = init_supabase()
        # Try finding by user_id first (UUID), fallback to uid (Text) for legacy
        resp     = (client.table("audit_log")
                          .select("*")
                          .or_(f"user_id.eq.{uid},uid.eq.{uid}")
                          .order("created_at", desc=True)
                          .limit(limit).execute())
        rows     = resp.data or []
        now_ts   = datetime.now(timezone.utc)
        result   = []
        for row in rows:
            ts_str = row.get("created_at", "")
            try:
                ts   = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                diff = int((now_ts - ts).total_seconds())
                ago  = (f"{diff}S AGO"       if diff < 60
                        else f"{diff//60}M AGO"   if diff < 3600
                        else f"{diff//3600}H AGO" if diff < 86400
                        else f"{diff//86400}D AGO")
            except Exception:
                ago = "JUST NOW"
            result.append({
                "action":      row.get("action", "Activity"),
                "description": row.get("description", ""),
                "patient_id":  row.get("patient_id", ""),
                "fraud_type":  row.get("fraud_type", ""),
                "amount":      row.get("amount", 0),
                "ago":         ago,
            })
        return result
    except Exception as e:
        print(f"[Supabase] fetch_audit_log error: {e}")
        return []


# ══════════════════════════════════════════════════════════════
#  SAVE PROCESSED RESULTS  (upsert — updates existing rows)
# ══════════════════════════════════════════════════════════════
def save_fraud_results_to_supabase(df: pd.DataFrame,
                                   table: str = "claims",
                                   chunk_size: int = 200,
                                   on_conflict: str = "PatientID",
                                   user_id: str = None) -> tuple[bool, str]:
    """Update existing rows with fraud analysis columns (upsert)."""
    try:
        client   = init_supabase()
        clean_df = _clean_df_for_json(df)
        if user_id:
            clean_df["user_id"] = user_id
        records  = clean_df.to_dict(orient="records")
        n_chunks = math.ceil(len(records) / chunk_size)
        for i in range(n_chunks):
            chunk = records[i * chunk_size : (i + 1) * chunk_size]
            client.table(table).upsert(chunk, on_conflict=on_conflict).execute()
        return True, f"✅ Saved {len(df):,} processed records to Supabase."
    except Exception as e:
        return False, f"❌ Could not save results: {str(e)[:300]}"


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def _clean_df_for_json(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
        else:
            df[col] = df[col].apply(_safe_value)
    return df


def _safe_value(v):
    if v is None:
        return None
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.floating,)):
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(v, np.bool_):       return bool(v)
    except ImportError:
        pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ══════════════════════════════════════════════════════════════
#  SYNC LOCAL CSV -> SUPABASE
# ══════════════════════════════════════════════════════════════
def sync_local_csv_to_supabase(csv_path: str, user_id: str = None) -> tuple[bool, str]:
    """
    Reads the local CSV and pushes all data to Supabase (new rows only).
    """
    try:
        if not os.path.exists(csv_path):
            return False, f"File not found: {csv_path}"
        
        df = pd.read_csv(csv_path)
        if df.empty:
            return False, "CSV file is empty."
            
        # Insert new rows
        res = insert_new_rows_only(df, user_id=user_id)
        
        if res.get("error"):
            return False, f"Sync error: {res['error']}"
            
        msg = f"✅ Sync Complete: {res['new_count']} new rows added, {res['skipped_count']} skipped."
        return True, msg
        
    except Exception as e:
        return False, f"❌ Sync failed: {str(e)}"


# ══════════════════════════════════════════════════════════════
#  DETECTED FRAUDS REPOSITORY
# ══════════════════════════════════════════════════════════════

def upsert_detected_frauds(df: pd.DataFrame, user_id: str = None) -> dict:
    """
    Specifically for the 'detected_frauds' table. Upserts all rows from df.
    """
    if df.empty:
        return {"new_count": 0, "status": "empty"}
        
    client = init_supabase()
    
    # 1. Clean for JSON
    clean_df = df.copy()
    if user_id: 
        clean_df["user_id"] = user_id
    
    # Ensure relevant columns exist
    cols = ["PatientID", "Age", "Primary_Diagnosis", "Final_Billed_Amount", 
            "Fraud_Type", "AI_Justification", "Risk_Score", "user_id"]
    existing_cols = [c for c in cols if c in clean_df.columns]
    
    payload_df = _clean_df_for_json(clean_df[existing_cols])
    payload = payload_df.to_dict(orient="records")
    
    # 2. Sequential upsert (can be chunked)
    try:
        resp = client.table("detected_frauds").upsert(payload, on_conflict="PatientID").execute()
        return {"status": "success", "count": len(payload)}
    except Exception as e:
        print(f"[Supabase] upsert_detected_frauds error: {e}")
        return {"status": "error", "error": str(e)}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_detected_frauds(user_id: str = None) -> pd.DataFrame:
    """
    Fetch high-priority fraud cases for the 'Fraud Audit Report' page.
    If user_id is provided, only fetches cases analyzed by that user.
    """
    try:
        client = init_supabase()
        q = client.table("detected_frauds").select("*").order("Risk_Score", desc=True)
        if user_id:
            try:
                resp = q.eq("user_id", user_id).execute()
                return pd.DataFrame(resp.data or [])
            except Exception as e:
                # If user_id column is missing in detected_frauds
                if "user_id" in str(e).lower():
                    resp = q.execute() # Fallback to global
                    return pd.DataFrame(resp.data or [])
                else:
                    raise e
        else:
            resp = q.execute()
            return pd.DataFrame(resp.data or [])
    except Exception as e:
        print(f"[Supabase] fetch_detected_frauds error: {e}")
        return pd.DataFrame()

def update_claim_status(patient_id: str, status: str, user_id: str = None) -> bool:
    """
    Update the investigation status of a specific claim.
    """
    try:
        client = init_supabase()
        # Update both claims and detected_frauds to keep synced
        client.table("claims").update({"Investigation_Status": status}).eq("PatientID", patient_id).execute()
        client.table("detected_frauds").update({"Investigation_Status": status}).eq("PatientID", patient_id).execute()
        
        # Log the action
        upsert_audit_log(
            uid=user_id,
            action="Status Update",
            description=f"Status for Patient {patient_id} changed to '{status}'.",
            patient_id=patient_id
        )
        return True
    except Exception as e:
        print(f"[Supabase] update_claim_status error: {e}")
        return False
