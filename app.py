"""
CardioGuard Flask app (full).
Features:
 - Clinical & Lifestyle predictions (CSV or manual)
 - Heuristic condition inference & suggestions
 - AI chat endpoint (Groq preferred, OpenAI fallback)
 - Optional Supabase chat-history persistence
 - Model auto-download from REMOTE_MODEL_BASE (if configured)
 - Health endpoint for uptime monitoring
 - Supabase Auth (email/password, OAuth)
 - Subscription management integrated into dashboards
 - Access control based on subscription plans
 - User profile editing with privacy settings (JSONB)
 - User account deletion
Requirements (install in your venv):
pip install flask pandas numpy scikit-learn joblib python-dotenv flask-mail requests supabase py-sdk-openai groq flask-cors
Note: package names may vary slightly for groq or supabase clients; adjust to what you actually use:
 - supabase: "supabase" or "supabase-py" (depending on venv)
 - groq: "groq" (if you plan to use Groq)
 - openai: "openai" (fallback)
"""
# ============================================================
# Imports & App Initialization
# ============================================================
import os
import bcrypt
import uuid
import time
import json
import requests
import markdown
import jwt
from datetime import datetime, timezone, timedelta
import hashlib
from typing import List, Tuple, Optional
from groq import Groq
from supabase import create_client, Client
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, jsonify, session
)
from flask_cors import CORS
from flask_mail import Mail, Message
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap
import joblib
from jinja2 import Environment
from datetime import datetime


from functools import wraps # For login_required decorator
# Optional SDKs — imported inside try/except so app still runs without them
try:
    from supabase import create_client as create_supabase_client
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False
try:
    # Groq client; if not installed, fallback to openai
    import groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
# RAG Imports
try:
    from rag.consult_agent import cardio_consult
    from rag.embedder import get_embedding
    from rag.retriever import KnowledgeRetriever
    RAG_MODULES_AVAILABLE = True
except Exception as e:
    print(f"⚠️ RAG modules not available: {e}")
    RAG_MODULES_AVAILABLE = False
# App initialization
load_dotenv()
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
app.config["TEMPLATES_AUTO_RELOAD"] = True
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])
# Email config (optional)
app.config.update(
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_USE_TLS=os.getenv("MAIL_USE_TLS", "True") == "True",
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_DEFAULT_SENDER=(
        os.getenv("MAIL_SENDER_NAME", "CardioGuard Contact"),
        os.getenv("MAIL_USERNAME")
    ),
)

app.jinja_env.globals['now'] = datetime.now

mail = Mail(app)
@app.context_processor
def inject_now():
    return {"current_year": datetime.now().year}
# ============================================================
# Custom Jinja2 Filters
# ============================================================
@app.template_filter('datetime_from_iso')
def datetime_from_iso_filter(date_string):
    """Convert an ISO format date string (YYYY-MM-DDTHH:MM:SS.ssssss) to a datetime object."""
    if not date_string:
        return None
    try:
        # Handle different potential formats, e.g., with/without microseconds, with/without 'Z'
        # The most common format from Supabase is YYYY-MM-DDTHH:MM:SS.ssssss+ZZ:ZZ
        # But the basic YYYY-MM-DDTHH:MM:SS also occurs
        # datetime.fromisoformat() is quite robust for standard ISO formats
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    except ValueError:
        # If parsing fails, return None or raise an error as appropriate
        print(f"Warning: Could not parse date string '{date_string}' using fromisoformat.")
        try:
            # Fallback: Try parsing without microseconds if the string contains them
            # This is less robust and assumes a specific format, better to rely on fromisoformat if possible
            return datetime.strptime(date_string.split('.')[0], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            print(f"Warning: Could not parse date string '{date_string}' with fallback method either.")
            return None
@app.template_filter('now')
def now_filter():
    """Return the current datetime object."""
    return datetime.now()
# ============================================================
# Configuration & Constants
# ============================================================
# Paths, model filenames & optional remote base
RESULTS_DIR = os.path.join("static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CLINICAL_MODEL_FILE = os.getenv("CLINICAL_MODEL_FILE", "heart_rf_clinical.pkl")
CLINICAL_SCALER_FILE = os.getenv("CLINICAL_SCALER_FILE", "heart_scaler_clinical.pkl")
CLINICAL_TEMPLATE_FILE = os.getenv("CLINICAL_TEMPLATE_FILE", "heart_user_template_clinical.csv")
LIFESTYLE_MODEL_FILE = os.getenv("LIFESTYLE_MODEL_FILE", "heart_rf_lifestyle.pkl")
LIFESTYLE_SCALER_FILE = os.getenv("LIFESTYLE_SCALER_FILE", "heart_scaler_lifestyle.pkl")
LIFESTYLE_TEMPLATE_FILE = os.getenv("LIFESTYLE_TEMPLATE_FILE", "heart_user_template_lifestyle.csv")
REMOTE_MODEL_BASE = os.getenv("REMOTE_MODEL_BASE", "").rstrip("/")
# Optional Supabase (chat history persistence, auth)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY and SUPABASE_AVAILABLE:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized.")
    except Exception as e:
        print("⚠️ Supabase init failed:", e)
        supabase = None
else:
    if SUPABASE_URL or SUPABASE_KEY:
        print("⚠️ Supabase credentials provided but 'supabase' package not available.")
    else:
        print("ℹ️ Supabase not configured; chat history persistence disabled.")
# AI provider configuration (Groq preferred, OpenAI fallback)
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
PAYSTACK_PUBLIC_KEY = os.getenv("PAYSTACK_PUBLIC_KEY")
use_groq = bool(GROQ_API_KEY and GROQ_AVAILABLE)
use_openai = bool(OPENAI_API_KEY and OPENAI_AVAILABLE)
if use_groq:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq configured for chat.")
elif use_openai:
    openai.api_key = OPENAI_API_KEY
    print("✅ OpenAI configured for chat fallback.")
else:
    print("⚠️ No AI provider configured (GROQ or OpenAI). Chat endpoint will return an error unless keys installed.")
# Initialize RAG System
retriever = None
if RAG_MODULES_AVAILABLE and OPENAI_AVAILABLE:
    try:
        print("🔄 Initializing RAG system...")
        kb_path = os.path.join("rag", "knowledge_base.txt")
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                kb_text = f.read()
            # Simple chunking by double newline
            kb_chunks = [c.strip() for c in kb_text.split("\n\n") if c.strip()]
            if kb_chunks:
                # Generate embeddings
                print(f"   - Generating embeddings for {len(kb_chunks)} chunks...")
                embeddings = [get_embedding(chunk) for chunk in kb_chunks]
                retriever = KnowledgeRetriever(embeddings, kb_chunks)
                print("✅ RAG system initialized.")
            else:
                print("⚠️ Knowledge base is empty.")
        else:
            print(f"⚠️ Knowledge base file not found at {kb_path}")
    except Exception as e:
        print(f"⚠️ RAG initialization failed: {e}")
        retriever = None
else:
    print("ℹ️ RAG system disabled (missing modules or OpenAI key).")
# Input columns (forms)
BASE_COLUMNS_CLINICAL = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
BASE_COLUMNS_LIFESTYLE = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]
# ============================================================
# Virtual Meeting System (Jitsi) Helpers
# ============================================================
def generate_jitsi_url(appointment_id, appointment_time_str):
    """
    Generates a unique Jitsi meeting URL based on appointment details.
    Uses a deterministic hash to ensure the same appointment gets the same room.
    """
    # Create a unique identifier for the meeting room
    # Combining appointment ID and time should be sufficient
    unique_identifier = f"{appointment_id}_{appointment_time_str}"
    # Use SHA-256 hash to create a deterministic, unique room name
    room_hash = hashlib.sha256(unique_identifier.encode()).hexdigest()
    # Truncate the hash for readability (e.g., first 12 characters)
    room_name = room_hash[:12]
    # Use a public Jitsi server or your own self-hosted one
    jitsi_server = os.getenv("JITSI_SERVER_URL", "https://meet.jit.si") # e.g., "https://your-jitsi-instance.com"
    return f"{jitsi_server}/{room_name}"
# ============================================================
# Notification System (Email) Helpers
# ============================================================
def send_appointment_reminder_email(appointment_id, user_email, user_name, doctor_name, appointment_time_str):
    """
    Sends an email reminder for an appointment.
    """
    try:
        msg = Message(
            subject="[CardioGuard] Appointment Reminder",
            recipients=[user_email],
            body=f"""
            Dear {user_name},
            This is a friendly reminder that you have an appointment scheduled with Dr. {doctor_name} on {appointment_time_str}.
            Please ensure you are ready for the session.
            Best regards,
            The CardioGuard Team
            """
        )
        mail.send(msg)
        print(f"✅ Reminder email sent to {user_email} for appointment {appointment_id}")
        # Optionally, log this to the 'notifications' table
        supabase.table("notifications").insert({
            "user_id": session.get("user_id"), # This might need adjustment if called outside a request context
            "appointment_id": appointment_id,
            "type": "appointment_reminder",
            "message": f"Reminder for appointment with Dr. {doctor_name} on {appointment_time_str}",
            "channel": "email",
            "status": "sent"
        }).execute()
        return True
    except Exception as e:
        print(f"❌ Failed to send reminder email for appointment {appointment_id}: {e}")
        # Optionally, log failure to the 'notifications' table
        try:
            supabase.table("notifications").insert({
                "user_id": session.get("user_id"), # This might need adjustment if called outside a request context
                "appointment_id": appointment_id,
                "type": "appointment_reminder",
                "message": f"Failed to send reminder for appointment with Dr. {doctor_name} on {appointment_time_str}",
                "channel": "email",
                "status": "failed",
                "message": str(e) # Store error details
            }).execute()
        except Exception as log_e:
            print(f"❌ Failed to log notification failure: {log_e}")
        return False
# ============================================================
# Utility Functions
# ============================================================
def ensure_model_files(timeout=60):
    required = [
        CLINICAL_MODEL_FILE, CLINICAL_SCALER_FILE, CLINICAL_TEMPLATE_FILE,
        LIFESTYLE_MODEL_FILE, LIFESTYLE_SCALER_FILE, LIFESTYLE_TEMPLATE_FILE,
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if not missing:
        return
    if not REMOTE_MODEL_BASE:
        raise FileNotFoundError(f"Missing model/template files: {missing}. Set REMOTE_MODEL_BASE to auto-download them.")
    print("🛰️ Missing files detected:", missing)
    for fname in missing:
        url = f"{REMOTE_MODEL_BASE}/{fname}"
        try:
            print(f"Downloading {fname} from {url} ...")
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            with open(fname, "wb") as fh:
                fh.write(r.content)
            print("✅", fname)
        except Exception as e:
            print("❌ failed to download", fname, e)
# Load models & templates
def load_models():
    print("🔄 Loading models & templates...")
    clinical_model = joblib.load(CLINICAL_MODEL_FILE)
    clinical_scaler = joblib.load(CLINICAL_SCALER_FILE)
    clinical_template_df = pd.read_csv(CLINICAL_TEMPLATE_FILE)
    CLINICAL_FEATURE_COLUMNS = clinical_template_df.columns.tolist()
    lifestyle_model = joblib.load(LIFESTYLE_MODEL_FILE)
    lifestyle_scaler = joblib.load(LIFESTYLE_SCALER_FILE)
    lifestyle_template_df = pd.read_csv(LIFESTYLE_TEMPLATE_FILE)
    LIFESTYLE_FEATURE_COLUMNS = lifestyle_template_df.columns.tolist()
    print("✅ Models loaded")
    return (clinical_model, clinical_scaler, CLINICAL_FEATURE_COLUMNS,
            lifestyle_model, lifestyle_scaler, LIFESTYLE_FEATURE_COLUMNS)
(
    clinical_model, clinical_scaler, CLINICAL_FEATURE_COLUMNS,
    lifestyle_model, lifestyle_scaler, LIFESTYLE_FEATURE_COLUMNS
) = load_models()
# Prediction helper (clinical & lifestyle) 
def prepare_and_predict(df_raw: pd.DataFrame, model_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if model_type not in ("clinical", "lifestyle"):
        raise ValueError("Invalid model_type")
    df = df_raw.copy()
    # If headerless (pandas will provide integer column names), map positional columns
    if all(isinstance(c, int) for c in df.columns):
        if model_type == "clinical":
            df = df.iloc[:, :len(BASE_COLUMNS_CLINICAL)]
            df.columns = BASE_COLUMNS_CLINICAL[:df.shape[1]]
        else:
            df = df.iloc[:, :len(BASE_COLUMNS_LIFESTYLE)]
            df.columns = BASE_COLUMNS_LIFESTYLE[:df.shape[1]]
    # Store raw features *before* scaling for SHAP
    raw_features_df = df.copy()
    # Clinical pipeline: one-hot align with CLINICAL_FEATURE_COLUMNS then scale
    if model_type == "clinical":
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        df_enc = pd.get_dummies(df, columns=cat_cols)
        df_enc = df_enc.reindex(columns=CLINICAL_FEATURE_COLUMNS, fill_value=0)
        X = clinical_scaler.transform(df_enc.values)
        model = clinical_model
        # Ensure raw_features_df aligns with CLINICAL_FEATURE_COLUMNS as well for SHAP (after dummies)
        raw_features_df = pd.get_dummies(raw_features_df, columns=cat_cols)
        raw_features_df = raw_features_df.reindex(columns=CLINICAL_FEATURE_COLUMNS, fill_value=0)
    else:
        # Lifestyle pipeline: ensure numeric & fillna then align
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(df.mean(numeric_only=True))
        df_enc = df.reindex(columns=LIFESTYLE_FEATURE_COLUMNS, fill_value=0)
        X = lifestyle_scaler.transform(df_enc.values)
        model = lifestyle_model
        # Ensure raw_features_df aligns with LIFESTYLE_FEATURE_COLUMNS for SHAP
        for col in raw_features_df.columns:
             raw_features_df[col] = pd.to_numeric(raw_features_df[col], errors="coerce")
        raw_features_df = raw_features_df.fillna(raw_features_df.mean(numeric_only=True))
        raw_features_df = raw_features_df.reindex(columns=LIFESTYLE_FEATURE_COLUMNS, fill_value=0)
    # predict probabilities & classes
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        # fallback - sigmoid of decision_function if available
        try:
            df_dec = model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-df_dec))
        except Exception:
            probs = model.predict(X).astype(float)  # last resort
    preds = model.predict(X)
    # Build output
    out_df = df_raw.copy() # Use original raw input for display
    out_df["Prediction"] = preds
    out_df["Prob_Pos"] = np.round(probs, 4)
    out_df["Risk_Level"] = out_df["Prob_Pos"].apply(lambda p: "High" if p > 0.66 else ("Medium" if p > 0.33 else "Low"))
    # Return both results and raw features for SHAP
    return out_df, raw_features_df
# NEW: Function to generate SHAP explanation
def generate_shap_explanation(raw_features_df_row, model, feature_names):
    """
    Calculates SHAP values for a single prediction row.
    Returns a dictionary containing feature names and their SHAP values.
    """
    try:
        import numpy as np # Ensure numpy is available
        # Ensure the input is a single row DataFrame (shape (1, n_features))
        if raw_features_df_row.shape[0] != 1:
             print(f"Warning: Expected 1 row for SHAP, got {raw_features_df_row.shape[0]}. Taking first row.")
             raw_features_df_row = raw_features_df_row.iloc[[0]] # Make sure it's still a DataFrame
        # Get the underlying numpy array for the explainer
        # Use .values to get the raw numerical array
        input_array = raw_features_df_row.values
        # Create an explainer object for the specific model type
        # TreeExplainer is efficient for tree-based models like Random Forest
        explainer = shap.TreeExplainer(model)
        # Calculate SHAP values for the specific input array
        # Pass the numpy array directly
        shap_values_raw = explainer.shap_values(input_array)
        # --- Handle potential list structure for multi-output models (like RandomForestClassifier for binary classification) ---
        # shap_values_raw might be a list of arrays for each class [class_0_values, class_1_values] for binary classification
        # or a single array if the model outputs probability for positive class directly or is single-output.
        # We typically want the SHAP values corresponding to the positive class (or the output used for probability).
        # For RandomForestClassifier.predict_proba(X)[:, 1], we usually want the SHAP values for class 1.
        if isinstance(shap_values_raw, list):
            # It's a list, likely [shap_values_class_0, shap_values_class_1] for binary classification
            if len(shap_values_raw) == 2:
                 # Assume index 1 corresponds to the positive class (probability output for class 1)
                 shap_values_for_output = shap_values_raw[1] # Take values for the positive class
            elif len(shap_values_raw) == 1:
                 # Only one class output, use that
                 shap_values_for_output = shap_values_raw[0]
            else:
                 # Unexpected number of outputs
                 print(f"Warning: SHAP returned {len(shap_values_raw)} lists, expected 1 or 2. Using the first.")
                 shap_values_for_output = shap_values_raw[0]
        else:
            # It's a single array (e.g., from a regressor or a classifier configured differently)
            shap_values_for_output = shap_values_raw
        # Ensure shap_values_for_output is a 1D array corresponding to features of the single input row
        # It should have shape (n_features,) after indexing the correct class if needed
        if shap_values_for_output.ndim > 1:
            # If it's still multi-dimensional (e.g., (1, n_features)), squeeze it to (n_features,)
            shap_values_for_output = shap_values_for_output.squeeze(axis=0) # Remove the batch dimension
            if shap_values_for_output.ndim != 1:
                 print(f"Warning: SHAP output shape {shap_values_for_output.shape} is unexpected after squeeze. Attempting to flatten.")
                 shap_values_for_output = shap_values_for_output.flatten() # Fallback to flatten if still wrong shape
        # At this point, shap_values_for_output should be a 1D numpy array of length n_features
        # Convert to a list for JSON serialization
        shap_values_list = shap_values_for_output.tolist()
        feature_names_list = feature_names # This should be the list of feature names corresponding to the columns used for the model
        if len(shap_values_list) != len(feature_names_list):
             print(f"Warning: SHAP values length ({len(shap_values_list)}) does not match feature names length ({len(feature_names_list)}).")
             # This might happen if the feature alignment failed somewhere. Proceed carefully or return empty.
             # For now, let's assume they align correctly based on the model's training feature order.
             # If lengths differ significantly, the explanation might be misleading.
        # Create a list of dictionaries for easier handling in the template
        shap_explanation = [
            {"feature": feat, "shap_value": val}
            for feat, val in zip(feature_names_list, shap_values_list)
        ]
        # Sort by absolute SHAP value to show most impactful features first
        shap_explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return shap_explanation
    except Exception as e:
        print(f"Error generating SHAP explanation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return [] # Return an empty list if explanation fails
# Heuristic diagnostic rules (your rules, expanded)
def get_likely_condition(
    age, cholesterol, resting_bp, max_heart_rate,
    fasting_blood_sugar=None, exercise_angina=None,
    chest_pain_type=None, oldpeak=None, st_slope=None,
    sex=None, smoking=None, obesity=None,
    alcohol=None, physical_activity=None
) -> Tuple[str, List[str]]:
    """Return (likely_condition, suggestions) based on simple heuristic rules."""
    likely_condition = "Generalized Cardiac Risk"
    suggestions: List[str] = []
    # normalize
    def to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default
    age_v = to_float(age, 0)
    chol_v = to_float(cholesterol, 0)
    bp_v = to_float(resting_bp, 0)
    mhr_v = to_float(max_heart_rate, 0)
    old_v = to_float(oldpeak, 0)
    fbs_v = to_float(fasting_blood_sugar, 0)
    # normalize some categorical inputs to lowercase strings for matching
    smoking_s = str(smoking).strip().lower() if smoking is not None else ""
    alcohol_s = str(alcohol).strip().lower() if alcohol is not None else ""
    physical_activity_s = str(physical_activity).strip().lower() if physical_activity is not None else ""
    st_slope_s = str(st_slope).strip().lower() if st_slope is not None else ""
    # Rule set (ordered — earlier matches stronger)
    if (
        (chol_v > 240 and bp_v > 140)
        or st_slope_s in {"down", "downsloping", "2"}
        or old_v > 2.0
    ):
        likely_condition = "Coronary Artery Disease (CAD)"
        suggestions = [
            "Adopt a low-fat, high-fiber diet and reduce saturated fats.",
            "Ask your physician about coronary CT angiography or stress testing.",
            "Start supervised, low-to-moderate intensity cardiovascular exercise."
        ]
    elif (mhr_v < 100 and age_v > 60) or (old_v > 1.5 and bp_v > 130):
        likely_condition = "Heart Failure (HF) — possible reduced cardiac output"
        suggestions = [
            "Monitor weight daily and report rapid gains (>2kg in 2 days).",
            "Request an echocardiogram (echo) to evaluate ejection fraction.",
            "Limit high-intensity exertion until evaluated."
        ]
    elif fbs_v > 120 and chol_v > 200:
        likely_condition = "Diabetic Cardiomyopathy risk"
        suggestions = [
            "Tight glycemic control and diet to lower fasting blood sugar.",
            "Get regular ECG and consider echocardiography.",
            "Coordinate care with an endocrinologist and cardiologist."
        ]
    elif mhr_v > 180 or (exercise_angina in [1, "1", "yes", "true"] and mhr_v > 150):
        likely_condition = "Suspected Arrhythmia / Tachycardia"
        suggestions = [
            "Avoid stimulants (caffeine, amphetamines) and alcohol.",
            "Consider ECG, ambulatory Holter monitor or event recorder.",
            "If palpitations are associated with syncope, seek urgent care."
        ]
    elif bp_v >= 160 and age_v > 40:
        likely_condition = "Hypertensive Heart Disease"
        suggestions = [
            "Start/optimize antihypertensive therapy as advised by your clinician.",
            "Daily BP monitoring and salt restriction are recommended.",
            "Evaluate with echocardiography for left ventricular hypertrophy."
        ]
    elif smoking_s in {"1", "yes", "true", "y"} and chol_v > 200:
        likely_condition = "Smoking-related coronary risk"
        suggestions = [
            "Immediate smoking cessation; consider pharmacotherapy (NRT, varenicline).",
            "Full lipid profile and stress testing if symptomatic.",
            "Lifestyle changes: exercise, diet, and smoking cessation programs."
        ]
    elif alcohol_s in {"1", "yes", "true", "y"} and age_v > 35:
        likely_condition = "Alcohol-related cardiomyopathy risk"
        suggestions = [
            "Strict alcohol abstinence and clinical Cardio follow-up.",
            "Check liver function and vitamin B/thiamine levels.",
            "Consider echocardiogram and cardiology referral."
        ]
    elif (
        obesity in ["1", "yes", "true", "y"]
        or physical_activity_s in {"low", "none", "0"}
        and float(bp_v) > 130
    ):
        likely_condition = "Obesity-related cardiometabolic risk"
        suggestions = [
            "Structured weight-loss program and increased physical activity.",
            "Dietary counseling and consider referral to nutritionist.",
            "Check for sleep apnea if obese; treat if present."
        ]
    elif (chol_v < 200 and bp_v < 120 and fbs_v < 100):
        likely_condition = "Low Cardiac Risk"
        suggestions = [
            "Maintain a balanced diet and regular exercise.",
            "Annual routine checkup and continue healthy lifestyle."
        ]
    else:
        likely_condition = "Generalized Cardiac Risk"
        suggestions = [
            "Discuss results with your primary care clinician.",
            "Consider targeted tests (lipid panel, ECG, echo) if concerned."
        ]
    return likely_condition, suggestions
# Chat helpers (Groq / OpenAI) + Supabase persistence
def save_chat_message(user_id: str, role: str, message: str):
    if supabase:
        try:
            supabase.table("chat_history").insert({
                "user_id": user_id,
                "role": role,
                "message": message
            }).execute()
        except Exception as e:
            # don't fail the whole request because of DB issues
            print("⚠️ Supabase insert failed:", e)
def call_groq_chat(user_message: str, system_prompt: Optional[str] = None) -> str:
    # Minimal Groq chat usage - adjust to your groq SDK version
    if not GROQ_API_KEY or not GROQ_AVAILABLE:
        raise RuntimeError("Groq not available/configured")
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        response = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama3-13b"),  # choose desired model
            messages=messages,
            temperature=float(os.getenv("GROQ_TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", 512)),
        )
        # the exact response structure may vary — adapt to your groq client
        text = response.choices[0].message["content"]
        return text
    except Exception as e:
        print("Groq chat error:", e)
        raise
def call_openai_chat(user_message: str, system_prompt: Optional[str] = None) -> str:
    if not OPENAI_API_KEY or not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not available/configured")
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 512)),
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        print("OpenAI chat error:", e)
        raise
# Authentication & Authorization Helpers
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id") and not session.get("doctor_id") and not session.get("admin_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function
def get_user_role(user_id):
    """Helper to fetch user role from Supabase.
    Checks the 'users', 'doctors', and 'admins' tables based on the Supabase Auth ID.
    Falls back to 'user' if the user is not found in any table.
    """
    try:
        # Check 'users' table first
        user_data = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if user_data.data:
            return user_data.data.get("role", "user")
    except Exception as e:
        # If not found in 'users', proceed to check other tables
        if "PGRST116" not in str(e) or "0 rows" not in str(e):
            print(f"Error checking 'users' table for {user_id}: {e}")
    try:
        # Check 'doctors' table
        doctor_data = supabase.table("doctors").select("id").eq("id", user_id).single().execute()
        if doctor_data.data:
            # If found in doctors table, their role is 'doctor'
            return "doctor"
    except Exception as e:
        # If not found in 'doctors', proceed to check 'admins'
        if "PGRST116" not in str(e) or "0 rows" not in str(e):
            print(f"Error checking 'doctors' table for {user_id}: {e}")
    try:
        # Check 'admins' table
        # The 'admins' table likely has 'user_id' column matching the Supabase Auth ID
        admin_data = supabase.table("admins").select("id").eq("id", user_id).single().execute()
        if admin_data.data:
            # If found in admins table, their role is 'admin'
            return "admin"
    except Exception as e:
        # If not found in 'admins' either, log error (if it's not the expected '0 rows')
        if "PGRST116" not in str(e) or "0 rows" not in str(e):
            print(f"Error checking 'admins' table for {user_id}: {e}")
    # If not found in any table, default to 'user' or handle as needed
    print(f"User {user_id} not found in any role table ('users', 'doctors', 'admins'). Assigning default role 'user'.")
    return "user"
def get_user_subscription_status(user_id):
    """
    Helper to fetch user's current subscription status from Supabase.
    Admin users are automatically considered unrestricted (active and paid).
    """
    try:
        # Step 1: Check if user is admin first
        role = get_user_role(user_id)
        if role == "admin":
            # Admins should have full access even without a subscription
            return "active", False
        # Step 2: For normal users, fetch actual subscription
        subs_data = (
            supabase.table("user_subscriptions")
            .select("status, subscription_plans(name, is_free)")
            .eq("user_id", user_id)
            .neq("status", "cancelled")
            .order("start_date", desc=True)
            .execute()
        )
        if subs_data.data:
            latest_sub = subs_data.data[0]
            plan_info = latest_sub.get("subscription_plans", {})
            is_free_plan = plan_info.get("is_free", True)
            return latest_sub["status"], is_free_plan
        else:
            # No active subscriptions found, default to free
            return "inactive", True
    except Exception as e:
        print(f"Error fetching subscription for user {user_id}: {e}")
        # Default to safest option (free/restricted)
        return "inactive", True
def handle_supabase_auth_session(session_data):
    """
    Process the session data returned from Supabase auth and set Flask session.
    Determines the user's role by checking the 'users', 'doctors', and 'admins' tables.
    """
    user_info = jwt.decode(session_data["access_token"], options={"verify_signature": False}, algorithms=["RS256"], audience="authenticated")
    user_id = user_info["sub"]
    email = user_info.get("email")
    user_name = user_info.get("user_metadata", {}).get("name") or email
    # Determine role by checking tables in a specific order
    # You might want to adjust the logic here depending on how you assign roles initially
    # e.g., via admin panel, registration form, or by checking the existence in specific tables.
    role = "user" # Default role
    # Check 'doctors' table first (or the order you prefer)
    try:
        doctor_data = supabase.table("doctors").select("id").eq("id", user_id).single().execute()
        if doctor_data.data:
            role = "doctor"
    except Exception as e:
        # If not found in doctors, continue checking
        if "PGRST116" not in str(e) or "0 rows" not in str(e):
            print(f"Error checking 'doctors' table for {user_id}: {e}")
    # Check 'admins' table if role is still default
    if role == "user": # Only check admins if not already determined to be a doctor
        try:
            admin_data = supabase.table("admins").select("id").eq("id", user_id).single().execute()
            if admin_data.data:
                role = "admin"
        except Exception as e:
            # If not found in admins, continue checking users or keep default
            if "PGRST116" not in str(e) or "0 rows" not in str(e):
                print(f"Error checking 'admins' table for {user_id}: {e}")
    # Check 'users' table if role is still default
    # This also ensures the user profile exists in the 'users' table if needed for other data
    if role == "user":
        user_meta = supabase.table("users").select("*").eq("id", user_id).execute().data
        if not user_meta:
            # First-time user (or user not in 'users' table but defaulting to 'user' role): create profile
            # This assumes the user should exist in the 'users' table regardless of their primary role for general data.
            try:
                supabase.table("users").insert({
                    "id": user_id, # Using Supabase Auth ID as primary key
                    "email": email,
                    "name": user_name,
                    "role": "user" # The role here might be less critical if determined above, but set for consistency
                }).execute()
                # Fetch the data back to ensure we have it
                user_meta = [{"id": user_id, "role": "user", "name": user_name, "email": email}]
            except Exception as e:
                print(f"Error creating user profile in 'users' table for {user_id}: {e}")
                # If creation fails, we might not have user details, but we have the role determined above
                user_meta = [{"id": user_id, "role": role, "name": user_name, "email": email}]
        else:
            # User exists in 'users' table, potentially update name/email if changed in auth
            # Or just use the data fetched
            pass
    # Set Flask session variables
    session.clear()
    session["user_id"] = user_id
    session["role"] = role # Use the role determined from the tables
    session["user_name"] = user_name
    # Optionally, store email if needed elsewhere
    session["user_email"] = email
# Access Control Decorators
def check_subscription_access(f):
    """
    Decorator to check if the user has access to restricted features based on their subscription.
    Applies to features like /form, chat, booking.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get("user_id")
        # If not logged in, treat as free user (restricted)
        if not user_id:
            # For non-logged-in users, check session-based limit for /form
            if request.endpoint == 'predict': # Only apply to the form submission route
                last_form_time = session.get('last_form_time')
                if last_form_time:
                    last_time = datetime.fromisoformat(last_form_time)
                    now = datetime.now()
                    if now - last_time < timedelta(days=30): # 30 days for example
                        flash("You can only submit the form once per month as a non-logged-in user.", "warning")
                        return redirect(url_for('form'))
            # Redirect to login or show restricted message for other features
            elif request.endpoint in ['chat', 'book_appointment']:
                flash("Please log in to access this feature.", "warning")
                return redirect(url_for("login"))
            # Allow access to the decorated function
            return f(*args, **kwargs)
        # User is logged in, fetch subscription status
        try:
            sub_status, is_free_plan = get_user_subscription_status(user_id)
        except Exception:
            # If there's an error fetching subscription, default to restricted
            flash("Error checking subscription status. Please try again later.", "danger")
            return redirect(url_for('user_dashboard'))
        if not is_free_plan and sub_status == "active":
            # Paid subscriber, allow access
            return f(*args, **kwargs)
        # Free subscriber or no active subscription
        if request.endpoint == 'predict':
            # Check usage limit for /form
            try:
                # Get the first record of the current month for this user
                start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                records = supabase.table("records").select("id").eq("user_id", user_id).gte("created_at", start_of_month.isoformat()).execute().data
                if len(records) >= 1: # Change 1 to your desired limit for paid users if different
                    flash("You have reached your monthly limit for form submissions.", "warning")
                    return redirect(url_for('user_dashboard'))
            except Exception as e:
                print(f"Error checking form limit for user {user_id}: {e}")
                flash("Error checking usage limit. Please try again.", "danger")
                return redirect(url_for('user_dashboard'))
        elif request.endpoint in ['chat', 'book_appointment']:
            # Restrict chat and booking for free users
            flash("This feature is available for paid subscribers only.", "warning")
            return redirect(url_for('user_dashboard'))
        # Allow access to the decorated function (e.g., for /form if limit not reached)
        return f(*args, **kwargs)
    return decorated_function
# ensure files on startup
ensure_model_files()
# ============================================================
# Routes - Authentication
# ============================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "user").strip().lower()  # expected: 'user', 'doctor' (or admin via admin UI)
        if not name or not email or not password:
            flash("Please fill in all required fields.", "danger")
            return redirect(url_for("register"))
        # Check if user exists
        try:
            existing = supabase.table("users").select("id, email").eq("email", email).execute()
            if existing and existing.data:
                flash("Email already registered. Try logging in.", "warning")
                return redirect(url_for("login"))
        except Exception as e:
            print("Supabase check error:", e)
            flash("Registration currently unavailable. Try again later.", "danger")
            return redirect(url_for("register"))
        # Hash password with bcrypt
        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        # Create user in users table (role included)
        try:
            res = supabase.table("users").insert({
                "name": name,
                "email": email,
                "password_hash": hashed_pw,
                "role": role
            }).execute()
            # extract created user id if available
            new_user = res.data[0] if res and res.data else None
        except Exception as e:
            print("Supabase insert user error:", e)
            flash("Failed to create account. Try again later.", "danger")
            return redirect(url_for("register"))
        # If doctor, create doctor profile row
        try:
            if role == "doctor":
                user_id = new_user.get("id") if new_user else None
                if user_id:
                    supabase.table("doctors").insert({
                        "user_id": user_id,
                        "specialization": request.form.get("specialization", "General Cardiology"),
                        "bio": request.form.get("bio", ""),
                        "consultation_fee": float(request.form.get("consultation_fee", 0))
                    }).execute()
        except Exception as e:
            print("Warning: failed to create doctor profile:", e)
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))
    # GET
    return render_template("register.html")
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Please provide both email and password.", "warning")
            return redirect(url_for("login"))
        # ---- STEP 1: Check Users Table ----
        try:
            user_resp = supabase.table("users").select("*").eq("email", email).limit(1).execute()
        except Exception as e:
            print("Supabase user query error:", e)
            flash("Login temporarily unavailable. Please try again later.", "danger")
            return redirect(url_for("login"))
        if user_resp and user_resp.data:
            user = user_resp.data[0]
            stored_hash = user.get("password_hash")
            if stored_hash:
                try:
                    if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                        session.clear()
                        session["user_id"] = user.get("id")
                        session["user_name"] = user.get("name")
                        # --- STEP 1A: Detect Role ---
                        # Default role is 'user', unless doctor or admin
                        role = user.get("role", "user")
                        # If not admin, check if this user is a doctor
                        if role != "admin":
                            doctor_check = supabase.table("doctors").select("id").eq("id", user["id"]).execute()
                            if doctor_check.data:
                                role = "doctor"
                                session["doctor_id"] = doctor_check.data[0]["id"]
                        session["role"] = role
                        # --- STEP 1B: Redirect Based on Role ---
                        if role == "admin":
                            flash("Welcome back, Admin!", "success")
                            return redirect(url_for("admin_dashboard"))
                        elif role == "doctor":
                            flash("Welcome Doctor!", "success")
                            return redirect(url_for("doctor_dashboard"))
                        else:
                            flash("Welcome back!", "success")
                            return redirect(url_for("user_dashboard"))
                except ValueError as e:
                    print("Password hash error:", e)
                    flash("Error verifying credentials. Contact support.", "danger")
                    return redirect(url_for("login"))
        # ---- STEP 2: Check Admins Table (Fallback) ----
        try:
            admin_resp = supabase.table("admins").select("*").eq("email", email).limit(1).execute()
            if admin_resp and admin_resp.data:
                admin = admin_resp.data[0]
                stored_hash = admin.get("password_hash")
                if stored_hash and bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                    session.clear()
                    session["user_id"] = admin.get("id")
                    session["role"] = "admin"
                    session["user_name"] = admin.get("name")
                    flash("Welcome back, Admin!", "success")
                    return redirect(url_for("admin_dashboard"))
        except Exception as e:
            print("Supabase admin lookup error:", e)
        flash("Invalid email or password.", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")
    # GET
    return render_template("login.html")
@app.route("/logout")
def logout():
    supabase.auth.sign_out() # Sign out from Supabase
    session.pop("user_id", None)
    session.pop("role", None)
    session.pop("user_name", None)
    session.pop("user_email", None) # Clear email
    flash("Logged out successfully.")
    return redirect(url_for("login"))
# --- Social Login Routes ---
@app.route("/auth/<provider>")
def social_login(provider):
    providers = {"google", "facebook", "github"}
    if provider not in providers:
        flash("Unsupported provider", "danger")
        return redirect(url_for("login"))
    redirect_url = url_for("auth_callback", provider=provider, _external=True)
    # Supabase OAuth URL
    auth_url = f"{SUPABASE_URL}/auth/v1/authorize?provider={provider}&redirect_to={redirect_url}"
    return redirect(auth_url)
@app.route("/auth/callback/<provider>")
def auth_callback(provider):
    code = request.args.get("code")
    if not code:
        flash("Authentication failed", "danger")
        return redirect(url_for("login"))
    # Exchange code for session
    res = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=authorization_code",
        json={"code": code, "redirect_uri": url_for("auth_callback", provider=provider, _external=True)},
        headers={"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
    )
    data = res.json()
    if "access_token" not in data:
        flash("Login failed", "danger")
        return redirect(url_for("login"))
    handle_supabase_auth_session(data) # Process session and set Flask session
    flash("Signed in successfully!", "success")
    role = session["role"]
    if role == "admin":
        return redirect(url_for("admin_dashboard"))
    elif role == "doctor":
        return redirect(url_for("doctor_dashboard"))
    else:
        return redirect(url_for("user_dashboard"))
# ============================================================
# Routes - Main UI Pages
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/resources")
def resources():
    return render_template("resources.html")
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")
        if not all([name, email, message]):
            flash("Please fill all fields.", "danger")
            return redirect(url_for("contact"))
        try:
            msg = Message(
                subject=f"[CardioGuard] {subject or 'New Message'} from {name}",
                recipients=[os.getenv("MAIL_DEFAULT_RECEIVER")],
                body=f"""Name: {name}
Email: {email}
Message: {message}"""
            )
            mail.send(msg)
            flash("✅ Message sent successfully!", "success")
        except Exception as e:
            print("Mail send error:", e)
            flash("❌ Failed to send message.", "danger")
        return redirect(url_for("contact"))
    return render_template("contact.html")
@app.route("/form")
def form():
    # Check access for non-logged-in users only
    if not session.get("user_id"):
        if last_form_time := session.get('last_form_time'):
            last_time = datetime.fromisoformat(last_form_time)
            now = datetime.now()
            if now - last_time < timedelta(days=30): # 30 days
                flash("You can only access the form once per month as a non-logged-in user.", "warning")
                return redirect(url_for('index'))
    return render_template(
        "form.html",
        BASE_COLUMNS_CLINICAL=BASE_COLUMNS_CLINICAL,
        BASE_COLUMNS_LIFESTYLE=BASE_COLUMNS_LIFESTYLE
    )
# ============================================================
# Routes - Prediction & AI
# ============================================================
@app.route("/predict", methods=["POST"])
#@check_subscription_access # Apply access control
def predict():
    try:
        raw_type = (request.form.get("model_type") or "clinical").lower()
        model_map = {"heart": "clinical", "clinical": "clinical", "cardio": "lifestyle", "lifestyle": "lifestyle"}
        model_type = model_map.get(raw_type, "clinical")
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename:
            df = pd.read_csv(uploaded_file)
            results, raw_features_df = prepare_and_predict(df, model_type) # Receive raw features
        else:
            base_cols = BASE_COLUMNS_CLINICAL if model_type == "clinical" else BASE_COLUMNS_LIFESTYLE
            user_data = {}
            for c in base_cols:
                # alias mapping: allow 'sex' -> 'gender' etc.
                val = request.form.get(c)
                if val is None:
                    if c == "gender":
                        val = request.form.get("sex")
                    if c == "cholesterol":
                        val = request.form.get("chol")
                user_data[c] = val
            df = pd.DataFrame([user_data])
            results, raw_features_df = prepare_and_predict(df, model_type) # Receive raw features
        # --- POST-SUCCESS LOGIC FOR ACCESS CONTROL ---
        user_id = session.get("user_id")
        if user_id:
            # Logged in user: record the prediction in the 'records' table
            # This implicitly tracks usage for paid users based on plan limits (handled in decorator)
            try:
                supabase.table("records").insert({
                    "user_id": user_id,
                    "consultation_type": model_type,
                    "health_score": float(results.iloc[0]["Prob_Pos"]), # Store probability as score
                    "recommendation": "See results page for details." # Or derive from results
                }).execute()
            except Exception as e:
                print(f"Error saving record for user {user_id}: {e}")
                # Flash a warning but allow results to be shown
                flash("⚠️ Warning: Result not saved to your history.", "warning")
        else:
            # Non-logged-in user: update session time
            session['last_form_time'] = datetime.now().isoformat()
        # --- END POST-SUCCESS LOGIC ---
        # Generate SHAP Explanation - NEW
        shap_explanation = []
        if not raw_features_df.empty: # Only calculate if raw features exist
            feature_names = raw_features_df.columns.tolist()
            # Pass the single row DataFrame to the SHAP function
            shap_explanation = generate_shap_explanation(raw_features_df.iloc[[0]], # Use iloc[[0]] to keep it as a DataFrame with one row
                                                        clinical_model if model_type == "clinical" else lifestyle_model,
                                                        feature_names)
        # Save results CSV
        fname = f"{model_type}_pred_{uuid.uuid4().hex[:8]}.csv"
        save_path = os.path.join(RESULTS_DIR, fname)
        results.to_csv(save_path, index=False)
        download_link = url_for("static", filename=f"results/{fname}")
        single = results.iloc[0]
        prob = float(single["Prob_Pos"])
        risk = single["Risk_Level"]
        readable = (
            "Heart Disease Detected" if model_type == "clinical" and single["Prediction"] == 1
            else "Elevated Cardiovascular Risk" if model_type == "lifestyle" and single["Prediction"] == 1
            else "No Heart Disease Detected"
        )
        # heuristic inference (use available fields; fallback to 0)
        likely_condition, suggestions = get_likely_condition(
            age=single.get("age"),
            cholesterol=single.get("chol", single.get("cholesterol", 0)),
            resting_bp=single.get("trestbps", single.get("ap_hi", 0)),
            max_heart_rate=single.get("thalach", single.get("ap_lo", 0)),
            fasting_blood_sugar=single.get("fbs", single.get("gluc", 0)),
            sex=single.get("sex"),
            smoking=single.get("smoke"),
            alcohol=single.get("alco"),
            physical_activity=single.get("active"),
            oldpeak=single.get("oldpeak"),
            st_slope=single.get("slope")
        )
        return render_template(
            "result.html",
            result=readable,
            prob=prob,
            risk=risk,
            likely_condition=likely_condition,
            suggestions=suggestions,
            tables=[results.to_html(classes="table table-striped", index=False)],
            download_link=download_link,
            model_type=model_type,
            shap_explanation=shap_explanation # Pass SHAP explanation to template
        )
    except Exception as e:
        print("Prediction error:", e)
        flash(f"Error processing prediction: {str(e)}", "danger")
        return redirect(url_for("form"))
# AI chat endpoint (JSON API)
@app.route("/consult", methods=["POST"])
def consult():
    """
    JSON input:
    { "user_id": "user-123", "message": "I feel chest pain when..." }
    Response:
    { "reply": "...", "saved": true/false }
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "missing 'message' in JSON body"}), 400
    user_msg = data["message"]
    user_id = data.get("user_id", f"anon-{uuid.uuid4().hex[:8]}")
    system_prompt = os.getenv("CHAT_SYSTEM_PROMPT", "You are CardioConsult, a medically informed assistant. Provide safe, conservative guidance and always advise seeing a clinician for definitive diagnosis.")
    # store user message (best-effort)
    try:
        save_chat_message(user_id, "user", user_msg)
    except Exception as e:
        print("Warning: save chat failed:", e)
    # call AI provider
    ai_reply = ""
    try:
        # Try RAG first if available
        if retriever:
            try:
                ai_reply = cardio_consult(user_msg, retriever=retriever)
            except Exception as e:
                print("RAG consult failed, falling back to direct chat:", e)
                # Fallback will happen below if ai_reply is empty
        if not ai_reply:
            if use_groq:
                ai_reply = call_groq_chat(user_msg, system_prompt=system_prompt)
            elif use_openai:
                ai_reply = call_openai_chat(user_msg, system_prompt=system_prompt)
            else:
                return jsonify({"error": "No AI provider configured (set GROQ_API_KEY or OPENAI_API_KEY)."}), 500
    except Exception as e:
        print("AI call failed:", e)
        return jsonify({"error": "AI provider error", "details": str(e)}), 500
    # persist assistant reply
    try:
        save_chat_message(user_id, "assistant", ai_reply)
    except Exception as e:
        print("Warning: save chat failed:", e)
    return jsonify({"reply": ai_reply, "saved": bool(supabase)}), 200
# ============================================================
# Routes - Notifications & Messaging
# ============================================================
@app.route("/api/notifications/count")
def get_notification_count():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"count": 0, "notifications": []})
    try:
        # Fetch unread notifications
        res = supabase.table("notifications").select("*").eq("user_id", user_id).eq("is_read", False).order("created_at", desc=True).limit(5).execute()
        notifications = res.data if res.data else []
        return jsonify({"count": len(notifications), "notifications": notifications})
    except Exception as e:
        print(f"Error fetching notifications: {e}")
        return jsonify({"count": 0, "notifications": []})
@app.route("/api/messages/unread_count")
def get_unread_message_count():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"count": 0})
    try:
        # Fetch unread messages count
        # Note: This assumes a 'messages' table exists with 'receiver_id' and 'is_read'
        res = supabase.table("messages").select("id", count="exact").eq("receiver_id", user_id).eq("is_read", False).execute()
        return jsonify({"count": res.count})
    except Exception as e:
        print(f"Error fetching message count: {e}")
        return jsonify({"count": 0})
@app.route("/messages")
@app.route("/messages/<conversation_id>")
@login_required
def messages_page(conversation_id=None):
    user_id = session.get("user_id")
    # 1. Fetch list of conversations (users interacted with)
    # This is complex in Supabase without a dedicated 'conversations' table.
    # We'll try to fetch distinct senders/receivers from messages.
    # For simplicity, we might just list doctors if user is patient, or patients if user is doctor.
    # Or we can query unique user IDs from messages where user is sender or receiver.
    conversations = []
    active_conversation_user = None
    messages = []
    try:
        # Simplified approach: If user is patient, show doctors they have appointments with.
        # If user is doctor, show patients.
        # AND show anyone they have exchanged messages with.
        # Let's just fetch recent messages to find conversation partners
        sent = supabase.table("messages").select("receiver_id").eq("sender_id", user_id).execute()
        received = supabase.table("messages").select("sender_id").eq("receiver_id", user_id).execute()
        contact_ids = set()
        if sent.data:
            contact_ids.update([m['receiver_id'] for m in sent.data])
        if received.data:
            contact_ids.update([m['sender_id'] for m in received.data])
        # Also add booked appointments participants
        if session.get("role") == "doctor":
            # Find patients
            # (Assuming 'appointments' table exists and links doctor_id (which is user_id here?) to user_id)
            pass 
        # Fetch user details for these contacts
        if contact_ids:
            # Supabase 'in' query
            users_res = supabase.table("users").select("id, name, email").in_("id", list(contact_ids)).execute()
            for u in users_res.data:
                conversations.append({
                    "other_user_id": u['id'],
                    "other_user_name": u['name'],
                    "last_message_time": "", # To be filled
                    "last_message_content": "Click to view chat"
                })
        # If conversation_id is provided, load messages
        if conversation_id:
            # Get user details
            u_res = supabase.table("users").select("name, email").eq("id", conversation_id).single().execute()
            if u_res.data:
                active_conversation_user = u_res.data
            # Fetch messages
            # (sender = me AND receiver = them) OR (sender = them AND receiver = me)
            # Supabase doesn't support OR easily in one query like that without raw SQL or stored procedures usually, 
            # but let's try .or_ syntax if available or two queries.
            # Query 1: Sent by me
            m1 = supabase.table("messages").select("*").eq("sender_id", user_id).eq("receiver_id", conversation_id).execute()
            # Query 2: Sent by them
            m2 = supabase.table("messages").select("*").eq("sender_id", conversation_id).eq("receiver_id", user_id).execute()
            all_msgs = (m1.data or []) + (m2.data or [])
            # Sort by created_at
            all_msgs.sort(key=lambda x: x['created_at'])
            messages = all_msgs
            # Mark as read
            if m2.data:
                unread_ids = [m['id'] for m in m2.data if not m.get('is_read')]
                if unread_ids:
                    supabase.table("messages").update({"is_read": True}).in_("id", unread_ids).execute()
    except Exception as e:
        print(f"Error loading messages: {e}")
        flash("Error loading messages.", "danger")
    return render_template(
        "messages.html",
        conversations=conversations,
        active_conversation_id=conversation_id,
        active_conversation_user=active_conversation_user,
        messages=messages
    )
@app.route("/messages/send", methods=["POST"])
@login_required
def send_message():
    sender_id = session.get("user_id")
    receiver_id = request.form.get("receiver_id")
    content = request.form.get("content")
    if not receiver_id or not content:
        flash("Message cannot be empty.", "warning")
        return redirect(url_for("messages_page", conversation_id=receiver_id))
    try:
        # Insert message
        supabase.table("messages").insert({
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "content": content
        }).execute()
        # Send Email Notification
        # Fetch receiver email
        receiver_data = supabase.table("users").select("email, name").eq("id", receiver_id).single().execute()
        if receiver_data.data:
            receiver_email = receiver_data.data['email']
            receiver_name = receiver_data.data['name']
            try:
                msg = Message(
                    subject="[CardioGuard] New Message",
                    recipients=[receiver_email],
                    body=f"""
                    Hello {receiver_name},
                    You have received a new message from {session.get('user_name', 'a user')} on CardioGuard.
                    "{content}"
                    Log in to reply: {url_for('login', _external=True)}
                    """
                )
                mail.send(msg)
            except Exception as e:
                print(f"Failed to send email notification: {e}")
        # Create in-app notification
        supabase.table("notifications").insert({
            "user_id": receiver_id,
            "message": f"New message from {session.get('user_name')}",
            "type": "message"
        }).execute()
    except Exception as e:
        print(f"Error sending message: {e}")
        flash("Failed to send message.", "danger")
    return redirect(url_for("messages_page", conversation_id=receiver_id))
@app.route("/chat", methods=["GET", "POST"])
@login_required
def chat():
    if request.method == "GET":
        chat_log = session.get("chat_log", [])
        return render_template("chat.html", chat_log=chat_log)
    else:
        data = request.get_json()
        user_message = data.get("message", "")
        chat_log = session.get("chat_log", [])
        chat_log.append({"role": "user", "message": user_message})
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": (
                        "You are CardioConsult, a compassionate AI cardiologist providing preventive health advice. "
                        "Do not give medical diagnoses; only provide general wellness guidance based on cardiovascular science."
                    )},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens = 1000
            )
            reply = response.choices[0].message.content.strip()
            formatted_reply = markdown.markdown(reply, extensions=["fenced_code", "nl2br"])
        except Exception as e:
            print("Groq connection error:", e)
            reply = "⚠️ Sorry, I'm having trouble connecting to my heart consultation engine."
        chat_log.append({"role": "assistant", "message": formatted_reply})
        session["chat_log"] = chat_log[-10:]
        try:
            supabase.table("chat_logs").insert({
                "user_id": session.get("user_id", "guest"),
                "user_message": user_message,
                "bot_reply": formatted_reply,
            }).execute()
        except Exception as e:
            print("Logging error:", e)
        return jsonify({"reply": formatted_reply})
@app.route("/api/chat", methods=["POST"])
@limiter.limit("5 per minute")  # tighter limit for chat API
def ai_chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400
    # Check if user is logged in and has access (for API calls, check session or token)
    user_id = request.json.get("id") # Assuming API sends user_id
    if user_id:
        # Fetch subscription for API user
        try:
            sub_status, is_free_plan = get_user_subscription_status(user_id)
            if is_free_plan or sub_status != "active":
                return jsonify({"error": "AI chat access denied. Upgrade your subscription."}), 403
        except Exception:
            return jsonify({"error": "Subscription check failed."}), 500
    else:
        # For anonymous API calls, deny access
        return jsonify({"error": "AI chat requires authentication and a paid subscription."}), 403
    # Retrieve session chat (if applicable for API, maybe use DB or token-based session)
    # For simplicity here, we'll proceed without session for the API endpoint
    # In a real app, you'd manage state differently for APIs.
    # Prepare Groq request
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",  # free + hosted
                "messages": [
                    {"role": "system", "content": "You are CardioConsult, an AI cardiovascular consultant. Provide informative, safe health responses and always encourage users to see a doctor if symptoms persist."},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7
            },
            timeout=15
        )
        data = response.json()
        ai_reply = data["choices"][0]["message"]["content"]
        # Basic logging (could log to DB instead)
        print(f"""[CHAT LOG] User: {user_input}
AI: {ai_reply}
---""")
        return jsonify({"reply": ai_reply})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "CardioConsult AI is currently unavailable. Please try again later."}), 500
# -----------------------------
# Utility: Clear chat
# -----------------------------
@app.route("/chat/clear")
def clear_chat():
    session.pop("chat_history", None)
    flash("Chat history cleared.")
    return redirect(url_for("chat"))
@app.route("/chat-history/<user_id>", methods=["GET"])
def chat_history(user_id):
    if not supabase:
        return jsonify({"error": "Chat history persistence is not configured."}), 400
    try:
        res = supabase.table("chat_history").select("*").eq("user_id", user_id).order("created_at", desc=False).execute()
        return jsonify({"history": res.data}), 200
    except Exception as e:
        print("Supabase fetch failed:", e)
        return jsonify({"error": "Failed to fetch history", "details": str(e)}), 500
# ============================================================
# Routes - Doctor Features
# ============================================================
@app.route("/doctor/availability", methods=["GET", "POST"])
def doctor_availability():
    if session.get("role") != "doctor":
        flash("Doctor access only.", "warning")
        return redirect(url_for("login"))
    user_id = session.get("user_id")
    # find doctor record
    doctor_data = supabase.table("doctors").select("id").eq("user_id", user_id).single().execute()
    if not doctor_data.data:
        flash("Doctor profile not found.", "danger")
        return redirect(url_for("doctor_dashboard"))
    doctor_id = doctor_data.data["id"]
    if request.method == "POST":
        date_str = request.form.get("available_date")
        start_time = request.form.get("start_time")
        end_time = request.form.get("end_time")
        try:
            available_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            supabase.table("doctor_availability").insert({
                "doctor_id": doctor_id,
                "available_date": str(available_date),
                "start_time": start_time,
                "end_time": end_time
            }).execute()
            flash("Availability added successfully.", "success")
        except Exception as e:
            print("Error adding availability:", e)
            flash("Failed to add availability.", "danger")
        return redirect(url_for("doctor_availability"))
    # GET request
    slots = supabase.table("doctor_availability").select("*").eq("doctor_id", doctor_id).order("available_date").execute().data
    return render_template("doctor_availability.html", slots=slots)
@app.route("/book", methods=["GET", "POST"])
@login_required
#@check_subscription_access # Apply access control
def book_appointment():
    if session.get("role") != "user":
        flash("Login as a user to book an appointment.", "warning")
        return redirect(url_for("login"))
    if request.method == "POST":
        slot_id = request.form.get("slot_id")
        # Fetch slot
        slot_data = supabase.table("doctor_availability").select("*").eq("id", slot_id).single().execute()
        if not slot_data:
            flash("Invalid slot selected.", "danger")
            return redirect(url_for("book_appointment"))
        slot = slot_data.data
        if slot.get("is_booked"):
            flash("This slot has already been booked.", "danger")
            return redirect(url_for("book_appointment"))
        # Create appointment
        doctor_id = slot["doctor_id"]
        user_id = session.get("user_id")
        dt_str = f"{slot['available_date']} {slot['start_time']}"
        appointment_time = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        # --- NEW: Generate Jitsi URL ---
        jitsi_url = generate_jitsi_url(slot_id, dt_str) # Use slot_id and time_str for uniqueness
        supabase.table("appointments").insert({
            "user_id": user_id,
            "doctor_id": doctor_id,
            "appointment_time": appointment_time.isoformat(),
            "status": "pending",
            "meeting_url": jitsi_url # Add the generated URL
        }).execute()
        # Mark slot as booked
        supabase.table("doctor_availability").update({"is_booked": True}).eq("id", slot_id).execute()
        # --- NEW: Send Appointment Reminder Email ---
        # Fetch doctor details for the email
        doctor_details = supabase.table("doctors").select("id, specialization").eq("id", doctor_id).single().execute().data
        user_details = supabase.table("users").select("name, email").eq("id", user_id).single().execute().data
        doctor_name = doctor_details.get("specialization", "Unknown") # Or fetch from user table if needed
        user_name = user_details.get("name")
        user_email = user_details.get("email")
        appointment_time_str = appointment_time.strftime("%Y-%m-%d at %H:%M")
        send_appointment_reminder_email(slot_id, user_email, user_name, doctor_name, appointment_time_str)
        flash("Appointment booked successfully!", "success")
        return redirect(url_for("user_dashboard"))
    # GET: get all available slots
    available_slots = supabase.table("doctor_availability").select(
        "id, doctor_id, available_date, start_time, end_time"
    ).eq("is_booked", False).execute().data
    # --- UPDATE: Fetch doctor details including rating ---
    doctors_data = supabase.table("doctors").select("id, specialization, bio, rating").execute().data # Add 'rating' here
    doctor_lookup = {d["id"]: d for d in doctors_data}
    # Build a frontend-friendly JSON object for JavaScript
    formatted_slots = []
    formatted_slots.extend(
        {
            "id": s["id"],
            "doctor_id": s["doctor_id"],
            "doctor": doctor_lookup.get(s["doctor_id"], {}).get(
                "specialization", "Unknown"
            ),
            # --- UPDATE: Include rating in the formatted slot data ---
            "rating": doctor_lookup.get(s["doctor_id"], {}).get("rating", "No ratings yet"), # Add rating info
            "date": s["available_date"],
            "start": s["start_time"],
            "end": s["end_time"],
        }
        for s in available_slots
    )
    return render_template("book_appointment.html", slots=formatted_slots)

@app.route("/my-bookings")
@login_required
def my_bookings():
    user_id = session.get("user_id")
    # --- NEW: Include meeting_url in query ---
    appointments = supabase.table("appointments").select(
        "id, appointment_time, status, doctor_id, meeting_url" # Include meeting_url
    ).eq("user_id", user_id).order("appointment_time", desc=True).execute().data
    doctor_lookup = {}
    try:
        if doctor_ids := list({a["doctor_id"] for a in appointments}):
            # --- UPDATE: Fetch doctor details including rating ---
            doctors = supabase.table("doctors").select("id, specialization, rating").in_("id", doctor_ids).execute().data # Add 'rating' here
            doctor_lookup = {d["id"]: d for d in doctors}
    except Exception as e:
        print("Doctor fetch error:", e)
    return render_template("my_bookings.html", appointments=appointments, doctors=doctor_lookup)

# ============================================================
# Routes - Subscription Management
# ============================================================
@app.route("/pricing")
def pricing():
    """Display available subscription plans."""
    try:
        # Fetch all active subscription plans from Supabase
        response = supabase.table("subscription_plans").select("*").eq("is_active", True).execute()
        plans = response.data or []
        # Sort by price (optional)
        plans.sort(key=lambda x: float(x["price"]))
    except Exception as e:
        print(f"Error fetching plans: {e}")
        flash("Error loading plans. Please try again later.", "danger")
        plans = [] # Fallback to empty list if fetch fails
    return render_template("pricing.html", plans=plans) 
@app.route("/subscribe/<plan_name>", methods=["GET", "POST"])
@login_required
def subscribe(plan_name):
    if session.get("role") != "user":
        flash("Only users can subscribe.", "danger")
        return redirect(url_for("user_dashboard"))
    # Fetch selected plan
    plan_response = supabase.table("subscription_plans").select("*").eq("name", plan_name).execute()
    plan = plan_response.data[0] if plan_response.data else None
    if not plan:
        flash("Plan not found.", "danger")
        return redirect(url_for("user_dashboard"))
    user_id = session["user_id"]
    if request.method == "POST":
        # Create Paystack transaction
        amount_kobo = int(float(plan["price"]) * 100)
        headers = {
            "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "email": session.get("user_email"), # Ensure email is stored in session during login
            "amount": amount_kobo,
            "callback_url": url_for("verify_payment", _external=True),
            "metadata": {
                "user_id": user_id,
                "plan_id": plan["id"],
                "plan_name": plan["name"]
            }
        }
        response = requests.post("https://api.paystack.co/transaction/initialize", json=data, headers=headers)
        res_data = response.json()
        if res_data.get("status"):
            auth_url = res_data["data"]["authorization_url"]
            return redirect(auth_url)
        else:
            flash("Failed to initialize payment.", "danger")
    return render_template("subscribe_confirm.html", plan=plan, PAYSTACK_PUBLIC_KEY=PAYSTACK_PUBLIC_KEY)
@app.route("/verify_payment")
@login_required
def verify_payment():
    reference = request.args.get("reference")
    if not reference:
        flash("Missing payment reference.", "danger")
        return redirect(url_for("user_dashboard"))
    headers = {"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"}
    response = requests.get(f"https://api.paystack.co/transaction/verify/{reference}", headers=headers)
    res_data = response.json()
    if res_data.get("status") and res_data["data"]["status"] == "success":
        metadata = res_data["data"]["metadata"]
        user_id = metadata["user_id"]
        plan_id = metadata["plan_id"]
        # Record in Supabase
        supabase.table("user_subscriptions").insert({
            "user_id": user_id,
            "plan_id": plan_id,
            "status": "active",
            "start_date": datetime.now(timezone.utc).isoformat()
        }).execute()
        flash("Subscription successful!", "success")
    else:
        flash("Payment verification failed.", "danger")
    return redirect(url_for("user_dashboard"))
@app.route("/user/subscriptions/cancel/<sub_id>", methods=["POST"])
@login_required
def cancel_subscription(sub_id):
    if session.get("role") != "user":
        flash("Unauthorized", "danger")
        return redirect(url_for("user_dashboard"))
    user_id = session.get("user_id")
    # Update status to cancelled
    supabase.table("user_subscriptions").update({"status": "cancelled"}).eq("id", sub_id).eq("user_id", user_id).execute()
    flash("Subscription cancelled.", "info")
    return redirect(url_for("user_dashboard"))
# ============================================================
# Routes - Dashboards
# ============================================================

@app.route("/doctor/dashboard")
@login_required
def doctor_dashboard():
    # Check if user is logged in as a doctor
    if session.get("role") != "doctor":
        flash("Access denied. Doctors only.", "danger")
        return redirect(url_for("login"))
    user_id = session.get("user_id")
    doctor_id = session.get("doctor_id") # Assuming this is set during login for doctors
    # If doctor_id isn't in session, fetch it from the doctors table
    if not doctor_id:
        try:
            doc_data = supabase.table("doctors").select("id").eq("id", user_id).single().execute()
            if doc_data and doc_data:
                doctor_id = doc_data.data["id"]
                session["doctor_id"] = doctor_id # Store for future use in this session
            else:
                flash("Doctor profile not found. Please contact support.", "danger")
                return redirect(url_for("login"))
        except Exception as e:
            print(f"Error fetching doctor ID: {e}")
            flash("Error accessing dashboard. Please try again later.", "danger")
            return redirect(url_for("login"))
    try:
        # Fetch doctor profile details (name, email, bio, specialization from users and doctors tables)
        user_data = supabase.table("users").select("name, email").eq("id", user_id).single().execute().data
        doctor_details = supabase.table("doctors").select("bio, specialization, consultation_fee").eq("id", doctor_id).single().execute().data
        # Combine user and doctor data
        profile_info = {**user_data, **doctor_details}
        # Fetch availability slots for this doctor
        availability_slots = supabase.table("doctor_availability").select("*").eq("doctor_id", doctor_id).order("available_date").execute().data
        # Fetch booked appointments for this doctor
        # --- NEW: Include meeting_url in query ---
        booked_appointments = supabase.table("appointments").select("id, user_id, appointment_time, status, meeting_url").eq("doctor_id", doctor_id).order("appointment_time").execute().data # Include meeting_url
        # Optionally, fetch user details for the booked appointments
        user_ids_needed = [appt["user_id"] for appt in booked_appointments]
        patient_details = {}
        if user_ids_needed:
            users_info = supabase.table("users").select("id, name, email").in_("id", user_ids_needed).execute().data
            patient_details = {u["id"]: u for u in users_info}
    except Exception as e:
        print(f"Error fetching doctor dashboard data: {e}")
        flash("Error loading dashboard data. Please try again later.", "danger")
        return redirect(url_for("login")) # Or render a partial template
    return render_template(
        "doctor_dashboard.html",
        profile=profile_info,
        availability_slots=availability_slots,
        booked_appointments=booked_appointments,
        patient_details=patient_details
    )
    
    
  
@app.route('/rate_doctor/<doctor_id>', methods=['POST'])
@login_required
def rate_doctor(doctor_id):
    if session.get("role") != "user":
        flash("Only patients can rate doctors.", "warning")
        return redirect(url_for("user_dashboard")) # Or appropriate redirect

    user_id = session.get("user_id")
    rating_str = request.form.get("rating")
    comment = request.form.get("comment", "").strip()

    try:
        rating = int(rating_str)
        if not (1 <= rating <= 5):
            flash("Please provide a valid rating (1–5).", "danger")
            return redirect(request.referrer or url_for("user_dashboard"))
    except (ValueError, TypeError):
        flash("Invalid rating value.", "danger")
        return redirect(request.referrer or url_for("user_dashboard"))

    try:
        # --- Fetch Current Rating and Count ---
        current_doc_res = supabase.table("doctors").select("rating, rating_count").eq("id", doctor_id).single().execute()
        current_doc = current_doc_res.data
        current_avg = current_doc.get("rating", 0.0) # Default to 0 if no rating exists yet
        current_count = current_doc.get("rating_count", 0) # Default to 0

        # --- Calculate New Average ---
        # Total points before = old_avg * old_count
        # Total points after = old_points + new_rating
        # New average = total_points_after / (old_count + 1)
        old_total_points = current_avg * current_count
        new_total_points = old_total_points + rating
        new_count = current_count + 1
        new_avg = new_total_points / new_count

        # --- Update Doctors Table ---
        supabase.table("doctors").update({
            "rating": round(new_avg, 2), # Round to 2 decimal places for display
            "rating_count": new_count
        }).eq("id", doctor_id).execute()

        flash("Thank you for your feedback!", "success")
    except Exception as e:
        print(f"Rating error: {e}")
        flash("Failed to submit rating.", "danger")

    # Redirect back to where the user came from (e.g., bookings page)
    return redirect(request.referrer or url_for("user_dashboard"))
   
    
@app.route("/user/dashboard")
@login_required
def user_dashboard():
    if session.get("role") != "user":
        flash("Access denied. Users only.", "danger")
        return redirect(url_for("login"))
    user_id = session.get("user_id")
    try:
        user_data = supabase.table("users").select("name, email, address, profile_picture_url, profile_settings").eq("id", user_id).single().execute().data
        # Store email in session for payment verification
        session["user_email"] = user_data["email"]
        # Fetch user's health records if applicable
        records = supabase.table("records").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data
        # Fetch user's booked appointments
        # --- NEW: Include meeting_url in query ---
        appointments = supabase.table("appointments").select(
            "id, doctor_id, appointment_time, status, meeting_url" # Include meeting_url
        ).eq("user_id", user_id).order("appointment_time", desc=True).execute().data
        # Fetch doctor details for the appointments
        doctor_ids_needed = [appt["doctor_id"] for appt in appointments]
        doctor_details = {}
        if doctor_ids_needed:
            doctors_info = supabase.table("doctors").select("id, specialization").in_("id", doctor_ids_needed).execute().data
            doctors_lookup = {d["id"]: d for d in doctors_info}
            users_info = supabase.table("users").select("id, name").in_("id", doctor_ids_needed).execute().data
            users_lookup = {u["id"]: u for u in users_info}
            # Combine doctor and user details
            for appt in appointments:
                doc_id = appt["doctor_id"]
                doc_info = doctors_lookup.get(doc_id, {})
                user_info = users_lookup.get(doc_id, {})
                doctor_details[doc_id] = {**doc_info, **user_info} # e.g., {'specialization': 'Cardiology', 'name': 'Dr. Smith'}
        # Prepare data for charts (example remains the same)
        if records:
            labels = [r["created_at"][:10] for r in records]
            scores = [r["health_score"] for r in records]
        else:
            labels, scores = [], []
        # --- NEW: Fetch user subscriptions ---
        user_subscriptions = supabase.table("user_subscriptions").select(
            "*, subscription_plans(name, price, duration_days, is_free)"
        ).eq("user_id", user_id).execute().data
        # Determine current plan status for UI hints
        current_plan_is_free = True
        if user_subscriptions:
            # Get the most recent active or non-cancelled subscription
            active_subs = [sub for sub in user_subscriptions if sub.get("status") != "cancelled"]
            if active_subs:
                # Sort by start_date descending to get the latest
                latest_sub = sorted(active_subs, key=lambda x: x.get("start_date", ""), reverse=True)[0]
                plan_info = latest_sub.get("subscription_plans", {})
                current_plan_is_free = plan_info.get("is_free", True)
    except Exception as e:
        print(f"Error fetching user dashboard  {e}")
        flash("Error loading dashboard data. Please try again later.", "danger")
        return redirect(url_for("login")) # Or render a partial template
    return render_template(
        "user_dashboard.html",
        user=user_data,
        records=records,
        chart_labels=json.dumps(labels),
        chart_scores=json.dumps(scores),
        appointments=appointments, # Pass appointments to the template
        doctor_details=doctor_details, # Pass doctor details for the template
        user_subscriptions=user_subscriptions, # Pass subscriptions to the template
        current_plan_is_free=current_plan_is_free # Pass plan status for UI
    )
@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if session.get("role") != "admin":
        flash("Access denied. Admins only.", "danger")
        return redirect(url_for("login"))
    try:
        # Fetch core counts
        total_users = len(supabase.table("users").select("id").execute().data)
        total_doctors = len(supabase.table("doctors").select("id").execute().data)
        total_admins = len(supabase.table("admins").select("id").execute().data)
        total_records = len(supabase.table("records").select("id").execute().data)
        total_chats = len(supabase.table("chat_logs").select("id").execute().data)
        total_appointments = len(supabase.table("appointments").select("id").execute().data)
        # Fetch counts for different appointment statuses
        appointment_statuses = supabase.table("appointments").select("status").execute().data
        status_counts = {}
        for appt in appointment_statuses:
            status = appt.get("status", "Unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        # Fetch admin details
        admins = supabase.table("admins").select("name, email").execute().data
        # Aggregate data for chart (records by consultation type)
        record_data = supabase.table("records").select("consultation_type").execute().data
        type_count = {}
        for r in record_data:
            ctype = r.get("consultation_type", "Unknown")
            type_count[ctype] = type_count.get(ctype, 0) + 1
        # Aggregate data for user role chart (if roles are stored in users table)
        user_roles = supabase.table("users").select("role").execute().data
        role_counts = {}
        for user in user_roles:
            role = user.get("role", "user") # Default to 'user' if role is missing
            role_counts[role] = role_counts.get(role, 0) + 1
        # --- Activity Chart Data ---
        # Example: Fetch user registrations over time (last 6 months)
        # Get the date 6 months ago
        six_months_ago = datetime.now(timezone.utc) - timedelta(days=6*30)
        six_months_ago_str = six_months_ago.strftime('%Y-%m-%d')
        # Fetch user creation dates within the last 6 months
        user_registrations_raw = supabase.table("users").select("created_at").gte("created_at", six_months_ago_str).execute().data
        # Extract just the date part and count occurrences
        registration_dates = [user["created_at"][:10] for user in user_registrations_raw]
        from collections import Counter
        registration_counts = Counter(registration_dates)
        # Sort the dates for the chart
        sorted_dates = sorted(registration_counts.keys())
        registration_counts_list = [registration_counts[date] for date in sorted_dates]
        # Example: Fetch appointment counts over time (last 6 months)
        appointment_counts_raw = supabase.table("appointments").select("appointment_time").gte("appointment_time", six_months_ago_str).execute().data
        appointment_dates = [appt["appointment_time"][:10] for appt in appointment_counts_raw]
        appointment_counts = Counter(appointment_dates)
        sorted_appt_dates = sorted(appointment_counts.keys())
        appointment_counts_list = [appointment_counts[date] for date in sorted_appt_dates]
        # Prepare labels for the chart (using the sorted unique dates)
        activity_labels = sorted(set(sorted_dates + sorted_appt_dates)) # Combine and sort unique dates
        # Prepare data for user registrations, aligning with activity_labels
        user_activity_data = [registration_counts.get(date, 0) for date in activity_labels]
        # Prepare data for appointments, aligning with activity_labels
        appt_activity_data = [appointment_counts.get(date, 0) for date in activity_labels]
        # --- Fetch all user subscriptions for admin ---
        user_subscriptions = supabase.table("user_subscriptions").select(
            "id, user_id, status, start_date, end_date, subscription_plans(name, price)"
        ).execute().data
        if user_ids_for_subs := list(
            {sub["user_id"] for sub in user_subscriptions}
        ):
            users_for_subs = supabase.table("users").select("id, name, email").in_("id", user_ids_for_subs).execute().data
            user_lookup_for_subs = {u["id"]: u for u in users_for_subs}
        else:
            user_lookup_for_subs = {}
    except Exception as e:
        print(f"Error fetching admin dashboard data: {e}")
        flash("Error loading dashboard data. Please try again later.", "danger")
        return redirect(url_for("admin_dashboard")) # Or render a partial template
    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        total_doctors=total_doctors,
        total_admins=total_admins,
        total_records=total_records,
        total_chats=total_chats,
        total_appointments=total_appointments,
        admins=admins,
        record_type_labels=json.dumps(list(type_count.keys())),
        record_type_counts=json.dumps(list(type_count.values())),
        user_role_labels=json.dumps(list(role_counts.keys())),
        user_role_counts=json.dumps(list(role_counts.values())),
        appointment_status_labels=json.dumps(list(status_counts.keys())),
        appointment_status_counts=json.dumps(list(status_counts.values())),
        # Pass activity chart data
        activity_labels=json.dumps(activity_labels),
        user_activity_data=json.dumps(user_activity_data),
        appt_activity_data=json.dumps(appt_activity_data),
        # Pass subscription data for admin
        user_subscriptions=user_subscriptions,
        user_lookup_for_subs=user_lookup_for_subs
    )
def log_activity(user_id, plan_id, activity_type):
    supabase.table("user_subscription_activity").insert({
        "user_id": user_id,
        "plan_id": plan_id,
        "activity_type": activity_type
    }).execute()

# ============================================================
# Routes - User Profile & Account Management
# ============================================================


def send_welcome_email_if_incomplete(user_id, email, name, role):
    """Sends a welcome email if the user's profile is incomplete."""
    if role != "doctor":
        return  # Only for doctors

    try:
        # Fetch doctor-specific details to check completeness
        doctor_data = supabase.table("doctors").select("bio, specialization, license_number").eq("id", user_id).single().execute().data
        bio = doctor_data.get("bio", "").strip()
        spec = doctor_data.get("specialization", "").strip()
        license_num = doctor_data.get("license_number", "").strip()

        if not bio or not spec or not license_num:
            msg = Message(
                subject="[CardioGuard] Welcome! Please Complete Your Profile",
                recipients=[email],
                body=f"""
                Hi {name},

                Welcome to CardioGuard! 🎉

                To start accepting appointments, please complete your professional profile:
                - Add your specialization
                - Write a short bio
                - Provide your license number
                - Set your consultation fee

                Visit your dashboard to get started:
                {url_for('doctor_dashboard', _external=True)}

                Best regards,
                The CardioGuard Team
                """
            )
            mail.send(msg)
            print(f"✅ Welcome email sent to {email}")
    except Exception as e:
        print(f"Failed to check doctor profile completeness or send email: {e}")

@app.route('/user/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    user_id = session.get("user_id")
    role = session.get("role")

    # Determine the correct dashboard based on role
    dashboard_redirect = f"{role}_dashboard" if role in ["user", "doctor", "admin"] else "login"

    if not user_id or role not in ["user", "doctor"]:
        flash("Access denied.", "danger")
        return redirect(url_for("login"))

    if request.method == 'POST':
        if role == "user":
            # --- Handle User Profile Updates ---
            new_address = request.form.get('address', '').strip()
            new_picture_url = request.form.get('profile_picture_url', '').strip()

            # --- Handle User Privacy Settings ---
            current_data_res = supabase.table("users").select("profile_settings").eq("id", user_id).single().execute()
            current_settings = current_data_res.data.get("profile_settings", {})
            new_address_public = request.form.get('address_public') == 'on'
            new_picture_public = request.form.get('picture_public') == 'on'
            updated_settings = {
                **current_settings,
                "address_public": new_address_public,
                "picture_public": new_picture_public
            }

            try:
                supabase.table("users").update({
                    "address": new_address,
                    "profile_picture_url": new_picture_url,
                    "profile_settings": updated_settings
                }).eq("id", user_id).execute()
                flash("User profile updated successfully!", "success")
            except Exception as e:
                print(f"Error updating user profile: {e}")
                flash("Failed to update profile. Please try again.", "danger")

        elif role == "doctor":
            # --- Handle Doctor Profile Updates ---
            name = request.form.get('name', '').strip()
            bio = request.form.get('bio', '').strip()
            specialization = request.form.get('specialization', '').strip()
            consultation_fee_str = request.form.get('consultation_fee', '0')
            license_number = request.form.get('license_number', '').strip()

            try:
                consultation_fee = float(consultation_fee_str)
            except ValueError:
                consultation_fee = 0.0

            try:
                # Update user table (name)
                supabase.table("users").update({"name": name}).eq("id", user_id).execute()
                # Update doctor table (specialization, bio, fee, license)
                supabase.table("doctors").update({
                    "bio": bio,
                    "specialization": specialization,
                    "consultation_fee": consultation_fee,
                    "license_number": license_number,
                }).eq("id", user_id).execute()
                flash("Doctor profile updated successfully!", "success")

                # Check if profile is now complete and send welcome email if it was incomplete before
                send_welcome_email_if_incomplete(user_id, session.get("user_email"), name, role)

            except Exception as e:
                print(f"Error updating doctor profile: {e}")
                flash("Failed to update profile. Please try again.", "danger")

        return redirect(url_for(dashboard_redirect)) # Redirect back to the appropriate dashboard after update

    # --- GET Request (or after POST redirect) ---
    try:
        user_data = supabase.table("users").select("*").eq("id", user_id).single().execute().data
        session["user_email"] = user_data.get("email") # Ensure email is in session for email helper

        if role == "user":
            profile_settings = user_data.get("profile_settings", {})
            address_public = profile_settings.get("address_public", False)
            picture_public = profile_settings.get("picture_public", False)
            return render_template('edit_profile.html',
                                   user=user_data,
                                   address_public=address_public,
                                   picture_public=picture_public)

        elif role == "doctor":
            doctor_data = supabase.table("doctors").select("*").eq("id", user_id).single().execute().data
            # Combine user and doctor data for the template context
            profile_info = {**user_data, **doctor_data}
            return render_template('edit_doctor_profile.html', doctor=profile_info)

    except Exception as e:
        print(f"Error fetching profile for edit: {e}")
        flash("Error loading profile data.", "danger")
        return redirect(url_for(dashboard_redirect))

@app.route('/user/account/delete', methods=['POST'])
@login_required
def delete_account():
    user_id = session.get("user_id")
    if not user_id or session.get("role") != "user":
         flash("Access denied.", "danger")
         return redirect(url_for("user_dashboard"))

    # IMPORTANT: Handle related data first!
    # This is a critical step. You must delete or anonymize all data associated with this user.
    # Example: Delete user's records, appointments, messages, chat history, etc.
    # You need to adapt these queries based on your actual table structure and relationships.
    # Be very careful with foreign key constraints.

    # Example deletions (uncomment and adapt as needed):
    try:
        # Delete user's health records
        supabase.table("records").delete().eq("user_id", user_id).execute()
        # Delete user's appointments (both as user and any created as doctor if applicable)
        supabase.table("appointments").delete().eq("user_id", user_id).execute()
        # Delete messages sent/received by this user
        supabase.table("messages").delete().eq("sender_id", user_id).execute()
        supabase.table("messages").delete().eq("receiver_id", user_id).execute()
        # Delete chat history
        supabase.table("chat_history").delete().eq("user_id", user_id).execute()
        # Delete chat logs
        supabase.table("chat_logs").delete().eq("user_id", user_id).execute()
        # Delete notifications for this user
        supabase.table("notifications").delete().eq("user_id", user_id).execute()
        # Delete user's subscription records
        supabase.table("user_subscriptions").delete().eq("user_id", user_id).execute()
        # Delete user's availability slots (if any, assuming user could be a doctor too - adjust logic)
        # supabase.table("doctor_availability").delete().eq("doctor_id", user_id).execute() # Only if user is doctor
        # Delete user's doctor profile (if applicable)
        # supabase.table("doctors").delete().eq("user_id", user_id).execute() # Only if user is doctor

        # Finally, delete the main user entry
        supabase.table("users").delete().eq("id", user_id).execute()

        # Clear session
        session.clear()
        flash("Your account has been permanently deleted.", "info")
        return redirect(url_for("index")) # Or a goodbye page
    except Exception as e:
        print(f"Error deleting account or related data: {e}")
        flash("Failed to delete account due to an error. Please contact support.", "danger")
        return redirect(url_for("edit_profile")) # Or back to settings - maybe don't clear session here on error

# ============================================================
# Routes - Utility
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()}), 200
# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    # For production use gunicorn (or fly/gunicorn)
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "False") == "True")
