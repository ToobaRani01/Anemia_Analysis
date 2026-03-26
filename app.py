"""
app.py — Streamlit UI for EYES-DEFY-ANEMIA
Run: streamlit run app.py
"""

import os
import sys
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# Make src/ importable when app.py lives inside src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import MODEL_PATH, SCALER_PATH, IMAGE_TYPES
from utils import (
    load_model, load_scaler,
    prepare_inference_inputs, predict_hemoglobin,
    classify_anemia, severity_color, severity_emoji
)





# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Eyes Defy Anemia",
    page_icon="👁️",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Root Variables */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --primary-light: #818cf8;
        --secondary: #ec4899;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-light: #f8fafc;
        --text-dark: #1e293b;
        --text-light: #64748b;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body, .stApp {
        background-color: #f0f4f8;
    }

    /* Main Container */
    .main {
        padding: 2rem 2.5rem;
    }

    /* Main Header */
    .main-header {
        text-align: center;
        background: radial-gradient(circle at top left, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 3.5rem 2rem;
        border-radius: 30px;
        margin: 0 0 2.5rem 0;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
    }

    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1.5px;
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
        letter-spacing: 0.2px;
    }

    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
        padding: 3rem 2.5rem;
        border-radius: 25px;
        margin-bottom: 3rem;
        border: 2px solid #c7d2fe;
    }

    /* Result Container */
    .output-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 30px;
        box-shadow: 0 10px 50px rgba(0, 0, 0, 0.05);
    }

    /* Result Cards - More Spacious */
    .result-card {
        background: #f8fafc;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }

    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.12);
        border-color: #818cf8;
        background: white;
    }

    .result-card h4 {
        font-size: 1.05rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6366f1;
        margin-bottom: 1.5rem;
    }

    .hgb-value {
        font-size: 3.8rem;
        font-weight: 900;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -2px;
        line-height: 1;
    }

    .result-card p {
        font-size: 1rem;
        color: #475569;
        margin: 0.8rem 0;
        font-weight: 500;
    }

    .result-card small {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Severity Badge */
    .severity-badge {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        color: white;
        font-size: 1.1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 1rem auto;
        width: fit-content;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.45);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* File Uploader */
    .stFileUploader {
        background: white;
        border-radius: 20px;
        border: 2px dashed #6366f1;
    }

    /* Info Box */
    .stInfo {
        background: linear-gradient(135deg, #ecf0ff 0%, #f3e8ff 100%);
        border-left: 5px solid #6366f1;
        border-radius: 15px;
        padding: 1.5rem;
    }

    /* Warning Box */
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fcd34d 5%);
        border-left: 5px solid #f59e0b;
        border-radius: 15px;
        padding: 1.5rem;
    }

    /* Success Box */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 5%);
        border-left: 5px solid #10b981;
        border-radius: 15px;
        padding: 1.5rem;
    }

    /* Error Box */
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 5%);
        border-left: 5px solid #ef4444;
        border-radius: 15px;
        padding: 1.5rem;
    }

    /* Sidebar */
    [data-testid="sidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    [data-testid="sidebar"] h3 {
        color: #6366f1;
        font-weight: 700;
        margin-top: 1.5rem;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #6366f1, transparent);
        margin: 2.5rem 0;
    }

    /* Footer Text */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-top: 3rem;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }

    .footer-text h4 {
        color: #6366f1;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }

    .footer-text p {
        line-height: 1.8;
        margin: 0.5rem 0;
    }

    /* Subheader */
    .subheader-styled {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
        margin: 2rem 0 1.2rem 0;
        letter-spacing: -0.5px;
    }

    /* Caption */
    .caption-styled {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* Number Input */
    input[type="number"] {
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        padding: 0.8rem;
        font-size: 1rem;
    }

    input[type="number"]:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }

    /* Radio */
    [data-testid="stRadio"] {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid #e2e8f0;
    }

    /* Spinner Text */
    .stSpinner > div > div {
        border-color: #6366f1;
    }

    /* Title and Headers */
    h1, h2, h3 {
        color: #1e293b;
    }

    p {
        color: #475569;
        line-height: 1.7;
    }

</style>
""", unsafe_allow_html=True)






# ──────────────────────────────────────────────────────────────
# LOAD MODEL + SCALER  (cached — loaded only once)
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading EfficientNetB0 model …")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading hemoglobin scaler …")
def get_scaler():
    return load_scaler(SCALER_PATH)


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>👁️ Eyes Defy Anemia</h1>
    <p>Hemoglobin Prediction & Anemia Screening powered by EfficientNet-B0</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SIDEBAR — PATIENT INFO
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧑‍⚕️ Patient Information")
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    with col2:
        gender = st.radio("Gender", options=["Male", "Female"], horizontal=True)
    
    is_pregnant = False
    if gender == "Female":
        is_pregnant = st.checkbox("🤰 Is Pregnant?", value=False)
    
    st.divider()
    
    st.markdown("### 📋 Hemoglobin Reference Ranges")
    st.markdown("""
    <style>
    .ref-table {
        font-size: 0.92rem;
        line-height: 1.8;
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #6366f1;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.1);
    }
    
    .ref-item {
        margin: 0.8rem 0;
        padding: 0.8rem;
        background: linear-gradient(90deg, #ecf0ff 0%, transparent 100%);
        border-radius: 10px;
        padding-left: 1rem;
    }
    
    .ref-label {
        font-weight: 700;
        color: #6366f1;
    }
    
    .ref-range {
        color: #475569;
        font-size: 0.9rem;
    }
    </style>
    
    <div class="ref-table">
    <div class="ref-item"><span class="ref-label">👶 Children (6-59m):</span> <span class="ref-range">≥ 11.0 g/dL</span></div>
    <div class="ref-item"><span class="ref-label">👦 Children (5-11y):</span> <span class="ref-range">≥ 11.5 g/dL</span></div>
    <div class="ref-item"><span class="ref-label">👧 Children (12-14y):</span> <span class="ref-range">≥ 12.0 g/dL</span></div>
    <div class="ref-item"><span class="ref-label">👩 Women (15y+):</span> <span class="ref-range">≥ 12.0 g/dL</span></div>
    <div class="ref-item"><span class="ref-label">🤰 Pregnant Women:</span> <span class="ref-range">≥ 11.0 g/dL</span></div>
    <div class="ref-item"><span class="ref-label">👨 Men (15y+):</span> <span class="ref-range">≥ 13.0 g/dL</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### ℹ️ About This Tool")
    st.info("""
    **Eyes Defy Anemia** is an AI-powered screening tool that analyzes conjunctiva images using deep learning (EfficientNet-B0) to predict hemoglobin levels.
    
    🔬 **Powered by:** Deep Learning + Medical AI
    
    ⚠️ **Important:** This tool is for screening purposes only, not a clinical diagnosis.
    """)


# ──────────────────────────────────────────────────────────────
# MAIN — IMAGE UPLOAD & ANALYSIS
# ──────────────────────────────────────────────────────────────

st.markdown('<h2 class="subheader-styled" style="margin-top: 0; color: #000000;">Anemic Analysis</h2>', unsafe_allow_html=True)
st.markdown('<p class="caption-styled">Upload a clear photo of the conjunctiva (inner eyelid) for deep learning analysis.</p>', unsafe_allow_html=True)

uf = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uf:
    img = Image.open(uf)
    
    # Create two columns: Left for Image Preview, Right for Analysis Trigger
    col_img, col_action = st.columns([1.2, 1], gap="large")

    with col_img:
        st.markdown("""
        <div style="border-radius: 24px; overflow: hidden; box-shadow: 0 15px 45px rgba(0,0,0,0.1); border: 2px solid #e2e8f0; background: white; padding: 10px;">
        """, unsafe_allow_html=True)
        st.image(img, use_container_width=True, caption="📸 Patient Image Preview")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_action:
        st.markdown("""
        <div style="padding: 2.5rem; background: white; border-radius: 24px; border: 1px solid #e2e8f0; box-shadow: 0 10px 30px rgba(0,0,0,0.03); display: flex; flex-direction: column; justify-content: center; height: 100%;">
            <h4 style="color: #6366f1; margin-bottom: 1rem;">✨ Image Ready</h4>
            <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6; margin-bottom: 2rem;">
                Successfully loaded <b>{uf.name}</b>. Our EfficientNet-B0 model is calibrated and ready to estimate hemoglobin levels.
            </p>
        """.format(uf=uf), unsafe_allow_html=True)
        
        if st.button("🚀 Run AI Analysis", use_container_width=True):
            try:
                with st.spinner("🧬 Deep Learning Inference in progress..."):
                    model = get_model()
                    scaler = get_scaler()
                    inputs = prepare_inference_inputs({IMAGE_TYPES[0]: img}, age, gender)
                    hgb = predict_hemoglobin(model, inputs, scaler)
                    result = classify_anemia(hgb, age, gender, is_pregnant)
                    
                    st.session_state.analysis_result = {
                        "hgb": hgb,
                        "result": result,
                        "color": severity_color(result["severity"]),
                        "emoji": severity_emoji(result["severity"]),
                        "sev": result["severity"],
                        "prob": result["anemia_probability"],
                        "min_hb": result["min_hb"],
                        "max_hb": result["max_hb"]
                    }
                st.success("✅ Analysis Complete!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Analysis Failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    if "analysis_result" in st.session_state:
        del st.session_state.analysis_result
    st.info("👆 **Awaiting entry.** Please upload an eye conjunctiva image to begin the screening process.")


# ──────────────────────────────────────────────────────────────
# RESULTS SECTION
# ──────────────────────────────────────────────────────────────
if "analysis_result" in st.session_state:
    res = st.session_state.analysis_result
    hgb    = res["hgb"]
    result = res["result"]
    color  = res["color"]
    emoji  = res["emoji"]
    sev    = res["sev"]
    prob   = res["prob"]
    m_min  = res["min_hb"]
    m_max  = res["max_hb"]

    st.divider()
    
    st.markdown('<div class="output-container">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color: #1e293b; text-align: center; font-size: 2.2rem; font-weight: 800; margin-bottom: 2.5rem;">📊 Screening Results</h3>', unsafe_allow_html=True)
    
    # Hero Metrics
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(f"""
        <div class="result-card">
            <h4>Hgb Level</h4>
            <div class="hgb-value">{hgb:.1f}</div>
            <p>g/dL</p>
            <small>Normal: {m_min:.1f}–{m_max:.1f}</small>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        status_text = "ANEMIC" if result["is_anemic"] else "NON-ANEMIC"
        st.markdown(f"""
        <div class="result-card">
            <h4>Clinical Status</h4>
            <div class="severity-badge" style="background:{color}">{emoji} {sev}</div>
            <p><strong style="font-size: 1.2rem;">{status_text}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        risk_label = "ANEMIC" if result["is_anemic"] or prob > 50 else "NON-ANEMIC"
        st.markdown(f"""
        <div class="result-card">
            <h4>Anemia Risk</h4>
            <div class="hgb-value">{prob}%</div>
            <p><strong>{risk_label}</strong></p>
            <small>Probability Score</small>
        </div>
        """, unsafe_allow_html=True)


    # Plotly Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = hgb,
        number = {"suffix": " g/dL", "font": {"size": 28, "color": color}},
        gauge = {
            "axis": {"range": [0, 20], "tickwidth": 1, "tickcolor": "#cbd5e1"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, m_min], "color": "#fee2e2"},
                {"range": [m_min, m_max], "color": "#dcfce7"},
                {"range": [m_max, 20], "color": "#fef9c3"}
            ],
            "threshold": {"line": {"color": color, "width": 4}, "value": m_min}
        }
    ))
    fig.update_layout(height=450, margin=dict(t=50, b=20, l=40, r=40), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"))
    st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────
    # DETAILED CLINICAL INTERPRETATION
    # ──────────────────────────────────────────────────────────────
    st.markdown('<hr style="margin: 3rem 0; opacity: 0.1;">', unsafe_allow_html=True)
    st.markdown('<h4 style="text-align: center; margin-bottom: 2.5rem; color: #000000; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">👨‍⚕️ Clinical Guidance</h4>', unsafe_allow_html=True)
    
    if result["is_anemic"]:
        deficit = m_min - hgb
        st.markdown(f"""
        <div style="color: #000000; background: #fff3cd; padding: 2rem; border-radius: 15px; border-left: 5px solid #f59e0b;">
        <h3 style="color: #dc2626; font-weight: 900; margin-bottom: 1rem;">🔴 Anemia Risk Detected</h3>
        <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">The AI analysis suggests your hemoglobin level (<strong style="color: #dc2626; font-size: 1.2rem;">{hgb:.1f} g/dL</strong>) is below the minimum threshold (<strong style="color: #2563eb; font-size: 1.2rem;">{m_min:.1f} g/dL</strong>) for your age and gender.</p>
        
        <h4 style="color: #7c3aed; font-weight: 800; margin: 1.5rem 0 1rem 0;">Key Findings:</h4>
        <ul style="font-size: 1rem; line-height: 1.8;">
        <li><strong style="color: #dc2626;">Deficit:</strong> <span style="color: #dc2626; font-weight: 700;">{deficit:.1f} g/dL</span> below the normal minimum.</li>
        <li><strong style="color: #ea580c;">Classification:</strong> <span style="color: #ea580c; font-weight: 700;">{sev} Anemia</span>.</li>
        <li><strong style="color: #059669;">Confidence:</strong> <span style="color: #059669; font-weight: 700;">{prob}%</span> probability of low hemoglobin.</li>
        </ul>
        
        <h4 style="color: #7c3aed; font-weight: 800; margin: 1.5rem 0 1rem 0;">Recommended Next Steps:</h4>
        <ol style="font-size: 1rem; line-height: 1.8;">
        <li><strong style="color: #dc2626;">Medical Consultation:</strong> Please share this report with a doctor or health professional.</li>
        <li><strong style="color: #2563eb;">Laboratory Testing:</strong> Request a Complete Blood Count (CBC) and serum ferritin test for a clinical diagnosis.</li>
        <li><strong style="color: #059669;">Nutrition:</strong> Increase intake of iron-rich foods (lean meats, leafy greens, legumes) and Vitamin C to aid absorption.</li>
        <li><strong style="color: #ea580c;">Follow-up:</strong> Do not start supplements without medical advice.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    elif result["is_high"]:
        st.markdown(f"""
        <div style="color: #000000; background: #dbeafe; padding: 2rem; border-radius: 15px; border-left: 5px solid #3b82f6;">
        <h3 style="color: #ea580c; font-weight: 900; margin-bottom: 1rem;">⚠️ Elevated Hemoglobin Level</h3>
        <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">Your predicted hemoglobin (<strong style="color: #2563eb; font-size: 1.2rem;">{hgb:.1f} g/dL</strong>) is slightly above the typical upper range (<strong style="color: #dc2626; font-size: 1.2rem;">{m_max:.1f} g/dL</strong>).</p>
        
        <h4 style="color: #7c3aed; font-weight: 800; margin: 1.5rem 0 1rem 0;">Points to Consider:</h4>
        <ul style="font-size: 1rem; line-height: 1.8;">
        <li>This can occur due to <strong style="color: #ea580c;">dehydration</strong>, high altitude, or certain lifestyle factors (e.g., smoking).</li>
        <li>In some cases, it may indicate <strong style="color: #dc2626;">erythrocytosis</strong> which should be monitored.</li>
        </ul>
        
        <h4 style="color: #7c3aed; font-weight: 800; margin: 1.5rem 0 1rem 0;">Recommendation:</h4>
        <p style="font-size: 1rem;">Maintain good hydration and mention these results at your next routine medical check-up.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="color: #000000; background: #d1fae5; padding: 2rem; border-radius: 15px; border-left: 5px solid #10b981;">
        <h3 style="color: #059669; font-weight: 900; margin-bottom: 1rem;">🟢 Healthy Range</h3>
        <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">Excellent! Your predicted hemoglobin level (<strong style="color: #059669; font-size: 1.2rem;">{hgb:.1f} g/dL</strong>) is well within the optimal range for your profile.</p>
        
        <h4 style="color: #7c3aed; font-weight: 800; margin: 1.5rem 0 1rem 0;">Summary:</h4>
        <ul style="font-size: 1rem; line-height: 1.8;">
        <li><strong style="color: #059669;">Status:</strong> <span style="color: #059669; font-weight: 700;">{sev} (Normal)</span></li>
        <li><strong style="color: #10b981;">Risk Level:</strong> <span style="color: #10b981; font-weight: 700;">Minimal probability of anemia ({prob}%)</span>.</li>
        </ul>
        
        <h4 style="color: #7c3aed; font-weight: 800; margin: 1.5rem 0 1rem 0;">Guidance:</h4>
        <ul style="font-size: 1rem; line-height: 1.8;">
        <li>Continue a <strong style="color: #059669;">balanced diet</strong> rich in iron, folic acid, and B12.</li>
        <li>Regular screening (every 6–12 months) is a good practice for proactive health monitoring.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div class="footer-text">
    <h4>👁️ Eyes Defy Anemia</h4>
    <p><strong>AI-Powered Hemoglobin Prediction & Anemia Screening System</strong></p>
    <p style="margin-top: 1.5rem; font-size: 0.95rem;">
    Powered by <strong>EfficientNet-B0 Deep Learning Model</strong> | Advanced Medical AI Technology
    </p>
    <p style="margin-top: 1.5rem; color: #64748b; font-size: 0.9rem; line-height: 1.8;">
    <strong>⚠️ Medical Disclaimer:</strong><br>
    This tool is designed for preliminary screening purposes only and is NOT a substitute for professional medical diagnosis. 
    The results provided are estimates based on image analysis and should never replace confirmatory blood tests and clinical evaluation by qualified healthcare professionals. 
    Always consult with a doctor before making any health-related decisions.
    </p>
    <p style="margin-top: 1.5rem; color: #cbd5e1; font-size: 0.8rem;">
    © 2026 Eyes Defy Anemia | All Rights Reserved | Built with ❤️ using Streamlit, TensorFlow & Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

