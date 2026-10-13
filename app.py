"""
Pune Public Transport Complaint Classifier & Analytics Portal
Features:
  - Sleek, Modern Compact Top Section with Live Transit Badges
  - Seamless Dark/Light Header Integration (eliminates white top bar)
  - Dynamic Dark / Light Mode with 100% Text Legibility
  - Interactive Transit Control Hub sidebar
  - Live Complaint Classifier (Service Category, Sentiment, Priority)
  - Multi-Model Benchmark & Confusion Matrix Heatmap
  - Theme-Matched Pune Transit Analytics & Searchable Dataset
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(
    page_title="Pune Transit Grievance AI",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Theme
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

# ==========================================================
# THEME TOGGLE (AT TOP OF SIDEBAR)
# ==========================================================
with st.sidebar:
    st.markdown("### 🎨 Display Theme")
    theme_choice = st.radio(
        "Select Theme Mode:",
        ["☀️ Light Mode", "🌙 Dark Mode"],
        index=0 if st.session_state.theme_mode == "Light" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    is_dark = (theme_choice == "🌙 Dark Mode")
    st.session_state.theme_mode = "Dark" if is_dark else "Light"

# ==========================================================
# DYNAMIC COLOR TOKENS
# ==========================================================
if is_dark:
    bg_color = "#0b0f19"          # Deep dark canvas
    card_bg = "#151e2e"           # Elevated dark card
    sidebar_bg = "#0f172a"        # Deep slate sidebar
    input_bg = "#1e293b"          # Text input background
    hover_bg = "#334155"          # Dropdown item hover
    text_color = "#f8fafc"        # Bright white text
    subtext_color = "#94a3b8"     # Light readable slate
    border_color = "#334155"      # Subtle separator border
    top_nav_bg = "linear-gradient(135deg, #151e2e 0%, #111827 100%)"
    top_badge_color = "#60a5fa"
    top_title_color = "#ffffff"
    chip_bg = "#1e293b"
    chip_border = "#334155"
    chip_text = "#cbd5e1"
    status_bg = "#064e3b"
    status_txt = "#6ee7b7"
    status_border = "#047857"
    help_box_bg = "#151e2e"
    tab_active = "#60a5fa"
    table_header_bg = "#1e293b"
    alert_bg = "#151e2e"
    fig_face = "#151e2e"
    fig_text = "#f8fafc"
    fig_grid = "#334155"
else:
    bg_color = "#f8fafc"          # Light slate canvas
    card_bg = "#ffffff"           # Crisp white card
    sidebar_bg = "#ffffff"        # Pure white sidebar
    input_bg = "#ffffff"          # White input background
    hover_bg = "#f1f5f9"          # Light hover
    text_color = "#0f172a"        # Deep navy-slate text
    subtext_color = "#64748b"     # High-contrast gray
    border_color = "#e2e8f0"      # Light border
    top_nav_bg = "linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%)"
    top_badge_color = "#2563eb"
    top_title_color = "#1e3a8a"
    chip_bg = "#eff6ff"
    chip_border = "#bfdbfe"
    chip_text = "#1e40af"
    status_bg = "#ecfdf5"
    status_txt = "#059669"
    status_border = "#a7f3d0"
    help_box_bg = "#f1f5f9"
    tab_active = "#2563eb"
    table_header_bg = "#f8fafc"
    alert_bg = "#f8fafc"
    fig_face = "#ffffff"
    fig_text = "#0f172a"
    fig_grid = "#e2e8f0"

# Inject High-Contrast CSS
st.markdown(f"""
<style>
    /* Fix Streamlit's Default White Top Bar in Dark Mode */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        color: {text_color} !important;
    }}
    div[data-testid="stDecoration"] {{
        display: none !important;
    }}
    div[data-testid="stToolbar"] {{
        color: {text_color} !important;
    }}

    /* Global App Canvas & Text */
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    
    /* Ensure All Headings & Paragraphs are Bright and Legible */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        font-weight: 700 !important;
    }}
    p, span, div[data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}
    
    /* Widget Labels */
    label, label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] p {{
        color: {text_color} !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }}
    
    /* Captions & Subtext */
    .stCaption, div[data-testid="stCaptionContainer"] p {{
        color: {subtext_color} !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{
        color: {text_color} !important;
    }}

    /* Text Area & Text Inputs */
    textarea, input[type="text"] {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}
    textarea:focus, input[type="text"]:focus {{
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }}
    textarea::placeholder, input::placeholder {{
        color: {subtext_color} !important;
        opacity: 0.8 !important;
    }}

    /* Dropdowns (Selectbox) */
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border-color: {border_color} !important;
        color: {text_color} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {{
        color: {text_color} !important;
    }}
    div[data-baseweb="select"] svg {{
        fill: {text_color} !important;
    }}
    ul[role="listbox"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    li[role="option"] {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
    }}
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
        background-color: {hover_bg} !important;
        color: {text_color} !important;
    }}

    /* Navigation Tabs */
    button[data-baseweb="tab"] {{
        border-radius: 6px 6px 0 0 !important;
    }}
    button[data-baseweb="tab"] p {{
        color: {subtext_color} !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] p {{
        color: {tab_active} !important;
        font-weight: 700 !important;
    }}

    /* === NEW SLEEK TOP SECTION === */
    .top-header-banner {{
        background: {top_nav_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, {"0.35" if is_dark else "0.05"});
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }}
    .top-header-main {{
        flex: 1;
        min-width: 280px;
    }}
    .top-eyebrow {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: {top_badge_color};
        margin-bottom: 0.25rem;
    }}
    .top-heading {{
        font-size: 1.55rem;
        font-weight: 800;
        color: {top_title_color} !important;
        margin: 0;
        letter-spacing: -0.4px;
        line-height: 1.25;
    }}
    .top-subheading {{
        font-size: 0.88rem;
        color: {subtext_color} !important;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }}
    .top-chips-group {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }}
    .meta-chip {{
        background: {chip_bg};
        color: {chip_text} !important;
        border: 1px solid {chip_border};
        padding: 0.4rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }}
    .meta-chip-live {{
        background: {"#064e3b" if is_dark else "#ecfdf5"};
        color: {"#6ee7b7" if is_dark else "#047857"} !important;
        border-color: {"#047857" if is_dark else "#a7f3d0"};
    }}

    /* Result Badges */
    .result-badge {{
        padding: 1.1rem;
        border-radius: 12px;
        text-align: center;
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}
    .badge-category {{
        border-top: 4px solid #3b82f6 !important;
    }}
    .badge-sentiment-neg {{
        border-top: 4px solid #ef4444 !important;
    }}
    .badge-sentiment-pos {{
        border-top: 4px solid #10b981 !important;
    }}
    .badge-sentiment-neu {{
        border-top: 4px solid #94a3b8 !important;
    }}
    .badge-prio-high {{
        border-top: 4px solid #dc2626 !important;
        background-color: {"#241215" if is_dark else "#fff5f5"} !important;
    }}
    .badge-prio-med {{
        border-top: 4px solid #f59e0b !important;
        background-color: {"#241b0a" if is_dark else "#fffbeb"} !important;
    }}
    .badge-prio-low {{
        border-top: 4px solid #10b981 !important;
        background-color: {"#0d2319" if is_dark else "#f0fdf4"} !important;
    }}

    /* Sidebar Brand Box */
    .sidebar-brand-box {{
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white !important;
        margin-bottom: 1.2rem;
    }}
    .sidebar-brand-box * {{
        color: white !important;
    }}
    
    .status-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        color: {status_txt} !important;
        background: {status_bg} !important;
        padding: 5px 12px;
        border-radius: 20px;
        border: 1px solid {status_border} !important;
    }}
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
    }}

    /* Metric Values & Labels */
    div[data-testid="stMetricValue"] {{
        color: {text_color} !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {subtext_color} !important;
        font-weight: 600 !important;
    }}

    /* Tables & Dataframes */
    div[data-testid="stTable"] table {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stTable"] th {{
        background-color: {table_header_bg} !important;
        color: {text_color} !important;
        font-weight: 700 !important;
        border-bottom: 2px solid {border_color} !important;
    }}
    div[data-testid="stTable"] td {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border-bottom: 1px solid {border_color} !important;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}

    /* Radio buttons & Checkboxes */
    div[data-testid="stRadio"] label p, div[data-testid="stCheckbox"] label p {{
        color: {text_color} !important;
        font-weight: 500 !important;
    }}

    /* Alert Banners */
    div[data-testid="stAlert"] {{
        background-color: {alert_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stAlert"] * {{
        color: {text_color} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Load Models & Resources
@st.cache_resource
def load_resources():
    models_dir = "models"
    with open(os.path.join(models_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(models_dir, "evaluation_metrics.json"), "r") as f:
        metrics = json.load(f)

    models = {}
    targets = ['service_category', 'sentiment', 'priority']
    algo_slugs = {
        "Logistic Regression": "logistic_regression",
        "Nearest Centroid": "nearest_centroid",
        "Linear SVM": "linear_svm",
        "Multinomial Naive Bayes": "multinomial_naive_bayes"
    }

    for target in targets:
        models[target] = {}
        for algo_name, slug in algo_slugs.items():
            path = os.path.join(models_dir, f"{target}_{slug}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    models[target][algo_name] = pickle.load(f)

    df_complaints = pd.read_csv("pune_transport_complaints.csv")
    return vectorizer, metrics, models, df_complaints

vectorizer, metrics_data, models_dict, df_complaints = load_resources()

# ==========================================================
# SIDEBAR: TRANSIT CONTROL & FILTER HUB
# ==========================================================
with st.sidebar:
    st.markdown("---")

    # Organization Branding Header
    st.markdown("""
    <div class="sidebar-brand-box">
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; opacity:0.85;">PUNE PUBLIC TRANSIT</div>
        <div style="font-size:1.15rem; font-weight:700; margin-top:2px;">Grievance AI Portal</div>
        <div style="font-size:0.8rem; opacity:0.9; margin-top:4px;">PMPML & Pune Metro Redressal Cell</div>
    </div>
    """, unsafe_allow_html=True)

    # System Status Pill
    st.markdown("""
    <div class="status-indicator">
        <span class="status-dot"></span> 4 NLP Models Operational
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Inference Settings")

    # Active Model Selector
    active_model = st.selectbox(
        "Active NLP Classifier:",
        ["Logistic Regression", "Nearest Centroid", "Linear SVM", "Multinomial Naive Bayes"],
        index=0,
        help="Select the machine learning model used to predict categories, sentiment, and priority."
    )

    model_stats = metrics_data["service_category"]["models"][active_model]
    st.caption(f"Category F1: **{model_stats['f1_weighted']*100:.1f}%** | Accuracy: **{model_stats['accuracy']*100:.1f}%**")

    st.markdown("---")
    st.markdown("### 📍 Depot & Corridor Filter")
    
    all_locations = ["All Pune Depots / Hubs"] + sorted(df_complaints["location"].unique().tolist())
    selected_depot = st.selectbox("Filter Analytics by Depot:", all_locations, index=0)

    all_modes = ["All Transit Modes"] + sorted(df_complaints["transit_mode"].unique().tolist())
    selected_mode = st.selectbox("Transit Fleet:", all_modes, index=0)

    st.markdown("---")
    st.markdown("### 📞 Transit Helplines")
    st.markdown(f"""
    <div style="font-size:0.82rem; color:{subtext_color}; background:{help_box_bg}; border:1px solid {border_color}; padding:0.75rem; border-radius:8px;">
        <b style="color:{text_color};">PMPML Toll-Free:</b> 1800-233-0808<br>
        <b style="color:{text_color};">Pune Metro Care:</b> 1800-270-5551<br>
        <b style="color:{text_color};">Swargate Central:</b> 020-24440417
    </div>
    """, unsafe_allow_html=True)

# Apply global sidebar filters to dataframe
filtered_df = df_complaints.copy()
if selected_depot != "All Pune Depots / Hubs":
    filtered_df = filtered_df[filtered_df["location"] == selected_depot]
if selected_mode != "All Transit Modes":
    filtered_df = filtered_df[filtered_df["transit_mode"] == selected_mode]

# ==========================================================
# REDESIGNED SLEEK TOP SECTION
# ==========================================================
st.markdown(f"""
<div class="top-header-banner">
    <div class="top-header-main">
        <div class="top-eyebrow">PUNE METROPOLITAN MOBILITY • AI GRIEVANCE REDRESSAL</div>
        <h1 class="top-heading">🚌 Pune Public Transport — Grievance Intelligence</h1>
        <p class="top-subheading">Automatic Service Categorization, Sentiment Tracking & Priority Triaging for PMPML & Pune Metro</p>
    </div>
    <div class="top-chips-group">
        <div class="meta-chip meta-chip-live">
            <span style="width:7px; height:7px; border-radius:50%; background:#10b981; display:inline-block;"></span>
            Live Redressal Cell
        </div>
        <div class="meta-chip">
            ⚡ 4 NLP Models
        </div>
        <div class="meta-chip">
            📍 PMPML & Metro
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "✍️ Classify Complaint",
    "📊 Model Benchmark & Confusion Matrix",
    "📈 Pune Transit Analytics & Dataset"
])

# ==========================================================
# TAB 1: LIVE COMPLAINT CLASSIFIER
# ==========================================================
with tab1:
    st.markdown("### 📝 Passenger Grievance Classifier")
    st.write("Enter an everyday Pune commuter complaint or pick a common scenario to test real-time classification.")

    sample_complaints = [
        "-- Select a pre-loaded commuter sample or type your own below --",
        "Bus driver on route 148 Swargate to Nigdi was driving recklessly and broke two red signals near Shivaji Nagar.",
        "AC in Hinjawadi electric bus was blowing hot air in 38 degree heat, passengers suffocating.",
        "Conductor refused to return change of 15 rupees at Swargate and gave rude abuses to passengers.",
        "Severe overcrowding at Hadapsar Gadital, commuters hanging dangerously on footboard.",
        "Digital route display board at Shivaji Nagar bus shelter was displaying wrong destination.",
        "Pune Metro train on Aqua Line arrived exactly on time and coaches were spotlessly clean."
    ]

    chosen_sample = st.selectbox("💡 Quick Fill Sample:", sample_complaints, index=0)

    user_text = ""
    if chosen_sample != sample_complaints[0]:
        user_text = chosen_sample

    col_input, col_action = st.columns([4, 1])

    with col_input:
        complaint_text = st.text_area(
            "Complaint Description:",
            value=user_text,
            height=110,
            placeholder="Type passenger complaint here (e.g., Driver was speaking on mobile phone while speeding in Katraj Ghat...)"
        )

    with col_action:
        st.write("")
        st.write("")
        classify_btn = st.button("🚀 Analyze Now", type="primary", use_container_width=True)
        st.caption(f"Active Model: **{active_model}**")

    if classify_btn or (complaint_text.strip() and complaint_text != ""):
        if not complaint_text.strip():
            st.warning("Please enter a complaint description above.")
        else:
            vec_input = vectorizer.transform([complaint_text])

            cat_clf = models_dict['service_category'][active_model]
            sent_clf = models_dict['sentiment'][active_model]
            prio_clf = models_dict['priority'][active_model]

            pred_cat = cat_clf.predict(vec_input)[0]
            cat_conf = float(np.max(cat_clf.predict_proba(vec_input)[0])) * 100

            pred_sent = sent_clf.predict(vec_input)[0]
            sent_conf = float(np.max(sent_clf.predict_proba(vec_input)[0])) * 100

            pred_prio = prio_clf.predict(vec_input)[0]
            prio_conf = float(np.max(prio_clf.predict_proba(vec_input)[0])) * 100

            st.markdown("#### 🎯 AI Triaging Results")
            c1, c2, c3 = st.columns(3)

            # Service Category Card
            with c1:
                st.markdown(f"""
                <div class="result-badge badge-category">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:{subtext_color}; font-weight:700;">SERVICE CATEGORY</div>
                    <div style="font-size:1.15rem; font-weight:700; color:{text_color}; margin:0.4rem 0;">{pred_cat}</div>
                    <div style="font-size:0.85rem; color:#3b82f6; font-weight:600;">{cat_conf:.1f}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(1.0, cat_conf / 100))

            # Sentiment Card
            with c2:
                sent_style = "badge-sentiment-neg" if pred_sent == "Negative" else ("badge-sentiment-pos" if pred_sent == "Positive" else "badge-sentiment-neu")
                sent_label = "⚠️ Negative" if pred_sent == "Negative" else ("✅ Positive" if pred_sent == "Positive" else "ℹ️ Neutral")
                st.markdown(f"""
                <div class="result-badge {sent_style}">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:{subtext_color}; font-weight:700;">SENTIMENT</div>
                    <div style="font-size:1.15rem; font-weight:700; color:{text_color}; margin:0.4rem 0;">{sent_label}</div>
                    <div style="font-size:0.85rem; color:{subtext_color}; font-weight:600;">{sent_conf:.1f}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(1.0, sent_conf / 100))

            # Priority Card
            with c3:
                prio_style = "badge-prio-high" if pred_prio == "High" else ("badge-prio-med" if pred_prio == "Medium" else "badge-prio-low")
                prio_label = "🚨 HIGH (Immediate Dispatch)" if pred_prio == "High" else ("⚡ MEDIUM (4h Resolution)" if pred_prio == "Medium" else "🟢 LOW (Standard Log)")
                st.markdown(f"""
                <div class="result-badge {prio_style}">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:{subtext_color}; font-weight:700;">PRIORITY LEVEL</div>
                    <div style="font-size:1.15rem; font-weight:700; color:{text_color}; margin:0.4rem 0;">{prio_label}</div>
                    <div style="font-size:0.85rem; color:{subtext_color}; font-weight:600;">{prio_conf:.1f}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(1.0, prio_conf / 100))

            st.markdown("<br>", unsafe_allow_html=True)
            if pred_prio == "High":
                st.error("🚨 **Urgent Action Protocol:** High-priority passenger safety concern. Automatic alert transmitted to Swargate Central Control Room and Traffic Officer.")
            elif pred_prio == "Medium":
                st.warning("⚡ **Standard Escalation:** Escalated to Depot Supervisor & Maintenance Engineer for resolution within 4 hours.")
            else:
                st.success("🟢 **Operational Record:** Logged into PMPML Customer Care database for weekly quality improvement review.")

# ==========================================================
# TAB 2: MODEL BENCHMARK & EVALUATION
# ==========================================================
with tab2:
    st.markdown("### 📊 Model Benchmark & Comparative Evaluation")
    st.write("Comprehensive evaluation of all **4 NLP Classifiers** across **Accuracy, Precision, Recall, F1-Score**, and **Confusion Matrix**.")

    target_option = st.selectbox(
        "Select Target Variable to Compare:",
        ["service_category", "sentiment", "priority"],
        format_func=lambda x: {
            "service_category": "Service Category (6 Classes)",
            "sentiment": "Sentiment (Negative, Neutral, Positive)",
            "priority": "Priority Level (High, Medium, Low)"
        }[x]
    )

    t_metrics = metrics_data[target_option]
    classes = t_metrics["classes"]
    models_metrics = t_metrics["models"]
    top_model = t_metrics.get("best_model", "Logistic Regression")

    st.info(f"🏆 **Best Performing Model for {target_option.replace('_', ' ').title()}:** **{top_model}**")

    # Metrics Table
    table_data = []
    for m_name, vals in models_metrics.items():
        table_data.append({
            "Model Name": m_name,
            "Accuracy": f"{vals['accuracy']*100:.2f}%",
            "Precision (Weighted)": f"{vals['precision_weighted']*100:.2f}%",
            "Recall (Weighted)": f"{vals['recall_weighted']*100:.2f}%",
            "F1-Score (Weighted)": f"{vals['f1_weighted']*100:.2f}%",
            "Macro F1": f"{vals['f1_macro']*100:.2f}%"
        })
    df_metrics = pd.DataFrame(table_data)
    st.table(df_metrics)

    # Dynamic Theme-Colored Bar Chart
    st.markdown("#### 📈 Comparative Metrics Chart")
    fig_bar, ax_bar = plt.subplots(figsize=(8.5, 3.8), facecolor=fig_face)
    ax_bar.set_facecolor(fig_face)

    m_names = list(models_metrics.keys())
    x = np.arange(len(m_names))
    width = 0.2

    metrics_tuples = [
        ("Accuracy", "accuracy", "#3b82f6"),
        ("Precision", "precision_weighted", "#10b981"),
        ("Recall", "recall_weighted", "#f59e0b"),
        ("F1-Score", "f1_weighted", "#ef4444")
    ]

    for i, (label, key, color) in enumerate(metrics_tuples):
        vals = [models_metrics[m][key] * 100 for m in m_names]
        ax_bar.bar(x + i * width, vals, width, label=label, color=color, alpha=0.9, edgecolor='none')

    ax_bar.set_ylabel("Score (%)", fontsize=10, color=fig_text, fontweight='bold')
    ax_bar.set_title(f"Performance Comparison for {target_option.replace('_', ' ').title()}", fontsize=11, fontweight='bold', color=fig_text)
    ax_bar.set_xticks(x + width * 1.5)
    ax_bar.set_xticklabels(m_names, fontsize=9.5, color=fig_text)
    ax_bar.set_ylim(0, 105)
    ax_bar.tick_params(colors=fig_text)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.25, color=fig_grid)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color(fig_grid)
    ax_bar.spines['bottom'].set_color(fig_grid)
    ax_bar.legend(loc='lower right', frameon=True, facecolor=fig_face, edgecolor=border_color, fontsize=9, labelcolor=fig_text)
    plt.tight_layout()
    st.pyplot(fig_bar)

    st.markdown("---")

    # Confusion Matrix Section
    st.markdown("#### 🔲 Confusion Matrix Heatmap")
    c_cm_m, c_cm_n = st.columns([2, 1])

    with c_cm_m:
        cm_chosen_model = st.selectbox("Select Model for Confusion Matrix:", m_names, index=0)
    with c_cm_n:
        cm_norm = st.checkbox("Normalize Matrix (%)", value=False)

    cm_raw = np.array(models_metrics[cm_chosen_model]["confusion_matrix"], dtype=float)
    if cm_norm:
        row_totals = cm_raw.sum(axis=1, keepdims=True)
        cm_raw = np.divide(cm_raw, row_totals, where=row_totals != 0) * 100

    clean_labels = [c if len(c) <= 16 else c[:14] + ".." for c in classes]

    fig_cm, ax_cm = plt.subplots(figsize=(6.5, 5.2), facecolor=fig_face)
    ax_cm.set_facecolor(fig_face)
    cax = ax_cm.imshow(cm_raw, cmap=plt.cm.Blues, interpolation='nearest')
    cbar = fig_cm.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(fig_grid)
    cbar.ax.yaxis.set_tick_params(color=fig_text)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=fig_text)

    ax_cm.set_xticks(np.arange(len(classes)))
    ax_cm.set_yticks(np.arange(len(classes)))
    ax_cm.set_xticklabels(clean_labels, rotation=35, ha='right', fontsize=8.5, color=fig_text)
    ax_cm.set_yticklabels(clean_labels, fontsize=8.5, color=fig_text)
    ax_cm.tick_params(colors=fig_text)
    ax_cm.set_xlabel("Predicted Label", fontweight='bold', fontsize=9.5, color=fig_text)
    ax_cm.set_ylabel("True Ground Truth Label", fontweight='bold', fontsize=9.5, color=fig_text)
    ax_cm.set_title(f"Confusion Matrix: {cm_chosen_model}", fontsize=11, fontweight='bold', color=fig_text)

    threshold = cm_raw.max() / 2.0
    for r in range(len(classes)):
        for c in range(len(classes)):
            val = cm_raw[r, c]
            txt = f"{val:.1f}%" if cm_norm else f"{int(val)}"
            ax_cm.text(c, r, txt, ha="center", va="center",
                       color="white" if val > threshold else ("#f8fafc" if is_dark else "#0f172a"),
                       fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig_cm)

    # Detailed Per-Class Breakdown
    st.markdown("#### 📋 Detailed Per-Class Metrics")
    rep = models_metrics[cm_chosen_model]["classification_report"]
    rep_list = []
    for cls_title, stats in rep.items():
        rep_list.append({
            "Class Name": cls_title,
            "Precision": f"{stats['precision']*100:.1f}%",
            "Recall": f"{stats['recall']*100:.1f}%",
            "F1-Score": f"{stats['f1_score']*100:.1f}%",
            "Support (Test Count)": stats['support']
        })
    st.table(pd.DataFrame(rep_list))

# ==========================================================
# TAB 3: DATASET & PUNE TRANSIT ANALYTICS
# ==========================================================
with tab3:
    st.markdown("### 📈 Pune Public Transport Grievance Analytics")
    if selected_depot != "All Pune Depots / Hubs" or selected_mode != "All Transit Modes":
        st.info(f"Showing filtered results for: **{selected_depot}** | **{selected_mode}** ({len(filtered_df)} records)")

    # Key Metrics
    k1, k2, k3, k4 = st.columns(4)
    tot = len(filtered_df)
    hi_prio = len(filtered_df[filtered_df["priority"] == "High"])
    neg_prio = len(filtered_df[filtered_df["sentiment"] == "Negative"])
    top_loc = filtered_df["location"].value_counts().index[0] if tot > 0 else "N/A"

    k1.metric("Grievances Analyzed", f"{tot}")
    k2.metric("Urgent High Priority", f"{hi_prio}", f"{(hi_prio/tot)*100:.1f}%" if tot else "0%")
    k3.metric("Negative Sentiment", f"{neg_prio}", f"{(neg_prio/tot)*100:.1f}%" if tot else "0%")
    k4.metric("Leading Hub / Depot", f"{top_loc}")

    st.markdown("---")

    # Matplotlib Theme-Adaptive Analytics Charts (No White Backing in Dark Mode)
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("**Complaints by Service Category:**")
        cat_counts = filtered_df["service_category"].value_counts()
        fig_c1, ax_c1 = plt.subplots(figsize=(6, 3.8), facecolor=fig_face)
        ax_c1.set_facecolor(fig_face)
        bars1 = ax_c1.bar(range(len(cat_counts)), cat_counts.values, color='#3b82f6', width=0.6)
        
        # Clean labels
        cat_labels = [c if len(c) <= 14 else c[:12] + ".." for c in cat_counts.index]
        ax_c1.set_xticks(range(len(cat_counts)))
        ax_c1.set_xticklabels(cat_labels, rotation=30, ha='right', fontsize=8.5, color=fig_text)
        ax_c1.tick_params(colors=fig_text)
        ax_c1.grid(axis='y', linestyle='--', alpha=0.25, color=fig_grid)
        ax_c1.spines['top'].set_visible(False)
        ax_c1.spines['right'].set_visible(False)
        ax_c1.spines['left'].set_color(fig_grid)
        ax_c1.spines['bottom'].set_color(fig_grid)
        plt.tight_layout()
        st.pyplot(fig_c1)

    with c_chart2:
        st.markdown("**Complaints by Priority Level:**")
        prio_counts = filtered_df["priority"].value_counts()
        fig_c2, ax_c2 = plt.subplots(figsize=(6, 3.8), facecolor=fig_face)
        ax_c2.set_facecolor(fig_face)
        prio_colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
        bar_cols = [prio_colors.get(p, '#3b82f6') for p in prio_counts.index]
        bars2 = ax_c2.bar(prio_counts.index, prio_counts.values, color=bar_cols, width=0.5)
        ax_c2.tick_params(colors=fig_text)
        ax_c2.grid(axis='y', linestyle='--', alpha=0.25, color=fig_grid)
        ax_c2.spines['top'].set_visible(False)
        ax_c2.spines['right'].set_visible(False)
        ax_c2.spines['left'].set_color(fig_grid)
        ax_c2.spines['bottom'].set_color(fig_grid)
        plt.tight_layout()
        st.pyplot(fig_c2)

    st.markdown("---")
    st.markdown("#### 🔍 Filter & Search Complaints")
    query = st.text_input("Filter complaints by keyword (e.g. Swargate, conductor, electric bus, ticket):")
    if query.strip():
        searched_df = filtered_df[filtered_df["complaint_text"].str.contains(query, case=False, na=False)]
    else:
        searched_df = filtered_df

    st.dataframe(searched_df, use_container_width=True)
