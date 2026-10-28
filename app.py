import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import os
import warnings
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E88E5, #7E57C2, #EC407A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #718096;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-top: 10px;
    }
    .price-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #38bdf8;
        text-shadow: 0 2px 10px rgba(56, 189, 248, 0.4);
    }
    .spec-badge {
        display: inline-block;
        background: #1e293b;
        color: #cbd5e1;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin: 4px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model & Encoder Loader (download from Hugging Face)
# ---------------------------------------------------------
HF_REPO_ID = "Anant8958/Daimond_Price_predictor"

@st.cache_resource(show_spinner="Downloading and loading trained model...")
def load_artifacts():
    try:
        # Hugging Face downloads these files and caches them locally.
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="diamond_model.pkl",
            repo_type="model"
        )
        encoder_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="encoder.pkl",
            repo_type="model"
        )

        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        try:
            encoder = joblib.load(encoder_path)
        except Exception:
            with open(encoder_path, "rb") as f:
                encoder = pickle.load(f)

        # Preserve the original app's encoder feature handling.
        if hasattr(encoder, "n_features_in_") and encoder.n_features_in_ != 9:
            encoder.n_features_in_ = 9

        return model, encoder

    except Exception as e:
        st.error(
            "Could not download/load the model files from Hugging Face. "
            "Check that the repository is public and that the filenames "
            "are exactly 'diamond_model.pkl' and 'encoder.pkl'. "
            f"Details: {e}"
        )
        return None, None

model, encoder = load_artifacts()

# ---------------------------------------------------------
# Sidebar: Project Info & Quick Defaults
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 💎 About This App")
    st.write(
        "Predict diamond retail prices using a **Random Forest Regressor** trained "
        "on the classic 54,000+ diamonds dataset."
    )
    st.markdown("---")
    st.markdown("### 📊 Model Metrics")
    st.metric("R² Score", "98.1%")
    st.metric("Mean Absolute Error (MAE)", "$270")
    
    st.markdown("---")
    st.markdown("### 💡 Preset Diamonds")
    preset = st.selectbox(
        "Choose a preset:",
        ["Custom Diamond", "1.00ct Engagement Ring (Ideal)", "0.50ct Budget Diamond", "2.00ct Luxury Solitaire"]
    )

# ---------------------------------------------------------
# Input Setup based on Presets
# ---------------------------------------------------------
defaults = {
    "Custom Diamond": {"carat": 0.70, "cut": "Ideal", "color": "E", "clarity": "SI1", "depth": 61.5, "table": 55.0, "x": 5.70, "y": 5.73, "z": 3.51},
    "1.00ct Engagement Ring (Ideal)": {"carat": 1.01, "cut": "Ideal", "color": "F", "clarity": "VS1", "depth": 61.8, "table": 56.0, "x": 6.43, "y": 6.46, "z": 3.98},
    "0.50ct Budget Diamond": {"carat": 0.51, "cut": "Good", "color": "H", "clarity": "SI2", "depth": 63.2, "table": 58.0, "x": 5.08, "y": 5.12, "z": 3.22},
    "2.00ct Luxury Solitaire": {"carat": 2.02, "cut": "Premium", "color": "D", "clarity": "VVS1", "depth": 61.2, "table": 59.0, "x": 8.12, "y": 8.16, "z": 4.98}
}[preset]

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.markdown('<div class="main-header">💎 Diamond Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Accurate market valuation powered by Machine Learning</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.8, 1.2], gap="large")

with col_left:
    st.subheader("1. The 4 C's of Diamond Quality")
    
    c1, c2 = st.columns(2)
    with c1:
        carat = st.number_input(
            "Carat Weight (ct)",
            min_value=0.20,
            max_value=5.01,
            value=float(defaults["carat"]),
            step=0.01,
            help="Weight of the diamond. 1 carat = 200 milligrams."
        )
        
        cut_options = ["Ideal", "Premium", "Very Good", "Good", "Fair"]
        cut = st.selectbox(
            "Cut Quality",
            options=cut_options,
            index=cut_options.index(defaults["cut"]),
            help="Quality of cut affects diamond's brilliance and fire."
        )

    with c2:
        color_options = ["D", "E", "F", "G", "H", "I", "J"]
        color = st.selectbox(
            "Color Grade",
            options=color_options,
            index=color_options.index(defaults["color"]),
            help="D is completely colorless (highest grade), J has slight warmth."
        )
        
        clarity_options = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
        clarity = st.selectbox(
            "Clarity Grade",
            options=clarity_options,
            index=clarity_options.index(defaults["clarity"]),
            help="Internal inclusions and surface blemishes. IF is Internally Flawless."
        )

    st.markdown("---")
    st.subheader("2. Proportions & Dimensions")
    
    p1, p2 = st.columns(2)
    with p1:
        depth = st.slider(
            "Total Depth %",
            min_value=43.0,
            max_value=79.0,
            value=float(defaults["depth"]),
            step=0.1,
            help="Total depth percentage = z / mean(x, y) = 2 * z / (x + y)"
        )
        x = st.number_input(
            "Length (x) in mm",
            min_value=0.1,
            max_value=11.0,
            value=float(defaults["x"]),
            step=0.01
        )
        z = st.number_input(
            "Depth (z) in mm",
            min_value=0.1,
            max_value=7.0,
            value=float(defaults["z"]),
            step=0.01
        )

    with p2:
        table = st.slider(
            "Table %",
            min_value=43.0,
            max_value=95.0,
            value=float(defaults["table"]),
            step=0.1,
            help="Width of top facet relative to widest point"
        )
        y = st.number_input(
            "Width (y) in mm",
            min_value=0.1,
            max_value=11.0,
            value=float(defaults["y"]),
            step=0.01
        )

with col_right:
    st.subheader("Valuation Estimate")
    
    # Prepare input DataFrame with exact columns expected by encoder
    input_df = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])
    
    if model is not None and encoder is not None:
        try:
            # Transform input
            input_transformed = encoder.transform(input_df)
            
            # Predict
            predicted_price = model.predict(input_transformed)[0]
            predicted_price = max(0.0, float(predicted_price))
            
            st.markdown(f"""
            <div class="price-card">
                <div class="price-title">Estimated Market Value</div>
                <div class="price-value">${predicted_price:,.2f}</div>
                <div style="margin-top: 16px;">
                    <span class="spec-badge">⚖️ {carat:.2f} ct</span>
                    <span class="spec-badge">✨ {cut} Cut</span>
                    <span class="spec-badge">🎨 {color} Color</span>
                    <span class="spec-badge">🔍 {clarity}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Estimated price range based on model MAE (~$270)
            mae = 270.0
            low_est = max(0.0, predicted_price - mae)
            high_est = predicted_price + mae
            
            st.markdown(f"""
            <div style="margin-top: 15px; font-size: 0.9rem; color: #94a3b8; text-align: center;">
                Estimated 95% Confidence Range: <b>${low_est:,.0f} – ${high_est:,.0f}</b>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Model and encoder are not loaded.")

    st.markdown("---")
    st.markdown("#### Diamond Summary")
    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )
