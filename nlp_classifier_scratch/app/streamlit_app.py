"""
Streamlit web app for text classification inference.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import joblib
import yaml
import os
import sys
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessor import TextPreprocessor

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")
with open(CONFIG_PATH) as f:
   config_path= yaml.safe_load(f)

AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

CLASS_COLORS = {
    "World":     "#4C72B0",
    "Sports":    "#DD8452",
    "Business":  "#55A868",
    "Sci/Tech":  "#C44E52",
}

SAMPLE_TEXTS = {
    "Sports":    "The NBA finals were decided in game 7 with a last-second three-pointer.",
    "Business":  "Tesla reported record quarterly earnings driven by strong EV demand in Asia.",
    "World":     "World leaders convened in Geneva to negotiate a landmark climate agreement.",
    "Sci/Tech":  "Researchers unveil a new transformer architecture that halves training costs.",
}

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

@st.cache_resource
def load_model():
    best_model=Path(config_path['paths']['models_dir']) / "best_model.joblib"
    if not best_model.exists() :
        return None
    return joblib.load(best_model)


@st.cache_resource
def load_preprocessor():
    return TextPreprocessor(config_path=CONFIG_PATH)

def classify(text: str, model, preprocessor, class_names):
    processed = preprocessor.preprocess(text)
    proba = model.predict_proba([processed])[0]
    
    # Get prediction from highest probability
    pred_idx = proba.argmax()
    label = class_names[pred_idx]
    
    # Create probability dictionary
    proba_dict = dict(zip(class_names, proba))
    
    return label, float(proba.max()), proba_dict
# def classify(text: str, model, preprocessor, class_names):
#     # processed = preprocessor.preprocess(text)
#     # pred = model.predict([processed])[0]
#     # proba = model.predict_proba([processed])[0]
#     # label = class_names[pred] if class_names else str(pred)
#     # return label, float(proba.max()), dict(zip(
#     #     class_names or [str(i) for i in range(len(proba))], proba
#     # )) 
#     processed = preprocessor.preprocess(text)
    
#     # Get the classifier from the pipeline
#     clf = model.named_steps['clf']
    
#     pred = model.predict([processed])[0]
#     proba = model.predict_proba([processed])[0]
    
#     # Map using model's internal class order
#     # model.classes_ = [0, 1, 2, 3] (the order the model uses)
#     label = class_names[pred] if class_names else str(pred)
    
#     # Create proba_dict correctly aligned with model's classes
#     proba_dict = {}
#     for i, class_idx in enumerate(clf.classes_):
#         proba_dict[class_names[class_idx]] = float(proba[i])
    
#     return label, float(proba.max()), proba_dict


def prob_bar_chart(proba_dict, predicted_label):
    labels = list(proba_dict.keys())
    values = [proba_dict[l] for l in labels]
    colors = [CLASS_COLORS.get(l, "#888888") for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(range=[0, 1.05], showticklabels=False, showgrid=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# -----------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------

st.set_page_config(page_title="Text Classifier", page_icon="📰", layout="wide")

st.title("📰 Text Classification Demo")
st.caption("Powered by TF-IDF + Logistic Regression ")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_preprocessed = st.checkbox("Show preprocessed text", value=False)
    st.divider()
    st.markdown("**Sample texts**")
    selected_sample = st.selectbox("Load a sample:", ["— choose —"] + list(SAMPLE_TEXTS.keys()))

# Load resources
model = load_model()
preprocessor = load_preprocessor()

if model is None:
    st.warning(
        "No trained model found. Run the pipeline first:\n\n"
        "```python\nfrom src.pipeline import TextClassificationPipeline\n"
        "p = TextClassificationPipeline()\np.run()\n```"
    )
    st.stop()


dataset_name = config_path["dataset"]["name"]
class_names = list(AG_NEWS_LABELS.values()) if dataset_name == "ag_news" else None

# Main input
default_text = SAMPLE_TEXTS.get(selected_sample, "") if selected_sample != "— choose —" else ""
user_text = st.text_area(
    "Enter a news headline or article snippet:",
    value=default_text,
    height=150,
    placeholder="Type or paste text here…"
)

col1, col2 = st.columns([1, 5])
classify_btn = col1.button("Classify", type="primary", use_container_width=True)

if classify_btn:
    if not user_text.strip():
        st.error("Please enter some text first.")
    else:
        with st.spinner("Classifying…"):
            label, confidence, proba_dict = classify(user_text, model, preprocessor, class_names)

        st.divider()
        res_col, chart_col = st.columns([1, 2])

        with res_col:
            color = CLASS_COLORS.get(label, "#333333")
            st.markdown(
                f"<div style='background:{color}22;border-left:5px solid {color};"
                f"padding:16px;border-radius:8px'>"
                f"<div style='font-size:1.1rem;color:{color};font-weight:600'>Predicted category</div>"
                f"<div style='font-size:2rem;font-weight:700;color:{color}'>{label}</div>"
                f"<div style='font-size:1rem;color:#555'>Confidence: {confidence:.1%}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        with chart_col:
            st.markdown("**Class probabilities**")
            st.plotly_chart(prob_bar_chart(proba_dict, label), use_container_width=True)

        if show_preprocessed:
            st.divider()
            preprocessed = preprocessor.preprocess(user_text)
            st.markdown("**Preprocessed text**")
            st.code(preprocessed, language=None)

# -----------------------------------------------------------------------
# Batch classification
# -----------------------------------------------------------------------
st.divider()
with st.expander("📊 Batch classification (upload CSV)"):
    uploaded = st.file_uploader("Upload a CSV with a 'Description' column", type="csv")
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        if "text" not in df_upload.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            with st.spinner(f"Classifying {len(df_upload)} rows…"):
                processed_texts = df_upload["text"].apply(preprocessor.preprocess)
                preds = model.predict(processed_texts)
                probas = model.predict_proba(processed_texts)
                df_upload["predicted_label"] = [
                    (class_names[p] if class_names else str(p)) for p in preds
                ]
                df_upload["confidence"] = probas.max(axis=1).round(4)

            st.success(f"Classified {len(df_upload)} texts!")
            st.dataframe(df_upload[["text", "predicted_label", "confidence"]], use_container_width=True)

            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download results", csv_out, "classified.csv", "text/csv")
