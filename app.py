import streamlit as st
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import pandas as pd
import plotly.express as px

# --- PAGE CONFIGURATION ---
# The theme is now controlled by the config.toml file
st.set_page_config(
    page_title="SafeGuard AI - Bullying Detector",
    page_icon="🛡️",
    layout="centered",
)

# --- CUSTOM CSS FOR A BETTER UI ---
st.markdown("""
<style>
    /* Making text area and button more visible in both themes */
    .stTextArea textarea {
        border: 2px solid #4A90E2;
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    /* Styling for the result cards */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-top: 20px;
        text-align: center;
    }
    .bullying {
        background-color: #D9534F; /* Red for Bullying */
    }
    .non-bullying {
        background-color: #5CB85C; /* Green for Non-Bullying */
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL AND TOKENIZER LOADING ---
@st.cache_resource
def load_model():
    model_path = "./BullyingDeploymentPackage/final_model_v2" # Using the improved v2 model
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure the 'BullyingDeploymentPackage/final_model_v2' directory is in the same folder as app.py.")
        return None, None

model, tokenizer = load_model()


# --- PREDICTION FUNCTION ---
def predict(text):
    if model is None or tokenizer is None:
        return None, None
    model.eval()
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities).item()
        confidence_scores = probabilities.flatten().tolist()
    return predicted_class_id, confidence_scores


# --- MAIN APP INTERFACE ---
st.title("🛡️ SafeGuard AI")
st.subheader("A Real-Time Bullying and Toxicity Detector")
st.markdown("Enter any text below to check if it contains harmful content. This tool is built on a fine-tuned BERT model to help promote safer online interactions.")

user_text = st.text_area("Enter your text here:", height=150, placeholder="e.g., 'You are so smart!' or 'I hate this, it's terrible.'")

if st.button("Analyze Text"):
    if user_text:
        with st.spinner("Analyzing..."):
            prediction_id, scores = predict(user_text)
            if prediction_id is not None:
                st.session_state['prediction_id'] = prediction_id
                st.session_state['scores'] = scores
    else:
        st.warning("Please enter some text to analyze.")

if 'prediction_id' in st.session_state:
    prediction_id = st.session_state['prediction_id']
    scores = st.session_state['scores']

    if prediction_id == 1:
        st.markdown('<div class="result-card bullying"><p class="result-text">🚨 Bullying Content Detected</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card non-bullying"><p class="result-text">✅ Looks Safe</p></div>', unsafe_allow_html=True)

    st.write("### Prediction Confidence")
    data = {'Category': ['Non-Bullying', 'Bullying'], 'Confidence': scores}
    df_scores = pd.DataFrame(data)
    fig = px.bar(df_scores, x='Category', y='Confidence',
                 color='Category',
                 color_discrete_map={'Non-Bullying': '#5CB85C', 'Bullying': '#D9534F'},
                 text_auto='.2%',
                 template='plotly_white')
    fig.update_layout(showlegend=False, yaxis_title="Confidence Score", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🔬 Learn More About The Model's Performance"):
    st.markdown("""
    This application is powered by a `bert-base-uncased` model that was fine-tuned on a diverse dataset of over 55,000 text samples labeled for toxicity and aggression.
    The model was trained to distinguish between two categories: 'Bullying' and 'Non-Bullying'. Here's a look at its performance on the unseen test data:
    """)
    try:
        st.image("./BullyingDeploymentPackage/results/confusion_matrix.png", caption="Confusion Matrix on Test Data")
    except Exception as e:
        st.warning("Could not load confusion matrix image.")
    try:
        with open("./BullyingDeploymentPackage/results/classification_report.txt") as f:
            report = f.read()
        st.text("Classification Report:")
        st.code(report, language='text')
    except Exception as e:
        st.warning("Could not load classification report.")

st.markdown("---")
st.markdown("Developed with ❤️ by a student passionate about safe AI.")
