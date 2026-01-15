import streamlit as st
import requests
from PIL import Image
import os
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CIFAKE Detector", layout="wide")

st.title("CIFAKE Detection System")

st.markdown("""
    <style>
        .block-container {padding-top: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.2rem;}
    </style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🧪 Multi-Model Analysis", "📊 Training Results"])

with tab1:
    st.markdown("### Detect Fake Images using Ensemble")

    col_input, col_image, col_results = st.columns([1, 1, 1.2], gap="medium")

    with col_input:
        st.info("Step 1: Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "Type": uploaded_file.type
            }
            st.write("📄 **File Details:**")
            st.json(file_details)

            analyze_button = st.button("🚀 Analyze with All Models", type="primary", width="stretch")
        else:
            analyze_button = False

    with col_image:
        st.info("Step 2: Preview")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Source Image", width="stretch")
        else:
            st.markdown("*Waiting for upload...*")

    with col_results:
        st.info("Step 3: Analysis Results")

        if uploaded_file and analyze_button:
            with st.spinner('Running inference on all models...'):
                try:
                    uploaded_file.seek(0)
                    files = {"file": uploaded_file}

                    response = requests.post(f"{API_URL}/predict_all", files=files)

                    if response.status_code == 200:
                        data = response.json()
                        results = data["results"]

                        for res in results:
                            model_name = res['model_name']
                            prediction = res['prediction']
                            conf = float(res['confidence'])

                            with st.container():
                                st.markdown(f"**{model_name}**")
                                r_col1, r_col2 = st.columns([1, 1])

                                with r_col1:
                                    if prediction == "REAL":
                                        st.success("REAL")
                                    else:
                                        st.error("FAKE")

                                with r_col2:
                                    st.metric("Confidence", f"{conf * 100:.1f}%")
                                    st.progress(conf)

                                st.divider()
                    else:
                        st.error(f"Server Error: {response.text}")

                except Exception as e:
                    st.error(f"Connection error: {e}")
        elif not uploaded_file:
            st.markdown("*Upload an image to see results.*")
        elif not analyze_button:
            st.markdown("*Click 'Analyze' to start.*")

with tab2:
    st.header("Model Performance & Metrics")

    data = [
        {"Model": "Conv CNN (Standard)", "Metric": "Precision", "Value": 0.9256},
        {"Model": "Conv CNN (Standard)", "Metric": "Recall", "Value": 0.9614},
        {"Model": "Conv CNN (Standard)", "Metric": "F1 Score", "Value": 0.9432},
        {"Model": "Conv CNN (Standard)", "Metric": "Time (s)", "Value": 374.67},

        {"Model": "ResNet-50", "Metric": "Precision", "Value": 0.9621},
        {"Model": "ResNet-50", "Metric": "Recall", "Value": 0.9591},
        {"Model": "ResNet-50", "Metric": "F1 Score", "Value": 0.9606},
        {"Model": "ResNet-50", "Metric": "Time (s)", "Value": 4877.81},

        {"Model": "Conv CNN (Tuned)", "Metric": "Precision", "Value": 0.9513},
        {"Model": "Conv CNN (Tuned)", "Metric": "Recall", "Value": 0.9413},
        {"Model": "Conv CNN (Tuned)", "Metric": "F1 Score", "Value": 0.9463},
        {"Model": "Conv CNN (Tuned)", "Metric": "Time (s)", "Value": 1289.65},
    ]
    df = pd.DataFrame(data)

    df_scores = df[df["Metric"] != "Time (s)"]
    fig_scores = px.bar(
        df_scores, x="Metric", y="Value", color="Model", barmode="group",
        text_auto=True, color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_scores.update_traces(textposition='outside')
    fig_scores.update_layout(yaxis_range=[0.8, 1.0])
    st.plotly_chart(fig_scores, width="stretch")

    st.subheader("⏱️ Training Efficiency")
    df_time = df[df["Metric"] == "Time (s)"]
    fig_time = px.bar(
        df_time, x="Value", y="Model", orientation='h', color="Model",
        text_auto=True, title="Training Time (seconds) - Lower is Faster",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_time, width="stretch")