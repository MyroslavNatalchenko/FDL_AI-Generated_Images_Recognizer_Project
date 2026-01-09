import streamlit as st
import requests
from PIL import Image
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CIFAKE Detector", layout="wide")

st.title("🕵️ CIFAKE Detection System")
st.markdown("Deep Learning system to distinguish between Real and AI-Generated images.")

tab1, tab2 = st.tabs(["🧪 Model Testing", "📊 Training Results"])

with tab1:
    st.header("Test Model on New Images")

    col1, col2 = st.columns([1, 1])

    with col1:
        model_option = st.selectbox(
            "Select Architecture",
            ("Conv CNN", "ResNet-50")
        )

        model_key = "conv_cnn" if model_option == "Conv CNN" else "resnet"

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)

            if st.button("Analyze Image", type="primary"):
                with st.spinner('Processing...'):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": uploaded_file}

                        response = requests.post(f"{API_URL}/predict/{model_key}", files=files)

                        if response.status_code == 200:
                            result = response.json()
                            pred = result['prediction']
                            conf = float(result['confidence'])

                            # Visual Feedback
                            if pred == "REAL":
                                st.success(f"**Prediction: REAL IMAGE**")
                            else:
                                st.error(f"**Prediction: AI-GENERATED (FAKE)**")

                            st.metric("Confidence Score", f"{conf * 100:.2f}%")
                        else:
                            st.error(f"Error: {response.text}")

                    except Exception as e:
                        st.error(f"Connection failed. Is the backend running? {e}")

with tab2:
    st.header("Model Performance & Metrics")

    def display_metrics(title, time, precision, recall, f1):
        st.subheader(title)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Time Taken", time)
        m2.metric("Precision", precision)
        m3.metric("Recall", recall)
        m4.metric("F1 Score", f1)

    display_metrics(
        "Architecture: Conv CNN (10 Epochs)",
        "273.76s", "0.9503", "0.9399", "0.9451"
    )

    conv_plot_path = os.path.join("training_results", "conv_cnn", "conv_cnn.png")
    if os.path.exists(conv_plot_path):
        st.image(conv_plot_path, caption="Conv CNN Training History & Confusion Matrix", use_column_width=True)
    else:
        st.warning(f"Plot not found at {conv_plot_path}")

    st.markdown("---")

    display_metrics(
        "Architecture: ResNet-50 (5 Epochs)",
        "2347.11s", "0.9444", "0.9614", "0.9528"
    )

    res_plot_path = os.path.join("training_results", "resnet-50", "resnet50.png")
    if os.path.exists(res_plot_path):
        st.image(res_plot_path, caption="ResNet-50 Training History & Confusion Matrix", use_column_width=True)
    else:
        st.warning(f"Plot not found at {res_plot_path}")