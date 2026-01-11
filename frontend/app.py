import streamlit as st
import requests
from PIL import Image
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CIFAKE Detector", layout="wide")

st.title("CIFAKE Detection System")

st.markdown("""
                <style>
                    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
                </style>
            """,
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🧪 Model Testing", "📊 Training Results"])

with tab1:
    st.header("Test Model on New Images")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("⚙️ Settings")

        model_display_names = {
            "Conv CNN (Standard)": "conv_cnn",
            "Conv CNN (Tuned)": "tuned_cnn",
            "ResNet-50": "resnet"
        }

        model_option = st.selectbox(
            "Select Architecture",
            list(model_display_names.keys())
        )
        model_key = model_display_names[model_option]

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        analyze_button = st.button("Analyze Image", type="primary", width="stretch")

    with col2:
        st.subheader("👁️ Preview & Results")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            if analyze_button:
                with st.spinner('Processing...'):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": uploaded_file}
                        response = requests.post(f"{API_URL}/predict/{model_key}", files=files)

                        if response.status_code == 200:
                            result = response.json()
                            pred = result['prediction']
                            conf = float(result['confidence'])

                            res_col1, res_col2 = st.columns([2, 1])

                            with res_col1:
                                if pred == "REAL":
                                    st.success(f"### Prediction: REAL IMAGE")
                                else:
                                    st.error(f"### Prediction: AI-GENERATED")

                            with res_col2:
                                st.metric("Confidence", f"{conf * 100:.2f}%")
                                st.progress(conf)
                                st.caption(f"Model used: {result['model']}")

                            st.divider()
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection failed. Is the backend running? {e}")

            st.image(image, caption="Uploaded Image", width="stretch")
        else:
            st.info("👈 Upload an image in the left panel to get started.")

with tab2:
    st.header("Model Performance & Metrics")


    def display_metrics(title, precision, recall, f1, time_taken=None, extra_info=None):
        st.subheader(title)

        if extra_info:
            with st.expander("🔍 View Hyperparameters (Best Configuration)", expanded=True):
                st.code(extra_info, language='yaml')

        cols = st.columns(4)
        cols[0].metric("Precision", precision)
        cols[1].metric("Recall", recall)
        cols[2].metric("F1 Score", f1)
        if time_taken:
            cols[3].metric("Training Time", time_taken)

    tuned_config_str = """
        Batch Size:    32
        Neurons (FC):  256
        Dropout:       0.29
        Learning Rate: 0.001
        Weight Decay:  1e-05
    """

    display_metrics(
        "Architecture: Conv CNN (Tuned) [Best Performer] 🏆",
        precision="0.9459",
        recall="0.9452",
        f1="0.9455",
        time_taken="1165.10s",
        extra_info=tuned_config_str
    )

    tuned_plot_path = os.path.join("training_results", "conv_cnn_tuner", "conv_cnn_tuner.png")
    if os.path.exists(tuned_plot_path):
        st.image(tuned_plot_path, caption="Tuned CNN Training History", width="stretch")
    else:
        st.warning(f"Plot not found at {tuned_plot_path}")

    st.markdown("---")

    display_metrics(
        "Architecture: Conv CNN (Standard)",
        precision="0.9503",
        recall="0.9399",
        f1="0.9451",
        time_taken="273.76s"
    )
    conv_plot_path = os.path.join("training_results", "conv_cnn", "conv_cnn.png")
    if os.path.exists(conv_plot_path):
        st.image(conv_plot_path, caption="Standard CNN Training History", width="stretch")

    st.markdown("---")

    display_metrics(
        "Architecture: ResNet-50",
        precision="0.9318",
        recall="0.9711",
        f1="0.9510",
        time_taken="2333.37s"
    )
    res_plot_path = os.path.join("training_results", "resnet-50", "resnet50.png")
    if os.path.exists(res_plot_path):
        st.image(res_plot_path, caption="ResNet-50 Training History", width="stretch")