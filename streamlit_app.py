# streamlit_app.py

import os
import tempfile

import streamlit as st
import torch

from inference import preprocess_video, run_inference, load_labels, MODEL_PATH, LABELS_TXT, NUM_CLASSES
from pytorch_i3d import InceptionI3d

@st.cache_resource
def load_model():
    """Load and cache the I3D model on GPU."""
    model = InceptionI3d(num_classes=400, in_channels=3)
    model.replace_logits(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.cuda().eval()
    return model

@st.cache_data
def get_labels():
    """Load and cache the class labels."""
    return load_labels(LABELS_TXT)

def main():
    st.set_page_config(page_title="Sign Language Inference", layout="wide")
    st.title("Sign Language Inference App")

    # Two‐column layout: left for upload/action, right for results
    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file…", type=["mp4", "mov"]
        )

        # Preview the video once loaded
        if uploaded_file:
            st.video(uploaded_file)

        # Start inference button, disabled until a file is chosen
        start_disabled = (uploaded_file is None)
        if st.button("▶ Start Inference", disabled=start_disabled):
            with st.spinner("Running inference…"):
                # Save uploaded file to a temp file so OpenCV can read it
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Preprocess frames
                video_tensor = preprocess_video(tmp_path)
                if video_tensor is None:
                    st.error("Failed to process video. Please check the file format.")
                    return

                # Run the model
                model = load_model()
                labels = get_labels()
                indices, confidences = run_inference(video_tensor, model)

                # Clean up temp file
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

                # Format results as (word, confidence%)
                results = []
                for idx, conf in zip(indices, confidences):
                    label = labels[idx] if idx < len(labels) else f"[{idx}]"
                    results.append((label, float(conf) * 100.0))

                # Store in session state to display in the right panel
                st.session_state.results = results

    with col2:
        st.header("Inference Results")
        results = st.session_state.get("results", None)

        if results:
            # Display each word + confidence + progress bar
            for word, conf in results:
                st.subheader(word)
                st.write(f"{conf:.2f}%")
                st.progress(conf / 100.0)

            # Prepare CSV for download
            csv_lines = ["word,confidence"]
            for word, conf in results:
                csv_lines.append(f"{word},{conf:.2f}")
            csv_data = "\n".join(csv_lines)

            st.download_button(
                "💾 Export Results",
                data=csv_data,
                file_name="results.csv",
                mime="text/csv"
            )
        else:
            st.info("No results yet. Upload a video and press ▶ Start Inference.")

if __name__ == "__main__":
    main()
