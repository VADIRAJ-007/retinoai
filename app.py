import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
from recommendations import cnv, dme, drusen, normal
import pandas as pd
from fpdf import FPDF
import datetime
import os
import pickle
import matplotlib.pyplot as plt
import io

# =============================
# 🎨 Page Config
# =============================
st.set_page_config(
    page_title="OCT Retinal Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# 📊 Session State Initialization
# =============================
if 'case_counts' not in st.session_state:
    st.session_state['case_counts'] = {
        "CNV": 0,
        "DME": 0,
        "DRUSEN": 0,
        "NORMAL": 0
    }

# =============================
# 🖌️ Custom CSS Styling
# =============================
st.markdown("""
    <style>
        /* Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        /* Headers */
        h1, h2, h3 {
            text-align: center;
            font-weight: 700;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #2C2F48 !important;
            color: white !important;
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Prediction Card */
        .prediction-card {
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin: 15px auto;
            max-width: 600px;
        }

        /* Center image with fixed width */
        .image-container {
            max-width: 600px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)


# =============================
# ⚡ Model Loader
# =============================
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model(
        "./Trained_Model.h5",
        custom_objects={"f1_score": f1_score}
    )
    return model


def model_prediction(test_image_path):
    model = load_model()
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)
    return prediction[0]  # Return probabilities


def f1_score(y_true, y_pred):
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), K.shape(y_true)[-1])
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)


# =============================
# 📄 PDF Report Generator
# =============================
def create_pdf(image_path, prediction_class, confidence_scores, recommendation_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Retinal OCT Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Date
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='R')
    pdf.ln(10)

    # Image
    pdf.image(image_path, x=60, w=90)
    pdf.ln(10)

    # Prediction
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Prediction: {prediction_class}", ln=True, align='C')
    pdf.ln(10)

    # Confidence Scores
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Confidence Scores:", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    for i, score in enumerate(confidence_scores):
        pdf.cell(200, 10, txt=f"{classes[i]}: {score*100:.2f}%", ln=True, align='L')

    # Recommendations
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Recommendations:", ln=True, align='L')
    pdf.ln(5)
    pdf.set_font("Arial", size=10)
    
    # Clean markdown syntax for PDF
    clean_text = recommendation_text.replace("**", "").replace("- ", "- ").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    pdf.multi_cell(0, 5, txt=clean_text)

    return pdf.output(dest='S').encode('latin-1')


# =============================
# 🧭 Sidebar Navigation
# =============================
st.sidebar.title("📌 Dashboard")
app_mode = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "ℹ️ About", "🩺 Disease Identification"]
)

# 📊 Session Statistics in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Session Stats")
stats_placeholder = st.sidebar.empty()

def update_sidebar_stats():
    with stats_placeholder.container():
        st.write(st.session_state['case_counts'])
        
        # Export to Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            pd.DataFrame(list(st.session_state['case_counts'].items()), columns=['Condition', 'Count']).to_excel(writer, index=False, sheet_name='Case Counts')
            
        st.download_button(
            label="📥 Download Stats (Excel Format)",
            data=buffer.getvalue(),
            file_name=f"oct_case_stats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_stats_{datetime.datetime.now().timestamp()}" # Unique key to prevent ID collisions
        )

# Initial Render
update_sidebar_stats()


# =============================
# 🏠 HOME
# =============================
if app_mode == "🏠 Home":
    st.markdown("<h1 style='color:#4B8BBE;'>👁️ OCT Retinal Analysis Platform</h1>",
                unsafe_allow_html=True)
    st.markdown("""
    Welcome to the **Retinal OCT Analysis Platform**!  
    Upload your **retinal OCT scans** and let our AI model assist in detecting common retinal conditions such as:  
    - 🟥 **Choroidal Neovascularization (CNV)**  
    - 🟨 **Diabetic Macular Edema (DME)**  
    - 🟩 **Drusen (Early AMD)**  
    - 🟦 **Normal Retina**
    
    ---
    """)


# =============================
# ℹ️ ABOUT
# =============================
elif app_mode == "ℹ️ About":
    st.markdown("<h1 style='color:#6A4C93;'>📖 About the Dataset & Model</h1>",
                unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Dataset Info", "Model Performance"])
    
    with tab1:
        st.markdown("""
        Retinal **Optical Coherence Tomography (OCT)** is a high-resolution imaging method.  
        Each year, **30M+ OCT scans** are performed globally, aiding in diagnosis of sight-threatening diseases.  
        
        - Dataset Size: **84,495 OCT images**  
        - Classes: **CNV, DME, Drusen, Normal**  
        - Sources: Multiple global hospitals (UCSD, Beijing Tongren, Shanghai First People’s Hospital, etc.)  
        - Verification: **Multi-tier grading by Specialized ophthalmologists**  
        """)
    
    with tab2:
        st.subheader("📊 Confusion Matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Model Confusion Matrix", use_column_width=True)
        else:
            st.warning("Confusion matrix image not found.")

        st.subheader("📈 Training History")
        if os.path.exists("Training_history.pkl"):
            try:
                with open("Training_history.pkl", "rb") as f:
                    history = pickle.load(f)
                
                # Plotting
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                
                # Accuracy
                if 'accuracy' in history and 'val_accuracy' in history:
                    ax[0].plot(history['accuracy'], label='Train Accuracy')
                    ax[0].plot(history['val_accuracy'], label='Val Accuracy')
                    ax[0].set_title('Model Accuracy')
                    ax[0].set_xlabel('Epoch')
                    ax[0].set_ylabel('Accuracy')
                    ax[0].legend()
                
                # Loss
                if 'loss' in history and 'val_loss' in history:
                    ax[1].plot(history['loss'], label='Train Loss')
                    ax[1].plot(history['val_loss'], label='Val Loss')
                    ax[1].set_title('Model Loss')
                    ax[1].set_xlabel('Epoch')
                    ax[1].set_ylabel('Loss')
                    ax[1].legend()
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not load training history: {e}")
        else:
            st.info("Training history file not found.")


# =============================
# 🩺 DISEASE IDENTIFICATION
# =============================
elif app_mode == "🩺 Disease Identification":
    st.markdown("<h1 style='color:#198754;'>🩺 Upload and Analyze OCT Scan</h1>",
                unsafe_allow_html=True)
    test_image = st.file_uploader(
        "📤 Upload an OCT Image", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as temp_file:
            temp_file.write(test_image.read())
            temp_file_path = temp_file.name

    if (st.button("🔍 Predict") and test_image is not None):
        with st.spinner("⚡ Analyzing your image..."):
            probabilities = model_prediction(temp_file_path)
            result_index = np.argmax(probabilities)
            
            class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
            colors = ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF"]
            result_color = colors[result_index]
            predicted_class = class_names[result_index]
            
            # Update Session Stats
            st.session_state['case_counts'][predicted_class] += 1
            update_sidebar_stats() # Realtime Update

            # 🎨 Styled prediction card with black text
            st.markdown(f"""
                <div class="prediction-card" style="background-color:{result_color}; color:black;">
                    ✅ Prediction: {predicted_class}
                </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # 📏 Uploaded image
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(test_image, caption="Uploaded OCT Scan", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                # 📊 Confidence Scores
                st.subheader("📊 Confidence Scores")
                df_scores = pd.DataFrame({
                    "Condition": class_names,
                    "Probability": probabilities
                })
                st.bar_chart(df_scores.set_index("Condition"))

            # 📥 Download Report
            # Get recommendation text based on prediction
            recommendation_map = {0: cnv, 1: dme, 2: drusen, 3: normal}
            rec_text = recommendation_map[result_index]
            
            pdf_bytes = create_pdf(temp_file_path, predicted_class, probabilities, rec_text)
            st.download_button(
                label="📄 Download Report (PDF)",
                data=pdf_bytes,
                file_name="oct_analysis_report.pdf",
                mime="application/pdf"
            )

            # Disease info
            with st.expander("📖 Learn More about this Condition"):
                if result_index == 0:
                    st.markdown("### 🟥 Choroidal Neovascularization (CNV)")
                    st.write("OCT scan showing **CNV with subretinal fluid**.")
                    st.markdown(cnv, unsafe_allow_html=True)

                elif result_index == 1:
                    st.markdown("### 🟨 Diabetic Macular Edema (DME)")
                    st.write(
                        "OCT scan showing **DME with retinal thickening and intraretinal fluid**.")
                    st.markdown(dme, unsafe_allow_html=True)

                elif result_index == 2:
                    st.markdown("### 🟩 Drusen (Early AMD)")
                    st.write("OCT scan showing **drusen deposits in early AMD**.")
                    st.markdown(drusen, unsafe_allow_html=True)

                elif result_index == 3:
                    st.markdown("### 🟦 Normal Retina")
                    st.write(
                        "OCT scan showing a **normal retina with preserved foveal contour**.")
                    st.markdown(normal, unsafe_allow_html=True)

    elif test_image is None:
        st.info("👆 Please upload an OCT image to begin analysis.")
