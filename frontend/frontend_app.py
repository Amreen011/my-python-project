import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Make sure to install this: pip install joblib

# --- Load your trained model ---
# This will now look for 'heart_disease_model.pkl' in the same directory.
try:
    model = joblib.load("heart_disease_model.pkl")
except FileNotFoundError:
    st.error("Model file 'heart_disease_model.pkl' not found. Please place it in the same directory as this script.")
    st.stop() # Stop the app if the model can't be found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Base styling for light theme */
    .stApp {
        background-color: #f0f2f6; /* A slightly off-white bg */
    }
    
    /* Main content block */
    .main .block-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    
    /* Title and Subtitle */
    .title {
        text-align: center;
        color: #002855; /* Dark blue */
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #4a4a4a; /* Dark gray */
        margin-bottom: 30px;
    }
    
    /* Header Icon */
    .header-icon {
        text-align: center;
        margin-bottom: 1rem;
    }
    .header-icon svg {
        width: 4rem;
        height: 4rem;
        color: #0077b6; /* Bright blue */
    }
    
    /* --- FIX FOR FAINT TEXT --- */
    /* Target the subheader ("Enter Patient Details") */
    [data-testid="stSubheader"] {
        color: #000000 !important; /* Force to black */
        opacity: 1 !important;
    }
    
    /* Target all widget labels (number_input, selectbox, etc.) */
    [data-testid="stWidgetLabel"] p {
        color: #0077b6 !important; /* Force to blue */
        opacity: 1 !important;
    }
    /* --- END OF FIX --- */

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    
    /* Prediction button specific style */
    .stButton>button[kind="primary"] {
        background-color: #0077b6;
        color: white;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #005f8a;
    }
    
    /* Chart button specific style */
    .stButton>button[kind="secondary"] {
        background-color: #e5e7eb; /* gray-200 */
        color: #374151; /* gray-700 */
    }
    .stButton>button[kind="secondary"]:hover {
        background-color: #d1d5db; /* gray-300 */
    }

    /* Prediction Result Box */
    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid;
        text-align: center;
        margin-top: 25px;
    }
    .prediction-box-high {
        background-color: #fff1f2; /* red-50 */
        border-color: #fecdd3; /* red-200 */
        color: #b91c1c; /* red-800 */
    }
    .prediction-box-low {
        background-color: #f0fdf4; /* green-50 */
        border-color: #bbf7d0; /* green-200 */
        color: #166534; /* green-800 */
    }
    .prediction-header {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .prediction-header svg {
        width: 30px;
        height: 30px;
        margin-right: 10px;
    }
    .prediction-prob {
        font-size: 48px;
        font-weight: 700;
        margin: 10px 0;
    }
    .prediction-desc {
        font-size: 16px;
        color: #4b5563; /* gray-600 */
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown(f"""
    <div class="header-icon">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12Z" />
        </svg>
    </div>
    <div class="title">Heart Disease Predictor</div>
    <div class="subtitle">Enter patient data to simulate a prediction.</div>
""", unsafe_allow_html=True)

# --- Input Form ---
st.subheader("Enter Patient Details")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3], 
        format_func=lambda x: {
            0: "Typical Angina (0)", 
            1: "Atypical Angina (1)", 
            2: "Non-Anginal Pain (2)", 
            3: "Asymptomatic (3)"
        }[x],
        help="Type of chest pain experienced by the patient."
    )

with col2:
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 120, help="Resting blood pressure when the patient was admitted.")
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200, help="Total cholesterol level in the blood.")
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], 
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Is the patient's fasting blood sugar higher than 120 mg/dl?"
    )
    
with col3:
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2], 
        format_func=lambda x: {
            0: "Normal (0)",
            1: "ST-T Abnormality (1)",
            2: "LV Hypertrophy (2)"
        }[x],
        help="Results of the resting electrocardiogram (ECG)."
    )
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150, help="The highest heart rate achieved during a stress test.")
    exang = st.selectbox("Exercise Induced Angina", [0, 1], 
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Did the patient experience chest pain (angina) during exercise?"
    )

# --- Second row of inputs ---
col4, col5, col6 = st.columns(3)
with col4:
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest. A measure of abnormality in the ECG during exercise.")
with col5:
    slope = st.selectbox("ST Segment Slope", [0, 1, 2], 
        format_func=lambda x: {
            0: "Upsloping (0)",
            1: "Flat (1)",
            2: "Downsloping (2)"
        }[x],
        help="The slope of the peak exercise ST segment on the ECG."
    )
with col6:
    ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3], help="Number of major blood vessels (0-3) colored by fluoroscopy (a type of X-ray). This shows blood flow.")

thal = st.selectbox("Thalassemia (Thal)", [0, 1, 2, 3], 
    format_func=lambda x: {
        0: "N/A (0)",
        1: "Fixed Defect (1)",
        2: "Normal (2)",
        3: "Reversible Defect (3)"
    }[x], 
    index=2, # Default to 'Normal'
    help="A blood disorder called thalassemia. 'Normal' is good, 'Fixed Defect' and 'Reversible Defect' are types of heart perfusion issues."
)


st.markdown("<br>", unsafe_allow_html=True) # Add some space

# --- Prediction Button ---
if st.button("Predict Heart Disease Risk", type="primary"):
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope, ca, thal
    ]])
    
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    try:
        # Get probability of high risk (class 1)
        probability = model.predict_proba(input_data)[0][1] * 100 
    except AttributeError:
        # Fallback if your model doesn't support 'predict_proba'
        # (e.g., some SVMs). It will show a default confidence.
        st.warning("Model doesn't support 'predict_proba'. Showing default confidence.")
        probability = 85 if prediction == 1 else 15
    except Exception as e:
        st.error(f"Could not get probability: {e}")
        probability = 85 if prediction == 1 else 15


    if prediction == 1:
        # High Risk
        prob_text = f"{probability:.0f}%"
        icon = """
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
        </svg>
        """
        st.markdown(f"""
            <div class="prediction-box prediction-box-high">
                <div class="prediction-header">{icon} High Risk</div>
                <div class="prediction-prob">{prob_text}</div>
                <p class="prediction-desc">The model predicts a high probability of heart disease based on the provided inputs.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Low Risk
        prob_text = f"{100-probability:.0f}%" # Show probability of low risk
        icon = """
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
        </svg>
        """
        st.markdown(f"""
            <div class="prediction-box prediction-box-low">
                <div class="prediction-header">{icon} Low Risk</div>
                <div class="prediction-prob">{prob_text}</div>
                <p class="prediction-desc">The model predicts a low probability of heart disease based on the provided inputs.</p>
            </div>
        """, unsafe_allow_html=True)


# --- Reference Chart Button ---
if st.button("Show Health Reference Chart", type="secondary"):
    st.subheader("Normal Health Parameter Ranges")
    ranges = {
        "Parameter": [
            "Age (years)", 
            "Resting Blood Pressure (mm Hg)", 
            "Serum Cholesterol (mg/dl)", 
            "Max Heart Rate Achieved", 
            "ST Depression (Oldpeak)"
        ],
        "Normal Range (Min)": [20, 90, 125, 100, 0.0],
        "Normal Range (Max)": [60, 120, 200, 180, 2.0],
    }

    df = pd.DataFrame(ranges)
    st.dataframe(df.set_index("Parameter"), use_container_width=True)