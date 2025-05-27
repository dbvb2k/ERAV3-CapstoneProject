# app.py

import os
import tempfile
import asyncio
import nest_asyncio
import requests
import streamlit as st
from PIL import Image

# Set OpenMP environment variable to handle multiple runtime libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure Streamlit page
st.set_page_config(
    page_title="Place Identifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration
NUM_PREDICTIONS = 3  # Number of predictions to show
API_URL = "http://localhost:8000/predict"  # API endpoint

# Main UI
st.title("📍 Place Identifier from Image")

# Create two columns
col1, col2 = st.columns(2)

# Left column for image upload and display
with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=False)

# Right column for predictions
with col2:
    st.subheader("Location Predictions")
    
    # Add confidence score legend
    st.markdown("""
    **Confidence Score Interpretation:**
    - 🟢 **Score > 100**: High confidence
    - 🟡 **Score 50-100**: Moderate confidence
    - 🔴 **Score < 50**: Low confidence
    """)
    
    if uploaded_file:
        try:
            with st.spinner("Identifying place..."):
                # Prepare the file for API request
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "image/jpeg"  # Set default content type if not provided
                    )
                }
                params = {"num_predictions": NUM_PREDICTIONS}
                
                # Make API request
                response = requests.post(API_URL, files=files, params=params)
                
                if response.status_code == 200:
                    predictions = response.json()
                    
                    # Display results
                    for pred in predictions:
                        confidence_score = pred["confidence_score"]
                        
                        # Determine confidence level color
                        if confidence_score > 100:
                            color = "green"
                        elif confidence_score >= 50:
                            color = "orange"
                        else:
                            color = "red"
                        
                        st.markdown(f"""
                        <div style='border: 2px solid {color}; padding: 5px; border-radius: 5px; margin-bottom: 5px;'>
                            <p style='margin: 0;'><b>Prediction {pred['prediction_number']}</b></p>
                            <p style='margin: 0;'>📍 <b>Location</b>: {pred['location_name']}</p>
                            <p style='margin: 0;'>🌍 <b>Latitude</b>: {pred['latitude']:.6f}, <b>Longitude</b>: {pred['longitude']:.6f}</p>
                            <p style='margin: 0;'><b>Confidence Score</b>: {confidence_score:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Error from API: {response.text}")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
