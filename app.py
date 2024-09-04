import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Custom CSS for a cleaner, modern look
custom_css = """
<style>
body {
    background-color: #f0f2f6;
    color: #333333;
}
.main {
    padding: 2rem;
    border-radius: 10px;
    background-color: #ffffff;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    max-width: 800px;
    margin: auto;
}
.stTextInput, .stNumberInput, .stSelectbox, .stSlider {
    margin-bottom: 1.5rem;
}
.stButton > button {
    margin-top: 1rem;
    background-color: #ff4b4b;
    color: #ffffff;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
}
.stButton > button:hover {
    background-color: #e84141;
}
h1, h2 {
    text-align: center;
    color: #ff4b4b;
    font-weight: bold;
}
footer {
    text-align: center;
    margin-top: 2rem;
    color: #888888;
    font-size: 0.9rem;
    font-weight: bold;
}
table {
    width: 100%;
    border: none;
}
th, td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #dddddd;
}
</style>
"""

# Apply the custom CSS to the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Title of the Streamlit app
st.title("Real Estate Price Prediction")

# Create layout using columns for better structure
col1, col2 = st.columns(2)

with col1:
    st.sidebar.header("Input Features")
    locations = [
        "1st Block Jayanagar", "1st Phase JP Nagar", "2nd Phase Judicial Layout", "2nd Stage Nagarbhavi",
        # (Add other locations as required)
        "Yelahanka", "Yelahanka New Town", "Yeshwanthpur", "other"
    ]
    location = st.sidebar.selectbox("Location", options=locations, index=0)
    total_sqft = st.sidebar.number_input("Total Square Feet", min_value=0.0, step=1.0)
    bath = st.sidebar.number_input("Number of Bathrooms", min_value=0, step=1)
    bhk = st.sidebar.number_input("Number of BHK", min_value=0, step=1)

    # Create an instance of CustomData
    custom_data = CustomData(location=location, total_sqft=total_sqft, bath=bath, bhk=bhk)

    # Return data as DataFrame
    input_df = custom_data.get_data_as_data_frame()

with col2:
    # Display user input features
    st.subheader("User Input Features")
    st.write(input_df.style.set_table_attributes('class="table"').set_properties(**{'font-weight': 'bold'}))

# Load prediction pipeline
predict_pipeline = PredictPipeline()

# Create button for prediction
if st.button("Predict Price"):
    prediction = predict_pipeline.predict(input_df)
    st.subheader("Predicted Price")
    st.write(f" ₹ {prediction[0]:.2f} LAKH RUPEES ")

# Footer with author's name
st.markdown("<footer>Developed by Jai Dixit</footer>", unsafe_allow_html=True)