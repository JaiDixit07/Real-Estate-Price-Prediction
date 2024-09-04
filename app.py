import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Custom CSS for styling
custom_css = """
<style>
body {
    background: rgba(0, 0, 0, 0.8) url("https://images.pexels.com/photos/730252/pexels-photo-730252.jpeg") no-repeat center center fixed;
    background-size: cover;
    color: #e0e0e0;
}
.main {
    padding: 2rem;
    border-radius: 10px;
    background-color: rgba(0, 0, 0, 0.7);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
}
.stTextInput, .stNumberInput, .stSelectbox, .stSlider {
    margin-bottom: 1rem;
}
.stButton > button {
    margin-top: 1rem;
}
h1, h2 {
    text-align: center;
    color: #ffffff;
    font-weight: bold;
}
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid #ffffff;
    text-align: left;
    padding: 8px;
}
tr:nth-child(even) {
    background-color: #333333;
}
tr:nth-child(odd) {
    background-color: #444444;
}
th {
    background-color: #555555;
    color: #ffffff;
}
footer {
    text-align: left;
    margin-top: 2rem;
    color: #ffffff;
    font-size: 0.9rem;
    font-weight: bold;
}
</style>
"""

# Apply the custom CSS to the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Title of the Streamlit app
st.title("Real Estate Price Prediction")

# Sidebar for user input features
st.sidebar.header("Input Features")

# List of locations
locations = [
    "1st Block Jayanagar", "1st Phase JP Nagar", "2nd Phase Judicial Layout", "2nd Stage Nagarbhavi",
    "5th Block Hbr Layout", "5th Phase JP Nagar", "6th Phase JP Nagar", "7th Phase JP Nagar",
    "8th Phase JP Nagar", "9th Phase JP Nagar", "AECS Layout", "Abbigere", "Akshaya Nagar",
    "Ambalipura", "Ambedkar Nagar", "Amruthahalli", "Anandapura", "Ananth Nagar", "Anekal",
    "Anjanapura", "Ardendale", "Arekere", "Attibele", "BEML Layout", "BTM 2nd Stage", "BTM Layout",
    "Babusapalaya", "Badavala Nagar", "Balagere", "Banashankari", "Banashankari Stage II",
    "Banashankari Stage III", "Banashankari Stage V", "Banashankari Stage VI", "Banaswadi",
    "Banjara Layout", "Bannerghatta", "Bannerghatta Road", "Basavangudi", "Basaveshwara Nagar",
    "Battarahalli", "Begur", "Begur Road", "Bellandur", "Benson Town", "Bharathi Nagar",
    "Bhoganhalli", "Billekahalli", "Binny Pete", "Bisuvanahalli", "Bommanahalli", "Bommasandra",
    "Bommasandra Industrial Area", "Bommenahalli", "Brookefield", "Budigere", "CV Raman Nagar",
    "Chamrajpet", "Chandapura", "Channasandra", "Chikka Tirupathi", "Chikkabanavar",
    "Chikkalasandra", "Choodasandra", "Cooke Town", "Cox Town", "Cunningham Road", "Dasanapura",
    "Dasarahalli", "Devanahalli", "Devarachikkanahalli", "Dodda Nekkundi", "Doddaballapur",
    "Doddakallasandra", "Doddathoguru", "Domlur", "Dommasandra", "EPIP Zone", "Electronic City",
    "Electronic City Phase II", "Electronics City Phase 1", "Frazer Town", "GM Palaya",
    "Garudachar Palya", "Giri Nagar", "Gollarapalya Hosahalli", "Gottigere", "Green Glen Layout",
    "Gubbalala", "Gunjur", "HAL 2nd Stage", "HBR Layout", "HRBR Layout", "HSR Layout",
    "Haralur Road", "Harlur", "Hebbal", "Hebbal Kempapura", "Hegde Nagar", "Hennur",
    "Hennur Road", "Hoodi", "Horamavu Agara", "Horamavu Banaswadi", "Hormavu", "Hosa Road",
    "Hosakerehalli", "Hoskote", "Hosur Road", "Hulimavu", "ISRO Layout", "ITPL", "Iblur Village",
    "Indira Nagar", "JP Nagar", "Jakkur", "Jalahalli", "Jalahalli East", "Jigani", "Judicial Layout",
    "KR Puram", "Kadubeesanahalli", "Kadugodi", "Kaggadasapura", "Kaggalipura", "Kaikondrahalli",
    "Kalena Agrahara", "Kalyan nagar", "Kambipura", "Kammanahalli", "Kammasandra", "Kanakapura",
    "Kanakpura Road", "Kannamangala", "Karuna Nagar", "Kasavanhalli", "Kasturi Nagar",
    "Kathriguppe", "Kaval Byrasandra", "Kenchenahalli", "Kengeri", "Kengeri Satellite Town",
    "Kereguddadahalli", "Kodichikkanahalli", "Kodigehaali", "Kodigehalli", "Kodihalli", "Kogilu",
    "Konanakunte", "Koramangala", "Kothannur", "Kothanur", "Kudlu", "Kudlu Gate",
    "Kumaraswami Layout", "Kundalahalli", "LB Shastri Nagar", "Laggere", "Lakshminarayana Pura",
    "Lingadheeranahalli", "Magadi Road", "Mahadevpura", "Mahalakshmi Layout", "Mallasandra",
    "Malleshpalya", "Malleshwaram", "Marathahalli", "Margondanahalli", "Marsur", "Mico Layout",
    "Munnekollal", "Murugeshpalya", "Mysore Road", "NGR Layout", "NRI Layout", "Nagarbhavi",
    "Nagasandra", "Nagavara", "Nagavarapalya", "Narayanapura", "Neeladri Nagar", "Nehru Nagar",
    "OMBR Layout", "Old Airport Road", "Old Madras Road", "Padmanabhanagar", "Pai Layout",
    "Panathur", "Parappana Agrahara", "Pattandur Agrahara", "Poorna Pragna Layout", "Prithvi Layout",
    "R.T. Nagar", "Rachenahalli", "Raja Rajeshwari Nagar", "Rajaji Nagar", "Rajiv Nagar",
    "Ramagondanahalli", "Ramamurthy Nagar", "Rayasandra", "Sahakara Nagar", "Sanjay nagar",
    "Sarakki Nagar", "Sarjapur", "Sarjapur  Road", "Sarjapura - Attibele Road", "Sector 2 HSR Layout",
    "Sector 7 HSR Layout", "Seegehalli", "Shampura", "Shivaji Nagar", "Singasandra",
    "Somasundara Palya", "Sompura", "Sonnenahalli", "Subramanyapura", "Sultan Palaya",
    "TC Palaya", "Talaghattapura", "Thanisandra", "Thigalarapalya", "Thubarahalli",
    "Thyagaraja Nagar", "Tindlu", "Tumkur Road", "Ulsoor", "Uttarahalli", "Varthur",
    "Varthur Road", "Vasanthapura", "Vidyaranyapura", "Vijayanagar", "Vishveshwarya Layout",
    "Vishwapriya Layout", "Vittasandra", "Whitefield", "Yelachenahalli", "Yelahanka",
    "Yelahanka New Town", "Yelenahalli", "Yeshwanthpur", "other"
]

# Function to get user input
def get_user_input():
    location = st.sidebar.selectbox("Location", options=locations, index=0)
    total_sqft = st.sidebar.number_input("Total Square Feet", min_value=0.0, step=1.0)
    bath = st.sidebar.number_input("Number of Bathrooms", min_value=0, step=1)
    bhk = st.sidebar.number_input("Number of BHK", min_value=0, step=1)
    
    # Create an instance of CustomData
    custom_data = CustomData(
        location=location,
        total_sqft=total_sqft,
        bath=bath,
        bhk=bhk
    )

    # Return data as DataFrame
    return custom_data.get_data_as_data_frame()

# Get user input data
input_df = get_user_input()

# Display user input features
st.subheader("User Input Features")
st.write(input_df.style.set_table_attributes('class="table"').set_properties(**{'font-weight': 'bold'}))

# Load prediction pipeline
predict_pipeline = PredictPipeline()

# Make predictions
if st.button("Predict"):
    prediction = predict_pipeline.predict(input_df)
    st.subheader("Predicted Price ")
    st.write(f"** ₹ {prediction[0]:.2f} LAKH RUPEES **")


# Footer with author's name
st.markdown("<footer>Developed by Jai Dixit</footer>", unsafe_allow_html=True)
