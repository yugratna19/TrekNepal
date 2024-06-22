import streamlit as st
import joblib
import numpy as np

# Load the model, encoders, and scaler
model = joblib.load('improved_trek_recommendation_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Trek Recommendation System')

# Create input widgets with both sliders and manual entry
trip_grade = st.selectbox('Trip Grade', label_encoders['Trip Grade'].classes_)
accomodation = st.selectbox('Accomodation', label_encoders['Accomodation'].classes_)
best_travel_time = st.selectbox('Best Travel Time', label_encoders['Best Travel Time'].classes_)
cost_range = st.selectbox('Cost Range', label_encoders['Cost Range'].classes_)
duration_category = st.selectbox('Duration Category', label_encoders['Duration Category'].classes_)
altitude_category = st.selectbox('Altitude Category', label_encoders['Altitude Category'].classes_)

# Define ranges based on categories
if duration_category == 'Short':
    time_range = (5, 7)
elif duration_category == 'Medium':
    time_range = (7, 14)
elif duration_category == 'Long':
    time_range = (14, 21)
elif duration_category == 'Very Long':
    time_range = (21, 27)
else:
    time_range = (5, 27)

if cost_range == 'Low':
    cost_range_val = (450.0, 1000.0)
elif cost_range == 'Medium':
    cost_range_val = (1000.0, 2000.0)
elif cost_range == 'High':
    cost_range_val = (2000.0, 3000.0)
elif cost_range == 'Very High':
    cost_range_val = (3000.0, 4000.0)
elif cost_range == 'Luxury':
    cost_range_val = (4000.0, 4200.0)
else:
    cost_range_val = (450.0, 4200.0)

if altitude_category == 'Low':
    altitude_range = (1550, 3000)
elif altitude_category == 'Moderate':
    altitude_range = (3000, 4000)
elif altitude_category == 'High':
    altitude_range = (4000, 5000)
elif altitude_category == 'Very High':
    altitude_range = (5000, 6000)
elif altitude_category == 'Extreme':
    altitude_range = (6000, 6340)
else:
    altitude_range = (1550, 6340)
    
if st.button('Get Recommendation'):
    # Encode categorical features
    trip_grade_enc = label_encoders['Trip Grade'].transform([trip_grade])[0]
    accomodation_enc = label_encoders['Accomodation'].transform([accomodation])[0]
    best_travel_time_enc = label_encoders['Best Travel Time'].transform([best_travel_time])[0]
    cost_range_enc = label_encoders['Cost Range'].transform([cost_range])[0]
    duration_category_enc = label_encoders['Duration Category'].transform([duration_category])[0]
    altitude_category_enc = label_encoders['Altitude Category'].transform([altitude_category])[0]

    # Create feature array
    time = np.mean(time_range)
    cost = np.mean(cost_range_val)
    max_altitude = np.mean(altitude_range)
    features = np.array([[time, cost, trip_grade_enc, max_altitude, accomodation_enc, best_travel_time_enc, cost_range_enc, duration_category_enc, altitude_category_enc]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict using the model
    probabilities = model.predict_proba(features_scaled)
    
    # Get the top N recommendations
    top_n = 5
    top_indices = np.argsort(probabilities[0])[::-1][:top_n]
    
    # Decode the recommendations
    recommendations = [label_encoders['Trek'].inverse_transform([i])[0] for i in top_indices]

    st.write('Recommended Treks:')
    for i, trek in enumerate(recommendations, 1):
        st.write(f"{i}. {trek}")
