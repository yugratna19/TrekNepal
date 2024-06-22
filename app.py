import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

# Load the processed trek data, vectorizer, and KNN model
trek_data = pd.read_csv('nepal-treking-dataset\Trek Data Modified.csv')

vectorizer = joblib.load('vectorizer.joblib')
knn = joblib.load('knn_model.joblib')

st.sidebar.header('Select Trek Preferences')

# Define options for user selection
cost_options = ['cheap', 'medium', 'expensive', 'luxury']
duration_options = ['short', 'medium', 'long', 'very long']
altitude_options = ['low', 'medium', 'high', 'very high']
difficulty_options = ['easy', 'moderate', 'hard']

# User selection using selectbox
selected_cost = st.sidebar.selectbox('Cost', cost_options)
selected_duration = st.sidebar.selectbox('Duration', duration_options)
selected_altitude = st.sidebar.selectbox('Altitude', altitude_options)
selected_difficulty = st.sidebar.selectbox('Difficulty', difficulty_options)

# Display selected options with formatting
st.markdown('### Selected Trek Preferences:')
st.markdown(f'- **Cost:** {selected_cost.capitalize()}')
st.markdown(f'- **Duration:** {selected_duration.capitalize()}')
st.markdown(f'- **Altitude:** {selected_altitude.capitalize()}')
st.markdown(f'- **Difficulty:** {selected_difficulty.capitalize()}')

# Combine user inputs into a feature string
user_input = ' '.join([selected_cost, selected_duration, selected_altitude, selected_difficulty])
user_vector = vectorizer.transform([user_input])

# Use KNN to find similar treks based on user preferences
st.write("Finding recommendations based on your preferences...")
X = vectorizer.transform(trek_data['Features'])
distances, indices = knn.kneighbors(user_vector)

# Get recommended treks
results = trek_data.iloc[indices[0]]

# Ensure unique treks are displayed
unique_treks = results.drop_duplicates(subset='Trek')

# Display the results
st.header('Recommended Treks')
if not unique_treks.empty:
    for _, row in unique_treks.iterrows():
        st.write(f"Trek: {row['Trek']}")
        st.write(f"Cost: USD {row['Cost']}")
        st.write(f"Duration: {row['Time']} days")
        st.write(f"Difficulty: {row['Trip Grade']}")
        st.write(f"Link: {row['Contact or Book your Trip']}")
        st.write("---")
else:
    st.write("No recommendations found.")
