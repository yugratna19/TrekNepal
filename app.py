import streamlit as st
import pandas as pd
import joblib
import json
import folium
from streamlit_folium import folium_static
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import numpy as np
import os
import traceback

# Set page config
st.set_page_config(
    page_title="Nepal Trek Recommender",
    page_icon="🏔️",
    layout="wide"
)

# App title and description
st.title("🏔️ Nepal Trek Recommender")
st.markdown("""
This app helps you find the perfect trekking route in Nepal based on your preferences.
Simply select your preferences in the sidebar and get personalized recommendations!
""")

# Modified load_data function with better error handling
@st.cache_resource
def load_data():
    try:
        # Check if model files exist
        if not os.path.exists('vectorizer.joblib'):
            st.error("Vectorizer file not found. Please run the model.ipynb notebook first.")
            st.stop()
            
        if not os.path.exists('knn_model.joblib'):
            st.error("KNN model file not found. Please run the model.ipynb notebook first.")
            st.stop()
        
        # Load dataset
        trek_data = pd.read_csv('nepal-treking-dataset\\Trek Data Modified.csv')
        
        # Load feature mappings
        with open('feature_mappings.json', 'r') as f:
            feature_mappings = json.load(f)
        
        # Load vectorizer with robust error handling
        try:
            vectorizer = joblib.load('vectorizer.joblib')
            
            # Verify vectorizer is properly fitted
            if not hasattr(vectorizer, 'vocabulary_'):
                st.warning("Vectorizer missing vocabulary. Attempting to recreate...")
                # Create a new vectorizer and fit it from the data
                vectorizer = TfidfVectorizer(
                    analyzer='word',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True
                )
                feature_texts = trek_data['Features'].values
                vectorizer.fit(feature_texts)
                # Save the fixed vectorizer
                joblib.dump(vectorizer, 'vectorizer.joblib')
                st.success("Vectorizer recreated successfully.")
        except Exception as e:
            st.error(f"Error loading vectorizer: {e}")
            st.error(traceback.format_exc())
            st.stop()
        
        # Load KNN model
        try:
            knn = joblib.load('knn_model.joblib')
        except Exception as e:
            st.error(f"Error loading KNN model: {e}")
            st.error(traceback.format_exc())
            st.stop()
            
        return trek_data, vectorizer, knn, feature_mappings
        
    except Exception as e:
        st.error(f"Failed to load application data: {e}")
        st.error(traceback.format_exc())
        st.stop()

# Load data
trek_data, vectorizer, knn, feature_mappings = load_data()

# Create reverse mappings for user input to model features
cost_reverse_mapping = {v: k for k, v in feature_mappings.get('cost_mapping', {}).items()}
duration_reverse_mapping = {v: k for k, v in feature_mappings.get('duration_mapping', {}).items()}
altitude_reverse_mapping = {v: k for k, v in feature_mappings.get('altitude_mapping', {}).items()}

# For difficulty mapping, we need to handle lists
difficulty_reverse_mapping = {}
for k, v_list in feature_mappings.get('difficulty_mapping', {}).items():
    for v in v_list:
        difficulty_reverse_mapping[v] = k

# Define options for user selection based on available data
# Add "Any" option for all dropdowns to make selection optional
cost_options = ["Any"] + list(cost_reverse_mapping.keys()) if cost_reverse_mapping else ["Any", "Budget", "Standard", "Premium"]
duration_options = ["Any"] + list(duration_reverse_mapping.keys()) if duration_reverse_mapping else ["Any", "1-5 days", "6-10 days", "10+ days"]
altitude_options = ["Any"] + list(altitude_reverse_mapping.keys()) if altitude_reverse_mapping else ["Any", "Below 3000m", "3000-4000m", "Above 4000m"]
difficulty_options = ["Any"] + list(difficulty_reverse_mapping.keys()) if difficulty_reverse_mapping else ["Any", "Easy", "Moderate", "Difficult", "Challenging"]

# Add default option to season_options if empty
season_options = ["Any"] + trek_data['Best Travel Time'].unique().tolist() if not trek_data.empty else ["Any", "Spring", "Summer", "Autumn", "Winter"]

# Sidebar for user input
st.sidebar.header('Your Trek Preferences')
st.sidebar.markdown("*Select 'Any' for parameters you don't have a preference for*")

# User selection with better descriptions
selected_cost = st.sidebar.selectbox('Cost Range', cost_options, 
                                    help="Select your budget preference or 'Any'")
selected_duration = st.sidebar.selectbox('Trek Duration', duration_options,
                                        help="How long do you want to trek? Select 'Any' for no preference")
selected_altitude = st.sidebar.selectbox('Maximum Altitude', altitude_options,
                                        help="What's your preferred maximum altitude? Select 'Any' for no preference")
selected_difficulty = st.sidebar.selectbox('Trek Difficulty', difficulty_options,
                                         help="Select your preferred difficulty level or 'Any'")
selected_season = st.sidebar.selectbox('Best Travel Time', season_options,
                                      help="When do you plan to go? Select 'Any' for no preference")

# Additional filters
st.sidebar.header('Additional Filters')
min_days = int(trek_data['Time'].min())
max_days = int(trek_data['Time'].max())
use_days_filter = st.sidebar.checkbox("Filter by specific days range", False)
if use_days_filter:
    days_range = st.sidebar.slider('Days Range', min_days, max_days, (min_days, max_days))
else:
    days_range = (min_days, max_days)

# Number of recommendations
num_recommendations = st.sidebar.slider('Number of Recommendations', 1, 20, 5)

# Search functionality
search_query = st.sidebar.text_input("Search for specific trek", "")

# Map user inputs to model features - only for selected parameters (not "Any")
# Build user_input string dynamically based on selected filters
user_input_parts = []
if selected_cost != "Any":
    model_cost = cost_reverse_mapping.get(selected_cost, "Medium")
    user_input_parts.append(model_cost)
    
if selected_duration != "Any":
    model_duration = duration_reverse_mapping.get(selected_duration, "Medium")
    user_input_parts.append(model_duration)

if selected_altitude != "Any":
    model_altitude = altitude_reverse_mapping.get(selected_altitude, "Moderate")
    user_input_parts.append(model_altitude)
    
if selected_difficulty != "Any":
    model_difficulty = difficulty_reverse_mapping.get(selected_difficulty, "Moderate")
    user_input_parts.append(model_difficulty)
    
if selected_season != "Any":
    user_input_parts.append(selected_season)

# Display selected options in main area (only those that aren't "Any")
col1, col2 = st.columns(2)

with col1:
    st.markdown('### Your Trek Preferences:')
    if selected_cost != "Any":
        st.markdown(f'- **Cost Range:** {selected_cost}')
    if selected_duration != "Any":
        st.markdown(f'- **Trek Duration:** {selected_duration}')
    if selected_altitude != "Any":
        st.markdown(f'- **Maximum Altitude:** {selected_altitude}')
    
with col2:
    st.markdown('###  ')  # For alignment
    if selected_difficulty != "Any":
        st.markdown(f'- **Trek Difficulty:** {selected_difficulty}')
    if selected_season != "Any":
        st.markdown(f'- **Best Travel Time:** {selected_season}')
    if use_days_filter:
        st.markdown(f'- **Days Range:** {days_range[0]} to {days_range[1]} days')

# Display message if no preferences are selected
if not user_input_parts and not use_days_filter and not search_query:
    st.info("No preferences selected. Please select at least one preference or filter to get recommendations.")

# Button to get recommendations
if st.button("Find My Perfect Trek"):
    # Start with all treks
    results = trek_data.copy()
    
    if user_input_parts:
        # Only use KNN if the user has selected at least one preference
        with st.spinner("Finding the best treks for you..."):
            # Join selected parameters into the user input string
            user_input = " ".join(user_input_parts)
            st.write(f"Searching for: {user_input}")
            
            # Add extra verification before transforming
            if not hasattr(vectorizer, 'vocabulary_'):
                st.error("Vectorizer is not properly fitted. Please run the model notebook again.")
                st.stop()
            
            # Request more neighbors to ensure we have enough unique options after filtering
            neighbors_to_request = min(len(trek_data), max(100, num_recommendations * 3))
            user_vector = vectorizer.transform([user_input])
            distances, indices = knn.kneighbors(user_vector, n_neighbors=neighbors_to_request)
            
            # Get recommended treks
            results = trek_data.iloc[indices[0]]
            
            # Add distance for sorting
            results = results.copy()
            results['similarity_score'] = distances[0][:len(results)]
    
    # Apply additional filters
    
    # Filter by days range if selected
    if use_days_filter:
        results = results[(results['Time'] >= days_range[0]) & (results['Time'] <= days_range[1])]
    
    # Filter by search query if provided
    if search_query:
        results = results[results['Trek'].str.lower().str.contains(search_query.lower())]
    
    # Sort results - if KNN was used, sort by similarity, otherwise by popularity metrics
    if user_input_parts:
        results = results.sort_values(by='similarity_score')
    else:
        # If no preferences selected, sort by other meaningful metrics
        # For example, sort by rating or popularity (simulated by Time for now)
        results = results.sort_values(by='Time')
    
    # Ensure recommendations are unique by dropping duplicates based on trek name
    results = results.drop_duplicates(subset=['Trek'])
    
    # Check if we have enough unique recommendations
    available_recommendations = len(results)
    if available_recommendations < num_recommendations:
        st.warning(f"Only {available_recommendations} unique treks match your criteria. Showing all available options.")
        actual_recommendations = available_recommendations
    else:
        actual_recommendations = num_recommendations

    # Display the results
    st.header('Recommended Treks')
    
    if not results.empty:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Trek List", "Comparison Chart", "Map View"])
        
        with tab1:
            # Show the top unique recommendations as cards without images
            for i, (_, row) in enumerate(results.head(actual_recommendations).iterrows()):
                st.subheader(f"{i+1}. {row['Trek']}")  # Add numbering for clarity
                
                # Create three columns for trek details
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.write(f"🕒 **Duration:** {row['Time']} days")
                    st.write(f"🏔️ **Max Altitude:** {row['Max Altitude']} m")
                with col_b:
                    st.write(f"💰 **Cost:** USD {row['Cost']:.2f}")
                    st.write(f"📈 **Difficulty:** {row['Trip Grade']}")
                with col_c:
                    st.write(f"🗓️ **Best Time:** {row['Best Travel Time']}")
                    if 'Contact or Book your Trip' in row:
                        st.write(f"[Book This Trek]({row['Contact or Book your Trip']})")
                
                st.markdown("---")
        
        with tab2:
            # Comparison chart of unique recommendations
            chart_data = results.head(actual_recommendations)[['Trek', 'Time', 'Cost', 'Max Altitude']]
            
            # Duration chart
            fig1 = px.bar(chart_data, x='Trek', y='Time', title="Trek Duration Comparison (Days)")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Cost chart
            fig2 = px.bar(chart_data, x='Trek', y='Cost', title="Trek Cost Comparison (USD)")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Add a altitude comparison chart
            fig3 = px.bar(chart_data, x='Trek', y='Max Altitude', title="Maximum Altitude Comparison (meters)")
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            # Map view with unique recommendations
            st.write("Map view of recommended treks")
            m = folium.Map(location=[28.3949, 84.1240], zoom_start=7)
            
            # Add markers with different colors for better visibility
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
            
            for i, (_, row) in enumerate(results.head(actual_recommendations).iterrows()):
                # Using random coordinates around Nepal for demonstration
                color_idx = i % len(colors)
                lat = 28.3949 + np.random.uniform(-0.5, 0.5)
                lon = 84.1240 + np.random.uniform(-0.5, 0.5)
                
                folium.Marker(
                    [lat, lon],
                    popup=f"<b>{row['Trek']}</b><br>{row['Time']} days<br>${row['Cost']:.2f}<br>{row['Trip Grade']}",
                    tooltip=f"{i+1}. {row['Trek']}",
                    icon=folium.Icon(color=colors[color_idx], icon='info-sign')
                ).add_to(m)
            folium_static(m)
    else:
        st.warning("No treks match your criteria. Please adjust your preferences and try again.")

# Add a footer with information
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This app uses machine learning to recommend treks based on your preferences. 
It analyzes trek features including cost, duration, altitude, difficulty, and season 
to find the most suitable options for you.
""")
