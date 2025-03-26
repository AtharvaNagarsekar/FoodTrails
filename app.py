import streamlit as st
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import re
import plotly.graph_objs as go
import plotly.express as px
import json

class DishRecommender:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.load_models()
    
    def load_models(self):
        """Load all pre-trained models and data"""
        try:
            # Load dataset
            self.df = pd.read_csv(f"{self.models_dir}/preprocessed_data.csv")
            
            # Load BERT
            with open(f"{self.models_dir}/bert_model_info.pkl", 'rb') as f:
                bert_info = pickle.load(f)
            self.bert = SentenceTransformer(bert_info['model_name'])
            
            # Load Autoencoder
            self.encoder = load_model(f"{self.models_dir}/encoder.keras")
            
            # Load Scaler
            with open(f"{self.models_dir}/taste_scaler.pkl", 'rb') as f:
                self.taste_scaler = pickle.load(f)
            
            # Load PCA
            with open(f"{self.models_dir}/pca_model.pkl", 'rb') as f:
                self.pca = pickle.load(f)
            
            # Load Best Clustering Info
            with open(f"{self.models_dir}/best_clustering_info.pkl", 'rb') as f:
                best_info = pickle.load(f)
            self.best_algo = best_info['algorithm']
            
            # Load Cluster Labels
            if self.best_algo in ['hierarchical', 'ensemble', 'weighted_ensemble']:
                # For these algorithms, we saved the labels directly
                self.cluster_labels = np.load(f"{self.models_dir}/{self.best_algo}_labels.npy")
            else:
                # For other algorithms, load the model and predict labels
                with open(f"{self.models_dir}/{self.best_algo}_model.pkl", 'rb') as f:
                    self.cluster_model = pickle.load(f)
                
                # Get reduced features for all dishes
                all_features = np.concatenate([
                    self.bert.encode(self.df['clean_ingredients'].tolist()),
                    self.taste_scaler.transform(
                        self.df[['spice_level','sweet_level','salty_level','sour_level','bitter_level']]
                    )
                ], axis=1)
                
                encoded = self.encoder.predict(all_features)
                reduced = self.pca.transform(encoded)
                
                # Predict cluster labels
                if self.best_algo == 'dbscan':
                    self.cluster_labels = self.cluster_model.fit_predict(reduced)
                else:
                    self.cluster_labels = self.cluster_model.predict(reduced)
            
            # Add cluster labels to dataframe
            self.df['cluster'] = self.cluster_labels
            
            # Load Similarity Matrix
            similarity_data = np.load(f"{self.models_dir}/similarity_matrices.npz")
            self.weighted_similarity = similarity_data['weighted']
            
            print("All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def clean_ingredients(self, text):
        """Preprocess ingredients text to match training format"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\b(tbsp|tsp|cup|cups|gram|g|kg)\b', '', text)
        text = re.sub(r'[^\w\s,]', '', text).strip()
        return text.split(',')[0].strip()
    
    def recommend(self, dish_data, top_n=5, state_filter=None, max_calories=None, exclude_allergies=None):
        """
        Get dish recommendations based on input dish
        
        Args:
            dish_data (dict): {
                'name': str,
                'ingredients': str,
                'spice_level': float (0-1),
                'sweet_level': float (0-1),
                'salty_level': float (0-1),
                'sour_level': float (0-1),
                'bitter_level': float (0-1)
            }
            top_n: Number of recommendations to return
            state_filter: Only recommend dishes from this state (optional)
            max_calories: Maximum calories per 100g (optional)
            exclude_allergies: Comma-separated list of allergens to exclude (optional)
        """
        try:
            # Preprocess and encode the new dish
            ingredients_clean = self.clean_ingredients(dish_data['ingredients'])
            
            # Get embeddings
            name_embed = self.bert.encode([dish_data['name'].lower()])
            ing_embed = self.bert.encode([ingredients_clean])
            
            # Scale taste features
            taste = np.array([
                dish_data['spice_level'],
                dish_data['sweet_level'],
                dish_data['salty_level'],
                dish_data['sour_level'],
                dish_data['bitter_level']
            ]).reshape(1, -1)
            taste_scaled = self.taste_scaler.transform(taste)
            
            # Combine features and reduce dimensions
            combined = np.concatenate([ing_embed, taste_scaled], axis=1)
            encoded = self.encoder.predict(combined)
            reduced = self.pca.transform(encoded)
            
            # Find target cluster
            if self.best_algo in ['hierarchical', 'ensemble', 'weighted_ensemble']:
                # For these algorithms, find the most similar existing dish's cluster
                all_reduced = self.pca.transform(self.encoder.predict(
                    np.concatenate([
                        self.bert.encode(self.df['clean_ingredients'].tolist()),
                        self.taste_scaler.transform(
                            self.df[['spice_level','sweet_level','salty_level','sour_level','bitter_level']]
                        )
                    ], axis=1)
                ))
                distances = np.linalg.norm(all_reduced - reduced, axis=1)
                closest_idx = np.argmin(distances)
                target_cluster = self.cluster_labels[closest_idx]
            else:
                # For other algorithms, predict cluster directly
                if self.best_algo == 'dbscan':
                    target_cluster = self.cluster_model.fit_predict(reduced)[0]
                else:
                    target_cluster = self.cluster_model.predict(reduced)[0]
            
            # Get candidate dishes from same cluster
            candidates = self.df[self.df['cluster'] == target_cluster].copy()
            
            # Apply filters
            if state_filter and state_filter != "All":
                candidates = candidates[candidates['state'].str.lower() == state_filter.lower()]
            if max_calories and 'calories_per_100g' in candidates.columns:
                candidates = candidates[candidates['calories_per_100g'] <= max_calories]
            if exclude_allergies:
                allergens = [a.strip().lower() for a in exclude_allergies.split(',')]
                for allergen in allergens:
                    candidates = candidates[~candidates['ingredients'].str.lower().str.contains(allergen)]
            
            # Calculate similarities if we have candidates
            if len(candidates) > 0:
                # Get candidate features
                candidate_features = np.concatenate([
                    self.bert.encode(candidates['clean_ingredients'].tolist()),
                    self.taste_scaler.transform(
                        candidates[['spice_level','sweet_level','salty_level','sour_level','bitter_level']]
                    )
                ], axis=1)
                
                # Encode and reduce candidate features
                candidate_encoded = self.encoder.predict(candidate_features)
                candidate_reduced = self.pca.transform(candidate_encoded)
                
                # Calculate similarities
                similarities = cosine_similarity(reduced, candidate_reduced)[0]
                candidates['similarity'] = similarities
                
                # Sort by similarity and get top N
                candidates = candidates.sort_values('similarity', ascending=False)
                top_n = min(top_n, len(candidates))
                results = candidates.head(top_n).to_dict('records')
                
                # Format results
                for rec in results:
                    rec['similarity'] = float(rec['similarity'])
                    rec['cluster'] = int(target_cluster)
                    if 'calories_per_100g' in rec:
                        rec['calories'] = float(rec['calories_per_100g'])
                    else:
                        rec['calories'] = None
            else:
                results = []
            
            return results
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return []

def generate_comprehensive_dish_details(dishes):
    """Generate comprehensive details for multiple dishes using Gemini API"""
    try:
        # Configure Gemini API
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare a structured prompt for detailed dish information
        prompt = """For each dish, provide a comprehensive JSON-formatted description with the following structure:
        {
            "dish_name": {
                "description": "Detailed culinary description",
                "ingredients": ["ingredient1", "ingredient2", ...],
                "taste_profile": {
                    "spicy": 0-10 score,
                    "sweet": 0-10 score,
                    "sour": 0-10 score,
                    "bitter": 0-10 score,
                    "umami": 0-10 score
                },
                "state": "Origin state",
                "cooking_method": "Traditional preparation method",
                "health_benefits": "Nutritional and health insights",
                "serving_suggestions": "Traditional serving recommendations",
                "cultural_significance": "Historical or cultural context"
            }
        }

        Dishes to describe:
        """ + "\n".join([f"- {dish['name']} from {dish['state']}" for dish in dishes])

        # Generate response
        response = model.generate_content(prompt)
        
        # Parse the JSON response
        try:
            dish_details = json.loads(response.text)
            return dish_details
        except json.JSONDecodeError:
            # Fallback parsing if JSON is not clean
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                try:
                    dish_details = json.loads(json_match.group(0))
                    return dish_details
                except:
                    st.error("Could not extract dish details.")
                    return {}
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return {}

def create_taste_radar_chart(dish_names, taste_data):
    """Create interactive radar charts for individual and combined taste profiles"""
    taste_categories = ['spicy', 'sweet', 'sour', 'bitter', 'umami']
    
    # Create individual radar charts
    individual_figs = []
    for name, data in taste_data.items():
        taste_values = [
            data.get('spice_level', 0) * 6,
            data.get('sweet_level', 0) * 6,
            data.get('sour_level', 0) * 6,
            data.get('bitter_level', 0) * 6,
            data.get('umami_level', 0) * 6
        ]
        
        fig = go.Figure(go.Scatterpolar(
            r=taste_values,
            theta=taste_categories,
            fill='toself',
            name=name
        ))
        
        fig.update_layout(
            title=f'Taste Profile: {name}',
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 6])
            )
        )
        
        individual_figs.append(fig)
    
    # Create combined radar chart
    combined_fig = go.Figure()
    
    for name, data in taste_data.items():
        taste_values = [
            data.get('spice_level', 0) * 6,
            data.get('sweet_level', 0) * 6,
            data.get('sour_level', 0) * 6,
            data.get('bitter_level', 0) * 6,
            data.get('umami_level', 0) * 6
        ]
        
        combined_fig.add_trace(go.Scatterpolar(
            r=taste_values,
            theta=taste_categories,
            fill='toself',
            name=name
        ))
    
    combined_fig.update_layout(
        title='Combined Taste Profile Comparison',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 6])
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return individual_figs, combined_fig

def create_calorie_comparison(dish_names, calorie_data):
    """Create bar chart comparing calories, excluding original dish"""
    fig = go.Figure(data=[
        go.Bar(
            x=dish_names,  # Use all dish names
            y=calorie_data,  # Use all calorie data 
            marker_color=px.colors.qualitative.Plotly
        )
    ])
    
    fig.update_layout(
        title='Calorie Comparison of Recommended Dishes',
        xaxis_title='Dish',
        yaxis_title='Calories per 100g',
        yaxis_range=[0, max(calorie_data) * 1.1]
    )
    
    return fig

def create_geographical_distribution(states):
    """Create pie chart for state distribution"""
    state_counts = pd.Series(states).value_counts()
    
    fig = px.pie(
        values=state_counts.values, 
        names=state_counts.index, 
        title='Recommended Dishes State Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    return fig

def modify_streamlit_ui():
    """Modify Streamlit UI to match the design"""
    # Dark theme custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }
    .stTextInput>div>div>input, 
    .stSelectbox>div>div>select {
        color: white;
        background-color: #1E1E1E;
        border: 1px solid #333;
    }
    .stSlider>div>div>div>div {
        background-color: #333 !important;
    }
    .stButton>button {
        background-color: #FF4500;
        color: white;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6347;
    }
    /* Sidebar styling */
    .css-1aumxhk {
        background-color: #1E1E1E;
    }
    /* Analytics card styling */
    .stExpander {
        border: 1px solid #333;
        background-color: #1E1E1E;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar setup
    st.sidebar.markdown("<h1>FoodTrails 🍽️</h1>", unsafe_allow_html=True)
    
    # Add "About" and "How to use" sections in sidebar
    st.sidebar.markdown("## About")
    st.sidebar.markdown("""
    This application uses a hybrid clustering approach to recommend Indian dishes 
    based on similarity metrics. It considers regional information, taste profiles, 
    and nutritional content to provide personalized recommendations.
    """)
    
    st.sidebar.markdown("## How to use")
    st.sidebar.markdown("""
    1. Select a dish from the dropdown menu
    2. Optionally specify a state or maximum calories
    3. Click "Get Recommendations" to see similar dishes
    4. Click on any dish card to see detailed information
    5. Explore the analytics to understand taste profiles
    """)
    
    st.sidebar.markdown("## Data Sources")
    st.sidebar.markdown("""
    - Dish data from comprehensive Indian cuisine database
    - Nutritional information from verified sources
    - State-specific culinary information from cultural repositories
    """)

def main():
    # Set page configuration
    st.set_page_config(page_title="FoodTrails", page_icon="🍽️", layout="wide")
    
    # Modify UI
    modify_streamlit_ui()
    
    # Initialize session state for recommender
    if 'recommender' not in st.session_state:
        st.session_state.recommender = DishRecommender()
    
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    
    if 'detailed_descriptions' not in st.session_state:
        st.session_state.detailed_descriptions = {}
    
    # Create page
    st.title("FoodTrails")
    
    # Dish selection
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        dish_names = st.session_state.recommender.df['name'].unique().tolist()
        selected_dish = st.selectbox("Select a dish:", dish_names)
    
    with col2:
        states = ["All"] + list(st.session_state.recommender.df['state'].unique())
        selected_state = st.selectbox("State (optional):", states)
    
    with col3:
        max_calories = st.slider("Max calories/100g", 0, 1000, 500)
    
    # Number of results
    num_results = st.slider("Number of results", 1, 12, 5)
    
    # Get selected dish details
    selected_dish_data = st.session_state.recommender.df[st.session_state.recommender.df['name'] == selected_dish].iloc[0].to_dict()
    
    # Prepare dish data for recommendation
    dish_recommendation_data = {
        'name': selected_dish,
        'ingredients': selected_dish_data['ingredients'],
        'spice_level': selected_dish_data['spice_level'],
        'sweet_level': selected_dish_data['sweet_level'],
        'salty_level': selected_dish_data['salty_level'],
        'sour_level': selected_dish_data['sour_level'],
        'bitter_level': selected_dish_data['bitter_level']
    }
    
    # Recommendation button
    if st.button("Get Recommendations", key="recommend_btn"):
        # Get recommendations
        recommendations = st.session_state.recommender.recommend(
            dish_recommendation_data,
            top_n=num_results,
            state_filter=selected_state,
            max_calories=max_calories
        )
        
        # Store recommendations in session state
        st.session_state.recommendations = recommendations
        
        # Fetch comprehensive details for all recommended dishes
        if recommendations:
            st.session_state.detailed_descriptions = generate_comprehensive_dish_details(recommendations)
    
    # Display recommendations if available
    if st.session_state.recommendations:
        st.subheader(f"Recommendations for {selected_dish}")
        
        # Create columns for recommendations
        recommendation_cols = st.columns(len(st.session_state.recommendations))
        
        for i, rec in enumerate(st.session_state.recommendations):
            with recommendation_cols[i]:
                st.markdown(f"""
                <div style="background-color: #2C2C2C; padding: 15px; border-radius: 10px; color: white;">
                <h3>{rec['name']}</h3>
                <p><strong>State:</strong> {rec['state']}</p>
                <p><strong>Similarity:</strong> {rec['similarity']:.2f}</p>
                <p><strong>Calories:</strong> {rec.get('calories', 'N/A')} per 100g</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Expandable Details Section
        st.header("Dish Details")
        for rec in st.session_state.recommendations:
            dish_name = rec['name']
            # Get details from pre-fetched descriptions
            dish_details = st.session_state.detailed_descriptions.get(dish_name, {})
            
            with st.expander(f"Details: {dish_name}"):
                if dish_details:
                    # Taste Profile
                    st.subheader("Taste Profile")
                    taste_profile = dish_details.get('taste_profile', {})
                    taste_cols = st.columns(5)
                    taste_labels = ['Spicy', 'Sweet', 'Sour', 'Bitter', 'Umami']
                    for col, label, key in zip(taste_cols, taste_labels, 
                                               ['spicy', 'sweet', 'sour', 'bitter', 'umami']):
                        col.metric(label, taste_profile.get(key, 'N/A'))
                    
                    # Description
                    st.subheader("Description")
                    st.write(dish_details.get('description', 'No description available.'))
                    
                    # Ingredients
                    st.subheader("Ingredients")
                    st.write(", ".join(dish_details.get('ingredients', [])))
                    
                    # Cooking Method
                    st.subheader("Cooking Method")
                    st.write(dish_details.get('cooking_method', 'No method details available.'))
                    
                    # Health Benefits
                    st.subheader("Health Benefits")
                    st.write(dish_details.get('health_benefits', 'No health benefits specified.'))
                    
                    # Serving Suggestions
                    st.subheader("Serving Suggestions")
                    st.write(dish_details.get('serving_suggestions', 'No serving suggestions available.'))
                    
                    # Cultural Significance
                    st.subheader("Cultural Significance")
                    st.write(dish_details.get('cultural_significance', 'No cultural context provided.'))
                else:
                    st.write("Detailed information not available.")
        
        # Dish Analytics Section
        st.header("Dish Analytics")

        # Prepare data for charts
        recommendation_names = [rec['name'] for rec in st.session_state.recommendations]
        dish_names = recommendation_names

        # Taste data
        taste_data = {}
        calorie_data = []
        states = []

        for dish_name in dish_names:
            # Find dish in recommendations
            dish_data = next((rec for rec in st.session_state.recommendations if rec['name'] == dish_name), None)
            
            if dish_data:
                taste_data[dish_name] = {
                    'spice_level': dish_data.get('spice_level', 0),
                    'sweet_level': dish_data.get('sweet_level', 0),
                    'sour_level': dish_data.get('sour_level', 0),
                    'bitter_level': dish_data.get('bitter_level', 0),
                    'umami_level': dish_data.get('umami_level', 0)
                }
                calorie_data.append(dish_data.get('calories', 0))
                states.append(dish_data.get('state', 'Unknown'))

        # Taste Profile Radar Charts
        st.subheader("Individual Taste Profiles")
        individual_taste_figs, combined_taste_fig = create_taste_radar_chart(dish_names, taste_data)

        # Display individual taste profile charts
        for fig in individual_taste_figs:
            st.plotly_chart(fig, use_container_width=True)

        # Display combined taste profile chart
        st.subheader("Combined Taste Profile Comparison")
        st.plotly_chart(combined_taste_fig, use_container_width=True)

        # Calorie Comparison
        st.subheader("Calorie Comparison")
        calorie_fig = create_calorie_comparison(dish_names, calorie_data)
        st.plotly_chart(calorie_fig, use_container_width=True)

        # Geographical Distribution
        st.subheader("Geographical Distribution")
        geo_fig = create_geographical_distribution(states)
        st.plotly_chart(geo_fig, use_container_width=True)

if __name__ == "__main__":
    main()