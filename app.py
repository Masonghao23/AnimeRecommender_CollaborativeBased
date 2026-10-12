import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    .stDecoration {display: none !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stApp > header {display: none !important;}
    .stApp > .main .block-container {padding-top: 1rem !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    .css-15zrgzn {display: none !important;}
    .css-eczf16 {display: none !important;}
    .css-jn99sy {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #4ECDC4;
    }
    
    .anime-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    
    .correlation-score {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .synopsis {
        color: #555;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .error-message {
        background: #FFE6E6;
        color: #D63031;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #D63031;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #E8F5E8;
        color: #00B894;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00B894;
        margin: 1rem 0;
    }
    
    .loading-message {
        background: #E3F2FD;
        color: #1976D2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976D2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the anime data"""
    try:
        # Load datasets
        anime = pd.read_csv('anime.csv', low_memory=False)
        rating = pd.read_csv('rating_complete.csv', low_memory=False)
        synopsis = pd.read_csv('anime_with_synopsis.csv', low_memory=False)
        
        if 'MAL_ID' in anime.columns:
            anime = anime.rename(columns={'MAL_ID': 'anime_id'})
        if 'MAL_ID' in synopsis.columns:
            synopsis = synopsis.rename(columns={'MAL_ID': 'anime_id'})
        
        return anime, rating, synopsis
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def preprocess_data(anime, rating, synopsis):
    """Preprocess data following the notebook logic"""
    try:
        # Create anime rating aggregation
        anime_rating = pd.DataFrame(
            rating.groupby('anime_id').agg(
                rating=('rating', 'mean'), 
                vote=('rating', 'sum')
            )
        ).reset_index()
        
        # Fill NaN values
        anime_rating['rating'] = anime_rating['rating'].fillna(0)
        
        # Filter qualified anime (top 20% by vote count)
        m = anime_rating['vote'].quantile(0.8)
        qualified_anime = anime_rating.copy().loc[anime_rating['vote'] >= m]
        
        # Filter ratings for qualified anime only
        filtered_rating = rating[rating['anime_id'].isin(qualified_anime['anime_id'])]
        
        # Create user-anime matrix
        user_anime_matrix = filtered_rating.pivot_table(
            index='user_id', 
            columns='anime_id', 
            values='rating'
        ).fillna(0)
        
        return anime, synopsis, user_anime_matrix, qualified_anime
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None, None, None

def get_recommendations(anime_name, anime, synopsis, user_anime_matrix, top_n=10):
    """Get anime recommendations based on collaborative filtering"""
    try:
        # Find anime by name (case-insensitive)
        matches = anime.loc[anime['Name'].str.lower() == anime_name.lower(), 'anime_id'].values
        
        if len(matches) == 0:
            # Try partial matching
            partial_matches = anime.loc[anime['Name'].str.contains(anime_name, case=False, na=False), 'anime_id'].values
            if len(partial_matches) == 0:
                return None, "Anime not found"
            else:
                anime_id = partial_matches[0]
                # Get the actual anime name for better user feedback
                actual_name = anime.loc[anime['anime_id'] == anime_id, 'Name'].iloc[0]
        else:
            anime_id = matches[0]
            actual_name = anime_name
        
        # Validate anime exists in database first
        if anime_id not in anime['anime_id'].values:
            return None, "Anime not found"
        
        # Check if anime is in the user-anime matrix (has enough ratings)
        if anime_id not in user_anime_matrix.columns:
            return None, "Anime not found"
        
        # Get user ratings for the selected anime
        users_ratings = user_anime_matrix[anime_id]
        
        # Calculate correlations with all other anime
        similar_animes = user_anime_matrix.corrwith(users_ratings)
        
        # Create DataFrame with correlations
        similar_animes = pd.DataFrame(similar_animes, columns=['correlation']).reset_index()
        
        # Remove NaN correlations and the input anime itself
        similar_animes = similar_animes.dropna()
        similar_animes = similar_animes[similar_animes['anime_id'] != anime_id]
        
        # Merge with anime names and synopsis
        df_name = anime[['anime_id', 'Name']]
        df_name = df_name[df_name['anime_id'].isin(similar_animes['anime_id'])]
        
        df_synopsis = synopsis[['anime_id', 'sypnopsis']]
        df_synopsis = df_synopsis[df_synopsis['anime_id'].isin(similar_animes['anime_id'])]
        
        # Merge all data
        df_anime = df_name.merge(df_synopsis, on='anime_id', how='left')
        similar_animes_with_name = similar_animes.merge(df_anime, on='anime_id', how='left')
        
        # Sort by correlation and get top N
        recommendations = similar_animes_with_name.sort_values(
            by=['correlation'], ascending=False
        ).head(top_n)
        
        return recommendations, None
        
    except Exception as e:
        return None, f"Error generating recommendations: {str(e)}"

def display_recommendations(recommendations):
    """Display recommendations in a modern card format"""
    for idx, row in recommendations.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="anime-title">{row['Name']}</div>
                <div class="correlation-score">Similarity: {row['correlation']:.3f}</div>
                <div class="synopsis">{row.get('sypnopsis', 'No synopsis available.')[:300]}{'...' if len(str(row.get('sypnopsis', ''))) > 300 else ''}</div>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎌 Anime Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover your next favorite anime using collaborative filtering</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading anime database...'):
        anime, rating, synopsis = load_data()
    
    if anime is None or rating is None or synopsis is None:
        st.error("Failed to load data. Please ensure all CSV files are in the correct directory.")
        return
    
    # Preprocess data
    with st.spinner('Processing data for recommendations...'):
        anime, synopsis, user_anime_matrix, qualified_anime = preprocess_data(anime, rating, synopsis)
    
    if user_anime_matrix is None:
        st.error("Failed to process data.")
        return
    
    # Sidebar with information
    with st.sidebar:
        st.header("📊 Dataset Information")
        st.metric("Total Anime", len(anime))
        st.metric("Qualified Anime", len(qualified_anime))
        st.metric("Total Ratings", len(rating))
        
        st.header("🔍 How it works")
        st.write("""
        This system uses **Item-Based Collaborative Filtering** to recommend anime:
        
        1. **Data Filtering**: Only anime with sufficient ratings (top 10% by vote count)
        2. **Similarity Calculation**: Uses correlation to find similar anime
        3. **Recommendations**: Shows anime most similar to your input
        """)
        
        st.header("💡 Tips")
        st.write("""
        - Try popular anime names like "Naruto", "One Piece", "Attack on Titan"
        - Use exact titles for best results
        - The system works best with well-known anime
        """)
    
    # Main interface
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Input section
        st.subheader("🔍 Find Similar Anime")
        
        # Anime name input
        anime_name = st.text_input(
            "Enter an anime name:",
            placeholder="e.g., Naruto, One Piece, Death Note...",
            help="Enter the name of an anime you like to get similar recommendations"
        )
        
        # show top 10 recommendations
        num_recommendations = 10
        
        # Search button
        if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
            if anime_name.strip():
                with st.spinner(f'Finding anime similar to "{anime_name}"...'):
                    recommendations, error = get_recommendations(
                        anime_name, anime, synopsis, user_anime_matrix, num_recommendations
                    )
                
                if error:
                    st.markdown(f'<div class="error-message">❌ {error}</div>', unsafe_allow_html=True)
                elif recommendations is not None and len(recommendations) > 0:
                    st.markdown(f'<div class="success-message">✅ Similar anime for \'{anime_name}\' found:</div>', unsafe_allow_html=True)
                    
                    st.subheader(f"🎌 Anime Similar to \"{anime_name}\"")
                    display_recommendations(recommendations)
                else:
                    st.markdown(f'<div class="error-message">❌ No similar anime found for "{anime_name}". This might happen if the anime has insufficient rating data or doesn\'t exist in our database. Try searching for popular anime titles like "Naruto", "One Piece", or "Attack on Titan".</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-message">❌ Please enter an anime name.</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Item-Based Collaborative Filtering</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()