# streamlit_movie_recommender.py
import streamlit as st
import pandas as pd
import numpy as np
import ast
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import requests
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dataset URLs
MOVIES_URL = "https://drive.google.com/uc?export=download&id=1GOuUEu1-KgepbjTxIOkbAU8VNJ5lfEg3"
CREDITS_URL = "https://drive.google.com/uc?export=download&id=10iuK9C87fYLyDLJhqT3bpVv1A2IErmHR"
RATINGS_URL = "https://drive.google.com/uc?export=download&id=122XJoryYXvv3AUa6F_y1KiCcYdXQjEp4"

# Load data from URL with optimizations
@st.cache_data
def load_data_from_url(url, sample_size=None):
    response = requests.get(url)
    response.raise_for_status()
    
    # Read only the first few rows for faster loading during development
    if sample_size:
        # Read the first sample_size rows
        df = pd.read_csv(StringIO(response.text), nrows=sample_size)
    else:
        df = pd.read_csv(StringIO(response.text))
    
    return df

# Load and preprocess datasets with optimizations
@st.cache_resource
def load_data():
    # Load datasets with smaller sample size for faster loading
    with st.spinner("Loading movies dataset (sampled for speed)..."):
        movies = load_data_from_url(MOVIES_URL, sample_size=5000)  # Reduced sample size
    
    with st.spinner("Loading credits dataset (sampled for speed)..."):
        credits = load_data_from_url(CREDITS_URL, sample_size=5000)  # Reduced sample size
    
    with st.spinner("Loading ratings dataset (sampled for speed)..."):
        ratings = load_data_from_url(RATINGS_URL, sample_size=10000)  # Reduced sample size

    # Filter movies with at least 50 votes
    movies = movies[movies['vote_count'] > 50]

    # Convert IDs to numeric - faster approach
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce', downcast='integer')
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce', downcast='integer')
    ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce', downcast='integer')

    # Drop missing IDs
    movies = movies.dropna(subset=['id'])
    credits = credits.dropna(subset=['id'])
    ratings = ratings.dropna(subset=['movieId'])

    # Merge datasets - inner join for faster processing
    movies = movies.merge(credits, on='id', how='inner')
    
    # Clean features - simplified approach
    movies['overview'] = movies['overview'].fillna('')
    movies['tagline'] = movies['tagline'].fillna('')
    movies['description'] = movies['overview'] + " " + movies['tagline']

    # Keep only needed columns
    movies = movies[['id','title','description','genres','cast','crew']].copy()

    # Parse JSON-like fields with error handling
    def parse_json_field(obj, field_name='name', max_items=3):
        try:
            parsed = ast.literal_eval(obj)
            if isinstance(parsed, list):
                return [item[field_name] for item in parsed[:max_items]]
            return []
        except:
            return []

    movies['genres'] = movies['genres'].apply(lambda x: " ".join(parse_json_field(x)))
    movies['cast'] = movies['cast'].apply(lambda x: " ".join(parse_json_field(x)))
    movies['crew'] = movies['crew'].apply(lambda x: " ".join(parse_json_field(x, 'name', 1)))  # Only director

    # Combine final features
    movies['final_features'] = (
        movies['description'] + ' ' +
        movies['genres'] + ' ' +
        movies['cast'] + ' ' +
        movies['crew']
    )
    
    return movies, ratings

@st.cache_resource
def create_tfidf_model(movies):
    # TF-IDF Vectorization with reduced features for faster processing
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)  # Reduced from 5000 to 1000
    vectors = tfidf.fit_transform(movies['final_features'])
    content_similarity = cosine_similarity(vectors)
    return tfidf, vectors, content_similarity

@st.cache_resource
def prepare_collaborative_data(movies, ratings):
    # Prepare movies metadata for collaborative filtering
    movies_cf = movies[['id', 'title']].copy()
    movies_cf = movies_cf.rename(columns={'id': 'movieId'})

    # Merge ratings with movie titles
    ratings = ratings.merge(movies_cf, on="movieId", how="inner")

    # User mapping
    user_mapping = {
        1: "Bob",
        2: "Alice",
        3: "Charlie",
        4: "Diana",
        5: "Eve"
    }
    ratings['user_name'] = ratings['userId'].replace(user_mapping)

    # Create user-item matrix with only top users for faster processing
    top_users = ratings['user_name'].value_counts().head(10).index  # Only top 10 users
    ratings_filtered = ratings[ratings['user_name'].isin(top_users)]
    
    user_item_matrix = ratings_filtered.pivot_table(
        index='user_name', columns='title', values='rating'
    ).fillna(0)

    # Compute user similarity
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    return ratings_filtered, user_item_matrix, user_sim_df

# Content-based recommendation function
def content_based_recommend(movie_title, movies, content_similarity, top_n=10):
    if movie_title not in movies['title'].values:
        return "❌ Movie not found in dataset.", []

    idx = movies[movies['title'] == movie_title].index[0]

    # Compute similarity only for this movie
    cosine_scores = linear_kernel(vectors[idx], vectors).flatten()

    similar_indices = cosine_scores.argsort()[-(top_n+1):-1][::-1]
    recommendations = movies.iloc[similar_indices].title.tolist()
    scores = cosine_scores[similar_indices].tolist()

    return "Content-based recommendations", [(title, float(score)) for title, score in zip(recommendations, scores)]

# Collaborative filtering recommendation function
def collaborative_recommend(user_name, user_item_matrix, user_sim_df, top_n=50):
    if user_name not in user_item_matrix.index:
        return {}

    # Find similar users
    sim_scores = user_sim_df[user_name].drop(user_name).sort_values(ascending=False)
    top_users = sim_scores.index[:3]  # Reduced from 5 to 3 for speed

    # Average ratings of top neighbors
    neighbor_ratings = user_item_matrix.loc[top_users].mean(axis=0)

    # Remove already watched movies
    watched = user_item_matrix.loc[user_name][user_item_matrix.loc[user_name] > 0].index
    neighbor_ratings = neighbor_ratings.drop(watched, errors='ignore')

    # Get top recommendations
    top_recs = neighbor_ratings.sort_values(ascending=False).head(top_n)
    return {title: score for title, score in top_recs.items()}

# Hybrid recommendation function
def hybrid_recommend(user_name, liked_movie, movies, content_similarity, user_item_matrix, user_sim_df, alpha=0.5, top_n=10):
    if not user_name or user_name.strip() == "" or user_name == "-":
        # Use a random existing user instead of creating a new one
        user_name = random.choice(user_item_matrix.index.tolist())
        
    collab_scores = collaborative_recommend(user_name, user_item_matrix, user_sim_df, top_n=30)  # Reduced from 50

    if liked_movie not in movies['title'].values:
        return user_name, [("❌ Movie not found", 0.0)]

    idx = movies.index[movies['title'] == liked_movie][0]
    cs = list(enumerate(content_similarity[idx]))
    cs = sorted(cs, key=lambda x: x[1], reverse=True)
    content_scores = {movies.iloc[i].title: s for i, s in cs[1:31]}  # Reduced from 51

    # Merge both approaches
    all_titles = set(collab_scores.keys()) | set(content_scores.keys())
    hybrid_scores = {}
    for t in all_titles:
        c = content_scores.get(t, 0.0)
        cf = collab_scores.get(t, 0.0)
        hybrid_scores[t] = alpha * c + (1 - alpha) * cf

    ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return user_name, [(t, float(s)) for t, s in ranked]

# Main app
def main():
    st.title("🎬 Movie Recommender System")
    
    # Load data with progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading data...")
    movies, ratings = load_data()
    progress_bar.progress(33)
    
    status_text.text("Creating TF-IDF model...")
    tfidf, vectors, content_similarity = create_tfidf_model(movies)
    progress_bar.progress(66)
    
    status_text.text("Preparing collaborative data...")
    ratings, user_item_matrix, user_sim_df = prepare_collaborative_data(movies, ratings)
    progress_bar.progress(100)
    status_text.text("Ready!")
    
    st.header("Movie Recommendations")
    
    # Recommendation type selection
    rec_type = st.radio("Recommendation Type", ["Content-Based", "Hybrid"])
    
    # User selection (only for hybrid)
    user_choices = ["-"] + user_item_matrix.index.tolist()
    if rec_type == "Hybrid":
        user_choice = st.selectbox("Select Movie Critic", user_choices)
    else:
        user_choice = "-"
    
    # Movie selection with search
    movie_choices = sorted(movies['title'].unique().tolist())
    movie_title = st.selectbox("Select a Movie", movie_choices)
    
    # Number of recommendations
    num_recs = st.slider("Number of Recommendations", 1, 15, 5)  # Reduced max from 20 to 15
    
    # Get recommendations button
    if st.button("Get Recommendations"):
        if not movie_title or movie_title.strip() == "":
            st.error("❌ Please select a movie first.")
        else:
            with st.spinner("Generating recommendations..."):
                if rec_type == "Content-Based":
                    method_name, recs = content_based_recommend(movie_title, movies, content_similarity, top_n=num_recs)
                    resolved_user = "Content-Based Filtering"
                else:  # Hybrid
                    user_input = "" if (user_choice is None or user_choice == "-") else user_choice
                    resolved_user, recs = hybrid_recommend(
                        user_input, movie_title, movies, content_similarity, 
                        user_item_matrix, user_sim_df, alpha=0.5, top_n=num_recs
                    )
                    method_name = "Hybrid recommendations"

            # Check if we got an error message
            if isinstance(recs, str) or len(recs) == 0:
                st.error(recs if isinstance(recs, str) else "❌ No recommendations found.")
            else:
                # Extract scores for normalization
                scores = [s for _, s in recs]
                max_score = max(scores) if scores else 1.0

                # Normalize scores so the top one is 100%
                rows = []
                for i, (title, score) in enumerate(recs, start=1):
                    norm_percentage = (score / max_score) * 100 if max_score > 0 else 0
                    rows.append([f"{i}. {title}", f"{norm_percentage:.1f}%"])

                # Display results
                user_display = resolved_user if rec_type == "Hybrid" else "Content-Based Filtering"
                
                st.subheader(f"🎭 {method_name} for {user_display} (based on {movie_title}):")
                
                # Create a dataframe for better display
                rec_df = pd.DataFrame(rows, columns=["Movie", "Score"])
                st.table(rec_df)

if __name__ == "__main__":
    main()
