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

# Load data from URL
@st.cache_data
def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

# Load and preprocess datasets
@st.cache_resource
def load_data():
    # Load datasets
    with st.spinner("Loading movies dataset..."):
        movies = load_data_from_url(MOVIES_URL)
    
    with st.spinner("Loading credits dataset..."):
        credits = load_data_from_url(CREDITS_URL)
    
    with st.spinner("Loading ratings dataset..."):
        ratings = load_data_from_url(RATINGS_URL)

    # Filter movies with at least 50 votes
    movies = movies[movies['vote_count'] > 50]

    # Convert IDs to numeric
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')

    # Drop missing IDs
    movies = movies.dropna(subset=['id'])
    credits = credits.dropna(subset=['id'])
    ratings = ratings.dropna(subset=['movieId'])

    movies['id'] = movies['id'].astype(int)
    credits['id'] = credits['id'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)

    # Merge datasets
    movies = movies.merge(credits, on='id', how='inner')
    
    # Clean features
    movies['overview'] = movies['overview'].fillna('')
    movies['tagline'] = movies['tagline'].fillna('')
    movies['description'] = movies['overview'] + " " + movies['tagline']

    # Keep only needed columns
    movies = movies[['id','title','description','genres','cast','crew']]

    # Parse JSON-like fields
    def parse_genres(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj)]
        except:
            return []

    def parse_cast(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj)[:3]]  # top 3 actors
        except:
            return []

    def parse_crew(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']
        except:
            return []

    movies['genres'] = movies['genres'].apply(parse_genres)
    movies['cast'] = movies['cast'].apply(parse_cast)
    movies['crew'] = movies['crew'].apply(parse_crew)

    # Convert lists → strings
    movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))
    movies['cast'] = movies['cast'].apply(lambda x: " ".join(x))
    movies['crew'] = movies['crew'].apply(lambda x: " ".join(x))

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
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
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

    # Create user-item matrix
    user_item_matrix = ratings.pivot_table(
        index='user_name', columns='title', values='rating'
    ).fillna(0)

    # Compute user similarity
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    return ratings, user_item_matrix, user_sim_df

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
    top_users = sim_scores.index[:5]

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
        random_id = random.randint(6, 671)
        user_name = f"User_{random_id}"
        
    collab_scores = collaborative_recommend(user_name, user_item_matrix, user_sim_df, top_n=50)

    if liked_movie not in movies['title'].values:
        return user_name, [("❌ Movie not found", 0.0)]

    idx = movies.index[movies['title'] == liked_movie][0]
    cs = list(enumerate(content_similarity[idx]))
    cs = sorted(cs, key=lambda x: x[1], reverse=True)
    content_scores = {movies.iloc[i].title: s for i, s in cs[1:51]}

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
    
    # Load data
    with st.spinner("Loading data and building models..."):
        movies, ratings = load_data()
        tfidf, vectors, content_similarity = create_tfidf_model(movies)
        ratings, user_item_matrix, user_sim_df = prepare_collaborative_data(movies, ratings)
    
    st.header("Movie Recommendations")
    
    # Recommendation type selection
    rec_type = st.radio("Recommendation Type", ["Content-Based", "Hybrid"])
    
    # User selection (only for hybrid)
    user_choices = ["-", "Bob", "Alice", "Charlie", "Diana", "Eve"]
    if rec_type == "Hybrid":
        user_choice = st.selectbox("Select Movie Critic", user_choices)
    else:
        user_choice = "-"
    
    # Movie selection
    movie_choices = sorted(movies['title'].unique().tolist())
    movie_title = st.selectbox("Select a Movie", movie_choices)
    
    # Number of recommendations
    num_recs = st.slider("Number of Recommendations", 1, 20, 10)
    
    # Get recommendations button
    if st.button("Get Recommendations"):
        if not movie_title or movie_title.strip() == "":
            st.error("❌ Please select a movie first.")
        else:
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
