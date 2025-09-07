import streamlit as st
import pandas as pd
import numpy as np
import ast
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests
from io import StringIO
import os
import pickle

# Streamlit page config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

MOVIES_URL = "https://drive.google.com/uc?export=download&id=1GOuUEu1-KgepbjTxIOkbAU8VNJ5lfEg3"
CREDITS_URL = "https://drive.google.com/uc?export=download&id=10iuK9C87fYLyDLJhqT3bpVv1A2IErmHR"
RATINGS_URL = "https://drive.google.com/uc?export=download&id=122XJoryYXvv3AUa6F_y1KiCcYdXQjEp4"

@st.cache_data
def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

@st.cache_resource
def load_data():
    with st.spinner("Loading datasets..."):
        movies = load_data_from_url(MOVIES_URL)
        credits = load_data_from_url(CREDITS_URL)
        ratings = load_data_from_url(RATINGS_URL)

    # Filter movies with more votes for faster dev
    movies = movies[movies['vote_count'] > 500]

    # Convert IDs
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    credits = credits.dropna(subset=['id'])
    ratings = ratings.dropna(subset=['movieId'])
    movies['id'] = movies['id'].astype(int)
    credits['id'] = credits['id'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)

    # Merge movies + credits
    movies = movies.merge(credits, on='id', how='inner')

    # Clean features
    movies['overview'] = movies['overview'].fillna('')
    movies['tagline'] = movies['tagline'].fillna('')
    movies['description'] = movies['overview'] + " " + movies['tagline']

    movies = movies[['id', 'title', 'description', 'genres', 'cast', 'crew']]

    # Parsing helpers
    def parse_genres(obj):
        try: return [i['name'] for i in ast.literal_eval(obj)]
        except: return []
    def parse_cast(obj):
        try: return [i['name'] for i in ast.literal_eval(obj)[:3]]
        except: return []
    def parse_crew(obj):
        try: return [i['name'] for i in ast.literal_eval(obj) if i['job']=='Director']
        except: return []

    movies['genres'] = movies['genres'].apply(parse_genres)
    movies['cast'] = movies['cast'].apply(parse_cast)
    movies['crew'] = movies['crew'].apply(parse_crew)

    # Convert lists → strings
    movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))
    movies['cast'] = movies['cast'].apply(lambda x: " ".join(x))
    movies['crew'] = movies['crew'].apply(lambda x: " ".join(x))

    movies['final_features'] = (
        movies['description'] + ' ' +
        movies['genres'] + ' ' +
        movies['cast'] + ' ' +
        movies['crew']
    )

    return movies, ratings

@st.cache_resource
def create_tfidf_model(movies, pickle_path="tfidf_model.pkl"):
    # If pickle exists, load it
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            tfidf, vectors = pickle.load(f)
    else:
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        vectors = tfidf.fit_transform(movies['final_features'])
        # Save to pickle
        with open(pickle_path, "wb") as f:
            pickle.dump((tfidf, vectors), f)
    return tfidf, vectors

@st.cache_resource
def prepare_collaborative_data(movies, ratings):
    movies_cf = movies[['id','title']].rename(columns={'id':'movieId'})
    ratings = ratings.merge(movies_cf, on="movieId", how="inner")

    user_mapping = {1: "Bob", 2: "Alice", 3: "Charlie", 4: "Diana", 5: "Eve"}
    ratings['user_name'] = ratings['userId'].replace(user_mapping)

    user_item_matrix = ratings.pivot_table(index='user_name', columns='title', values='rating').fillna(0)

    from sklearn.metrics.pairwise import cosine_similarity
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

    return ratings, user_item_matrix, user_sim_df

# Content-based
def content_based_recommend(movie_title, movies, vectors, top_n=10):
    if movie_title not in movies['title'].values:
        return "❌ Movie not found", []
    idx = movies[movies['title']==movie_title].index[0]
    scores = linear_kernel(vectors[idx], vectors).flatten()
    indices = scores.argsort()[-(top_n+1):-1][::-1]
    return "Content-based recommendations", [(movies.iloc[i].title, float(scores[i])) for i in indices]

# Collaborative
def collaborative_recommend(user_name, user_item_matrix, user_sim_df, top_n=50):
    if user_name not in user_item_matrix.index:
        return {}
    sim_scores = user_sim_df[user_name].drop(user_name).sort_values(ascending=False)
    top_users = sim_scores.index[:5]
    neighbor_ratings = user_item_matrix.loc[top_users].mean(axis=0)
    watched = user_item_matrix.loc[user_name][user_item_matrix.loc[user_name]>0].index
    neighbor_ratings = neighbor_ratings.drop(watched, errors='ignore')
    top_recs = neighbor_ratings.sort_values(ascending=False).head(top_n)
    return {title: score for title, score in top_recs.items()}

# Hybrid
def hybrid_recommend(user_name, liked_movie, movies, vectors, user_item_matrix, user_sim_df, alpha=0.5, top_n=10):
    if not user_name or user_name.strip()=="" or user_name=="-":
        user_name = f"User_{random.randint(6,671)}"
    collab_scores = collaborative_recommend(user_name, user_item_matrix, user_sim_df, top_n=50)
    if liked_movie not in movies['title'].values:
        return user_name, [("❌ Movie not found", 0.0)]
    idx = movies.index[movies['title']==liked_movie][0]
    cs = linear_kernel(vectors[idx], vectors).flatten()
    content_scores = {movies.iloc[i].title: float(cs[i]) for i in cs.argsort()[-51:-1]}
    all_titles = set(collab_scores.keys()) | set(content_scores.keys())
    hybrid_scores = {t: alpha*content_scores.get(t,0)+ (1-alpha)*collab_scores.get(t,0) for t in all_titles}
    ranked = sorted(hybrid_scores.items(), key=lambda x:x[1], reverse=True)[:top_n]
    return user_name, [(t,float(s)) for t,s in ranked]

def main():
    st.title("🎬 Movie Recommender System")

    # Load data and models
    movies, ratings = load_data()
    tfidf, vectors = create_tfidf_model(movies)
    ratings, user_item_matrix, user_sim_df = prepare_collaborative_data(movies, ratings)

    st.header("Movie Recommendations")

    rec_type = st.radio("Recommendation Type", ["Content-Based", "Hybrid"])
    user_choices = ["-", "Bob", "Alice", "Charlie", "Diana", "Eve"]
    user_choice = st.selectbox("Select Movie Critic", user_choices) if rec_type=="Hybrid" else "-"

    movie_choices = sorted(movies['title'].unique())
    movie_title = st.selectbox("Select a Movie", movie_choices)
    num_recs = st.slider("Number of Recommendations", 1, 20, 10)

    if st.button("Get Recommendations"):
        if not movie_title.strip():
            st.error("❌ Please select a movie first.")
            return
        if rec_type=="Content-Based":
            method_name, recs = content_based_recommend(movie_title, movies, vectors, num_recs)
            user_display = "Content-Based Filtering"
        else:
            user_input = "" if user_choice=="-" else user_choice
            user_display, recs = hybrid_recommend(user_input, movie_title, movies, vectors, user_item_matrix, user_sim_df, top_n=num_recs)
            method_name = "Hybrid recommendations"

        if isinstance(recs,str) or len(recs)==0:
            st.error(recs if isinstance(recs,str) else "❌ No recommendations found")
        else:
            max_score = max([s for _,s in recs]) if recs else 1.0
            rows = [[f"{i}. {title}", f"{(score/max_score)*100:.1f}%"] for i,(title,score) in enumerate(recs,start=1)]
            st.subheader(f"🎭 {method_name} for {user_display} (based on {movie_title}):")
            st.table(pd.DataFrame(rows, columns=["Movie","Score"]))

if __name__=="__main__":
    main()
