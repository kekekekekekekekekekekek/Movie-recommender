# -*- coding: utf-8 -*-
"""Movie Recommender System"""

# Cell 1: Setup
import os

# Try to get the current script directory; fallback to current working directory
try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = os.getcwd()

DATA_PATH = os.path.join(BASE_DIR, "data")

# Cell 2: Import libraries
import pandas as pd
import numpy as np
import ast
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.metrics import mean_squared_error
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Cell 3: Load and preprocess datasets
os.makedirs(DATA_PATH, exist_ok=True)

def download_file(url, dest_path):
    """Download a file from a URL if it doesn't exist locally."""
    if not os.path.exists(dest_path):
        st.write(f"⬇️ Downloading {os.path.basename(dest_path)}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.write(f"✅ Saved to {dest_path}")

MOVIES_URL = "https://drive.google.com/uc?export=download&id=1GOuUEu1-KgepbjTxIOkbAU8VNJ5lfEg3"
CREDITS_URL = "https://drive.google.com/uc?export=download&id=10iuK9C87fYLyDLJhqT3bpVv1A2IErmHR"
RATINGS_URL = "https://drive.google.com/uc?export=download&id=122XJoryYXvv3AUa6F_y1KiCcYdXQjEp4"

movies_path = os.path.join(DATA_PATH, "movies_metadata.csv")
credits_path = os.path.join(DATA_PATH, "credits_small.csv")  # ✅ small file
ratings_path = os.path.join(DATA_PATH, "ratings_small.csv")

download_file(MOVIES_URL, movies_path)
download_file(CREDITS_URL, credits_path)
download_file(RATINGS_URL, ratings_path)

movies = pd.read_csv(movies_path, low_memory=False)
credits = pd.read_csv(credits_path)
ratings = pd.read_csv(ratings_path)

# Load
movies = pd.read_csv(movies_path, low_memory=False)
credits = pd.read_csv(credits_path)
ratings = pd.read_csv(ratings_path)

# ✅ Fix ids before merging
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna(subset=['id'])
movies['id'] = movies['id'].astype(int)

# Merge works now
movies = movies.merge(credits, on="id", how="inner")

# Cell 4: Clean and prepare features
movies['overview'] = movies['overview'].fillna('')
movies['tagline'] = movies['tagline'].fillna('')
movies['description'] = movies['overview'] + " " + movies['tagline']

movies = movies[['id','title','description','genres','cast','crew']]

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

movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))
movies['cast'] = movies['cast'].apply(lambda x: " ".join(x))
movies['crew'] = movies['crew'].apply(lambda x: " ".join(x))

movies['final_features'] = (
    movies['description'] + ' ' +
    movies['genres'] + ' ' +
    movies['cast'] + ' ' +
    movies['crew']
)

# Cell 5: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
vectors = tfidf.fit_transform(movies['final_features'])
content_similarity = cosine_similarity(vectors)

# Cell 6: Prepare collaborative filtering data
movies_cf = movies[['id', 'title']].copy()
movies_cf = movies_cf.rename(columns={'id': 'movieId'})
ratings = ratings.merge(movies_cf, on="movieId", how="inner")

user_mapping = {1: "Bob", 2: "Alice", 3: "Charlie", 4: "Diana", 5: "Eve"}
ratings['user_name'] = ratings['userId'].replace(user_mapping)

user_item_matrix = ratings.pivot_table(
    index='user_name', columns='title', values='rating'
).fillna(0)

user_sim = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

# Cell 7: Recommendation functions
def content_based_recommend(movie_title, top_n=10):
    if movie_title not in movies['title'].values:
        return "❌ Movie not found in dataset.", []
    idx = movies[movies['title'] == movie_title].index[0]
    cosine_scores = linear_kernel(vectors[idx], vectors).flatten()
    similar_indices = cosine_scores.argsort()[-(top_n+1):-1][::-1]
    recommendations = movies.iloc[similar_indices].title.tolist()
    scores = cosine_scores[similar_indices].tolist()
    return "Content-based recommendations", [(title, float(score)) for title, score in zip(recommendations, scores)]

def collaborative_recommend(user_name, top_n=50):
    if user_name not in user_item_matrix.index:
        return {}
    sim_scores = user_sim_df[user_name].drop(user_name).sort_values(ascending=False)
    top_users = sim_scores.index[:5]
    neighbor_ratings = user_item_matrix.loc[top_users].mean(axis=0)
    watched = user_item_matrix.loc[user_name][user_item_matrix.loc[user_name] > 0].index
    neighbor_ratings = neighbor_ratings.drop(watched, errors='ignore')
    top_recs = neighbor_ratings.sort_values(ascending=False).head(top_n)
    return {title: score for title, score in top_recs.items()}

def hybrid_recommend(user_name, liked_movie, alpha=0.5, top_n=10):
    if not user_name or user_name.strip() == "" or user_name == "-":
        random_id = random.randint(6, 671)
        user_name = f"User_{random_id}"
        ratings['user_name'] = ratings['userId'].replace(user_mapping)
        ratings.loc[ratings['userId'] == random_id, 'user_name'] = user_name
        global user_item_matrix, user_sim_df
        user_item_matrix = ratings.pivot_table(index='user_name', columns='title', values='rating').fillna(0)
        user_sim = cosine_similarity(user_item_matrix)
        user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

    collab_scores = collaborative_recommend(user_name, top_n=50)
    if liked_movie not in movies['title'].values:
        return user_name, [("❌ Movie not found", 0.0)]

    idx = movies.index[movies['title'] == liked_movie][0]
    cs = list(enumerate(content_similarity[idx]))
    cs = sorted(cs, key=lambda x: x[1], reverse=True)
    content_scores = {movies.iloc[i].title: s for i, s in cs[1:51]}

    all_titles = set(collab_scores.keys()) | set(content_scores.keys())
    hybrid_scores = {t: alpha * content_scores.get(t, 0.0) + (1 - alpha) * collab_scores.get(t, 0.0) for t in all_titles}
    ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return user_name, [(t, float(s)) for t, s in ranked]

# Cell 9: Streamlit UI
def run_streamlit():
    st.title("🎬 Movie Recommender System")
    st.sidebar.header("Controls")
    rec_type = st.sidebar.radio("Recommendation Type", ["Content-Based", "Hybrid"])
    user_choice = None
    if rec_type == "Hybrid":
        user_choice = st.sidebar.selectbox("Select Movie Critic", ["-"] + ["Bob", "Alice", "Charlie", "Diana", "Eve"])
    movie_title = st.sidebar.selectbox("Select a Movie", sorted(movies['title'].unique().tolist()))
    num_recs = st.sidebar.slider("Number of Recommendations", 1, 20, 10)

    if st.sidebar.button("✨ Get Recommendations"):
        result = get_recommendations(rec_type, user_choice, movie_title, num_recs)
        if isinstance(result, str):
            st.error(result)
        else:
            st.markdown(result, unsafe_allow_html=True)

def get_recommendations(recommendation_type, user_choice, movie_title, num_recs):
    if not movie_title or movie_title.strip() == "":
        return "❌ Please select a movie first."
    if recommendation_type == "Content-Based":
        method_name, recs = content_based_recommend(movie_title, top_n=num_recs)
        resolved_user = "Content-Based Filtering"
    else:
        user_input = "" if (user_choice is None or user_choice == "-") else user_choice
        resolved_user, recs = hybrid_recommend(user_input, movie_title, alpha=0.5, top_n=num_recs)
        method_name = "Hybrid recommendations"
    if isinstance(recs, str) or len(recs) == 0:
        return recs if isinstance(recs, str) else "❌ No recommendations found."
    scores = [s for _, s in recs]
    max_score = max(scores) if scores else 1.0
    rows = []
    for i, (title, score) in enumerate(recs, start=1):
        norm_percentage = (score / max_score) * 100 if max_score > 0 else 0
        rows.append([f"{i}. {title}", f"{norm_percentage:.1f}%"])
    rows_html = "".join(
        f"<tr><td style='padding:6px 12px;border-bottom:1px solid #eee;'>{movie}</td>"
        f"<td style='padding:6px 12px;text-align:center;border-bottom:1px solid #eee;'>{score}</td></tr>"
        for movie, score in rows
    )
    table_html = (
        f"<div style='font-weight:600;margin:6px 0;'>🎭 {method_name} for {resolved_user} (based on {movie_title}):</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:15px;'>"
        "<thead><tr><th style='text-align:left;padding:6px 12px;border-bottom:2px solid #ccc;'>Movie</th>"
        "<th style='text-align:center;padding:6px 12px;border-bottom:2px solid #ccc;'>Score</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )
    return table_html

if __name__ == "__main__":
    import sys
    if any("streamlit" in arg for arg in sys.argv):
        run_streamlit()
