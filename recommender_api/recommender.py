import requests
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
import sqlite3


OMDB_API_KEY = "d28146e9"

# --- Load data from SQLite ---
conn = sqlite3.connect("movies.db")
movies = pd.read_sql("SELECT * FROM movies", conn)
ratings = pd.read_sql("SELECT * FROM ratings", conn)
tags = pd.read_sql("SELECT * FROM tags", conn)
links = pd.read_sql("SELECT * FROM links", conn)
conn.close()

# --- Preprocess content ---
tag_txt = tags.groupby("movieId")["tag"].agg(" ".join)
movies = movies.merge(tag_txt, on="movieId", how="left")
movies["content"] = movies["genres"].str.replace('|', ' ', regex=False) + ' ' + movies["tag"].fillna('')

# --- Time-aware weights ---
HALF_LIFE_DAYS = 730
lambda_ = np.log(2) / HALF_LIFE_DAYS
now = ratings["timestamp"].max()
delta_days = (now - ratings["timestamp"]) / 86400
ratings["weight"] = np.exp(-lambda_ * delta_days)

# --- Build user-item matrix ---
user_item = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
user_idx_map = {uid: idx for idx, uid in enumerate(user_item.index)}
movie_idx_map = {mid: idx for idx, mid in enumerate(user_item.columns)}
idx_user_map = {idx: uid for uid, idx in user_idx_map.items()}
idx_movie_map = {idx: mid for mid, idx in movie_idx_map.items()}

user_item_sparse = sparse.csr_matrix(user_item.values)

# --- Matrix Factorization with TruncatedSVD ---
n_factors = 64
svd = TruncatedSVD(n_components=n_factors, random_state=42)
user_factors = svd.fit_transform(user_item_sparse)
item_factors = svd.components_.T
cf_pred_matrix = np.dot(user_factors, item_factors.T)


# Filter movies that are present in user_item matrix (rated movies)
rated_movie_ids = user_item.columns
rated_movies = movies[movies["movieId"].isin(rated_movie_ids)].copy()
rated_movies = rated_movies.reset_index(drop=True)

# Map movieId to position
movieid_to_index = {mid: i for i, mid in enumerate(rated_movies["movieId"])}
index_to_movieid = {i: mid for mid, i in movieid_to_index.items()}

# TF-IDF on filtered movies
tfidf = TfidfVectorizer(stop_words="english")
content_mat = tfidf.fit_transform(rated_movies["content"])
cb_sim = cosine_similarity(content_mat, dense_output=False)


# --- Train hybrid meta-model (Ridge Regression) ---
rows = []
for uidx in range(user_factors.shape[0]):
    for iidx, mid in enumerate(rated_movie_ids):  # iidx now matches cb_sim
        if user_item_sparse[uidx, iidx] > 0:
            cf_score = cf_pred_matrix[uidx, iidx]
            cb_score = cb_sim[iidx].dot(user_item_sparse[uidx].T).sum()
            true_rating = user_item_sparse[uidx, iidx]
            rows.append((cf_score, cb_score, true_rating))

df_meta = pd.DataFrame(rows, columns=["cf", "cb", "true"])
meta = Ridge(alpha=1.0)
meta.fit(df_meta[["cf", "cb"]].values, df_meta["true"].values)

# --- Recommend function ---
def recommend(user_id: int, top_n: int = 20):
    if user_id not in user_idx_map:
        raise KeyError("User ID not found")

    uidx = user_idx_map[user_id]
    seen = user_item_sparse[uidx].nonzero()[1]

    cf_scores = cf_pred_matrix[uidx]
    cb_scores = cb_sim.dot(user_item_sparse[uidx].T).toarray().ravel()

    hybrid_scores = meta.predict(np.vstack([cf_scores, cb_scores]).T)
    hybrid_scores[seen] = -np.inf  # mask already watched

    top_indices = np.argpartition(-hybrid_scores, top_n)[:top_n]
    top_scores = hybrid_scores[top_indices]
    top_movie_ids = [user_item.columns[i] for i in top_indices]

    top_movies = movies[movies["movieId"].isin(top_movie_ids)][["movieId", "title"]]
    top_movies["score"] = top_scores
    top_movies = top_movies.merge(links[["movieId", "imdbId"]], on="movieId", how="left")
    top_movies = top_movies.sort_values("score", ascending=False)

    recommendations = []
    for _, row in top_movies.iterrows():
        imdb_id = row['imdbId']
        poster_url = None

        if pd.notnull(imdb_id):
            imdb_str = f"tt{int(imdb_id):07d}"
            try:
                res = requests.get(f"https://www.omdbapi.com/?i={imdb_str}&apikey={OMDB_API_KEY}")
                if res.ok:
                    data = res.json()
                    poster_url = data.get('Poster') if data.get('Poster') != "N/A" else None
            except Exception:
                pass

        recommendations.append({
            "title": row['title'],
            "imdbId": f"tt{int(imdb_id):07d}" if pd.notnull(imdb_id) else None,
            "poster_url": poster_url
        })

    return recommendations
