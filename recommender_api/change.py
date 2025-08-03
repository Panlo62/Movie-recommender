import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge

#Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")

# Merge tags into one text field per movie
tag_txt = tags.groupby("movieId")["tag"].agg(" ".join)
movies = movies.merge(tag_txt, on="movieId", how="left")


# Create content string (genres + tags)
movies['content'] = movies['genres'].str.replace('|',' ') + ' ' + movies['tag'].fillna('')


# Time-aware Weights
HALF_LIFE_DAYS = 730   # ~2 years
lambda_ = np.log(2) / HALF_LIFE_DAYS

now = ratings['timestamp'].max()           # latest interaction in the file
delta_days = (now - ratings['timestamp']) / 86_400  # seconds → days
ratings['weight'] = np.exp(-lambda_ * delta_days)


# Build Dataset and Interaction Matrix
ds = Dataset()
ds.fit(ratings["userId"].unique(), ratings["movieId"].unique())
(interactions, _) = ds.build_interactions(
    (u, i, r) for u, i, r in ratings[["userId", "movieId", "rating"]].itertuples(index=False)
)

uid_map, uf_map, iid_map, if_map = ds.mapping()
rows = ratings["userId"].map(uid_map).values
cols = ratings["movieId"].map(iid_map).values
data = ratings["weight"].values
mask = (~pd.isna(rows)) & (~pd.isna(cols))
rows, cols, data = rows[mask].astype(int), cols[mask].astype(int), data[mask]
sample_weight = sparse.coo_matrix((data, (rows, cols)), shape=interactions.shape)


# Train Collaborative Filtering Model (LightFM)
model_cf = LightFM(no_components=64, loss='warp')
model_cf.fit(interactions, sample_weight=sample_weight, epochs=20, num_threads=4)


# Content-Based Filtering (TF-IDF over genres + tags)
tfidf = TfidfVectorizer(stop_words="english")
content_mat = tfidf.fit_transform(movies["content"])
sim_mat = cosine_similarity(content_mat, dense_output=False)



# Meta-model: Learn hybrid weights using validation set
val_r = ratings

uid_map, iid_map = ds.mapping()[0], ds.mapping()[2]
def inner_uid(u): return uid_map[u]
def inner_iid(i): return iid_map[i]

val_rows = []
for u, i, r in val_r[['userId','movieId','rating']].itertuples(index=False):
    if u in uid_map and i in iid_map:
        uid, iid = inner_uid(u), inner_iid(i)
        cf_score  = model_cf.predict(uid, np.array([iid]))[0]
        cb_score  = sim_mat[iid].max()
        val_rows.append((cf_score, cb_score, r))

val_df = pd.DataFrame(val_rows, columns=['cf','cb','true'])

meta = Ridge(alpha=1.0)
meta.fit(val_df[['cf', 'cb']], val_df['true'])


# Recommend Function
def recommend(user_id: int, top_n: int = 10):
    if user_id not in uid_map:
        raise KeyError(f"user_id {user_id} not found")

    uid = uid_map[user_id]
    user_interactions = interactions.tocsr()[uid]
    cf_preds = model_cf.predict(uid, np.arange(content_mat.shape[0]))
    cb_preds = sim_mat.dot(user_interactions.T).toarray().ravel()
    scores = meta.coef_[0] * cf_preds + meta.coef_[1] * cb_preds + meta.intercept_

    seen = user_interactions.indices
    scores[seen] = -np.inf

    top_idx = np.argpartition(-scores, top_n)[:top_n]
    return movies.iloc[top_idx][['title']].assign(score=scores[top_idx]).sort_values('score', ascending=False)