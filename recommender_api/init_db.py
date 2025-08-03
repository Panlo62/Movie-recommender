import pandas as pd
import sqlite3

# Load CSVs
movies = pd.read_csv("./data/movies.csv")
ratings = pd.read_csv("./data/ratings.csv")
tags = pd.read_csv("./data/tags.csv")
links = pd.read_csv("./data/links.csv")

# Connect to SQLite database (creates file if not exists)
conn = sqlite3.connect("movies.db")

# Save to SQL tables
movies.to_sql("movies", conn, if_exists="replace", index=False)
ratings.to_sql("ratings", conn, if_exists="replace", index=False)
tags.to_sql("tags", conn, if_exists="replace", index=False)
links.to_sql("links", conn, if_exists="replace", index=False)

conn.close()
print("Database created successfully.")