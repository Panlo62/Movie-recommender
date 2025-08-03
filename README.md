Movie Recommender System (Hybrid) — React + FastAPI

A full-stack movie recommender system combining collaborative filtering, content-based filtering, and meta learning. Built using React on the frontend and FastAPI on the backend. This system returns personalized movie recommendations along with movie posters using the OMDB API.

Features
- Personalized recommendations (hybrid: CF + CB)
- Time-aware user rating weights
- Cold-start logic for new users/items
- Movie poster integration via OMDB API
- Responsive React frontend
- FastAPI backend with CORS support
- IMDB ID mapping via links.csv
  
Tech Stack
- Frontend:
+ React
+ Axios
- Backend:
+ FastAPI
+ Pandas, NumPy, Scikit-learn, SciPy
+ Ridge Regression (meta model)
+ SQLite for data storage
+ CORS Middleware
+ Uvicorn (development server)

Getting Started
1. Clone the Repo
git clone https://github.com/your-username/movie-recommender-hybrid.git
cd movie-recommender-hybrid

2. Backend Setup
- Create virtual environment & install dependencies
+ cd backend
+ python -m venv venv
+ source venv/bin/activate  # or venv\Scripts\activate on Windows
+ pip install -r requirements.txt

- Prepare the SQLite Database
+ python init_db.py

- Start FastAPI Server
+ uvicorn main:app --reload

3. Frontend Setup
+ cd frontend
+ npm install
+ npm start

API Endpoints: GET /recommend
Query Parameters:
- user_id (int) – user ID
- top_n (int) – number of movies (default 20)
Response:
{
  "user_id": 1,
  "recommendations": [
    {
      "title": "Inception",
      "imdbId": "1375666",
      "poster": "https://...jpg"
    }
  ]
}


Movie Posters
Posters are fetched from: https://www.omdbapi.com/?i=tt<tmdbId>&apikey=<YOUR_OMDB_API_KEY>

To-Do / Improvements
- Add user onboarding with genre preferences
- Save user sessions & ratings from frontend
- Replace OMDB with TMDB for more robust data
- Deploy backend on Render / Railway / GCP
- Use a real database like PostgreSQL
