import React, { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [movies, setMovies] = useState([]);
  const [userId, setUserId] = useState(1);
  const [loading, setLoading] = useState(false);

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/recommend?user_id=${userId}&top_n=20`);
      const data = await response.json();
      setMovies(data.recommendations || []);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setMovies([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, []);

  return (
    <div className="app">
      <h1>Cine Match</h1>
      <p>The perfect place to find your next movie</p>

      <div className="user-input">
        <input
          type="number"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          placeholder="Enter User ID"
        />
        <button onClick={fetchRecommendations}>Recommend</button>
      </div>

      {loading ? (
        <p>Loading...</p>
      ) : (
        <div className="movies-grid">
          {movies.map((movie, index) => (
            <div key={index} className="movie-card">
              <img
                src={movie.poster_url || "./../public/placeholder.jpg"}
                alt={movie.title}
              />
              <h3>{movie.title}</h3>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
