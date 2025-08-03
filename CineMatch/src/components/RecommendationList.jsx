import React, { useState } from "react";
import axios from "axios";

const RecommendationList = () => {
  const [userId, setUserId] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState("");

  const fetchRecommendations = async () => {
    try {
      const res = await axios.get("http://localhost:8000/recommend", {
        params: { user_id: userId, top_n: 20 },
      });
      if (res.data.recommendations) {
        setRecommendations(res.data.recommendations);
        setError("");
      } else {
        setRecommendations([]);
        setError(res.data.error || "No recommendations.");
      }
    } catch (err) {
      console.error(err);
      setError("Error fetching recommendations.");
      setRecommendations([]);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h2>Movie Recommender</h2>
      <input
        type="number"
        placeholder="Enter user ID"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
        style={{ padding: "8px", marginRight: "8px" }}
      />
      <button onClick={fetchRecommendations} style={{ padding: "8px 12px" }}>
        Get Recommendations
      </button>

      {error && <p style={{ color: "red", marginTop: "16px" }}>{error}</p>}

      <ul style={{ marginTop: "20px" }}>
        {recommendations.map((movie) => (
          <li key={movie.movieId}>
            <strong>{movie.title}</strong>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default RecommendationList;