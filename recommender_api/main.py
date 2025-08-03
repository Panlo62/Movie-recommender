from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from recommender import recommend

app = FastAPI()

# Enable CORS for frontend access (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for React dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def get_recommendations(user_id: int = Query(...), top_n: int = Query(20)):
    try:
        recommendations = recommend(user_id, top_n)
        return {"user_id": user_id, "recommendations": recommendations}
    except KeyError as e:
        return {"error": f"{e} user_id {user_id} not found or no recommendations"}