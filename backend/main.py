
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .api import routes

app = FastAPI(title="NeuroQuant MVP", description="RL-based Trading Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)

if __name__ == "__main__":
    print("🚀 Starting NeuroQuant MVP...")
    print("📊 Features: DQN Agent + Technical Indicators + Sentiment")
    print("🌐 FastAPI Backend running on http://localhost:8000")
    print("📖 API Docs available at http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
