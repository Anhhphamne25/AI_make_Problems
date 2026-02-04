from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI
from AI_make_Problems.api.problems import router as problems_router

app = FastAPI(title="AI Make Problems API")

app.include_router(
    problems_router,
    prefix="/api",
    tags=["Problems"]
)

@app.get("/")
def health():
    return {"status": "ok"}

# python -m uvicorn AI_make_Problems.main:app --reload --workers 1
