# api/generate.py
from fastapi import APIRouter
from AI_make_Problems.schemas.problem import GenerateRequest
from AI_make_Problems.graph.workflow import generate_problem
from AI_make_Problems.models.respon import ProgrammingProblem

router = APIRouter()

@router.post("/generate", response_model=ProgrammingProblem)
def generate_api(req: GenerateRequest):
    return generate_problem(
        topic=req.topic,
        human_feedback=None,
        problem=None
    )
