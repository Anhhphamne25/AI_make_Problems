# api/edit.py
from fastapi import APIRouter
from AI_make_Problems.schemas.problem import EditRequest
from AI_make_Problems.graph.workflow import generate_problem
from AI_make_Problems.models.respon import ProgrammingProblem

router = APIRouter()

@router.post("/edit", response_model=ProgrammingProblem)
def edit_api(req: EditRequest):
    return generate_problem(
        topic=req.topic,
        problem=req.problem,
        human_feedback=req.human_feedback
    )
