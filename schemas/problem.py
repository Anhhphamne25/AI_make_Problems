from pydantic import BaseModel
from typing import Optional
from AI_make_Problems.models.respon import ProgrammingProblem

class GenerateRequest(BaseModel):
    topic: str


class EditRequest(BaseModel):
    topic: str
    problem: ProgrammingProblem
    human_feedback: str