from typing import TypedDict, Optional
from AI_make_Problems.models.respon import ProgrammingProblem, ProgrammingCritic

class GraphState(TypedDict):
    topic: str
    problem: Optional[ProgrammingProblem]
    critic_result: Optional[ProgrammingCritic]
    iteration: int