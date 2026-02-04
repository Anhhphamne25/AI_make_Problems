from typing import List, Dict,Literal
from pydantic import BaseModel, Field

class ProgrammingProblem(BaseModel):
    name: str
    problemStatement: str
    inputFormat: str
    outputFormat: str
    constraints: str
    note: str = ""
    difficulty: str
    tags: List[str] = Field(default_factory=list)
    multipleFiles: Dict = Field(default_factory=dict)
    languages: List[str] = Field(default_factory=list)
    status: str = "draft"

class ProgrammingCritic(BaseModel):
    status: Literal["Approved", "Rejected"]
    feedback: str