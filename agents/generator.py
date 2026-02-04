from AI_make_Problems.core.prompts import GENERATOR_PROMPT
from AI_make_Problems.models.respon import ProgrammingProblem
from AI_make_Problems.core.llm import llm

class GeneratorAgent:
    def run(self, topic: str) -> ProgrammingProblem:
        chain = GENERATOR_PROMPT | llm
        return chain.invoke({"topic": topic})