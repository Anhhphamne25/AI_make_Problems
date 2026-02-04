from AI_make_Problems.core.prompts import CRITIC_PROMPT
from AI_make_Problems.models.respon import ProgrammingProblem, ProgrammingCritic
from AI_make_Problems.core.llm import critic_llm

class CriticAgent:
    def run(self, problem: ProgrammingProblem) -> ProgrammingCritic:
        chain = CRITIC_PROMPT | critic_llm
        return chain.invoke({
            "problem_text": problem.model_dump_json(ensure_ascii=False)
        })