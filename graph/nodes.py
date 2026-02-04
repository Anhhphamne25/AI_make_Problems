from AI_make_Problems.models.state import GraphState
from AI_make_Problems.agents.critic import CriticAgent
from AI_make_Problems.agents.generator import GeneratorAgent

MAX_ITER = 3

def generate_node(state: GraphState):
    generator = GeneratorAgent()
    iteration = state.get("iteration", 0) + 1

    if state.get("human_feedback") is not None:
        problem = generator.run(
            f"""
            Chủ đề gốc:
            {state["topic"]}

            Đề bài hiện tại:
            {state["problem"].model_dump_json(ensure_ascii=False)}

            Góp ý cần chỉnh (NGƯỜI DÙNG):
            {state["human_feedback"]}

            Hãy chỉnh lại đề bài theo ĐÚNG góp ý người dùng.
            """
        )

        print("\n--- Human Edited Problem ---\n")
        print(problem.model_dump_json(indent=2, ensure_ascii=False))
        print("\n----------------------------\n")

        return {
            "problem": problem,
            "iteration": MAX_ITER,
            "human_feedback": None
        }

    if state.get("critic_result") is None:
        problem = generator.run(state["topic"])

    else:
        problem = generator.run(
            f"""
            Chủ đề gốc:
            {state["topic"]}

            Góp ý cần chỉnh (AI):
            {state["critic_result"].feedback}

            Hãy chỉnh lại đề bài cho đầy đủ, rõ ràng và đúng chuẩn học thuật.
            """
        )

    print("\n--- Generated Problem ---\n")
    print(problem.model_dump_json(indent=2, ensure_ascii=False))
    print("\n-------------------------\n")

    return {
        "problem": problem,
        "iteration": iteration
    }

def critic_node(state: GraphState):
    critic = CriticAgent()
    result = critic.run(state["problem"])
    print("\n--- Critic Feedback ---\n")
    print("STATUS:", result.status)
    print("FEEDBACK:")
    print(result.feedback)
    print("\n-----------------------\n")
    return {"critic_result": result}

def should_continue(state: GraphState) -> str:
    if  state["iteration"] >= 3:
        return "end"
    if state["critic_result"].status == "Approved":
        return "end"
    return "revise"
