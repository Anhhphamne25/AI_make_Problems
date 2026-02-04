from AI_make_Problems.models.state import GraphState
from langgraph.graph import StateGraph, END
from AI_make_Problems.models.respon import ProgrammingProblem
from AI_make_Problems.graph.nodes import generate_node, critic_node, should_continue

def generate_problem(
    topic: str,
    human_feedback: str | None = None,
    problem: ProgrammingProblem | None = None,
) -> ProgrammingProblem:
    graph = StateGraph(GraphState)

    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "critic")

    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "end": END,
            "revise": "generate",
        }
    )

    app = graph.compile()

    result = app.invoke({
        "topic": topic,
        "problem": problem,
        "human_feedback": human_feedback,
        "critic_result": None,
        "iteration": 0,
    })

    return result["problem"]