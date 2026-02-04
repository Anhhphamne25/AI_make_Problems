

def generate_problem(topic: str) -> ProgrammingProblem:
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
        "problem": None,
        "critic_result": None,
        "iteration": 0,
    })

    return result["problem"]