from langgraph.graph import StateGraph
from src.rag.rag_pipeline import generate_proposal

# ✅ Optional typed state (can extend later)
class GraphState(dict):
    pass

def create_graph():


    # ==========================================================
    # ✅ Retrieve Step
    # ==========================================================

    def retrieve(state):
        state["docs"] = state["vector_db"].similarity_search(
            state["query"], k=5
        )
        return state

    def generate(state):
        context = "\n\n".join([d.page_content for d in state["docs"]])
        state["answer"] = state["generate_fn"](context)
        return state

    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")

    return graph.compile()