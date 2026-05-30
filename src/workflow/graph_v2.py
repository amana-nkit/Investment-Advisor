from langgraph.graph import StateGraph
from typing import TypedDict, List, Any
from langchain_core.documents import Document
from src.rag.rag_pipeline import generate_proposal


# ✅ Proper typed state
class GraphState(TypedDict, total=False):
    query: str
    vector_db: Any
    generate_fn: Any
    docs: List[Document]
    answer: str

def create_graph():

    # ==========================================================
    # ✅ Retrieve Step
    # ==========================================================
    def retrieve(state: GraphState):
        vector_db = state.get("vector_db")
        # query = state.get("query")

        if vector_db is None:
            raise ValueError("❌ vector_db is missing in state")

        # if not query:
        #     raise ValueError("❌ query is missing in state")

        docs = vector_db.similarity_search(state["query"], k=5)

        if not docs:
            print("⚠️ No documents retrieved")

        # state["docs"] = docs
        return {
            **state,
            "docs":docs
        }


    # ==========================================================
    # ✅ Generate Step
    # ==========================================================
    def generate(state: GraphState):
        docs = state.get("docs", [])

        if not docs:
            raise ValueError("❌ No documents available for generation")

        # ✅ Build context from retrieved docs
        context = "\n\n".join([
            d.page_content for d in docs if hasattr(d, "page_content")
        ])

        generate_fn = state.get("generate_fn")

        if generate_fn is None:
            raise ValueError("❌ generate_fn missing in state")

        # ✅ Generate final answer
        state["answer"] = generate_fn(context)

        return state


    # ==========================================================
    # ✅ Build Graph
    # ==========================================================
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")

    return graph.compile()

    st.write("Graph Output:", result)