from langgraph.graph import StateGraph
from typing import TypedDict, List, Any
from langchain_core.documents import Document

from rag.rag_pipeline import generate_proposal
from models.document_schema import InvestmentDocument
from utilities.logger import logger


# ==========================================================
# ✅ Typed State (CRITICAL FIX)
# ==========================================================
class GraphState(TypedDict, total=False):
    query: str
    vector_db: Any
    generate_fn: Any
    docs: List[Document]
    validated_docs: List[InvestmentDocument]
    answer: str


# ==========================================================
# ✅ Create Graph
# ==========================================================
def create_graph():

    # ==========================================================
    # ✅ RETRIEVE STEP
    # ==========================================================
    def retrieve(state: GraphState):
        vector_db = state.get("vector_db")
        query = state.get("query")

        if vector_db is None:
            raise ValueError("❌ vector_db is missing in state")

        if not query:
            raise ValueError("❌ query is missing in state")

        docs = vector_db.similarity_search(query, k=5)

        logger.info(f"Retrieved {len(docs)} documents from vector DB")

        return {
            **state,
            "docs": docs
        }


    # ==========================================================
    # ✅ VALIDATE STEP (NEW 🔥)
    # ==========================================================
    def validate(state: GraphState):
        docs = state.get("docs", [])

        validated_docs = []

        for d in docs:
            try:
                clean_text = d.page_content.replace("\n", " ").strip()

                validated_doc = InvestmentDocument(
                    content=clean_text,
                    source=d.metadata.get("source"),
                    doc_type=d.metadata.get("type"),
                )

                validated_docs.append(validated_doc)

            except Exception as e:
                logger.warning(f"Invalid document skipped: {e}")

        if not validated_docs:
            raise ValueError("❌ No valid documents after validation")

        logger.info(f"Validated {len(validated_docs)} documents")

        return {
            **state,
            "validated_docs": validated_docs
        }


    # ==========================================================
    # ✅ GENERATE STEP
    # ==========================================================
    def generate(state: GraphState):
        validated_docs = state.get("validated_docs", [])

        if not validated_docs:
            raise ValueError("❌ No validated documents available")

        # ✅ Clean structured context
        context = "\n\n".join([doc.content for doc in validated_docs])

        generate_fn = state.get("generate_fn")

        if generate_fn is None:
            raise ValueError("❌ generate_fn missing")

        answer = generate_fn(context)

        logger.info("Proposal generated successfully")

        return {
            **state,
            "answer": answer
        }


    # ==========================================================
    # ✅ BUILD GRAPH
    # ==========================================================
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("validate", validate)     # ✅ NEW step
    graph.add_node("generate", generate)

    # ✅ Flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "validate")
    graph.add_edge("validate", "generate")

    return graph.compile()