"""
workflow/graph.py

LangGraph pipeline — 6 nodes:

  load_memory   → retrieve   → enrich
       ↓              ↓           ↓
  generate  →  validate_output  →  persist_memory  → END

Memory layers touched per node:
  load_memory     : L3 (profile + history), L4 (cross-session recall)
  retrieve        : L4 (vector store similarity search)
  enrich          : web context (optional)
  generate        : L2 (chat history injected into prompt)
  validate_output : Pydantic schema check
  persist_memory  : L3 (save proposal to DB), L4 (embed proposal text)
"""
from __future__ import annotations

import json
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from core.config import get_settings
from loader.loaders import load_web_context
from memory.memory_manager import MemoryManager, _get_embeddings
from models.schemas import InvestmentProposal
from utilities.logger import get_logger
from workflow.state import ProposalState

log = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Cached LLM singleton
# ---------------------------------------------------------------------------

@st.cache_resource
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openai_api_key,
    )


# ---------------------------------------------------------------------------
# Node 1 — Load memory (L3 + L4)
# ---------------------------------------------------------------------------

def load_memory_node(state: ProposalState) -> ProposalState:
    mm = MemoryManager(state["customer_id"])

    # L3: structured profile + history
    state["customer_profile"] = mm.load_profile()
    state["proposal_history"] = mm.get_proposal_history(limit=3)
    log.info("[load_memory] profile=%s, history_count=%d",
             bool(state["customer_profile"]), len(state["proposal_history"]))

    # L4: cross-session semantic recall
    recalled = mm.cross_session_recall(state["session_question"], k=4)
    state["recalled_memories"] = [d.page_content for d in recalled]
    log.info("[load_memory] recalled %d cross-session memories", len(state["recalled_memories"]))

    # L2: conversation history from current session
    state["chat_history"] = mm.get_chat_history()

    return state


# ---------------------------------------------------------------------------
# Node 2 — Retrieve (L4 vector store)
# ---------------------------------------------------------------------------

def retrieve_node(state: ProposalState) -> ProposalState:
    mm = MemoryManager(state["customer_id"])
    vector_store = mm.get_session_vector_store()

    if vector_store is None:
        log.warning("[retrieve] No vector store in session — skipping retrieval")
        state["document_chunks"] = []
        return state

    docs = mm.multi_query_retrieve(vector_store, top_k=state.get("top_k", settings.retrieval_top_k))
    state["document_chunks"] = [d.page_content for d in docs]
    log.info("[retrieve] %d unique chunks retrieved", len(state["document_chunks"]))
    return state


# ---------------------------------------------------------------------------
# Node 3 — Enrich (web market context)
# ---------------------------------------------------------------------------

def enrich_node(state: ProposalState) -> ProposalState:
    web_docs = load_web_context()
    state["market_context"] = [d.page_content[:800] for d in web_docs]
    log.info("[enrich] %d web context snippets loaded", len(state["market_context"]))
    return state


# ---------------------------------------------------------------------------
# Node 4 — Generate (LLM with structured output + all memory layers in prompt)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a SEBI-aware senior wealth manager with complete memory of this client.

=== CUSTOMER PROFILE (L3 memory) ===
{customer_profile}

=== PREVIOUS PROPOSALS — most recent first (L3 memory) ===
{proposal_history}

=== RECALLED MEMORIES FROM PRIOR SESSIONS (L4 semantic memory) ===
{recalled_memories}

=== DOCUMENT CONTEXT (current PDF) ===
{document_chunks}

=== MARKET CONTEXT ===
{market_context}

Instructions:
- Build on previous proposals — acknowledge what has changed.
- Address any follow-up or amendment raised in the current question.
- Ensure asset allocation percentages sum exactly to 100%.
- Include the mandatory SEBI disclaimer.
- Use formal, client-ready language.
"""

def generate_node(state: ProposalState) -> ProposalState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{session_question}"),
    ])

    structured_llm = get_llm().with_structured_output(InvestmentProposal)
    chain = prompt | structured_llm

    try:
        proposal: InvestmentProposal = chain.invoke({
            "customer_profile":  json.dumps(state.get("customer_profile") or {}, indent=2),
            "proposal_history":  json.dumps(state.get("proposal_history") or [], indent=2),
            "recalled_memories": "\n---\n".join(state.get("recalled_memories") or []) or "None",
            "document_chunks":   "\n\n".join(state.get("document_chunks") or []) or "No document uploaded.",
            "market_context":    "\n\n".join(state.get("market_context") or []) or "Not available.",
            "chat_history":      state.get("chat_history") or [],
            "session_question":  state["session_question"],
        })

        state["proposal_final"] = proposal.model_dump()
        state["proposal_text"]  = _proposal_to_text(proposal)
        state["validation_errors"] = []
        log.info("[generate] Proposal generated successfully")

    except Exception as e:
        log.error("[generate] LLM call failed: %s", e)
        state["validation_errors"] = [str(e)]
        state["proposal_final"] = None
        state["proposal_text"]  = ""

    return state


def _proposal_to_text(p: InvestmentProposal) -> str:
    """Convert structured proposal to readable string for PDF and L4 ingestion."""
    lines = [
        f"INVESTMENT PROPOSAL — {p.customer_name}",
        f"Date: {p.prepared_date}",
        f"Risk Appetite: {p.risk_appetite}",
        f"Horizon: {p.investment_horizon_years} years",
        f"Financial Health Score: {p.financial_health_score}/10",
        "",
        "EXECUTIVE SUMMARY",
        p.executive_summary,
        "",
        "ASSET ALLOCATION",
    ]
    for a in p.asset_allocation:
        lines.append(f"  {a.asset_class}: {a.percentage:.1f}% — {a.rationale}")
    lines += [
        "",
        "RECOMMENDED PRODUCTS",
        *[f"  - {r}" for r in p.recommended_products],
        "",
        f"EXPECTED ANNUAL RETURN: {p.expected_annual_return_pct:.1f}%",
        "",
        "KEY RISKS",
        *[f"  - {r}" for r in p.key_risks],
        "",
        "MITIGATION STRATEGIES",
        *[f"  - {m}" for m in p.mitigation_strategies],
        "",
        f"NEXT REVIEW: {p.next_review_months} months",
        "",
        "DISCLAIMER",
        p.disclaimer,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node 5 — Validate output (Pydantic re-check)
# ---------------------------------------------------------------------------

def validate_output_node(state: ProposalState) -> ProposalState:
    if not state.get("proposal_final"):
        state["validation_errors"].append("No proposal was generated.")
        return state
    try:
        InvestmentProposal(**state["proposal_final"])
        log.info("[validate_output] Proposal passed Pydantic validation")
    except Exception as e:
        state["validation_errors"].append(f"Schema validation failed: {e}")
        log.warning("[validate_output] %s", e)
    return state


# ---------------------------------------------------------------------------
# Node 6 — Persist memory (L3 + L4 write-back)
# ---------------------------------------------------------------------------

def persist_memory_node(state: ProposalState) -> ProposalState:
    if state.get("validation_errors") or not state.get("proposal_final"):
        log.warning("[persist_memory] Skipped — proposal invalid or missing")
        return state

    mm = MemoryManager(state["customer_id"])

    # L3: save structured JSON + text
    proposal_id = mm.save_proposal_to_db(
        proposal_json=state["proposal_final"],
        proposal_text=state["proposal_text"],
    )

    # L4: embed proposal text for cross-session recall
    mm.ingest_past_proposal(
        proposal_text=state["proposal_text"],
        proposal_id=proposal_id,
    )

    # L2: record this generation as an assistant turn
    mm.add_turn(
        human=state["session_question"],
        ai=state.get("proposal_text", "")[:500],  # store summary in chat buffer
    )

    log.info("[persist_memory] Proposal %s persisted to L3 + L4", proposal_id)
    return state


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(ProposalState)

    builder.add_node("load_memory",    load_memory_node)
    builder.add_node("retrieve",       retrieve_node)
    builder.add_node("enrich",         enrich_node)
    builder.add_node("generate",       generate_node)
    builder.add_node("validate_output",validate_output_node)
    builder.add_node("persist_memory", persist_memory_node)

    builder.set_entry_point("load_memory")
    builder.add_edge("load_memory",     "retrieve")
    builder.add_edge("retrieve",        "enrich")
    builder.add_edge("enrich",          "generate")
    builder.add_edge("generate",        "validate_output")
    builder.add_edge("validate_output", "persist_memory")
    builder.add_edge("persist_memory",  END)

    return builder.compile()


proposal_graph = build_graph()
