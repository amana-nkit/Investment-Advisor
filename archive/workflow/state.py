"""
workflow/state.py
TypedDict that flows through every LangGraph node.
All four memory layers are represented here.
"""
from __future__ import annotations
from typing import TypedDict, List, Optional


class ProposalState(TypedDict):
    # Input
    customer_id: str
    session_question: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    # L2 — conversation history (injected by MemoryManager)
    chat_history: List[dict]

    # L3 — structured long-term memory
    customer_profile: Optional[dict]
    proposal_history: List[dict]

    # L4 — semantic long-term memory (cross-session recall)
    recalled_memories: List[str]

    # Retrieved document chunks (current session)
    document_chunks: List[str]

    # Web enrichment context
    market_context: List[str]

    # Validation
    validation_errors: List[str]

    # Generated output
    proposal_draft: str
    proposal_final: Optional[dict]   # structured InvestmentProposal dict
    proposal_text: str               # human-readable string for PDF + L4 ingestion
