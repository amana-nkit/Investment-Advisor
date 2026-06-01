"""
memory/memory_manager.py

MemoryManager — four-layer memory system for the Investment Advisor.

  L1  st.session_state          in-process, page-session scope
  L2  ConversationBufferWindow  rolling 10-turn chat history
  L3  PostgreSQL (SQLAlchemy)   persistent structured client facts + proposals
  L4  ChromaDB (persistent)     semantic embeddings of docs & past proposals

Usage:
    mm = MemoryManager(customer_id="cust_abc123")
    mm.init_session()
    profile = mm.load_profile()
    store   = mm.get_or_build_vector_store(split_docs)
    ...
    mm.persist_proposal(proposal_json, proposal_text, pdf_path)
"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from core.config import get_settings
from models.db_models import CustomerProfile, ProposalRecord, ConsentRecord, init_db
from utilities.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Module-level singletons (cached across Streamlit reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


@st.cache_resource
def _get_db_engine():
    return init_db()


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Unified interface to all four memory layers for a given customer."""

    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self._engine = _get_db_engine()
        self._embeddings = _get_embeddings()

    # -----------------------------------------------------------------------
    # L1 — Streamlit session_state
    # -----------------------------------------------------------------------

    def init_session(self) -> None:
        """Initialise session_state keys with safe defaults."""
        defaults: dict = {
            f"vs_{self.customer_id}": None,       # vector store
            f"file_hash_{self.customer_id}": None, # last ingested file hash
            "chat_memory": None,
            "conversation_history": [],
            "current_proposal": None,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val
        log.info("Session initialised for customer %s", self.customer_id)

    def _vs_key(self) -> str:
        return f"vs_{self.customer_id}"

    def _hash_key(self) -> str:
        return f"file_hash_{self.customer_id}"

    def get_session_vector_store(self) -> Optional[Chroma]:
        return st.session_state.get(self._vs_key())

    def set_session_vector_store(self, store: Chroma) -> None:
        st.session_state[self._vs_key()] = store

    @staticmethod
    def file_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def is_file_already_ingested(self, file_data: bytes) -> bool:
        return st.session_state.get(self._hash_key()) == self.file_hash(file_data)

    def mark_file_ingested(self, file_data: bytes) -> None:
        st.session_state[self._hash_key()] = self.file_hash(file_data)

    # -----------------------------------------------------------------------
    # L2 — Conversation buffer (ConversationBufferWindowMemory)
    # -----------------------------------------------------------------------

    def get_chat_memory(self) -> ConversationBufferWindowMemory:
        if not st.session_state.get("chat_memory"):
            st.session_state["chat_memory"] = ConversationBufferWindowMemory(
                k=settings.conversation_window_k,
                memory_key="chat_history",
                return_messages=True,
            )
        return st.session_state["chat_memory"]

    def add_turn(self, human: str, ai: str) -> None:
        mem = self.get_chat_memory()
        mem.save_context({"input": human}, {"output": ai})
        st.session_state["conversation_history"].append(
            {"role": "user", "content": human}
        )
        st.session_state["conversation_history"].append(
            {"role": "assistant", "content": ai}
        )
        log.debug("Turn saved. History length: %d", len(st.session_state["conversation_history"]))

    def get_chat_history(self) -> list:
        return (
            self.get_chat_memory()
            .load_memory_variables({})
            .get("chat_history", [])
        )

    def clear_chat_memory(self) -> None:
        st.session_state["chat_memory"] = None
        st.session_state["conversation_history"] = []
        log.info("Chat memory cleared for customer %s", self.customer_id)

    # -----------------------------------------------------------------------
    # L3 — PostgreSQL structured memory
    # -----------------------------------------------------------------------

    def load_profile(self) -> Optional[dict]:
        with Session(self._engine) as s:
            p = s.get(CustomerProfile, self.customer_id)
            if not p:
                log.warning("No profile found for customer %s", self.customer_id)
                return None
            return {
                "name": p.name,
                "email": p.email,
                "age": p.age,
                "annual_income_inr": p.annual_income_inr,
                "risk_appetite": p.risk_appetite,
                "investment_horizon_years": p.investment_horizon_years,
                "financial_goals": p.financial_goals or [],
            }

    def upsert_profile(self, data: dict) -> None:
        with Session(self._engine) as s:
            existing = s.get(CustomerProfile, self.customer_id)
            if existing:
                for k, v in data.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
                existing.updated_at = datetime.now(timezone.utc)
            else:
                s.add(CustomerProfile(id=self.customer_id, **data))
            s.commit()
        log.info("Profile upserted for customer %s", self.customer_id)

    def save_proposal_to_db(
        self,
        proposal_json: dict,
        proposal_text: str,
        pdf_path: str = "",
    ) -> str:
        pid = str(uuid.uuid4())
        with Session(self._engine) as s:
            s.add(ProposalRecord(
                id=pid,
                customer_id=self.customer_id,
                proposal_json=proposal_json,
                proposal_text=proposal_text,
                pdf_path=pdf_path,
            ))
            s.commit()
        log.info("Proposal %s saved for customer %s", pid, self.customer_id)
        return pid

    def get_proposal_history(self, limit: int = 3) -> list[dict]:
        with Session(self._engine) as s:
            rows = (
                s.query(ProposalRecord)
                .filter_by(customer_id=self.customer_id)
                .order_by(ProposalRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [r.proposal_json for r in rows if r.proposal_json]

    def record_consent(self, ip_address: str = "") -> None:
        """Log DPDP Act consent before persisting any financial data."""
        with Session(self._engine) as s:
            s.add(ConsentRecord(
                customer_id=self.customer_id,
                ip_address=ip_address,
            ))
            s.commit()
        log.info("Consent recorded for customer %s", self.customer_id)

    # -----------------------------------------------------------------------
    # L4 — ChromaDB persistent semantic memory
    # -----------------------------------------------------------------------

    def _get_chroma_store(self) -> Chroma:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_path)
        return Chroma(
            client=client,
            collection_name=f"customer_{self.customer_id}",
            embedding_function=self._embeddings,
        )

    def get_or_build_vector_store(
        self,
        split_docs: list[Document],
        file_data: bytes,
    ) -> Chroma:
        """
        Return cached L1 store if the file hasn't changed, otherwise
        embed the new chunks and persist to ChromaDB (L4).
        """
        if self.is_file_already_ingested(file_data):
            cached = self.get_session_vector_store()
            if cached is not None:
                log.info("Returning cached vector store (file unchanged)")
                return cached

        log.info("Embedding %d chunks for customer %s", len(split_docs), self.customer_id)
        for doc in split_docs:
            doc.metadata["customer_id"] = self.customer_id
            doc.metadata["source_type"] = "pdf"
            doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()

        store = self._get_chroma_store()
        store.add_documents(split_docs)

        self.set_session_vector_store(store)
        self.mark_file_ingested(file_data)
        log.info("Vector store built and persisted to ChromaDB")
        return store

    def ingest_past_proposal(self, proposal_text: str, proposal_id: str) -> None:
        """Embed a past proposal text so future sessions can recall it semantically."""
        doc = Document(
            page_content=proposal_text,
            metadata={
                "customer_id": self.customer_id,
                "source_type": "past_proposal",
                "proposal_id": proposal_id,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        store = self._get_chroma_store()
        store.add_documents([doc])
        log.info("Past proposal %s embedded into L4 for customer %s", proposal_id, self.customer_id)

    def cross_session_recall(self, query: str, k: int = 4) -> list[Document]:
        """Retrieve semantically relevant memories across ALL past sessions."""
        store = self._get_chroma_store()
        try:
            results = store.similarity_search(query, k=k)
            log.info("Cross-session recall: %d docs for query '%s'", len(results), query[:60])
            return results
        except Exception as e:
            log.warning("Cross-session recall failed: %s", e)
            return []

    def multi_query_retrieve(self, vector_store: Chroma, top_k: int = 5) -> list[Document]:
        """
        Run multiple targeted queries against the vector store and
        return deduplicated results — far better than a single generic query.
        """
        queries = [
            "income salary assets net worth investments",
            "risk tolerance conservative aggressive preferences",
            "financial goals retirement education home purchase",
            "liabilities debt EMI obligations tax",
            "current portfolio existing holdings mutual funds",
        ]
        docs: list[Document] = []
        for q in queries:
            docs.extend(vector_store.similarity_search(q, k=top_k))

        seen: set[str] = set()
        unique: list[Document] = []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique.append(d)
        log.info("Multi-query retrieval: %d unique chunks", len(unique))
        return unique

    # -----------------------------------------------------------------------
    # Full memory erasure (DPDP Act / GDPR right to erasure)
    # -----------------------------------------------------------------------

    def delete_all_memory(self) -> None:
        """Irreversibly delete all data for this customer across L3 + L4."""
        with Session(self._engine) as s:
            s.query(ProposalRecord).filter_by(customer_id=self.customer_id).delete()
            s.query(CustomerProfile).filter_by(id=self.customer_id).delete()
            s.query(ConsentRecord).filter_by(customer_id=self.customer_id).delete()
            s.commit()

        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_path)
        try:
            client.delete_collection(f"customer_{self.customer_id}")
        except Exception:
            pass

        self.clear_chat_memory()
        log.info("All memory deleted for customer %s", self.customer_id)
