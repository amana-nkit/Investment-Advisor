"""
ui/app.py
Streamlit UI — thin shell only.
All business logic lives in workflow/graph.py and memory/memory_manager.py.

Run:
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import sys
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from core.config import get_settings
from loader.loaders import validate_and_read_pdf, load_and_split_pdf, PDFValidationError
from memory.memory_manager import MemoryManager
from models.db_models import init_db
from utilities.email_service import send_proposal_email, EmailValidationError
from utilities.pdf_export import create_pdf
from workflow.graph import proposal_graph

settings = get_settings()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Investment Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DB init (idempotent)
# ---------------------------------------------------------------------------
init_db()

# ---------------------------------------------------------------------------
# Sidebar — configuration + customer selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    customer_id = st.text_input(
        "Customer ID",
        value="demo_customer_001",
        help="Unique identifier — all memory is scoped to this ID.",
    )

    st.divider()
    st.subheader("RAG settings")
    chunk_size    = st.slider("Chunk size",    200, 3000, settings.chunk_size,    step=100)
    chunk_overlap = st.slider("Chunk overlap", 0,   500,  settings.chunk_overlap, step=50)
    top_k         = st.slider("Top-K results", 2,   10,   settings.retrieval_top_k)

    st.divider()
    st.subheader("🧠 Memory layers")
    st.markdown("""
| Layer | Backend | Scope |
|-------|---------|-------|
| L1 | session_state | Page session |
| L2 | ConversationBuffer | Rolling 10 turns |
| L3 | SQLite / PostgreSQL | Permanent |
| L4 | ChromaDB | Permanent |
""")

    st.divider()
    if st.button("🗑️ Erase all memory", type="secondary", use_container_width=True):
        mm = MemoryManager(customer_id)
        mm.delete_all_memory()
        st.success("All memory erased for this customer.")
        st.rerun()

# ---------------------------------------------------------------------------
# Initialise memory for this customer
# ---------------------------------------------------------------------------
mm = MemoryManager(customer_id)
mm.init_session()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("📊 AI Investment Advisor")
st.caption(
    "Generates structured, memory-aware investment proposals from client PDFs."
)

col_upload, col_profile = st.columns([1.4, 1], gap="large")

# ── Left column: PDF upload ──────────────────────────────────────────────
with col_upload:
    st.subheader("1. Upload client document")
    uploaded_file = st.file_uploader(
        "Upload client financial profile (PDF, max 10 MB)",
        type=["pdf"],
        help="The PDF is validated, chunked, and embedded into ChromaDB.",
    )

    if uploaded_file:
        try:
            file_data = validate_and_read_pdf(uploaded_file)

            if not mm.is_file_already_ingested(file_data):
                with st.spinner("Validating and embedding document…"):
                    split_docs = load_and_split_pdf(
                        file_data,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    mm.get_or_build_vector_store(split_docs, file_data)
                st.success(f"✅ Document embedded — {len(split_docs)} chunks stored in ChromaDB (L4).")
            else:
                st.info("📎 Document already embedded for this session.")

        except PDFValidationError as e:
            st.error(f"❌ {e}")
            st.stop()

# ── Right column: customer profile from L3 ───────────────────────────────
with col_profile:
    st.subheader("2. Customer profile (L3 memory)")
    profile = mm.load_profile()
    if profile:
        st.json(profile, expanded=False)
        history = mm.get_proposal_history(limit=3)
        if history:
            st.caption(f"📋 {len(history)} prior proposal(s) in memory")
    else:
        st.info("No profile found — will be created after first proposal.")

st.divider()

# ── Proposal generation ──────────────────────────────────────────────────
st.subheader("3. Generate proposal")

question = st.text_area(
    "Question / instruction",
    value="Generate a comprehensive investment proposal based on the uploaded document.",
    height=80,
)

generate_btn = st.button("🚀 Generate Investment Proposal", type="primary", use_container_width=True)

if generate_btn:
    if not mm.get_session_vector_store():
        st.warning("⚠️ Please upload a client PDF first.")
        st.stop()

    with st.spinner("Running memory-aware LangGraph pipeline…"):
        result = proposal_graph.invoke({
            "customer_id":     customer_id,
            "session_question": question,
            "chunk_size":      chunk_size,
            "chunk_overlap":   chunk_overlap,
            "top_k":           top_k,
            "chat_history":    [],
            "customer_profile": None,
            "proposal_history": [],
            "recalled_memories": [],
            "document_chunks":  [],
            "market_context":   [],
            "validation_errors": [],
            "proposal_draft":  "",
            "proposal_final":  None,
            "proposal_text":   "",
        })

    if result.get("validation_errors"):
        st.error("Pipeline errors:\n" + "\n".join(result["validation_errors"]))
        st.stop()

    proposal = result["proposal_final"]
    proposal_text = result["proposal_text"]
    st.session_state["current_proposal"] = proposal

    st.success("✅ Proposal generated and saved to L3 + L4 memory.")

    # Display structured output
    st.subheader("📄 Investment Proposal")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Risk Appetite",    proposal.get("risk_appetite", "—"))
    kpi2.metric("Horizon",          f"{proposal.get('investment_horizon_years', '—')} yrs")
    kpi3.metric("Health Score",     f"{proposal.get('financial_health_score', '—')}/10")
    kpi4.metric("Expected Return",  f"{proposal.get('expected_annual_return_pct', 0):.1f}%")

    with st.expander("Executive Summary", expanded=True):
        st.write(proposal.get("executive_summary", ""))

    with st.expander("Asset Allocation"):
        for a in proposal.get("asset_allocation", []):
            st.markdown(
                f"**{a['asset_class']}** — `{a['percentage']:.1f}%`  \n"
                f"{a['rationale']}"
            )

    with st.expander("Recommended Products"):
        for p in proposal.get("recommended_products", []):
            st.markdown(f"- {p}")

    with st.expander("Key Risks & Mitigation"):
        risks = proposal.get("key_risks", [])
        mitigations = proposal.get("mitigation_strategies", [])
        for i, risk in enumerate(risks):
            st.markdown(f"**Risk:** {risk}")
            if i < len(mitigations):
                st.markdown(f"*Mitigation: {mitigations[i]}*")

    st.caption(proposal.get("disclaimer", ""))

    # Download PDF
    st.divider()
    st.subheader("4. Download / Send")
    col_dl, col_email = st.columns(2)

    with col_dl:
        pdf_path = create_pdf(proposal)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF",
                data=f,
                file_name=f"proposal_{customer_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col_email:
        with st.form("email_form"):
            to_email = st.text_input("Client email address")
            send_btn = st.form_submit_button("📧 Send via Email", use_container_width=True)
            if send_btn:
                if not to_email:
                    st.warning("Enter an email address.")
                else:
                    try:
                        pdf_path = create_pdf(proposal)
                        ok = send_proposal_email(
                            to_email=to_email,
                            customer_name=proposal.get("customer_name", "Client"),
                            pdf_path=pdf_path,
                        )
                        if ok:
                            st.success(f"✅ Email sent to {to_email}")
                        else:
                            st.error("Email send failed — check logs.")
                    except EmailValidationError as e:
                        st.error(f"❌ {e}")

st.divider()

# ── Multi-turn chat (L2 memory) ───────────────────────────────────────────
st.subheader("5. Follow-up questions (L2 conversation memory)")
st.caption("Ask follow-ups like 'Make it more conservative' or 'What if I retire in 5 years?'")

for turn in st.session_state.get("conversation_history", []):
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

followup = st.chat_input("Ask a follow-up or request changes…")
if followup:
    if not mm.get_session_vector_store():
        st.warning("Upload a document first.")
    else:
        with st.chat_message("user"):
            st.write(followup)
        with st.spinner("Thinking…"):
            fu_result = proposal_graph.invoke({
                "customer_id":      customer_id,
                "session_question": followup,
                "chunk_size":       chunk_size,
                "chunk_overlap":    chunk_overlap,
                "top_k":            top_k,
                "chat_history":     mm.get_chat_history(),
                "customer_profile": None,
                "proposal_history": [],
                "recalled_memories": [],
                "document_chunks":  [],
                "market_context":   [],
                "validation_errors": [],
                "proposal_draft":   "",
                "proposal_final":   None,
                "proposal_text":    "",
            })
        response_text = fu_result.get("proposal_text", "")
        with st.chat_message("assistant"):
            st.write(response_text[:1500] + ("…" if len(response_text) > 1500 else ""))
        mm.add_turn(followup, response_text[:500])
        st.rerun()
