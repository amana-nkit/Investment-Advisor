import streamlit as st
import tempfile
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ✅ Imports (clean, using src package)
from src.loader.loaders import load_pdf, load_web
from src.vector.vector_store import create_vector_store
# from vector.vector_store import create_vector_store
from src.rag.rag_pipeline import generate_proposal
from src.utilities.utils import create_pdf, send_email
from src.workflow.graph import create_graph


# ✅ Streamlit Page Config
st.set_page_config(page_title="AI Investment Proposal", layout="wide")

st.title("💰 AI Investment Proposal (LangGraph + RAG)")

st.markdown("""
Upload client financial data → Generate proposal → Download PDF → Email client  
""")

# ✅ File Upload
uploaded_file = st.file_uploader("📄 Upload Customer Financial PDF", type=["pdf"])

# ✅ Session Persistence
if "proposal" not in st.session_state:
    st.session_state["proposal"] = None

if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None

# ==========================================================
# MAIN FLOW
# ==========================================================
if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.success("✅ File uploaded successfully")

    if st.button("🚀 Generate Investment Proposal"):

        with st.spinner("🔄 Processing..."):

            try:
                # ✅ Step 1: Load data
                pdf_docs = load_pdf(file_path)
                web_docs = load_web()

                all_docs = pdf_docs + web_docs

                st.write("📊 Total Docs:", len(all_docs))  # ✅ DEBUG

                # ✅ Step 2: Create vector DB
                vector_db = create_vector_store(all_docs)

                st.write("✅ Vector DB:", type(vector_db))  # ✅ DEBUG

                # ✅ Step 3: LangGraph execution
                graph = create_graph()

                result = graph.invoke({
                    "query": "investment strategy, risk profile, mutual fund allocation",
                    "vector_db": vector_db,
                    "generate_fn": generate_proposal
                    # "generate_fn": lambda ctx: generate_proposal(vector_db)
                })

                st.write("✅ Graph Output:", result)  # ✅ DEBUG

                proposal = result.get("answer", "No output generated.")

                # ✅ Save results
                st.session_state["proposal"] = proposal

                # ✅ Step 4: Generate PDF
                pdf_path = create_pdf(proposal)
                st.session_state["pdf_path"] = pdf_path

            except Exception as e:
                st.error(f"❌ Error occurred: {e}")

# ==========================================================
# DISPLAY OUTPUT
# ==========================================================
if st.session_state["proposal"]:

    st.subheader("📑 Investment Proposal")
    st.write(st.session_state["proposal"])

    # ✅ Download PDF
    with open(st.session_state["pdf_path"], "rb") as f:
        st.download_button(
            label="📥 Download Proposal PDF",
            data=f,
            file_name="Investment_Proposal.pdf"
        )

    # ✅ Email Section
    st.subheader("📧 Email Proposal")

    email = st.text_input("Enter Client Email")

    if st.button("Send Email"):

        if not email:
            st.warning("⚠️ Please enter a valid email address")
        else:
            try:
                send_email(email, st.session_state["pdf_path"])
                st.success("✅ Email sent successfully")
            except Exception as e:
                st.error(f"❌ Email failed: {e}")

# ==========================================================
# FOOTER
# ==========================================================
st.divider()

st.markdown("""
### 🚀 Features
✅ LangGraph Workflow  
✅ Hybrid RAG (PDF + Investment Websites)  
✅ Intelligent Proposal Generation  
✅ PDF Export  
✅ Email Automation  

### 🧠 Tech Stack
- Streamlit  
- LangGraph  
- LangChain  
- FAISS / ChromaDB  
- OpenAI  
""")