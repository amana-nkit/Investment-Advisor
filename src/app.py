import streamlit as st
import tempfile
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ✅ Logger
from utilities.logger import logger

# ✅ Imports
from loader.loaders import load_pdf, load_web
from vector.vector_store import create_vector_store
from rag.rag_pipeline import generate_proposal
from utilities.utils import create_pdf, send_email
from archive.graph_v2 import create_graph

# ==========================================================
# ✅ Streamlit Config
# ==========================================================
st.set_page_config(page_title="AI Investment Proposal", layout="wide")

st.title("💰 AI Investment Proposal (LangGraph + RAG)")

st.markdown("""
Upload client financial data → Generate proposal → Download PDF → Email client  
""")

# ==========================================================
# ✅ File Upload
# ==========================================================
uploaded_file = st.file_uploader("📄 Upload Customer Financial PDF", type=["pdf"])

# ==========================================================
# ✅ Session State
# ==========================================================
if "proposal" not in st.session_state:
    st.session_state["proposal"] = None

if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None

# ==========================================================
# ✅ MAIN FLOW
# ==========================================================
if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.success("✅ File uploaded successfully")
    logger.info("PDF uploaded successfully")

    if st.button("🚀 Generate Investment Proposal"):

        with st.spinner("🔄 Processing..."):

            try:
                # ✅ Load data
                pdf_docs = load_pdf(file_path)
                web_docs = load_web()

                all_docs = pdf_docs + web_docs
                logger.info(f"Total documents loaded: {len(all_docs)}")

                # ✅ Create vector DB
                vector_db = create_vector_store(all_docs)

                if vector_db is None:
                    raise ValueError("Vector store creation failed")

                logger.info("Vector DB created successfully")

                # ✅ LangGraph execution
                graph = create_graph()

                result = graph.invoke({
                    "query": "customer financial profile investment strategy risk appetite asset allocation",
                    "vector_db": vector_db,
                    "generate_fn": generate_proposal
                })

                logger.info("LangGraph execution successful")

                proposal = result.get("answer", "")

                if not proposal:
                    raise ValueError("Empty proposal generated")

                # ✅ Save proposal
                st.session_state["proposal"] = proposal

                # ✅ Create PDF
                pdf_path = create_pdf(proposal)
                st.session_state["pdf_path"] = pdf_path

                logger.info("PDF generated successfully")

            except Exception as e:
                logger.error(f"Error in proposal generation: {str(e)}", exc_info=True)
                st.error("❌ Something went wrong. Please check logs.")

# ==========================================================
# ✅ DISPLAY RESULT
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
                logger.info(f"Proposal emailed to {email}")
                st.success("✅ Email sent successfully")
            except Exception as e:
                logger.error(f"Email sending failed: {str(e)}", exc_info=True)
                st.error("❌ Failed to send email. Check logs.")

# ==========================================================
# ✅ FOOTER
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