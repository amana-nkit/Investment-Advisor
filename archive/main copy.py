# ==========================================================
# AI INVESTMENT PROPOSAL GENERATOR (MAY 2026 READY)
# ==========================================================

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# LANGCHAIN IMPORTS
# ==========================================================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ==========================================================
# PDF GENERATION
# ==========================================================
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ==========================================================
# EMAIL
# ==========================================================
import smtplib
from email.message import EmailMessage

# ==========================================================
# STREAMLIT CONFIG
# ==========================================================
st.set_page_config(page_title="AI Investment Proposal", layout="wide")

st.title("💰 AI Investment Proposal Generator")

st.markdown("""
Upload customer financial data → Generate proposal → Download PDF → Email client  
""")

# ==========================================================
# SIDEBAR CONFIG
# ==========================================================
st.sidebar.header("⚙️ Settings")

chunk_size = st.sidebar.slider("Chunk Size", 500, 3000, 1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200)
top_k = st.sidebar.slider("Top K Context", 2, 10, 5)

# ==========================================================
# FILE UPLOAD
# ==========================================================
uploaded_file = st.file_uploader("📄 Upload Customer Investment", type=["pdf"])

# ==========================================================
# FUNCTIONS
# ==========================================================

def create_vector_store(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        file_path = tmp_file.name

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)

    return vector_store


def generate_proposal(vector_store):
    docs = vector_store.similarity_search("customer financial profile", k=top_k)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate.from_template("""
You are a senior wealth manager.

Generate a professional Investment Proposal.

Customer Data:
{context}

Output format:

1. Customer Profile Summary
2. Financial Health Analysis
3. Risk Appetite (Low/Medium/High)
4. Investment Strategy
5. Asset Allocation (% split)
6. Recommended Products
7. Expected Returns
8. Risks & Mitigation
9. Conclusion

Keep the tone formal and client-ready.
""")

    llm = ChatOpenAI(model="gpt-4.1")

    chain = prompt | llm
    response = chain.invoke({"context": context})

    return response.content


def create_pdf(text, filename="proposal.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    y = height - 40

    for line in text.split("\n"):
        c.drawString(40, y, line[:100])
        y -= 15

        if y < 40:
            c.showPage()
            y = height - 40

    c.save()
    return filename


def send_email(to_email, pdf_path):
    msg = EmailMessage()
    msg["Subject"] = "Your Investment Proposal"
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = to_email
    msg.set_content("Please find your investment proposal attached.")

    with open(pdf_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename="Investment_Proposal.pdf"
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        smtp.send_message(msg)


# ==========================================================
# MAIN FLOW
# ==========================================================

if uploaded_file:

    st.success("✅ File uploaded successfully")

    if st.button("🚀 Create Investment Proposal"):

        with st.spinner("Processing..."):

            # STEP 1: VECTOR STORE
            vector_store = create_vector_store(uploaded_file)

            # STEP 2: GENERATE PROPOSAL
            proposal_text = generate_proposal(vector_store)

            st.subheader("📑 Investment Proposal")
            st.write(proposal_text)

            # STEP 3: CREATE PDF
            pdf_file = create_pdf(proposal_text)

            # STEP 4: DOWNLOAD BUTTON
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "📥 Download PDF",
                    f,
                    file_name="Investment_Proposal.pdf"
                )

            # STEP 5: EMAIL
            st.subheader("📧 Email Proposal")
            email = st.text_input("Client Email")

            if st.button("Send Email"):
                try:
                    send_email(email, pdf_file)
                    st.success("✅ Email sent successfully")
                except Exception as e:
                    st.error(f"Error sending email: {e}")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
### 🚀 Tech Stack
- Streamlit
- LangChain v1
- OpenAI GPT-4.1
- FAISS
- RAG Pipeline

### ✅ Features
✔ Proposal Generation  
✔ PDF Export  
✔ Email Automation  
✔ Wealth Manager Ready  
""")