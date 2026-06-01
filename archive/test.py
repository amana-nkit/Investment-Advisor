# To execute command : streamlit run streamlit_langchain_06.py

# ==========================================================

# FINTECH STREAMLIT + RAG + LANGCHAIN APP (MAY 2026)

# ==========================================================

#

# FEATURES

# ----------------------------------------------------------

# ✓ Streamlit UI

# ✓ PDF Upload

# ✓ RAG Pipeline

# ✓ FAISS Vector Database

# ✓ OpenAI Embeddings

# ✓ Financial Question Answering

# ✓ Source Context Retrieval

#

# SAFE FOR MAY 2026

# ----------------------------------------------------------

# ✓ LangChain v1+

# ✓ No deprecated imports

# ✓ No package conflicts

# ✓ No classic package dependency

#

# ==========================================================

# INSTALLATION

# ==========================================================
 
"""

pip install -U \

# streamlit \

# langchain \

# langchain-core \

# langchain-community \

# langchain-openai \

# langchain-text-splitters \

# faiss-cpu \

# pypdf \

# python-dotenv \

# tiktoken

"""
 
# ==========================================================

# .env

# ==========================================================
 
# OPENAI_API_KEY=your_api_key
 
# ==========================================================

# RUN APP

# ==========================================================
 
"""

streamlit run streamlit_langchain_06.py

"""
 
# ==========================================================

# IMPORTS

# ==========================================================
 
import os

import tempfile
 
from dotenv import load_dotenv
 
load_dotenv()
os.environ["OPEN_API_KEY"]=os.getenv("OPEN_API_KEY")
 
# ==========================================================

# STREAMLIT

# ==========================================================
 
import streamlit as st
 
# ==========================================================

# LANGCHAIN IMPORTS

# ==========================================================
 
# OpenAI

from langchain_openai import (

    ChatOpenAI,

    OpenAIEmbeddings,

)
 
# PDF Loader

from langchain_community.document_loaders import (

    PyPDFLoader,

)
 
# Text Splitter

from langchain_text_splitters import (

    RecursiveCharacterTextSplitter,

)
 
# Vector Store

from langchain_community.vectorstores import (

    FAISS,

)
 
# Prompt

from langchain_core.prompts import (

    PromptTemplate,

)
 
# Output Parser

from langchain_core.output_parsers import (

    StrOutputParser,

)
 
# ==========================================================

# STREAMLIT CONFIG

# ==========================================================
 
st.set_page_config(

    page_title="FinTech RAG Assistant",

    page_icon="💰",

    layout="wide",

)
 
# ==========================================================

# PAGE TITLE

# ==========================================================
 
st.title("💰 FinTech AI RAG Assistant")
 
st.markdown(

    """

Upload a financial report PDF and ask questions using:
 
- Retrieval-Augmented Generation (RAG)

- LangChain

- OpenAI GPT

- Financial Document Intelligence

"""

)
 
# ==========================================================

# SIDEBAR

# ==========================================================
 
st.sidebar.header("⚙️ Configuration")
 
chunk_size = st.sidebar.slider(

    "Chunk Size",

    min_value=500,

    max_value=2000,

    value=1000,

    step=100,

)
 
chunk_overlap = st.sidebar.slider(

    "Chunk Overlap",

    min_value=50,

    max_value=500,

    value=150,

    step=25,

)
 
top_k = st.sidebar.slider(

    "Top K Retrieval",

    min_value=1,

    max_value=10,

    value=4,

)
 
# ==========================================================

# FILE UPLOADER

# ==========================================================
 
uploaded_file = st.file_uploader(

    "Upload Financial PDF",

    type=["pdf"],

)
 
# ==========================================================

# PROCESS PDF

# ==========================================================
 
if uploaded_file:
 
    with st.spinner("Processing PDF..."):
 
        # ==================================================

        # SAVE TEMP FILE

        # ==================================================
 
        with tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".pdf"

        ) as tmp_file:
 
            tmp_file.write(uploaded_file.read())
 
            pdf_path = tmp_file.name
 
        # ==================================================

        # LOAD DOCUMENT

        # ==================================================
 
        loader = PyPDFLoader(pdf_path)
 
        documents = loader.load()
 
        st.success(

            f"Loaded {len(documents)} pages"

        )
 
        # ==================================================

        # TEXT SPLITTING

        # ==================================================
 
        """

        RecursiveCharacterTextSplitter is a smart text-splitting 

        tool used in RAG pipelines to break large documents into smaller chunks while preserving meaning.
 
        It follows this order (default behavior):
 
        1. Paragraphs (\n\n)

        2. Lines (\n)

        3. Sentences (.)

        4. Words (space)

        5. Characters
 
        """
 
        splitter = RecursiveCharacterTextSplitter(

            chunk_size=chunk_size,

            chunk_overlap=chunk_overlap,

            separators=[

                "\n\n",

                "\n",

                ".",

                " ",

                "",

            ],

        )
 
        docs = splitter.split_documents(

            documents

        )
 
        st.success(

            f"Created {len(docs)} chunks"

        )
 
        # ==================================================

        # EMBEDDINGS

        # ==================================================
 
        embedding_model = OpenAIEmbeddings(

            model="text-embedding-3-large"

        )
 
        # ==================================================

        # VECTOR STORE

        # ==================================================
 
        vectorstore = FAISS.from_documents(

            documents=docs,

            embedding=embedding_model,

        )
 
        retriever = vectorstore.as_retriever(

            search_type="similarity",

            search_kwargs={"k": top_k},

        )
 
        # ==================================================

        # LLM

        # ==================================================
 
        llm = ChatOpenAI(

            model="gpt-4.1-mini",

            temperature=0.1,

        )
 
        st.success("RAG pipeline initialized")
 
        # ==================================================

        # QUESTION INPUT

        # ==================================================
 
        st.divider()
 
        question = st.text_input(

            "Ask a financial question"

        )
 
        # ==================================================

        # ASK QUESTION

        # ==================================================
 
        if question:
 
            with st.spinner(

                "Generating financial analysis..."

            ):
 
                # ==========================================

                # RETRIEVE DOCUMENTS

                # ==========================================
 
                retrieved_docs = retriever.invoke(

                    question

                )
 
                # ==========================================

                # CONTEXT

                # ==========================================
 
                context = "\n\n".join(
 
                    [

                        doc.page_content

                        for doc in retrieved_docs

                    ]

                )
 
                # ==========================================

                # PROMPT

                # ==========================================
 
                prompt = PromptTemplate(

                    input_variables=[

                        "context",

                        "question",

                    ],

                    template="""

You are a senior fintech AI assistant.
 
Use the provided financial context

to answer the user's question.
 
Financial Context:

{context}
 
Question:

{question}
 
Provide:

1. Financial explanation

2. Risk analysis

3. Business interpretation

4. Final answer

""",

                )
 
                # ==========================================

                # CHAIN

                # ==========================================
 
                chain = (

                    prompt

                    | llm

                    | StrOutputParser()

                )
 
                # ==========================================

                # GENERATE RESPONSE

                # ==========================================
 
                response = chain.invoke(

                    {

                        "context": context,

                        "question": question,

                    }

                )
 
                # ==========================================

                # DISPLAY RESPONSE

                # ==========================================
 
                st.subheader(

                    "📊 Financial Analysis"

                )
 
                st.write(response)
 
                # ==========================================

                # SOURCE DOCUMENTS

                # ==========================================
 
                with st.expander(

                    "📄 Retrieved Source Chunks"

                ):
 
                    for i, doc in enumerate(

                        retrieved_docs,

                        start=1

                    ):
 
                        st.markdown(

                            f"### Chunk {i}"

                        )
 
                        st.write(

                            doc.page_content[:1200]

                        )
 
                        st.divider()
 
# ==========================================================

# FOOTER

# ==========================================================
 
st.divider()
 
st.markdown(

    """

### 🚀 Tech Stack
 
- Streamlit

- LangChain

- OpenAI GPT-4.1

- FAISS Vector Database

- RAG Architecture

- FinTech AI
 
### ✅ May 2026 Compatible
 
- No deprecated imports

- No package conflicts

- LangChain v1 safe

"""

)
 