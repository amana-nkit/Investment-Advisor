from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_vector_store(docs):

    if not docs:
        raise ValueError("❌ No documents provided")

    # ✅ Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    if not split_docs:
        raise ValueError("❌ No document chunks created")

    # ✅ Create embeddings
    embeddings = OpenAIEmbeddings()

    # ✅ Create Chroma vector DB
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="data/vector_db"
    )

    # ✅ Optional but recommended (persist to disk)
    vector_db.persist()

    return vector_db