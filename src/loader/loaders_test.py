import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_web():
    urls = [
        "https://groww.in",
        "https://zerodha.com",
        "https://www.moneycontrol.com"
    ]

    docs = []

    for url in urls:
        try:
            res = requests.get(url, headers={"User-Agent": "Mozilla"})
            soup = BeautifulSoup(res.text, "html.parser")

            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text()

            docs.append(Document(
                page_content=text,
                metadata={"source": url}
            ))
        except:
            pass
    
    

    return docs