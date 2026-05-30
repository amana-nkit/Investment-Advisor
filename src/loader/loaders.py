import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
import time

# ✅ Logger import (IMPORTANT)
from src.utilities.logger import logger


# ==========================================================
# ✅ Load PDF
# ==========================================================
def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        logger.info(f"Loaded {len(docs)} PDF documents")

        if docs:
            logger.debug(f"Sample PDF Data: {docs[0].page_content[:200]}")

        return docs

    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}", exc_info=True)
        return []


# ==========================================================
# ✅ Extract Main Text
# ==========================================================
def extract_main_text(soup):
    """
    Extract meaningful content from webpage
    """
    try:
        main_content = soup.find("article") or soup.find("main")

        if main_content:
            return main_content.get_text(separator=" ", strip=True)

        return soup.get_text(separator=" ", strip=True)

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""


# ==========================================================
# ✅ Load Web Data
# ==========================================================
def load_web():

    urls = [
        "https://zerodha.com/varsity/module/personalfinance/"
        "https://zerodha.com/varsity/module/fundamental-analysis/",
        "https://groww.in/blog/",
        "https://www.moneycontrol.com/news/",
        "https://www.investopedia.com/"
    ]

    docs = []

    for url in urls:
        for attempt in range(2):  # ✅ Retry mechanism
            try:
                logger.info(f"Fetching URL: {url} (Attempt {attempt+1})")

                res = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10
                )

                if res.status_code != 200:
                    logger.warning(f"Failed {url} (Status: {res.status_code})")
                    time.sleep(2)
                    continue

                soup = BeautifulSoup(res.text, "html.parser")

                # ✅ Remove unwanted tags
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()

                text = extract_main_text(soup)

                if len(text) > 300:  # ✅ Filter noise
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": url,
                                "type": "web"
                            }
                        )
                    )
                else:
                    logger.warning(f"Low-content page skipped: {url}")

                break  # ✅ Exit retry loop if success

            except Exception as e:
                logger.error(f"Error loading {url}: {str(e)}", exc_info=True)
                time.sleep(2)

    logger.info(f"Total Web Documents Loaded: {len(docs)}")

    if docs:
        logger.debug(f"Sample Web Data: {docs[0].page_content[:200]}")

    return docs


# ==========================================================
# ✅ Optional: Structured Market Data (Future Ready)
# ==========================================================
def load_market_data():
    try:
        logger.info("Loading sample market data")

        dummy_data = """
        Market Overview:
        Investors are allocating funds across equity, debt and gold.

        Recommended Strategies:
        - SIP in mutual funds
        - Diversification across assets
        - Long-term compounding approach
        """

        return [
            Document(
                page_content=dummy_data,
                metadata={"source": "internal_market_data", "type": "api"}
            )
        ]

    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}", exc_info=True)
        return []


# ==========================================================
# ✅ MAIN (ONLY FOR DEBUG — NO UI IMPACT)
# ==========================================================
if __name__ == "__main__":
    logger.info("Running loader standalone test")

    docs = load_web()

    logger.info(f"Test complete. Loaded docs: {len(docs)}")