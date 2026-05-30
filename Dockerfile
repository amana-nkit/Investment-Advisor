# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -U \
    streamlit \
    langchain \
    langchain-core \
    langchain-community \
    langchain-openai \
    langchain-text-splitters \
    faiss-cpu \
    pypdf \
    python-dotenv \
    tiktoken \
    reportlab

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]