# Instructions

## Create and Activate Virtual environment

python -m venv .venv
 
activate:
.venv\Scripts\activate.bat

## Install all required python packages

pip install -r requirements.txt

python -m streamlit run main.py

## Docker login

 docker login -u amanankit  

## Build & Run Locally

### Build
docker build -t ai-investment-app .

### Run
docker run -p 8501:8501 ai-investment-app

http://localhost:8501

## Download Azure CLI

https://aka.ms/installazurecliwindows

Run the .msi file
Follow default steps (Next → Finish)
