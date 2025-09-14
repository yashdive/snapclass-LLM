pip install fastapi uvicorn requests PyPDF2 langchain sentence-transformers chromadb


ollama run llama3.2:3b


uvicorn server.main:app --reload --port 8000


Go to http://127.0.0.1:8000/docs#/default/ask_model_ask_post
