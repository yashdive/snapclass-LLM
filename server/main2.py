# server.py


from fastapi import FastAPI
from pydantic import BaseModel
import requests
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

app = FastAPI()


class Query(BaseModel):
    prompt: str

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
PDF_PATH = "SnapManual.pdf"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_pdf_text(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

def build_prompt(context: str, question: str):
    return f"""You are Snap Assistant.
    Use the following manual context to answer:

    {context}

    Question: {question}
    Answer:"""

# -------------------------
# RAG Setup
# -------------------------
print("Loading Snap manual and initializing RAG...")

manual_text = extract_pdf_text(PDF_PATH)
chunks = create_chunks(manual_text)
print(f"{len(chunks)} chunks created")

# Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma vector DB
client = chromadb.Client()
collection = client.create_collection("snap_manual")

# Insert chunks with embeddings
for i, chunk in enumerate(chunks):
    embedding = embedder.encode(chunk).tolist()
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("RAG setup complete ✅")

# -------------------------
# FastAPI endpoints
# -------------------------
@app.post("/ask")
async def ask_model(req: Query):
    # 1️⃣ Embed user query
    embedding = embedder.encode(req.prompt).tolist()

    # 2️⃣ Retrieve top relevant chunks
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )
    context = "\n\n".join(results["documents"][0])

    # 3️⃣ Build prompt for LLM
    prompt = build_prompt(context, req.prompt)

    # 4️⃣ Call Ollama HTTP API
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.2,
        "max_tokens": 500,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    result = response.json()
    answer = result.get("response", "")
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "FastAPI + Llama3.2:3b is running 🚀"}
