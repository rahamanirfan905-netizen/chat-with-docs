import os, time, uuid
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
NOT_FOUND = "I could not find this in the document."

# Loaded ONCE when the server starts
llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("documents")

app = FastAPI(title="DocChat RAG Service")


def split_text(text, chunk_size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def generate_with_retry(prompt, tries=5):
    for attempt in range(tries):
        try:
            return llm.models.generate_content(model=MODEL, contents=prompt).text
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                time.sleep(2 ** attempt * 3)
            else:
                raise
    raise HTTPException(status_code=503, detail="Gemini is busy, try again later")


class AskRequest(BaseModel):
    question: str
    doc_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        reader = PdfReader(BytesIO(file.file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this PDF")

    doc_id = str(uuid.uuid4())
    docs, metas, ids = [], [], []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for j, chunk in enumerate(split_text(text)):
            if chunk.strip():
                docs.append(chunk)
                metas.append({"doc_id": doc_id, "page": i + 1, "filename": file.filename})
                ids.append(f"{doc_id}_p{i+1}_c{j}")

    if not docs:
        raise HTTPException(status_code=422, detail="No readable text found (scanned PDF?)")

    embeddings = embedder.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)

    return {"doc_id": doc_id, "filename": file.filename,
            "pages": len(reader.pages), "chunks": len(docs)}


@app.post("/ask")
def ask(req: AskRequest):
    q_emb = embedder.encode([req.question]).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=4,
        where={"doc_id": req.doc_id},
    )
    chunks = res["documents"][0]
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found")

    metas = res["metadatas"][0]
    context = "\n\n".join(f"[Page {m['page']}]\n{c}" for m, c in zip(metas, chunks))
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "{NOT_FOUND}"

Context:
{context}

Question: {req.question}"""

    answer = generate_with_retry(prompt)

    # Hide misleading citations when the model found nothing
    if NOT_FOUND.lower() in answer.lower():
        return {"answer": answer, "sources": []}

    sources = [{"page": m["page"], "text": c[:200]} for m, c in zip(metas, chunks)]
    return {"answer": answer, "sources": sources}


@app.delete("/document/{doc_id}")
def delete_document(doc_id: str):
    collection.delete(where={"doc_id": doc_id})
    return {"deleted": doc_id}