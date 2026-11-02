import os, time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

load_dotenv()
client_llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-flash-lite-latest"

model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("docs")

def generate_with_retry(prompt, tries=5):
    for attempt in range(tries):
        try:
            return client_llm.models.generate_content(
                model=MODEL, contents=prompt
            ).text
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                wait = 2 ** attempt * 3
                print(f"Busy ({attempt+1}/{tries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini is still busy, try again later")

def ask(question):
    q_emb = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=4)
    chunks = res["documents"][0]
    pages = [m["page"] for m in res["metadatas"][0]]

    context = "\n\n".join(f"[Page {p}]\n{c}" for p, c in zip(pages, chunks))
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the document."

Context:
{context}

Question: {question}"""

    return generate_with_retry(prompt), sorted(set(pages))

question = "What is the OSI model?"
answer, pages = ask(question)
print("\nAnswer:", answer)
print("Sources: pages", pages)