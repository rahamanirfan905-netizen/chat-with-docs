from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Read and chunk (same as Step 2)
reader = PdfReader("sample.pdf")

def split_text(text, chunk_size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

docs, metas, ids = [], [], []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    for j, chunk in enumerate(split_text(text)):
        if chunk.strip():
            docs.append(chunk)
            metas.append({"page": i + 1})
            ids.append(f"p{i+1}_c{j}")

print(f"Chunks to embed: {len(docs)}")

# 2. Embed (downloads a small model the first time)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, show_progress_bar=True).tolist()

# 3. Store in Chroma (saved on disk in ./chroma_db)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("docs")
collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
print("Stored in Chroma!")

# 4. Test a search
question = "What is the OSI model?"
q_emb = model.encode([question]).tolist()
results = collection.query(query_embeddings=q_emb, n_results=3)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"\n--- Page {meta['page']} ---")
    print(doc[:300])