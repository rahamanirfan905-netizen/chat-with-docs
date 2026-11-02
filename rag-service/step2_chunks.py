from pypdf import PdfReader

# Read the PDF page by page
reader = PdfReader("sample.pdf")
pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    pages.append({"page": i + 1, "text": text})

print(f"Total pages: {len(pages)}")

#  Split each page into chunks (with overlap)
def split_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = []
for p in pages:
    for chunk in split_text(p["text"]):
        if chunk.strip():
            all_chunks.append({"page": p["page"], "text": chunk})

print(f"Total chunks: {len(all_chunks)}")

#  Look at the first 3 chunks
for c in all_chunks[:3]:
    print("\n--- Page", c["page"], "---")
    print(c["text"])