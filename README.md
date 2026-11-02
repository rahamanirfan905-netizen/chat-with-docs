“A full-stack app that lets users upload PDFs and ask questions, with answers grounded in the documents and cited by page.”

# chat-with-docs
A full-stack app that lets users upload PDFs and ask questions, with answers grounded in the documents and cited by page.

## Tech stack
Python, FastAPI, ChromaDB, Sentence-Transformers, Gemini API

## Status
- [x] PDF reading and chunking
- [x] Embeddings and vector search
- [x] Grounded answers with page citations
- [ ] FastAPI service
- [ ] Express backend
- [ ] React frontend
Tested questions:
- "What is the OSI model?" → correct answer with pages
- "Who won the cricket world cup?" → "I could not find this in the document."

