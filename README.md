# Chat with PDF (RAG)

A backend that lets you upload a PDF and ask questions about it, using
Retrieval-Augmented Generation (RAG) with Node.js, Gemini and Qdrant.

## How it works
PDF upload -> text extraction -> paragraph chunking -> embeddings (Gemini)
-> stored in Qdrant -> question embedded -> semantic search -> Gemini answers
using the retrieved chunk as context.

## Tech stack
Node.js, Express, Multer, pdf-parse, Google Gemini (@google/genai),
Qdrant vector database.

## Setup
1. cd server && npm install
2. Copy .env.example to .env and add GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
3. npm start
4. Open http://localhost:3000 or POST to /upload with form-data:
   pdf (File) and question (Text)

## Routes
- GET /health: server check
- GET /create-collection: creates the Qdrant collection
- POST /upload: runs the full RAG pipeline and returns the answer

