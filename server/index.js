require("dotenv").config();

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const pdfParse = require("pdf-parse");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { GoogleGenAI } = require("@google/genai");
const { QdrantClient } = require("@qdrant/js-client-rest");

// ---------- Config ----------
const PORT = process.env.PORT || 3000;
const COLLECTION = "pdf-docs";
const EMBEDDING_MODEL = "gemini-embedding-2";
const CHAT_MODEL = "gemini-2.5-flash-lite";
const TOP_K = 1; // number of chunks retrieved (try 3 or 5 later)

// ---------- Clients ----------
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

// ---------- App + Multer ----------
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public"))); // tiny test UI

const upload = multer({ dest: "uploads/" });

// ---------- Helpers ----------
async function createEmbedding(text) {
  const response = await ai.models.embedContent({
    model: EMBEDDING_MODEL,
    contents: text,
  });
  return response.embeddings[0].values;
}

// Real cosine similarity (kept for learning / manual comparison).
// Qdrant does this for us in the final pipeline.
function cosineSimilarity(vecA, vecB) {
  if (vecA.length !== vecB.length) {
    throw new Error("Vector size mismatch");
  }
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Create the collection if it does not exist yet.
async function ensureCollection(size) {
  const { collections } = await qdrant.getCollections();
  const exists = collections.some((c) => c.name === COLLECTION);
  if (!exists) {
    await qdrant.createCollection(COLLECTION, {
      vectors: { size, distance: "Cosine" },
    });
  }
}

// ---------- Routes ----------
app.get("/health", (req, res) => {
  res.send("Server running");
});

// Manual collection creation (from the PDF, Chapter 16).
// Optional: /upload also auto-creates it.
app.get("/create-collection", async (req, res) => {
  try {
    await ensureCollection(3072);
    res.send("Collection created (or already exists)");
  } catch (error) {
    console.log(error);
    res.status(500).send("Error creating collection");
  }
});

// Upload PDF + ask question (multipart/form-data: pdf=file, question=text)
app.post("/upload", upload.single("pdf"), async (req, res) => {
  try {
    console.log(req.body);

    if (!req.file) return res.status(400).send("No PDF uploaded (field name must be 'pdf')");
    const question = (req.body.question || "").trim();
    if (!question) return res.status(400).send("Missing 'question' field");

    // 1. PDF -> buffer -> text
    const dataBuffer = fs.readFileSync(req.file.path);
    const pdfData = await pdfParse(dataBuffer);
    const text = pdfData.text;

    // 2. Chunking (paragraphs, drop empty ones)
    const chunks = text.split("\n\n").filter((chunk) => chunk.trim() !== "");
    if (chunks.length === 0) {
      return res.status(400).send("No readable text found in this PDF");
    }

    // 3. Embed every chunk
    const chunkEmbeddings = [];
    for (const chunk of chunks) {
      const embedding = await createEmbedding(chunk);
      chunkEmbeddings.push({ text: chunk, embedding });
    }
    console.log("Vector size:", chunkEmbeddings[0].embedding.length);

    // 4. Store in Qdrant (UUID ids so multiple PDFs never collide)
    await ensureCollection(chunkEmbeddings[0].embedding.length);

    const points = chunkEmbeddings.map((item) => ({
      id: crypto.randomUUID(),
      vector: item.embedding,
      payload: { text: item.text, source: req.file.originalname },
    }));

    await qdrant.upsert(COLLECTION, { points });

    // 5. Question -> embedding -> semantic search in Qdrant
    const questionEmbedding = await createEmbedding(question);

    const searchResult = await qdrant.search(COLLECTION, {
      vector: questionEmbedding,
      limit: TOP_K,
    });

    const bestChunk = searchResult.map((r) => r.payload.text).join("\n\n");
    console.log("Best chunk:", bestChunk);
    console.log("Score:", searchResult[0].score);

    // 6. Gemini answers using retrieved context
    const response = await ai.models.generateContent({
      model: CHAT_MODEL,
      contents: `
Answer the question using this context:
${bestChunk}

Question:
${question}
`,
    });

    res.send(response.text);
  } catch (error) {
    console.log(error);
    res.status(500).send("Error processing PDF");
  } finally {
    if (req.file) fs.unlink(req.file.path, () => {}); // clean temp upload
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
