import express from "express";
import multer from "multer";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { initRag, upsertText, ask } from "./rag.js";

const why = (e) => (e && e.stack) ? e.stack : String(e?.message || e);

// For ES modules, we need to get __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3005;

// Use Multer in-memory storage (no leftover files in ./data)
const upload = multer({ storage: multer.memoryStorage() });

// Init RAG pipeline on startup
await initRag();

// Health check
app.get("/health", (_req, res) => {
  res.json({ ok: true, message: "API is running" });
});

// Upload a PDF and ask a question
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    // Validate inputs
    if (!req.file) {
      return res.status(400).json({ error: "Missing file field 'file' (multipart/form-data)" });
    }
    const question = (req.body?.question || "").trim();
    if (!question) {
      return res.status(400).json({ error: "Missing 'question' field (form-data text)" });
    }

    // Create temp directory if it doesn't exist
    const tempDir = './temp';
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    // Write buffer to temp file for PDFLoader
    const tempPath = join(tempDir, `${Date.now()}_${req.file.originalname}`);
    fs.writeFileSync(tempPath, req.file.buffer);

    try {
      // Use PDFLoader to parse PDF
      const loader = new PDFLoader(tempPath);
      const docs = await loader.load();
      const text = docs.map(doc => doc.pageContent).join('\n').trim();

      if (!text) {
        return res.status(422).json({ error: "No extractable text found in PDF" });
      }

      // Insert into vector store
      const chunksAdded = await upsertText(req.file.originalname, text);

      // Run retrieval + generation
      const answer = await ask(question, 4);

      return res.json({
        answer,
        chunks_added: chunksAdded,
        file: req.file.originalname,
      });

    } finally {
      // Clean up temp file
      if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath);
      }
    }

  } catch (err) {
    console.error("[/upload] error:\n" + why(err));
    return res.status(500).json({ error: "Internal error", where: "/upload", detail: why(err) });
  }
});

// Ask a question without uploading (uses stored docs)
app.post("/ask", express.json(), async (req, res) => {
  try {
    const question = (req.body?.question || "").trim();
    if (!question) {
      return res.status(400).json({ error: "Missing 'question' field in JSON body" });
    }

    const answer = await ask(question, 4);
    return res.json({ answer });
  } catch (err) {
    console.error("[/ask] error:\n" + why(err));
    return res.status(500).json({ error: "Internal error", where: "/ask", detail: why(err) });
  }
});

app.get("/diag", async (_req, res) => {
  try {
    const ping = await fetch("http://localhost:11434/api/tags")
      .then(r => r.json())
      .catch(e => ({ error: String(e) }));

    const buf = Buffer.from("%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF");
    let pdfOK = true;
    try { await pdfParse(buf); } catch { pdfOK = false; }

    let askOK = true;
    try { await ask("ping", 1); } catch { askOK = false; }

    res.json({ ok: true, ollama: ping, pdfParse: pdfOK, askDryRun: askOK });
  } catch (e) {
    res.status(500).json({ ok: false, detail: why(e) });
  }
});

app.listen(PORT, () => {
  console.log(`API is running on http://localhost:${PORT}`);
});