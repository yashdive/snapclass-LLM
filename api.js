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
  const tmp = path.join(os.tmpdir(), `${Date.now()}-${req.file?.originalname || "upload"}.pdf`);
  try {
    if (!req.file) return res.status(400).json({ error: "Missing file 'file'" });
    const question = (req.body?.question || "").trim();
    if (!question) return res.status(400).json({ error: "Missing 'question'" });

    fs.writeFileSync(tmp, req.file.buffer);
    const loader = new PDFLoader(tmp);
    const docs = await loader.load();

    const answer = await askWithDocs(question, docs, 4);
    return res.json({ answer, file: req.file.originalname });
  } catch (e) {
    console.error("[/upload] error:", e?.stack || e);
    return res.status(500).json({ error: "Internal error", detail: String(e?.message || e) });
  } finally {
    try { fs.unlinkSync(tmp); } catch {}
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
  // helper to stringify errors nicely
  const why = (e) => (e && e.stack) ? e.stack : String(e?.message || e);

  try {
    // 1) Check Ollama daemon & models
    let ollama = { ok: false, status: 0, models: [], error: "" };
    try {
      const r = await fetch(`${OLLAMA_URL}/api/tags`);
      const j = await r.json();
      ollama.ok = r.ok;
      ollama.status = r.status;
      ollama.models = Array.isArray(j?.models) ? j.models.map(m => m.name) : [];
    } catch (e) {
      ollama.error = why(e);
    }

    // 2) Check PDFLoader path (write tiny temp PDF, then load)
    let pdfLoaderOK = false;
    let pdfLoaderDetail = "";
    const tmp = path.join(os.tmpdir(), `diag-${Date.now()}.pdf`);
    try {
      const minimalPdf = Buffer.from(
        "%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n" +
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n" +
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>\nendobj\n" +
        "4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 72 120 Td (hello) Tj ET\nendstream\nendobj\n" +
        "xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000062 00000 n \n0000000123 00000 n \n0000000221 00000 n \n" +
        "trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n320\n%%EOF"
      );
      fs.writeFileSync(tmp, minimalPdf);

      const loader = new PDFLoader(tmp);
      const docs = await loader.load();
      pdfLoaderOK = Array.isArray(docs) && docs.length > 0;
      if (!pdfLoaderOK) pdfLoaderDetail = "PDFLoader returned no docs";
    } catch (e) {
      pdfLoaderOK = false;
      pdfLoaderDetail = why(e);
    } finally {
      try { fs.unlinkSync(tmp); } catch {}
    }

    // 3) Try a dry-run of ask() (should not throw even without docs)
    let askDryRun = false;
    let askDetail = "";
    try {
      const maybe = await ask("ping", 1);
      askDryRun = true;
      // include a small preview if it returns text
      if (typeof maybe === "string") askDetail = maybe.slice(0, 120);
    } catch (e) {
      askDryRun = false;
      askDetail = why(e);
    }

    return res.json({
      ok: true,
      ollama,
      pdfLoader: { ok: pdfLoaderOK, detail: pdfLoaderDetail },
      askDryRun: { ok: askDryRun, detail: askDetail }
    });
  } catch (e) {
    return res.status(500).json({ ok: false, detail: (e && e.stack) ? e.stack : String(e) });
  }
});

app.listen(PORT, () => {
  console.log(`API is running on http://localhost:${PORT}`);
});