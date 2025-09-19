import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://127.0.0.1:11434";

export const loadAndSplitTheDocs = async (file_path) => {
  // load the uploaded file data
  const loader = new PDFLoader(file_path);
  const docs = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const allSplits = await textSplitter.splitDocuments(docs);
  return allSplits;
};

export const vectorSaveAndSearch = async (splits,question) => {
  const embeddings = new OllamaEmbeddings({
    model: process.env.EMBED_MODEL || "nomic-embed-text:latest", 
    baseUrl: OLLAMA_URL,
  });
    const vectorStore = await MemoryVectorStore.fromDocuments(
        splits,
        embeddings
    );

    const searches = await vectorStore.similaritySearch(question);
    return searches;
};

export const generatePrompt = async (searches,question) =>
{
    let context = "";
    searches.forEach((search) => {
        context = context + "\n\n" + search.pageContent;
    });

    const prompt = PromptTemplate.fromTemplate(`
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
`);

    const formattedPrompt = await prompt.format({
        context: context,
        question: question,
    });
    return formattedPrompt;
}


export const generateOutput = async (prompt) =>
{
  const llm = new Ollama({
    model: process.env.OLLAMA_MODEL || "llama3.2:3b",
    baseUrl: OLLAMA_URL,
  });

    const response = await ollamaLlm.invoke(prompt);
    return response;
}

// Reuse existing objects/functions if present on the module scope:
const _has = (name) => typeof globalThis[name] !== "undefined" || typeof eval?.(name) !== "undefined";
const _maybe = (name) => {
  try { return eval?.(name); } catch { return undefined; }
};

// 1) Splitter: prefer an existing one; else create a simple default
let __splitter = _maybe("splitter");
async function __splitText(raw, chunkSize = 1000, chunkOverlap = 150) {
  if (__splitter && typeof __splitter.splitText === "function") {
    return await __splitter.splitText(raw);
  }
  // Fallback: try LangChain text splitter if available, else naive splitter
  try {
    const { RecursiveCharacterTextSplitter } = await import("@langchain/textsplitters").catch(() => ({}));
    if (RecursiveCharacterTextSplitter) {
      __splitter = new RecursiveCharacterTextSplitter({ chunkSize, chunkOverlap });
      return await __splitter.splitText(raw);
    }
  } catch {}
  // Naive: split on ~chunkSize chars by sentence-ish boundaries
  const parts = [];
  let buf = "";
  for (const piece of (raw || "").split(/(?<=[\.\?\!])\s+/)) {
    if ((buf + " " + piece).length > chunkSize && buf) {
      parts.push(buf.trim());
      buf = piece;
    } else {
      buf = buf ? buf + " " + piece : piece;
    }
  }
  if (buf) parts.push(buf.trim());
  return parts;
}

// 2) Vector store & embeddings: reuse if present; else create a minimal one
let __vectorStore = _maybe("vectorStore");
let __embeddings = _maybe("embeddings");

async function __ensureEmbeddings() {
  if (__embeddings) return __embeddings;
  // Prefer @langchain/ollama embeddings if installed
  try {
    const { OllamaEmbeddings } = await import("@langchain/ollama");
    __embeddings = new OllamaEmbeddings({ model: process.env.EMBED_MODEL || "nomic-embed-text" });
    return __embeddings;
  } catch {}
  // Soft fallback: simple JS embedding (very poor quality, avoids crashes if libs missing)
  __embeddings = {
    embedQuery: async (q) => Array.from(q).map(c => c.charCodeAt(0) % 13), // toy hash
    embedDocuments: async (docs) => docs.map((d) => Array.from(d).map(c => c.charCodeAt(0) % 13)),
  };
  return __embeddings;
}

async function __ensureVectorStore() {
  if (__vectorStore) return __vectorStore;
  try {
    // Try Chroma vector store from @langchain/community
    const { Chroma } = await import("@langchain/community/vectorstores/chroma");
    const embeddings = await __ensureEmbeddings();
    // If you’re running a Chroma server, set CHROMA_URL; otherwise this will use local persistence when available
    const options = {};
    if (process.env.CHROMA_URL) options.url = process.env.CHROMA_URL;
    __vectorStore = await Chroma.fromTexts([], [], embeddings, {
      collectionName: process.env.CHROMA_COLLECTION || "docs",
      ...options,
    });
    return __vectorStore;
  } catch (e) {
    // Final fallback: in-memory basic store
    const _docs = [];
    const _vecs = [];
    const emb = await __ensureEmbeddings();
    __vectorStore = {
      addDocuments: async (docs) => {
        for (const d of docs) {
          _docs.push(d);
        }
        const vecs = await emb.embedDocuments(docs.map((d) => d.pageContent));
        _vecs.push(...vecs);
      },
      asRetriever: (k = 4) => ({
        getRelevantDocuments: async (q) => {
          const qv = await emb.embedQuery(q);
          const scored = _vecs.map((v, i) => [i, __cosSim(qv, v)]);
          scored.sort((a, b) => b[1] - a[1]);
          return scored.slice(0, k).map(([i]) => _docs[i]);
        },
      }),
    };
    return __vectorStore;
  }
}

// Tiny cosine similarity for the fallback store
function __cosSim(a, b) {
  const n = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na || 1) * Math.sqrt(nb || 1));
}

// 3) Ollama caller: reuse if you have one (e.g., callOllama), else minimal fetch
async function __callOllama(prompt, model, opts = {}) {
  const existing = _maybe("callOllama") || _maybe("ollamaCall");
  if (typeof existing === "function") return await existing(prompt, model, opts);

  const url = process.env.OLLAMA_URL || "http://localhost:11434/api/generate";
  const body = {
    model: model || process.env.OLLAMA_MODEL || "llama3.2:3b",
    prompt,
    temperature: opts.temperature ?? 0.2,
    max_tokens: opts.max_tokens ?? 500,
    stream: false,
  };
  const fetcher = globalThis.fetch ?? (await import("node-fetch")).default;
  const r = await fetcher(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`Ollama error ${r.status}: ${await r.text().catch(() => "")}`);
  const j = await r.json();
  return j?.response ?? "";
}




export async function initRag(opts = {}) {
    if (opts.splitter && typeof opts.splitter.splitText === "function") __splitter = opts.splitter;
    if (opts.embeddings) __embeddings = opts.embeddings;
    __vectorStore = opts.vectorStore || (await __ensureVectorStore());
    return { vectorStore: __vectorStore };
  }
  
  /** Upsert raw text into the store: split -> embed -> add */
  export async function upsertText(sourceId, rawText) {
    if (!rawText || !rawText.trim()) return 0;
    const chunks = await __splitText(rawText);
    if (!chunks.length) return 0;
    const docs = chunks.map((t, i) => ({ pageContent: t, metadata: { sourceId, idx: i } }));
    const store = await __ensureVectorStore();
    await store.addDocuments(docs);
    return docs.length;
  }

  export const askWithDocs = async (question, docs, k = 4) => {
    if (!question?.trim()) return "";
    const { topKText } = await vectorSaveAndSearch(docs, question, k);
    if (!topKText?.trim()) return "I couldn't find relevant context in the document.";
    const prompt = await generatePrompt(question, topKText);
    const out = await generateOutput(prompt);
    return typeof out === "string" ? out : (out?.content ?? out?.response ?? "");
  };
  
  /** Ask a question using retrieved context, returning a concise answer string */
  export async function ask(question, k = 4, opts = {}) {
    if (!question || !question.trim()) return "";
    const store = await __ensureVectorStore();
    const retriever = store.asRetriever ? store.asRetriever(k) : { getRelevantDocuments: async () => [] };
    const docs = await retriever.getRelevantDocuments(question);
    const context = (docs || []).map((d, i) => `[Doc ${i + 1}]\n${d.pageContent}`).join("\n\n");
    const prompt =
      `You are a helpful assistant. Use the context to answer concisely.\n\n` +
      `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
    return await __callOllama(prompt, opts.model, opts);
  }
