// rag.js
import JSZip from "jszip";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://127.0.0.1:11434";


const embeddings = new OllamaEmbeddings({
  baseUrl: OLLAMA_URL,
  model: process.env.EMBED_MODEL || "nomic-embed-text:latest",
});
const vectorStore = new MemoryVectorStore(embeddings);

function blockToLine(block) {
  if (!block || typeof block !== "object") return "";
  const { opcode, fields = {}, inputs = {} } = block;

  const fieldStr = Object.entries(fields)
    .map(([k, v]) => `${k}=${Array.isArray(v) ? JSON.stringify(v[0]) : JSON.stringify(v)}`)
    .join(", ");

  const inputStr = Object.entries(inputs)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ");

  return `[${opcode}] ${fieldStr}${fieldStr && inputStr ? " | " : ""}${inputStr}`;
}


function summarizeTarget(tgt) {
  const lines = [];
  lines.push(`Target: ${tgt.name} (${tgt.isStage ? "Stage" : "Sprite"})`);

  if (tgt.variables) {
    const vars = Object.values(tgt.variables).map(v => (Array.isArray(v) ? v[0] : v)).join(", ");
    if (vars) lines.push(`Variables: ${vars}`);
  }

  if (tgt.lists) {
    const lists = Object.values(tgt.lists).map(v => (Array.isArray(v) ? v[0] : v)).join(", ");
    if (lists) lines.push(`Lists: ${lists}`);
  }

  if (tgt.broadcasts) {
    const bcasts = Object.values(tgt.broadcasts).join(", ");
    if (bcasts) lines.push(`Broadcasts: ${bcasts}`);
  }

  if (tgt.blocks && typeof tgt.blocks === "object") {
    lines.push("Blocks:");
    for (const [id, blk] of Object.entries(tgt.blocks)) {
      if (blk && blk.opcode) lines.push(`  - ${blockToLine(blk)}`);
    }
  }

  return lines.join("\n");
}


function parseScratchProject(projectJsonStr, filename = "upload.sb3") {
  const proj = JSON.parse(projectJsonStr);

  const meta = [
    `Scratch Project: ${filename}`,
    proj.meta ? `Meta: ${JSON.stringify(proj.meta)}` : null,
  ].filter(Boolean).join("\n");

  const docs = [meta];

  if (Array.isArray(proj.targets)) {
    for (const tgt of proj.targets) {
      docs.push(summarizeTarget(tgt));
    }
  }
  return docs;
}


async function extractProjectJsonFromSb3(buffer) {
  const zip = await JSZip.loadAsync(buffer);
  const projectFile = zip.file("project.json");
  if (!projectFile) throw new Error("project.json not found in .sb3");
  return await projectFile.async("string");
}


async function chunkAndUpsert(texts, sourceId = "scratch") {
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 800, chunkOverlap: 120 });
  const docs = [];
  for (const t of texts) {
    const splits = await splitter.splitText(t);
    for (const s of splits) docs.push({ pageContent: s, metadata: { source: sourceId } });
  }
  await vectorStore.addDocuments(docs);
}

/** PUBLIC: upsert a .sb3 Buffer */
export async function upsertSb3(buffer, filename = "upload.sb3") {
  const projectJson = await extractProjectJsonFromSb3(buffer);
  const texts = parseScratchProject(projectJson, filename);
  await chunkAndUpsert(texts, filename);
  return { ok: true, chunks: texts.length };
}

/** PUBLIC: your existing ask() (unchanged) */
export async function ask(query, k = 6) {
  const retr = await vectorStore.similaritySearch(query, k);
  const context = retr.map((d, i) => `[[${i+1}]] ${d.pageContent}`).join("\n\n");

  const prompt = PromptTemplate.fromTemplate(
`You are a helpful assistant answering questions about a Scratch 3 project.
Use the context below; cite snippet numbers like [1], [2] if helpful.

Question: {question}

Context:
{context}

Answer:`);

  const llm = new ChatOllama({ baseUrl: OLLAMA_URL, model: process.env.GEN_MODEL || "llama3.2:3b" });
  const finalPrompt = await prompt.format({ question: query, context });
  const resp = await llm.invoke(finalPrompt);
  return { answer: resp?.content ?? String(resp), context: retr };
}
