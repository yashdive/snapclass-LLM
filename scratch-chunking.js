const fs = require("fs");
const AdmZip = require("adm-zip");

// 🔹 1. Load project.json from .sb3
function loadSb3Project(sb3Path) {
  const zip = new AdmZip(sb3Path);
  const projectJson = zip.readAsText("project.json");
  return JSON.parse(projectJson);
}

// 🔹 2. Convert a block to human-readable text
function blockToText(block) {
  const opcode = block.opcode;
  const inputs = block.inputs || {};

  const mapping = {
    motion_movesteps: (i) => `move ${getInput(i.STEPS, "10")} steps`,
    motion_turnright: (i) =>
      `turn clockwise ${getInput(i.DEGREES, "15")} degrees`,
    motion_turnleft: (i) =>
      `turn counterclockwise ${getInput(i.DEGREES, "15")} degrees`,
    control_repeat: (i) => `repeat ${getInput(i.TIMES, "10")} times`,
    looks_sayforsecs: (i) =>
      `say ${getInput(i.MESSAGE, "Hello!")} for ${getInput(i.SECS, "2")} seconds`,
    event_whenflagclicked: () => "when green flag clicked",
    event_whenkeypressed: (i) =>
      `when ${getInput(i.KEY_OPTION, "space")} key pressed`,
  };

  return mapping[opcode] ? mapping[opcode](inputs) : opcode;
}


function getInput(inputArr, defaultVal) {
  if (!inputArr || !Array.isArray(inputArr)) return defaultVal;
  if (inputArr.length > 1) return inputArr[1];
  return defaultVal;
}


function scriptToText(startBlockId, blocks) {
  const lines = [];
  let currentId = startBlockId;

  while (currentId) {
    const block = blocks[currentId];
    if (!block) break;
    lines.push(blockToText(block));
    currentId = block.next;
  }

  return lines;
}

// 🔹 4. Chunk Scratch project by sprite + script
function chunkScratchProject(project) {
  const chunks = {};

  for (const target of project.targets) {
    const spriteName = target.name;
    const blocks = target.blocks || {};
    const spriteChunks = [];

    for (const [blockId, block] of Object.entries(blocks)) {
      if (block.topLevel && block.opcode.startsWith("event_")) {
        const scriptLines = scriptToText(blockId, blocks);
        spriteChunks.push(scriptLines.join("\n"));
      }
    }

    chunks[spriteName] = spriteChunks;
  }

  return chunks;
}

// 🔹 Example usage
const sb3Path = "example.sb3"; // path to your Scratch project
const project = loadSb3Project(sb3Path);
const chunks = chunkScratchProject(project);

// Print results
for (const [sprite, scripts] of Object.entries(chunks)) {
  console.log(`\n=== SPRITE: ${sprite} ===`);
  scripts.forEach((script, i) => {
    console.log(`\n--- Script ${i + 1} ---\n${script}`);
  });
}
