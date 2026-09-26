import { performance } from "node:perf_hooks";
import { preprocessImageBlocks } from "../dist/protocol/image-preprocessor.js";

const imageCount = Number(process.argv[2] ?? 3);
const delayMs = Number(process.argv[3] ?? 70);
const repetitions = Number(process.argv[4] ?? 7);
let activeCalls = 0;
let peakConcurrentCalls = 0;
const client = {
  async callTool(_name, args) {
    activeCalls += 1;
    peakConcurrentCalls = Math.max(peakConcurrentCalls, activeCalls);
    await new Promise(resolve => setTimeout(resolve, delayMs));
    activeCalls -= 1;
    return { content: [{ type: "text", text: args.image_source }] };
  },
  async dispose() {},
};
const blocks = Array.from({ length: imageCount }, (_, index) => ({
  type: "image",
  mimeType: "image/png",
  uri: "fixture://image-" + index,
}));
const samples = [];
for (let index = 0; index < repetitions; index += 1) {
  const start = performance.now();
  const result = await preprocessImageBlocks(blocks, client);
  samples.push(Number((performance.now() - start).toFixed(1)));
  if (result.blocks.length !== imageCount) throw new Error("preprocessor returned an unexpected block count");
}
const sortedWarm = samples.slice(1).sort((a, b) => a - b);
console.log(JSON.stringify({ node: process.version, imageCount, delayMsPerAnalysis: delayMs, repetitions, firstRunMs: samples[0], warmRunSamplesMs: samples.slice(1), warmMedianMs: sortedWarm[Math.floor(sortedWarm.length / 2)], peakConcurrentVisionCalls: peakConcurrentCalls }, null, 2));
