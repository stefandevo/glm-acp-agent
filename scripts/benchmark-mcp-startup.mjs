import { createServer } from "node:http";
import { once } from "node:events";
import { performance } from "node:perf_hooks";
import { connectSessionMcpServers } from "../dist/tools/session-mcp-client.js";

const serverCount = Number(process.argv[2] ?? 3);
const delayMs = Number(process.argv[3] ?? 60);
const repetitions = Number(process.argv[4] ?? 6);
const fixtures = [];
let activeRequests = 0;
let peakActiveRequests = 0;

for (let index = 0; index < serverCount; index += 1) {
  const fixture = createServer(async (request, response) => {
    if (request.method === "DELETE") {
      response.writeHead(202).end();
      return;
    }
    let text = "";
    for await (const chunk of request) text += chunk;
    const message = JSON.parse(text);
    activeRequests += 1;
    peakActiveRequests = Math.max(peakActiveRequests, activeRequests);
    await new Promise(resolve => setTimeout(resolve, delayMs));
    activeRequests -= 1;
    response.setHeader("Content-Type", "application/json");
    if (message.method === "notifications/initialized") {
      response.writeHead(202).end();
      return;
    }
    let result;
    if (message.method === "initialize") {
      response.setHeader("MCP-Session-Id", "fixture-" + index);
      result = { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: "delayed-" + index, version: "1" } };
    } else if (message.method === "tools/list") {
      result = { tools: [{ name: "echo", inputSchema: { type: "object", properties: {} } }] };
    } else {
      response.writeHead(400).end();
      return;
    }
    response.writeHead(200).end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }));
  });
  fixture.listen(0, "127.0.0.1");
  await once(fixture, "listening");
  fixtures.push(fixture);
}

const servers = fixtures.map((fixture, index) => ({
  type: "http",
  name: "fixture-" + index,
  url: "http://127.0.0.1:" + fixture.address().port,
  headers: [],
}));
const samples = [];
for (let index = 0; index < repetitions; index += 1) {
  const start = performance.now();
  const tools = await connectSessionMcpServers(servers);
  samples.push(Number((performance.now() - start).toFixed(1)));
  await tools.dispose();
}
const sortedWarm = samples.slice(1).sort((a, b) => a - b);
const warmMedian = sortedWarm[Math.floor(sortedWarm.length / 2)];
console.log(JSON.stringify({ node: process.version, serverCount, delayMsPerRequest: delayMs, requestsPerServer: 3, repetitions, firstRunMs: samples[0], warmRunSamplesMs: samples.slice(1), warmMedianMs: warmMedian, peakConcurrentFixtureRequests: peakActiveRequests }, null, 2));
await Promise.all(fixtures.map(fixture => new Promise((resolve, reject) => fixture.close(error => error ? reject(error) : resolve()))));
