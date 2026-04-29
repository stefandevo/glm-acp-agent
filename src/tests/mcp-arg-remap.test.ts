import test from "node:test";
import assert from "node:assert/strict";
import { remapArguments, resolveToolName } from "../tools/mcp-arg-remap.js";

// ---------------------------------------------------------------------------
// remapArguments
// ---------------------------------------------------------------------------

test("remapArguments: passes through unchanged when targetProperties is empty (no schema)", () => {
  const args = { query: "hello", count: 5 };
  const result = remapArguments(args, []);
  assert.deepEqual(result, { query: "hello", count: 5 });
});

test("remapArguments: passes through when key already matches a target property", () => {
  const result = remapArguments({ search_query: "hello" }, ["search_query", "count"]);
  assert.deepEqual(result, { search_query: "hello" });
});

test("remapArguments: remaps query → search_query when search_query is in targetProperties", () => {
  const result = remapArguments({ query: "hello", count: 3 }, ["search_query", "count"]);
  assert.deepEqual(result, { search_query: "hello", count: 3 });
});

test("remapArguments: passes through unrecognised key that has no alias", () => {
  const result = remapArguments({ unknown_key: "val" }, ["search_query"]);
  assert.deepEqual(result, { unknown_key: "val" });
});

test("remapArguments: passes through alias key when alias not in targetProperties", () => {
  // query → search_query alias exists, but target schema doesn't have search_query
  const result = remapArguments({ query: "hello" }, ["q"]);
  assert.deepEqual(result, { query: "hello" });
});

test("remapArguments: canonical key wins over alias (key already in targetProperties)", () => {
  // If caller somehow passes search_query directly and it matches, no remapping occurs
  const result = remapArguments({ search_query: "direct" }, ["search_query"]);
  assert.deepEqual(result, { search_query: "direct" });
});

// ---------------------------------------------------------------------------
// resolveToolName
// ---------------------------------------------------------------------------

test("resolveToolName: exact match takes priority", () => {
  const result = resolveToolName("webSearchPrime", ["webSearchPrime", "webReader"], "test-endpoint");
  assert.equal(result, "webSearchPrime");
});

test("resolveToolName: returns requestedName unchanged when availableTools is empty (fail-open)", () => {
  const result = resolveToolName("image_analysis", [], "test-endpoint");
  assert.equal(result, "image_analysis");
});

test("resolveToolName: keyword fallback matches search tool with different name", () => {
  const result = resolveToolName("webSearch", ["SearchToolV2", "readerTool"], "test-endpoint");
  assert.equal(result, "SearchToolV2");
});

test("resolveToolName: keyword fallback matches reader tool with different name", () => {
  const result = resolveToolName("webReader", ["searchTool", "ContentReaderV2"], "test-endpoint");
  assert.equal(result, "ContentReaderV2");
});

test("resolveToolName: keyword fallback matches vision/image tool with different name", () => {
  const result = resolveToolName("image_analysis", ["analyzeImage", "otherTool"], "test-endpoint");
  assert.equal(result, "analyzeImage");
});

test("resolveToolName: throws descriptive error when no match found", () => {
  assert.throws(
    () => resolveToolName("unknownTool", ["webSearch", "webReader"], "https://api.example.com"),
    (err: Error) => {
      assert.ok(err.message.includes('"unknownTool"'));
      assert.ok(err.message.includes("webSearch"));
      assert.ok(err.message.includes("webReader"));
      return true;
    }
  );
});
