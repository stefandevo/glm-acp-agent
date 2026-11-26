import test from "node:test";
import assert from "node:assert/strict";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  SESSION_SCHEMA_VERSION,
  SessionStore,
  type PersistedSession,
} from "../protocol/session-store.js";
import type { GoalState } from "../protocol/goal.js";

function makeDir(): string {
  return mkdtempSync(join(tmpdir(), "glm-acp-session-store-"));
}

function cleanup(dir: string): void {
  rmSync(dir, { recursive: true, force: true });
}

function validSession(overrides: Partial<PersistedSession> = {}): PersistedSession {
  return {
    sessionId: "session-1",
    cwd: "/tmp/project",
    messages: [
      { role: "system", content: "You are a coding assistant." },
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" },
    ],
    title: "A session",
    updatedAt: "2026-09-16T10:00:00.000Z",
    model: "glm-5.3",
    mode: "default",
    thoughtLevel: "max",
    ...overrides,
  };
}

function writeRaw(dir: string, sessionId: string, value: unknown): void {
  writeFileSync(join(dir, `${sessionId}.json`), JSON.stringify(value), "utf8");
}

test("load and listMetadataAsync skip null, primitive, and array JSON roots", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    for (const [index, value] of [null, 7, "session", true, []].entries()) {
      const id = `invalid-root-${index}`;
      writeRaw(dir, id, value);
      assert.equal(store.load(id), undefined, `load should reject ${String(value)}`);
    }

    await assert.doesNotReject(() => store.listMetadataAsync());
    assert.deepEqual(await store.listMetadataAsync(), []);
  } finally {
    cleanup(dir);
  }
});

test("load rejects malformed metadata, mismatched ids, and unsupported versions", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const base = validSession();
    const malformed: Array<[string, unknown]> = [
      ["missing-session-id", { ...base, sessionId: undefined }],
      ["mismatched-id", { ...base, sessionId: "some-other-session" }],
      ["bad-cwd", { ...base, cwd: 12 }],
      ["bad-title", { ...base, title: 12 }],
      ["bad-updated-at", { ...base, updatedAt: null }],
      ["bad-model", { ...base, model: false }],
      ["bad-mode", { ...base, mode: "all" }],
      ["bad-thought-level", { ...base, thoughtLevel: "ultra" }],
      ["bad-display-text", { ...base, displayText: { "0": 12 } }],
      ["bad-display-text-root", { ...base, displayText: [] }],
      ["bad-version-type", { ...base, schemaVersion: "4" }],
      ["unsupported-version", { ...base, schemaVersion: SESSION_SCHEMA_VERSION + 1 }],
      ["old-version-zero", { ...base, schemaVersion: 0 }],
    ];

    for (const [id, value] of malformed) {
      writeRaw(dir, id, value);
      assert.equal(store.load(id), undefined, `load should reject ${id}`);
    }

    await assert.doesNotReject(() => store.listMetadataAsync());
    assert.deepEqual(await store.listMetadataAsync(), []);
  } finally {
    cleanup(dir);
  }
});

test("load rejects malformed messages but accepts supported GlmMessage shapes", () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const invalidMessages: Array<[string, unknown]> = [
      ["messages-null", null],
      ["messages-object", { role: "user", content: "hello" }],
      ["message-null", [null]],
      ["message-primitive", ["hello"]],
      ["message-bad-role", [{ role: "narrator", content: "hello" }]],
      ["message-bad-content", [{ role: "user", content: 42 }]],
      ["message-bad-content-part", [{ role: "user", content: [{}] }]],
      [
        "message-empty-file-part",
        [{ role: "user", content: [{ type: "file", file: {} }] }],
      ],
      [
        "message-file-non-string-filename",
        [{ role: "user", content: [{ type: "file", file: { file_id: "id", filename: 42 } }] }],
      ],
      [
        "message-file-null-data",
        [{ role: "user", content: [{ type: "file", file: { file_id: "id", file_data: null } }] }],
      ],
      [
        "message-file-empty-id",
        [{ role: "user", content: [{ type: "file", file: { file_id: "" } }] }],
      ],
      [
        "message-user-refusal-part",
        [{ role: "user", content: [{ type: "refusal", refusal: "no" }] }],
      ],
      [
        "message-assistant-file-part",
        [{ role: "assistant", content: [{ type: "file", file: { file_id: "id" } }] }],
      ],
      ["message-missing-content", [{ role: "user" }]],
      ["message-tool-missing-id", [{ role: "tool", content: "result" }]],
      ["message-tool-null-content", [{ role: "tool", tool_call_id: "call-1", content: null }]],
      [
        "message-bad-tool-calls",
        [{ role: "assistant", content: null, tool_calls: [{ id: 42 }] }],
      ],
      [
        "message-tool-call-missing-function",
        [{ role: "assistant", content: null, tool_calls: [{ id: "call-1", type: "function" }] }],
      ],
      [
        "message-malformed-function-call",
        [{ role: "assistant", content: null, function_call: { name: 42, arguments: null } }],
      ],
      [
        "message-user-tool-calls",
        [{ role: "user", content: "hello", tool_calls: [] }],
      ],
    ];
    for (const [id, messages] of invalidMessages) {
      writeRaw(dir, id, { ...validSession({ sessionId: id }), messages });
      assert.equal(store.load(id), undefined, `load should reject ${id}`);
    }

    const supported = validSession({
      sessionId: "supported-shapes",
      messages: [
        { role: "developer", content: [{ type: "text", text: "rules" }] },
        {
          role: "user",
          content: [
            { type: "text", text: "inspect this" },
            { type: "image_url", image_url: { url: "data:image/png;base64,AA==" } },
            { type: "file", file: { file_data: "encoded-file", filename: "notes.txt" } },
          ],
        },
        {
          role: "assistant",
          content: [{ type: "refusal", refusal: "not needed" }],
        },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call-1",
              type: "function",
              function: { name: "read_file", arguments: '{"path":"x"}' },
            },
          ],
        },
        { role: "tool", tool_call_id: "call-1", content: "file contents" },
      ],
    });
    writeRaw(dir, supported.sessionId, supported);
    assert.deepEqual(store.load(supported.sessionId)?.messages, supported.messages);
  } finally {
    cleanup(dir);
  }
});

test("load migrates valid v1 through v4 records without losing fields", () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const records: Array<[string, Record<string, unknown>]> = [
      ["v1", { ...validSession({ sessionId: "v1" }), schemaVersion: undefined, mode: undefined, thoughtLevel: undefined }],
      ["v2", { ...validSession({ sessionId: "v2" }), schemaVersion: 2, thoughtLevel: undefined }],
      ["v3", { ...validSession({ sessionId: "v3" }), schemaVersion: 3, displayText: undefined }],
      ["v4", { ...validSession({ sessionId: "v4" }), schemaVersion: 4, displayText: { "1": "hello" }, goal: undefined }],
    ];
    for (const [id, value] of records) {
      writeRaw(dir, id, value);
      const loaded = store.load(id);
      assert.equal(loaded?.schemaVersion, SESSION_SCHEMA_VERSION, id);
      assert.equal(loaded?.sessionId, id);
      assert.equal(loaded?.thoughtLevel, "max", id);
      assert.equal(loaded?.mode, "default", id);
      assert.equal(loaded?.goal, undefined, id);
      if (id === "v4") assert.deepEqual(loaded?.displayText, { "1": "hello" });
    }
  } finally {
    cleanup(dir);
  }
});

test("a v5 record round-trips its goal and rejects a malformed one", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const goal: GoalState = {
      condition: "all tests pass",
      status: "paused",
      judgeModel: "glm-5.3-flash",
      setAt: "2026-11-27T10:00:00.000Z",
      rounds: 2,
      maxRounds: 20,
      noProgressRounds: 0,
      lastVerdict: "not_met",
      lastReason: "refresh still fails",
      pauseReason: "budget",
      tokens: { inputTokens: 11, outputTokens: 7, totalTokens: 18 },
    };
    const session = validSession({ goal });
    await store.save(session);
    assert.deepEqual(store.load(session.sessionId), { ...session, schemaVersion: SESSION_SCHEMA_VERSION });

    for (const [index, broken] of [
      { ...goal, status: "done" },
      { ...goal, rounds: -1 },
      { ...goal, maxRounds: 0 },
      { ...goal, pauseReason: "because" },
      { ...goal, tokens: { inputTokens: 1 } },
      "goal",
    ].entries()) {
      const id = `bad-goal-${index}`;
      writeRaw(dir, id, validSession({ sessionId: id, goal: broken as PersistedSession["goal"] }));
      assert.equal(store.load(id), undefined, JSON.stringify(broken));
    }
    // Validation runs before any I/O, so the rejection is synchronous.
    assert.throws(
      () => store.save(validSession({ goal: { ...goal, status: "finished" } as unknown as GoalState })),
      /Invalid persisted session/
    );
  } finally {
    cleanup(dir);
  }
});

test("save atomically replaces a broad-mode record and keeps the file private", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const first = validSession({ updatedAt: "2026-09-16T10:00:00.000Z" });
    await store.save(first);
    const path = join(dir, `${first.sessionId}.json`);
    chmodSync(path, 0o666);

    const second = validSession({
      title: "Replaced session",
      updatedAt: "2026-09-16T11:00:00.000Z",
      messages: [...first.messages, { role: "user", content: "new turn" }],
    });
    await store.save(second);

    // POSIX exposes the mode bits; Windows does not provide this permission
    // contract, while the replacement/load/temporary-file assertions remain
    // meaningful on every supported platform.
    if ((process.platform as string) !== "win32") {
      assert.equal(statSync(path).mode & 0o777, 0o600);
    }
    assert.deepEqual(store.load(first.sessionId), { ...second, schemaVersion: SESSION_SCHEMA_VERSION });
    assert.deepEqual(readdirSync(dir).filter((name) => name.includes(".tmp")), []);
  } finally {
    cleanup(dir);
  }
});

test("save uses invocation-time snapshots and orders concurrent writes per session", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const first = validSession({ sessionId: "ordered", title: "first snapshot" });
    const expectedMessages = structuredClone(first.messages);
    const firstSave = store.save(first);
    first.messages.push({ role: "user", content: "mutated after save invocation" });
    await firstSave;
    assert.deepEqual(
      store.load(first.sessionId)?.messages,
      expectedMessages,
      "the persisted snapshot must reflect invocation time, not later mutation",
    );

    // Artificially delay the first queued write by pre-seeding its FIFO slot,
    // the way a slow predecessor save would: the second snapshot is enqueued
    // while the first is still pending and must still land last.
    const pendingSaves = (store as unknown as { pendingSaves: Map<string, Promise<void>> }).pendingSaves;
    pendingSaves.set("ordered", new Promise<void>((resolve) => { setTimeout(resolve, 40); }));
    const older = store.save(validSession({ sessionId: "ordered", title: "older queued snapshot" }));
    const newer = store.save(validSession({ sessionId: "ordered", title: "newer queued snapshot" }));
    await Promise.all([older, newer]);
    assert.equal(store.load(first.sessionId)?.title, "newer queued snapshot");
  } finally {
    cleanup(dir);
  }
});

test("flush drains pending saves including ones queued during the flush", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const pendingSaves = (store as unknown as { pendingSaves: Map<string, Promise<void>> }).pendingSaves;
    pendingSaves.set("flushed", new Promise<void>((resolve) => { setTimeout(resolve, 40); }));
    const early = store.save(validSession({ sessionId: "flushed", title: "early snapshot" }));
    // The flush's first drain only sees `early`; `late` is queued mid-flush
    // and must extend the drain rather than being left in the queue.
    const flushing = store.flush();
    const late = store.save(validSession({ sessionId: "flushed", title: "late snapshot" }));
    await flushing;
    assert.equal(store.load("flushed")?.title, "late snapshot");
    await Promise.all([early, late]);
  } finally {
    cleanup(dir);
  }
});

test("save cleans up its temporary file when replacement fails", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const session = validSession({ sessionId: "rename-fails" });
    const target = join(dir, `${session.sessionId}.json`);
    mkdirSync(target);

    await assert.rejects(store.save(session));
    assert.equal(existsSync(target), true);
    assert.equal(statSync(target).isDirectory(), true);
    assert.deepEqual(
      readdirSync(dir).filter((name) => name.includes(".tmp")),
      [],
      "failed atomic save must not leave temporary artifacts"
    );
  } finally {
    cleanup(dir);
  }
});

test("flush reports a failed accepted save even after a later snapshot succeeds", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const failed = validSession({ sessionId: "failed-flush" });
    mkdirSync(join(dir, `${failed.sessionId}.json`));
    await assert.rejects(store.save(failed));
    rmSync(join(dir, `${failed.sessionId}.json`), { recursive: true });
    await store.save(validSession({ sessionId: failed.sessionId, title: "recovered" }));
    await assert.rejects(store.flush(), /./);
    assert.equal(store.load(failed.sessionId)?.title, "recovered");
    await store.flush();
  } finally {
    cleanup(dir);
  }
});

test("flush drains successful writes before rejecting a concurrently accepted failed write", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    mkdirSync(join(dir, "failed-during-flush.json"));
    const failed = store.save(validSession({ sessionId: "failed-during-flush" }));
    const flushing = assert.rejects(store.flush());
    const successful = store.save(validSession({ sessionId: "saved-during-flush", title: "durable" }));
    await flushing;
    assert.equal(store.load("saved-during-flush")?.title, "durable");
    await assert.rejects(failed);
    await successful;
  } finally {
    cleanup(dir);
  }
});

test("concurrent flush callers both observe a failed accepted write", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    mkdirSync(join(dir, "shared-failure.json"));
    const rejectedSave = assert.rejects(store.save(validSession({ sessionId: "shared-failure" })));
    await Promise.all([
      assert.rejects(store.flush()),
      assert.rejects(store.flush()),
      rejectedSave,
    ]);
    await store.flush();
  } finally {
    cleanup(dir);
  }
});

test("save preserves the prior record when serializing the replacement fails", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const first = validSession({ sessionId: "serialization-fails" });
    await store.save(first);
    const path = join(dir, `${first.sessionId}.json`);
    const before = readFileSync(path, "utf8");
    const cyclicMessage = { role: "user", content: "bad" } as Record<string, unknown>;
    cyclicMessage.self = cyclicMessage;

    assert.throws(() =>
      store.save(validSession({ sessionId: first.sessionId, messages: [cyclicMessage] as never }))
    );
    assert.equal(readFileSync(path, "utf8"), before);
    assert.deepEqual(readdirSync(dir).filter((name) => name.includes(".tmp")), []);
  } finally {
    cleanup(dir);
  }
});

test("loadAsync waits for the in-flight queued save of the same session (read-your-writes)", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    const old = validSession({ sessionId: "racing", title: "old record" });
    await store.save(old);

    // Start the replacement save WITHOUT awaiting it: the snapshot is queued
    // in the session's FIFO and its multi-step write (open/write/sync/rename)
    // is still in flight. loadAsync chains on that in-flight entry, so the
    // read cannot race ahead of the rename and observe the pre-rename file
    // holding the OLD record.
    const updated = validSession({ sessionId: "racing", title: "new record" });
    const pending = store.save(updated);
    const loaded = await store.loadAsync(old.sessionId);
    assert.deepEqual(loaded, { ...updated, schemaVersion: SESSION_SCHEMA_VERSION });
    await pending;
  } finally {
    cleanup(dir);
  }
});

test("listMetadataAsync lists sessions in parallel, sorted by updatedAt desc, skipping invalid entries", async () => {
  const dir = makeDir();
  try {
    const store = new SessionStore(dir);
    // More files than the concurrency bound so the parallel path is exercised.
    const ids = Array.from({ length: 12 }, (_, index) => `par-${index}`);
    for (const id of ids) {
      const seconds = String(Number(id.slice("par-".length))).padStart(2, "0");
      writeRaw(dir, id, validSession({ sessionId: id, updatedAt: `2026-09-01T00:00:${seconds}.000Z` }));
    }
    // Non-JSON file: excluded by the extension filter.
    writeFileSync(join(dir, "notes.txt"), "not a session", "utf8");
    // Malformed JSON root: parsed but rejected.
    writeFileSync(join(dir, "broken-json.json"), "{ not json", "utf8");
    // Session id mismatch between filename and record: rejected by the parser.
    writeRaw(dir, "mismatched", validSession({ sessionId: "some-other-session" }));

    const metadata = await store.listMetadataAsync();
    assert.equal(metadata.length, ids.length, "only the valid session files are listed");
    assert.deepEqual(
      metadata.map((entry) => entry.sessionId),
      [...ids].reverse(),
      "entries are sorted by updatedAt descending",
    );
  } finally {
    cleanup(dir);
  }
});
