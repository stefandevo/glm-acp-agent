import test from "node:test";
import assert from "node:assert/strict";
import { SessionLifecycle } from "../protocol/session-lifecycle.js";

test("a transition lease synchronously excludes prompts and stale releases", () => {
  const lifecycle = new SessionLifecycle();
  assert.equal(lifecycle.acceptsPrompts(), true);

  const restoring = lifecycle.begin("restoring");
  assert.equal(lifecycle.phase, "restoring");
  assert.equal(lifecycle.acceptsPrompts(), false);
  assert.throws(() => lifecycle.begin("snapshotting"), /transition in progress/i);

  restoring.release();
  assert.equal(lifecycle.phase, "open");
  assert.equal(lifecycle.acceptsPrompts(), true);

  const closing = lifecycle.begin("closing");
  restoring.release();
  assert.equal(lifecycle.phase, "closing", "an old lease cannot reopen a newer transition");
  assert.equal(lifecycle.owns(closing), true);
});
