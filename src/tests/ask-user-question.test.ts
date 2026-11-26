import test from "node:test";
import assert from "node:assert/strict";
import { ToolExecutor } from "../tools/executor.js";
import {
  ASK_DECLINE_OPTION_ID,
  buildAskElicitation,
  foldAskResponse,
  parseAskQuestions,
  type AskQuestion,
} from "../tools/ask-question.js";
import { TOOL_DEFINITIONS } from "../tools/definitions.js";
import { validateToolArguments } from "../tools/argument-validation.js";

const FORM_CAPS = { elicitation: { form: {} } };

type Response = Record<string, unknown> | ((params: Record<string, unknown>) => Record<string, unknown>);

function askConnection(response: Response) {
  const updates: Array<Record<string, unknown>> = [];
  const elicitations: Array<Record<string, unknown>> = [];
  const permissionRequests: Array<Record<string, unknown>> = [];
  const reply = (params: Record<string, unknown>) =>
    typeof response === "function" ? response(params) : response;
  return {
    updates,
    elicitations,
    permissionRequests,
    async sessionUpdate(payload: Record<string, unknown>) {
      updates.push(payload);
    },
    async unstable_createElicitation(params: Record<string, unknown>) {
      elicitations.push(params);
      return reply(params);
    },
    async requestPermission(params: Record<string, unknown>) {
      permissionRequests.push(params);
      return reply(params);
    },
  };
}

const SINGLE_ARGS = {
  questions: [
    {
      question: "Which library should we use for date formatting?",
      header: "Library",
      multiSelect: false,
      options: [
        { label: "chrono", description: "The de facto standard." },
        { label: "time", description: "Lighter, no timezone database." },
      ],
    },
  ],
};

const MULTI_ARGS = {
  questions: [
    {
      question: "Which features do you want to enable?",
      header: "Features",
      multiSelect: true,
      options: [
        { label: "replays", description: "Record and replay runs." },
        { label: "bench", description: "Performance budgets." },
      ],
    },
  ],
};

const QUESTION_TEXT = "Which library should we use for date formatting?";

function parsed(args: Record<string, unknown>): AskQuestion[] {
  const questions = parseAskQuestions(args);
  assert.ok(typeof questions !== "string", typeof questions === "string" ? questions : "");
  return questions;
}

test("the tool is defined and its arguments validate", () => {
  const definition = TOOL_DEFINITIONS.find((entry) => entry.function.name === "ask_user_question");
  assert.ok(definition);
  const schema = definition!.function.parameters as Record<string, Record<string, Record<string, unknown>>>;
  const questions = schema["properties"]["questions"];
  assert.equal(questions["minItems"], 1);
  assert.equal(questions["maxItems"], 4);

  assert.equal(validateToolArguments("ask_user_question", JSON.stringify(SINGLE_ARGS)).ok, true);
  assert.equal(validateToolArguments("ask_user_question", JSON.stringify(MULTI_ARGS)).ok, true);

  const nonBoolean = structuredClone(SINGLE_ARGS);
  (nonBoolean.questions[0] as Record<string, unknown>).multiSelect = "true";
  const validation = validateToolArguments("ask_user_question", JSON.stringify(nonBoolean));
  assert.equal(validation.ok, false);
  assert.match((validation as { message: string }).message, /multiSelect must be a boolean/);

  const tooMany = structuredClone(SINGLE_ARGS);
  tooMany.questions = [SINGLE_ARGS.questions[0], SINGLE_ARGS.questions[0], SINGLE_ARGS.questions[0], SINGLE_ARGS.questions[0], SINGLE_ARGS.questions[0]];
  assert.equal(validateToolArguments("ask_user_question", JSON.stringify(tooMany)).ok, false);
});

test("parseAskQuestions enforces the bounds", () => {
  assert.ok(typeof parseAskQuestions({ questions: "no" }) === "string");
  assert.ok(typeof parseAskQuestions({ questions: [] }) === "string");

  const five = { questions: [1, 2, 3, 4, 5].map(() => structuredClone(SINGLE_ARGS.questions[0])) };
  assert.match(parseAskQuestions(five) as string, /1-4 questions/);

  const oneOption = structuredClone(SINGLE_ARGS);
  oneOption.questions[0].options = [SINGLE_ARGS.questions[0].options[0]];
  assert.match(parseAskQuestions(oneOption) as string, /2-4 options/);

  const noDescription = structuredClone(SINGLE_ARGS);
  delete (noDescription.questions[0].options[0] as { description?: string }).description;
  assert.match(parseAskQuestions(noDescription) as string, /description is required/);
});

test("buildAskElicitation renders one question as a titled enum", () => {
  const request = buildAskElicitation(parsed(SINGLE_ARGS), "s1", "t1") as Record<string, unknown>;
  assert.equal(request["mode"], "form");
  assert.equal(request["sessionId"], "s1");
  assert.equal(request["toolCallId"], "t1");
  assert.equal(request["message"], QUESTION_TEXT);
  const schema = request["requestedSchema"] as Record<string, unknown>;
  const properties = schema["properties"] as Record<string, Record<string, unknown>>;
  const field = properties["question_0"];
  assert.equal(field["type"], "string");
  assert.equal(field["title"], "Library");
  assert.equal(field["description"], undefined);
  const oneOf = field["oneOf"] as Array<Record<string, unknown>>;
  assert.equal(oneOf[0]["const"], "chrono");
  assert.equal(oneOf[0]["title"], "chrono");
  assert.equal(oneOf[0]["description"], "The de facto standard.");
  const custom = properties["question_0_custom"];
  assert.equal(custom["title"], "Other");
  // the marker the claude-agent-acp adapter sets, so clients recognize the box
  assert.deepEqual(custom["_meta"], {
    _askUserQuestionCustomAnswer: { questionId: "question_0", isCustomAnswer: true },
  });
  assert.equal(schema["required"], undefined);
});

test("buildAskElicitation carries each question's text with several", () => {
  const args = {
    questions: [SINGLE_ARGS.questions[0], MULTI_ARGS.questions[0]],
  };
  const request = buildAskElicitation(parsed(args), "s1", "t1") as Record<string, unknown>;
  assert.equal(request["message"], "Please answer the following questions.");
  const properties = (request["requestedSchema"] as Record<string, unknown>)["properties"] as Record<
    string,
    Record<string, unknown>
  >;
  assert.equal(properties["question_0"]["description"], QUESTION_TEXT);
  const multi = properties["question_1"];
  assert.equal(multi["type"], "array");
  const items = multi["items"] as Record<string, unknown>;
  assert.equal(Array.isArray(items["anyOf"]), true);
  assert.equal(
    (properties["question_1_custom"]["description"] as string).includes("add to your selection"),
    true
  );
});

test("an accepted form folds into answers and completes the call", async () => {
  const conn = askConnection({ action: "accept", content: { question_0: "chrono" } });
  const exec = new ToolExecutor(conn as never, "s1", FORM_CAPS);
  const result = await exec.execute("t1", "ask_user_question", JSON.stringify(SINGLE_ARGS));
  assert.equal(result.outcome, "completed");
  assert.deepEqual(JSON.parse(result.content), { answers: { [QUESTION_TEXT]: "chrono" } });

  assert.equal(conn.elicitations.length, 1);
  const statuses = conn.updates.map((update) => (update["update"] as Record<string, unknown>)["status"]);
  assert.deepEqual(statuses.filter(Boolean), ["pending", "completed"]);
});

test("decline yields empty answers; cancel never starts the call", async () => {
  const conn = askConnection({ action: "decline" });
  const exec = new ToolExecutor(conn as never, "s1", FORM_CAPS);
  const result = await exec.execute("t1", "ask_user_question", JSON.stringify(SINGLE_ARGS));
  assert.equal(result.outcome, "completed");
  assert.deepEqual(JSON.parse(result.content), {
    answers: {},
    annotations: { [QUESTION_TEXT]: { unanswered: "dismissed" } },
  });

  const cancelled = askConnection({ action: "cancel" });
  const cancelledExec = new ToolExecutor(cancelled as never, "s1", FORM_CAPS);
  const cancelledResult = await cancelledExec.execute(
    "t1",
    "ask_user_question",
    JSON.stringify(SINGLE_ARGS)
  );
  assert.equal(cancelledResult.outcome, "not_started");
  assert.match(cancelledResult.content, /dismissed/);
});

test("a multi-select joins picks and quotes typed answers", async () => {
  const conn = askConnection({
    action: "accept",
    content: { question_0: ["replays", "bench"], question_0_custom: "Redis, not Memcached" },
  });
  const exec = new ToolExecutor(conn as never, "s1", FORM_CAPS);
  const result = await exec.execute("t1", "ask_user_question", JSON.stringify(MULTI_ARGS));
  assert.equal(
    JSON.parse(result.content).answers["Which features do you want to enable?"],
    'replays, bench, "Redis, not Memcached"'
  );
});

test("typed text answers or rides along as notes", () => {
  const questions = parsed(SINGLE_ARGS);
  const typed = foldAskResponse(
    { action: "accept", content: { question_0_custom: "  hifitime  " } },
    questions
  );
  assert.equal(typed.action, "answered");
  if (typed.action === "answered") {
    assert.deepEqual(typed.answers, { [QUESTION_TEXT]: "hifitime" });
    assert.deepEqual(typed.annotations, {});
  }

  const noted = foldAskResponse(
    { action: "accept", content: { question_0: "time", question_0_custom: "if it stays light" } },
    questions
  );
  if (noted.action === "answered") {
    assert.deepEqual(noted.answers, { [QUESTION_TEXT]: "time" });
    assert.deepEqual(noted.annotations, { [QUESTION_TEXT]: { notes: "if it stays light" } });
  }
});

test("without elicitation a single question becomes permission buttons", async () => {
  const conn = askConnection({
    outcome: { outcome: "selected", optionId: "chrono" },
  });
  const exec = new ToolExecutor(conn as never, "s1", {});
  const result = await exec.execute("t1", "ask_user_question", JSON.stringify(SINGLE_ARGS));
  assert.equal(result.outcome, "completed");
  assert.deepEqual(JSON.parse(result.content), { answers: { [QUESTION_TEXT]: "chrono" } });

  assert.equal(conn.elicitations.length, 0);
  assert.equal(conn.permissionRequests.length, 1);
  const request = conn.permissionRequests[0];
  const options = request["options"] as Array<Record<string, unknown>>;
  assert.deepEqual(
    options.map((option) => option["optionId"]),
    ["chrono", "time", ASK_DECLINE_OPTION_ID]
  );
  assert.equal(options.at(-1)?.["name"], "None of these");
});

test("the permission fallback reads a decline or dismissal as dismissed", async () => {
  for (const response of [
    { outcome: { outcome: "selected", optionId: ASK_DECLINE_OPTION_ID } },
    { outcome: { outcome: "cancelled" } },
  ]) {
    const conn = askConnection(response);
    const exec = new ToolExecutor(conn as never, "s1", {});
    const result = await exec.execute("t1", "ask_user_question", JSON.stringify(SINGLE_ARGS));
    assert.equal(result.outcome, "completed");
    assert.deepEqual(JSON.parse(result.content), {
      answers: {},
      annotations: { [QUESTION_TEXT]: { unanswered: "dismissed" } },
    });
  }
});

test("without elicitation a multi-select tells the model to ask in text", async () => {
  const conn = askConnection({ action: "accept", content: {} });
  const exec = new ToolExecutor(conn as never, "s1", {});
  const result = await exec.execute("t1", "ask_user_question", JSON.stringify(MULTI_ARGS));
  assert.equal(conn.permissionRequests.length, 0);
  assert.match(result.content, /cannot show questions interactively/);
  // the pending card was announced, so the failure closes it instead of
  // emitting a second tool_call for the same id
  const updates = conn.updates.map((update) => update["update"] as Record<string, unknown>);
  assert.deepEqual(
    updates.map((update) => `${update["sessionUpdate"]}:${update["status"]}`),
    ["tool_call:pending", "tool_call_update:failed"]
  );
});

test("a transport failure closes the announced card", async () => {
  const conn = askConnection(() => {
    throw new Error("connection reset");
  });
  const exec = new ToolExecutor(conn as never, "s1", FORM_CAPS);
  const result = await exec.execute("t1", "ask_user_question", JSON.stringify(SINGLE_ARGS));
  assert.match(result.content, /could not present the question to the user: connection reset/);
  const updates = conn.updates.map((update) => update["update"] as Record<string, unknown>);
  assert.deepEqual(
    updates.map((update) => `${update["sessionUpdate"]}:${update["status"]}`),
    ["tool_call:pending", "tool_call_update:failed"]
  );
});
