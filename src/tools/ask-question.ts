/**
 * ask_user_question: the AskUserQuestion tool Claude Code has, for this agent.
 *
 * The model fills in structured questions; the client renders them and the
 * user's answers come back as the tool's result. The questions become an ACP
 * form elicitation — the same mapping Zed's claude-code-acp adapter performs —
 * so a client that advertises form elicitation shows the inline question card:
 * one dropdown per single-select question (checkboxes for a multi-select),
 * each option with its description, a per-question "Other" free-text box, and
 * Submit/Decline.
 */
import type {
  CreateElicitationRequest,
  CreateElicitationResponse,
  ElicitationContentValue,
  ElicitationPropertySchema,
  EnumOption,
} from "@agentclientprotocol/sdk";

export interface AskQuestionOption {
  label: string;
  description: string;
}

export interface AskQuestion {
  question: string;
  header: string;
  multiSelect: boolean;
  options: AskQuestionOption[];
}

/** Option id for the permission-button fallback's decline choice. */
export const ASK_DECLINE_OPTION_ID = "__none_of_these__";

/**
 * Parse and check the tool's arguments. A string return is the error message;
 * the schema bounds (1-4 questions, 2-4 options) are checked again here
 * because the model may call with more than the schema allows.
 */
export function parseAskQuestions(args: Record<string, unknown>): AskQuestion[] | string {
  const raw = args["questions"];
  if (!Array.isArray(raw)) return "`questions` must be an array of questions.";
  if (raw.length < 1 || raw.length > 4) return "`questions` must hold 1-4 questions.";
  const questions: AskQuestion[] = [];
  for (let index = 0; index < raw.length; index += 1) {
    const entry: unknown = raw[index];
    if (typeof entry !== "object" || entry === null) return `questions[${index}] must be an object.`;
    const record = entry as Record<string, unknown>;
    const question = typeof record["question"] === "string" ? record["question"].trim() : "";
    if (!question) return `questions[${index}].question is required.`;
    const header = typeof record["header"] === "string" ? record["header"].trim() : "";
    if (!header) return `questions[${index}].header is required.`;
    const multiSelect = record["multiSelect"];
    if (typeof multiSelect !== "boolean") return `questions[${index}].multiSelect must be a boolean.`;
    const rawOptions = record["options"];
    if (!Array.isArray(rawOptions)) return `questions[${index}].options must be an array.`;
    if (rawOptions.length < 2 || rawOptions.length > 4) {
      return `questions[${index}].options must hold 2-4 options.`;
    }
    const options: AskQuestionOption[] = [];
    for (let optionIndex = 0; optionIndex < rawOptions.length; optionIndex += 1) {
      const rawOption: unknown = rawOptions[optionIndex];
      if (typeof rawOption !== "object" || rawOption === null) {
        return `questions[${index}].options[${optionIndex}] must be an object.`;
      }
      const optionRecord = rawOption as Record<string, unknown>;
      const label = typeof optionRecord["label"] === "string" ? optionRecord["label"].trim() : "";
      if (!label) return `questions[${index}].options[${optionIndex}].label is required.`;
      if (typeof optionRecord["description"] !== "string") {
        return `questions[${index}].options[${optionIndex}].description is required.`;
      }
      options.push({ label, description: optionRecord["description"] });
    }
    questions.push({ question, header, multiSelect, options });
  }
  return questions;
}

/** An enum option whose description travels alongside title and const. */
type TitledOption = EnumOption & { description?: string };

/**
 * The `elicitation/create` params for the questions.
 *
 * Each question is one field keyed `question_<n>` — single-select a titled
 * `oneOf` enum, multi-select an array with a titled `anyOf` item enum — with
 * the enum `const` always the option's label, since that is what the tool
 * records as the answer. Each is followed by its own optional free-text
 * "Other" field (`question_<n>_custom`). Nothing is required, so the user can
 * also just skip.
 */
export function buildAskElicitation(
  questions: AskQuestion[],
  sessionId: string,
  toolCallId: string
): CreateElicitationRequest {
  const single = questions.length === 1;
  const properties: Record<string, ElicitationPropertySchema> = {};
  questions.forEach((question, index) => {
    const options: TitledOption[] = question.options.map((option) => ({
      const: option.label,
      title: option.label,
      ...(option.description ? { description: option.description } : {}),
    }));
    const title = question.header || undefined;
    // With one question the prompt rides on `message`; with several, each
    // field carries its own question text.
    const description = single ? undefined : question.question;
    properties[`question_${index}`] = question.multiSelect
      ? { type: "array", title, description, items: { anyOf: options } }
      : { type: "string", title, description, oneOf: options };
    properties[`question_${index}_custom`] = {
      type: "string",
      title: "Other",
      description: question.multiSelect
        ? "Type your own answer to add to your selection above (optional)."
        : "Type your own answer, or add a note to the option you chose above (optional).",
    };
  });
  return {
    mode: "form",
    sessionId,
    toolCallId,
    message: single ? questions[0].question : "Please answer the following questions.",
    requestedSchema: { type: "object", properties },
  };
}

export type AskAnnotations = Record<string, { notes?: string; unanswered?: "dismissed" | "skipped" }>;

export type AskOutcome =
  | { action: "answered"; answers: Record<string, string>; annotations: AskAnnotations }
  | { action: "cancelled" };

/**
 * Fold the client's response into the tool's answer.
 *
 * `decline` yields empty answers — the model is told the user skipped, the
 * turn does not abort. A multi-select's typed "Other" joins its picks; a
 * single-select's typed text is the answer when nothing was picked and
 * otherwise rides along as `notes`, so a client that presents the box as a
 * notes field cannot make the selection disappear. Each question left without
 * an answer gets an `unanswered` annotation: "dismissed" when the user
 * declined the form, "skipped" for one left blank in an accepted form.
 */
export function foldAskResponse(
  response: CreateElicitationResponse,
  questions: AskQuestion[]
): AskOutcome {
  if (response.action === "cancel") return { action: "cancelled" };
  const answers: Record<string, string> = {};
  const annotations: AskAnnotations = {};
  if (response.action === "decline") {
    for (const question of questions) {
      annotations[question.question] = { unanswered: "dismissed" };
    }
    return { action: "answered", answers, annotations };
  }
  const content = response.content ?? {};
  questions.forEach((question, index) => {
    const custom = (typeof content[`question_${index}_custom`] === "string"
      ? (content[`question_${index}_custom`] as string)
      : ""
    ).trim();
    const picks = picksOf(content[`question_${index}`]);
    if (question.multiSelect) {
      const items = custom === "" ? picks : [...picks, custom];
      if (items.length > 0) answers[question.question] = joinMultiSelect(items);
      return;
    }
    const picked = picks.join(", ");
    if (picked === "") {
      if (custom !== "") answers[question.question] = custom;
      return;
    }
    answers[question.question] = picked;
    if (custom !== "") annotations[question.question] = { notes: custom };
  });
  for (const question of questions) {
    if (!Object.hasOwn(answers, question.question)) {
      annotations[question.question] = { unanswered: "skipped" };
    }
  }
  return { action: "answered", answers, annotations };
}

/** A pick is any scalar the client submitted; arrays hold a multi-select's. */
function picksOf(value: ElicitationContentValue | undefined | null): string[] {
  if (value === undefined || value === null) return [];
  const values = Array.isArray(value) ? value : [value];
  return values
    .map(String)
    .filter((text) => text !== "");
}

/**
 * Multi-select answers are comma-joined the way Claude Code's own UI does it:
 * an item that itself contains the separator (or a double quote) is
 * JSON-quoted, so a typed answer like `Redis, not Memcached` stays one item
 * instead of reading as two more picks.
 */
function joinMultiSelect(items: string[]): string {
  return items
    .map((item) => (item.includes(", ") || item.includes('"') ? JSON.stringify(item) : item))
    .join(", ");
}
