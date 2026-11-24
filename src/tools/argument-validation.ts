import { TOOL_DEFINITIONS } from "./definitions.js";

export type ValidatedArguments =
  | { ok: true; value: Record<string, unknown> }
  | { ok: false; message: string };

type UnknownRecord = Record<string, unknown>;

const SUPPORTED_SCHEMA_KEYWORDS = new Set([
  "type",
  "properties",
  "required",
  "enum",
  "minimum",
  "maximum",
  "items",
  "description",
  "minItems",
  "maxItems",
]);

const BUILTIN_SCHEMAS = new Map(
  TOOL_DEFINITIONS.map((definition) => [definition.function.name, definition.function.parameters])
);

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function unsupportedKeyword(schema: unknown, path: string): string | undefined {
  if (!isRecord(schema)) return `${path} must be an object.`;
  for (const key of Object.keys(schema)) {
    if (!SUPPORTED_SCHEMA_KEYWORDS.has(key)) return `${path} uses unsupported schema keyword \`${key}\`.`;
  }
  if (
    schema["type"] !== undefined &&
    !["object", "array", "string", "number", "integer", "boolean"].includes(schema["type"] as string)
  ) {
    return `${path} uses unsupported schema type \`${String(schema["type"])}\`.`;
  }
  if (schema["required"] !== undefined && (!Array.isArray(schema["required"]) || !schema["required"].every((key) => typeof key === "string"))) {
    return `${path}.required must be an array of strings.`;
  }
  if (schema["enum"] !== undefined && !Array.isArray(schema["enum"])) return `${path}.enum must be an array.`;
  if (schema["minimum"] !== undefined && typeof schema["minimum"] !== "number") return `${path}.minimum must be a number.`;
  if (schema["maximum"] !== undefined && typeof schema["maximum"] !== "number") return `${path}.maximum must be a number.`;
  if (schema["properties"] !== undefined) {
    if (!isRecord(schema["properties"])) return `${path}.properties must be an object.`;
    for (const [name, property] of Object.entries(schema["properties"])) {
      const error = unsupportedKeyword(property, `${path}.properties.${name}`);
      if (error) return error;
    }
  }
  if (schema["items"] !== undefined) {
    const error = unsupportedKeyword(schema["items"], `${path}.items`);
    if (error) return error;
  }
  return undefined;
}

function assertSupportedBuiltinSchemas(): void {
  for (const definition of TOOL_DEFINITIONS) {
    const error = unsupportedKeyword(definition.function.parameters, `Tool ${definition.function.name}`);
    if (error) throw new Error(error);
  }
}

function typeError(path: string, expected: string): string {
  return `${path} must be a ${expected}.`;
}

function validateSchema(value: unknown, schema: unknown, path: string): string | undefined {
  if (!isRecord(schema)) return "Tool schema is invalid.";
  const type = schema["type"];
  if (type !== undefined) {
    const valid =
      (type === "object" && isRecord(value)) ||
      (type === "array" && Array.isArray(value)) ||
      (type === "string" && typeof value === "string") ||
      (type === "number" && typeof value === "number" && Number.isFinite(value)) ||
      (type === "integer" && typeof value === "number" && Number.isFinite(value) && Number.isInteger(value)) ||
      (type === "boolean" && typeof value === "boolean");
    if (!valid) return typeError(path, String(type));
  }

  if (Array.isArray(schema["enum"]) && !schema["enum"].some((entry) => Object.is(entry, value))) {
    return `${path} must be one of ${schema["enum"].map((entry) => JSON.stringify(entry)).join(", ")}.`;
  }
  if (typeof schema["minimum"] === "number" && typeof value === "number" && value < schema["minimum"]) {
    return `${path} must be at least ${schema["minimum"]}.`;
  }
  if (typeof schema["maximum"] === "number" && typeof value === "number" && value > schema["maximum"]) {
    return `${path} must be at most ${schema["maximum"]}.`;
  }
  if (typeof schema["minItems"] === "number" && Array.isArray(value) && value.length < schema["minItems"]) {
    return `${path} must have at least ${schema["minItems"]} items.`;
  }
  if (typeof schema["maxItems"] === "number" && Array.isArray(value) && value.length > schema["maxItems"]) {
    return `${path} must have at most ${schema["maxItems"]} items.`;
  }

  if (isRecord(value) && isRecord(schema["properties"])) {
    const required = schema["required"];
    if (Array.isArray(required)) {
      for (const key of required) {
        if (typeof key === "string" && !Object.hasOwn(value, key)) return `${path}.${key} is required.`;
      }
    }
    for (const [key, propertySchema] of Object.entries(schema["properties"])) {
      if (!Object.hasOwn(value, key)) continue;
      const error = validateSchema(value[key], propertySchema, `${path}.${key}`);
      if (error) return error;
    }
  }
  if (Array.isArray(value) && schema["items"] !== undefined) {
    for (let index = 0; index < value.length; index += 1) {
      const error = validateSchema(value[index], schema["items"], `${path}[${index}]`);
      if (error) return error;
    }
  }
  return undefined;
}

function requiredNonBlank(value: Record<string, unknown>, property: string): string | undefined {
  return typeof value[property] === "string" && value[property].trim().length > 0
    ? undefined
    : `arguments.${property} is required and must be a non-empty string.`;
}

function validateBuiltinConstraints(toolName: string, value: Record<string, unknown>): string | undefined {
  const nonBlankProperty = new Map<string, string>([
    ["read_file", "path"],
    ["write_file", "path"],
    ["edit_file", "path"],
    ["list_files", "path"],
    ["run_command", "command"],
    ["web_search", "query"],
    ["web_reader", "url"],
    ["image_analysis", "image_source"],
  ]).get(toolName);
  if (nonBlankProperty) {
    const error = requiredNonBlank(value, nonBlankProperty);
    if (error) return error;
  }
  if (toolName === "edit_file" && value["old_text"] === "") {
    return "arguments.old_text must be a non-empty string.";
  }
  if (toolName === "todowrite" && Array.isArray(value["todos"]) && value["todos"].length === 0) {
    return "arguments.todos must not be empty.";
  }
  return undefined;
}

/**
 * Parse model-supplied arguments and enforce the subset of the advertised
 * schemas that the local built-in implementations rely on. Session MCP tools
 * only require an object here; the MCP server retains ownership of its schema.
 */
export function validateToolArguments(toolName: string, rawArguments: string): ValidatedArguments {
  let parsed: unknown;
  try {
    parsed = rawArguments.trim() === "" ? {} : JSON.parse(rawArguments);
  } catch {
    return { ok: false, message: "could not parse tool arguments as JSON." };
  }
  if (!isRecord(parsed)) {
    return { ok: false, message: "Tool arguments must be a JSON object." };
  }

  const schema = BUILTIN_SCHEMAS.get(toolName);
  if (!schema) return { ok: true, value: parsed };
  const schemaError = validateSchema(parsed, schema, "arguments");
  if (schemaError) return { ok: false, message: schemaError };
  const constraintError = validateBuiltinConstraints(toolName, parsed);
  if (constraintError) return { ok: false, message: constraintError };
  return { ok: true, value: parsed };
}

// Keep implementation validation aligned with the schemas the model sees.
assertSupportedBuiltinSchemas();
