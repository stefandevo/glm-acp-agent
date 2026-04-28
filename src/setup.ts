import { createInterface } from "node:readline/promises";
import { credentialsPath, writeCredentials } from "./llm/credentials.js";

/**
 * Interactive setup flow that prompts the user for a Z.AI API key on stdin and
 * persists it to the credentials file. Invoked when the binary is run with
 * `--setup` so users can configure the agent without leaking the key into
 * their shell history.
 */
export async function runSetup(input: NodeJS.ReadableStream = process.stdin, output: NodeJS.WritableStream = process.stdout): Promise<void> {
  const path = credentialsPath();

  output.write("GLM ACP agent setup\n");
  output.write("===================\n\n");
  output.write("Get a Z.AI API key from https://z.ai/manage-apikey/apikey-list\n\n");

  const rl = createInterface({ input, output, terminal: false });
  let apiKey: string;
  try {
    const answer = await rl.question("Z_AI_API_KEY: ");
    apiKey = answer.trim();
  } finally {
    rl.close();
  }

  if (apiKey.length === 0) {
    throw new Error("No API key entered; aborting setup.");
  }

  writeCredentials(apiKey);
  output.write(`\nAPI key saved to ${path}\n`);
  output.write("You can now run `glm-acp-agent` from your ACP client.\n");
}
