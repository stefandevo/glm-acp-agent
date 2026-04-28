import { createInterface } from "node:readline/promises";
import { credentialsPath, writeCredentials } from "./llm/credentials.js";

/**
 * Interactive setup flow that prompts the user for a Z.AI API key on stdin and
 * persists it to the credentials file. Invoked when the binary is run with
 * `--setup` so users can configure the agent without leaking the key into
 * their shell history.
 */
export async function runSetup(
  input: NodeJS.ReadableStream = process.stdin,
  output: NodeJS.WritableStream = process.stdout
): Promise<void> {
  const path = credentialsPath();

  output.write("GLM ACP agent setup\n");
  output.write("===================\n\n");
  output.write("Get a Z.AI API key from https://z.ai/manage-apikey/apikey-list\n\n");

  const apiKey = (await readSecret(input, output, "Z_AI_API_KEY: ")).trim();

  if (apiKey.length === 0) {
    throw new Error("No API key entered; aborting setup.");
  }

  writeCredentials(apiKey);
  output.write(`\nAPI key saved to ${path}\n`);
  output.write("You can now run `glm-acp-agent` from your ACP client.\n");
}

/**
 * Read a single line from `input`, masking each character with `*` when stdin
 * is a TTY so the typed key doesn't appear in scrollback. When stdin is piped
 * (the common test/CI case) we fall back to a plain readline read — the key
 * isn't on a screen anyway.
 */
async function readSecret(
  input: NodeJS.ReadableStream,
  output: NodeJS.WritableStream,
  prompt: string
): Promise<string> {
  const stream = input as NodeJS.ReadStream;
  const isTTY = stream.isTTY === true && typeof stream.setRawMode === "function";

  if (!isTTY) {
    const rl = createInterface({ input, output, terminal: false });
    try {
      return await rl.question(prompt);
    } finally {
      rl.close();
    }
  }

  output.write(prompt);
  stream.setRawMode(true);
  stream.resume();

  return new Promise<string>((resolve, reject) => {
    let buf = "";
    const cleanup = () => {
      stream.off("data", onData);
      stream.off("close", onClose);
      stream.off("error", onError);
      stream.setRawMode(false);
      stream.pause();
    };
    const onClose = () => {
      cleanup();
      reject(new Error("Input stream closed before a key was entered."));
    };
    const onError = (err: Error) => {
      cleanup();
      reject(err);
    };
    const onData = (data: Buffer | string) => {
      const chunk = typeof data === "string" ? data : data.toString("utf8");
      for (const ch of chunk) {
        if (ch === "\n" || ch === "\r") {
          output.write("\n");
          cleanup();
          resolve(buf);
          return;
        }
        if (ch === "") {
          // Ctrl-C
          cleanup();
          reject(new Error("Setup cancelled by user."));
          return;
        }
        if (ch === "" || ch === "\b") {
          // Backspace / DEL
          if (buf.length > 0) {
            buf = buf.slice(0, -1);
            output.write("\b \b");
          }
          continue;
        }
        // Ignore other non-printable control characters silently.
        if (ch.charCodeAt(0) < 0x20) continue;
        buf += ch;
        output.write("*");
      }
    };
    stream.on("data", onData);
    stream.on("close", onClose);
    stream.on("error", onError);
  });
}
