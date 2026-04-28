# Registry submission

This directory holds the files that will be copied into a fork of
[`agentclientprotocol/registry`](https://github.com/agentclientprotocol/registry)
to list `glm-acp-agent` in the official ACP agent registry.

## Submission steps

The actions below are external to this repo and require credentials the agent
process does not have, so they are executed by hand:

1. **Publish the npm package**
   ```sh
   npm run build
   npm publish --access public
   npm view glm-acp-agent version   # confirm the version is live
   ```

2. **Fork** `agentclientprotocol/registry` and create a working branch.

3. **Copy the contents of `registry/glm-acp-agent/`** in this repo into a
   `glm-acp-agent/` directory at the registry repo root.

4. **Validate locally** (from the registry repo root):
   ```sh
   uv run --with jsonschema .github/workflows/build_registry.py
   SKIP_URL_VALIDATION=1 uv run --with jsonschema .github/workflows/build_registry.py
   python3 .github/workflows/verify_agents.py --auth-check --agent glm-acp-agent
   ```

5. **Open a pull request** against `main`.

The version in `agent.json` and the npm tag pinned in `distribution.npx.package`
must match the version published to npm.
