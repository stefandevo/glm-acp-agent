# Releasing

Publishing to npm is automated via [`.github/workflows/publish.yml`](./.github/workflows/publish.yml),
which fires when a GitHub Release is published. Auth uses npm Trusted Publishing
(OIDC) — no `NPM_TOKEN` secret is required.

## Cut a release

```bash
# Bump version + create a git tag (creates a "chore(release): vX.Y.Z" commit)
npm version patch -m "chore(release): %s"   # or: minor / major / X.Y.Z

# Push the commit and the new tag
git push --follow-tags
```

Then on GitHub:

1. Go to **Releases** → **Draft a new release**.
2. Pick the tag you just pushed (e.g. `v1.0.1`).
3. Click **Generate release notes**.
4. Click **Publish release**.

Watch the **Actions** tab. When `Publish to npm` goes green:

```bash
npm view glm-acp-agent version    # should match the new tag
```

## Notes

- `package.json` version and the git tag must agree. `npm version` keeps them
  in sync; don't tag manually.
- Trusted Publisher is configured at
  https://www.npmjs.com/package/glm-acp-agent/access — if the workflow file
  is renamed or the repo moves, update it there.
- `--provenance` in the publish step requires a public repo or paid npm org.

## Troubleshooting: `bun x` failures

If `bun x glm-acp-agent@<version>` fails with `ERR_MODULE_NOT_FOUND` /
`ERR_UNSUPPORTED_DIR_IMPORT` on one machine but succeeds with a fresh
`TMPDIR`, the likely cause is a stale or corrupted Bun temp install
(Bun caches `bun x` packages in a deterministic temp directory).

**Recovery:** clear the cached install for that package/version:

```bash
# Find and remove the cached bunx install
rm -rf /private/var/folders/.../T/bunx-<uid>-glm-acp-agent@<version>
# Or run with a clean temp directory:
TMPDIR=$(mktemp -d) bun x glm-acp-agent@<version>
```

This is an operational recovery for corrupted local state — it does **not**
mean the published tarball is missing `main`, `package.json`, or
dependencies. Fresh `bun x` installs resolve all dependencies correctly.
