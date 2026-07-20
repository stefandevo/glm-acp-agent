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

## Troubleshooting: `bun x` verify step warns after publish

The publish workflow runs a post-publish `bun x` smoke test. It is
`continue-on-error` and only emits a **warning** — it never fails the release,
because `npm publish` has already validated the tarball. If you see

```
error: No version matching "<version>" found for specifier "glm-acp-agent" (but package exists)
```

the just-published version simply hasn't propagated yet. Two caches lag behind
a fresh publish:

1. **npm's registry CDN** serves a stale package manifest for ~1–2 minutes.
2. **Bun** caches that manifest in `~/.bun/install/cache` and reuses it, so a
   single stale fetch would poison every retry. Changing only `TMPDIR` does
   **not** help — that's the extract dir, not the metadata cache.

The workflow handles this by giving each retry a fresh `BUN_INSTALL_CACHE_DIR`
(forcing a re-fetch) and looping while the CDN catches up. To reproduce the
resolution manually:

```bash
BUN_INSTALL_CACHE_DIR=$(mktemp -d) TMPDIR=$(mktemp -d) \
  bun x --package "glm-acp-agent@<version>" glm-acp-agent
```

A warning here does **not** mean the published tarball is broken — confirm the
release with `npm view glm-acp-agent version`.
