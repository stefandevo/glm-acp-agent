# Releasing

Publishing to npm is automated via
[`.github/workflows/publish.yml`](./.github/workflows/publish.yml), which fires
when a GitHub Release is published. Auth uses npm Trusted Publishing (OIDC) —
no `NPM_TOKEN` secret is required.

## Full release procedure

### 1. Sync the local registry manifest (if needed)

The pinned versions in `registry/glm-acp-agent/agent.json` stay in sync
automatically: the `version` npm lifecycle script
([`src/registry/sync-manifest.ts`](./src/registry/sync-manifest.ts)) rewrites
both `version` and `distribution.npx.package` to match `package.json` and
stages the file, so `npm version` (step 2) folds the manifest bump into the
release commit. CI enforces the invariant — a `--check` step in
[`ci.yml`](./.github/workflows/ci.yml) and
[`publish.yml`](./.github/workflows/publish.yml) fails the build if the two
ever drift apart.

Manual work is only needed when other fields in `agent.json` change — the
`description` (model catalog, feature set, capabilities), `icon`, or
distribution metadata. Open a PR for those and merge it to `main` before
proceeding.

> **Note:** After the initial registry submission is merged upstream, version
> bumps are automatic — the registry runs an hourly cron that picks up the latest
> npm version and commits it directly. So this step is only needed when other
> fields in `agent.json` change (description, icon, distribution metadata).

### 2. Cut the release

From `main`, run:

```bash
git pull                          # make sure main is up to date
npm version minor -m "chore(release): %s"   # or: patch / major / X.Y.Z
git push --follow-tags
```

`npm version` does four things atomically:
- bumps `"version"` in `package.json` (and writes `package-lock.json`),
- runs the `version` lifecycle script, which syncs
  `registry/glm-acp-agent/agent.json` to the new version and stages it,
- creates a `chore(release): vX.Y.Z` commit that includes the manifest, and
- tags it `vX.Y.Z`.

After pushing, the tag and commit arrive on GitHub.

### 3. Publish the GitHub Release

1. Go to **Releases** → **Draft a new release**.
2. Pick the tag you just pushed (e.g. `v1.4.0`).
3. Click **Generate release notes**.
4. Click **Publish release**.

Publishing the release triggers the **Publish to npm** workflow.

### 4. Verify

Watch the **Actions** tab. When `Publish to npm` goes green:

```bash
npm view glm-acp-agent version    # should match the new tag
```

`registry/glm-acp-agent/agent.json` should already show the same version — it
rode along in the release commit (see step 2).

The workflow also runs a post-publish `bun x` smoke test; see the
[Troubleshooting](#troubleshooting-bun-x-verify-step-warns-after-publish)
section below if it emits a warning.

## Notes

- `package.json` version and the git tag must agree. `npm version` keeps them
  in sync; don't tag manually.
- `package.json`, the git tag, and the registry manifest must all agree. The
  first two are handled by `npm version`; the manifest is synced by the
  `version` lifecycle script and guarded by the CI `--check` step.
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
