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
