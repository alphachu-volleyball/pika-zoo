---
name: prepare-release
description: Prepare a release branch for merging to main — version bump, lock sync, commit, and PR
---

# Prepare Release

Prepare the current `release/*` branch for merging to main.

## Steps

1. Verify current branch is `release/*`, extract version from branch name (e.g. `release/v1.1.0` → `1.1.0`)
2. Run `uv version {version}` to set the version in pyproject.toml
3. Run `uv lock` to sync the lock file
4. Commit both changes: `chore: bump version to {version}`
5. Push and create PR targeting `main` with title `release: v{version}`
