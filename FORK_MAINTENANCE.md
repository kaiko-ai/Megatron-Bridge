# Fork Maintenance Guide

This document explains how we maintain the `kaiko-ai/Megatron-Bridge` fork of `NVIDIA-NeMo/Megatron-Bridge`, including our branching model, contribution workflow, and sync procedures.

## Why we fork

We want kaiko-specific Bridge patches to live as real commits (not runtime monkey-patches in `projects/mllm_midtrain`), and we want to adopt newer Bridge fixes without waiting for the next NeMo container release. The fork is the home for both. The `kmbridge-nemo` Docker image in `kaiko-eng` installs Bridge from it. See [kaiko-ai/kaiko-eng#33267](https://github.com/kaiko-ai/kaiko-eng/issues/33267) for the rationale.

## TL;DR

- **Mirrors** (`main`, `r0.5.0`) are pristine copies of upstream. Fast-forward only. They never carry our commits.
- **kaiko branches** (`kaiko-main`, `kaiko-r0.5.0`) carry our commits on top of the matching upstream base. All kaiko work lives here.
- **Naming rule:** no `kaiko-` prefix ⇒ pristine mirror (safe to fast-forward, never touched by us). Has `kaiko-` prefix ⇒ carries our work.
- **Contributing:** branch off a kaiko branch, PR back into it, **squash-merge**.
- **Syncing upstream:** fast-forward the mirror branch from upstream, create a temporary sync branch from the corresponding kaiko branch, merge the updated mirror into the sync branch, and open a PR back into the kaiko branch. Merge the PR with a merge commit — never squash, and never rebase-and-force-push a shared branch.

## Branch model

### Naming and roles

| Branch         | Tracks             | Carries kaiko commits? | Update method     | Force-push? |
| -------------- | ------------------ | ---------------------- | ----------------- | ----------- |
| `main`         | `upstream/main`    | No — pristine mirror   | fast-forward only | Never       |
| `r0.5.0`       | `upstream/r0.5.0`  | No — pristine mirror   | fast-forward only | Never       |
| `kaiko-main`   | `main` + our work  | Yes                    | merge (sync PR)   | Never       |
| `kaiko-r0.5.0` | `r0.5.0` + our work | Yes                    | merge (sync PR)   | Never       |

- We keep the upstream branch names (`main`, `r0.5.0`, …) for the pristine mirrors and prefix everything that carries our commits with `kaiko-`.
- The branch name alone tells you whether it's safe to fast-forward: anything without the `kaiko-` prefix is a mirror and must only ever be fast-forwarded from upstream.
- A `kaiko-release` tag points at the tip of the current release-tracking kaiko branch, so consumers have a stable "latest kaiko release" reference.
- Release branches follow the same pattern for every upstream release we track: `rX.Y.Z` (mirror) and `kaiko-rX.Y.Z` (kaiko).

**Mirrors** are exact copies of the corresponding upstream branch. They exist so we have a stable local reference to upstream state and a clean base to diff and sync against. They are **only ever fast-forwarded** from `upstream/*`. They never receive our commits, are never rebased, and are never force-pushed.

**kaiko branches** are the upstream base plus our commits on top. This is where all kaiko development lands and where upstream changes are merged in via sync PRs.

## First-time setup

```bash
git clone --recurse-submodules https://github.com/kaiko-ai/Megatron-Bridge.git
cd Megatron-Bridge
git remote add upstream https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

# Format/lint hooks Bridge uses upstream
pip install pre-commit
pre-commit install
```

After this:

- `origin` = our fork, `upstream` = NVIDIA-NeMo.
- Megatron-Core is vendored at `3rdparty/Megatron-LM` and initialized by `--recurse-submodules`. We do not fork Core separately — it follows Bridge's submodule pin. If you cloned without the flag, run `git submodule update --init`.
- Every `git commit` runs ruff format / lint and the other hooks Bridge uses upstream.

## Workflows

### Adding a kaiko change (feature work)

Branch off the relevant kaiko branch, open a PR back into it:

```bash
git switch kaiko-main
git pull origin kaiko-main
git switch -c kaiko-<feature>     # e.g. kaiko-fix-validation-loss-name
# ... make changes, commit ...
git push -u origin kaiko-<feature>
# open PR: kaiko-<feature> -> kaiko-main
```

Use Bridge's `[area] type: description` format per their [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md), e.g.:

```bash
git commit -s -m "[training] feat: add step-0 validation flag"
```

Sign-off is required (see below if you forgot).

#### Forgot to sign off?

NVIDIA rejects unsigned commits. Fix the whole branch in one go:

```bash
git rebase --signoff kaiko-main
git push --force-with-lease
```

**Squash-merge feature PRs.** One commit per feature keeps kaiko history compact and makes our commits trivially identifiable against the mirror:

```bash
git log main..kaiko-main --oneline    # exactly our commits, nothing else
```

After merge, bump `MEGATRON_BRIDGE_SHA` in kaiko-eng's `kmbridge-nemo` Dockerfile and ship a new image version. When ready, submit a PR to upstream (`NVIDIA-NeMo/Megatron-Bridge`).

### Syncing a kaiko branch with upstream

This brings the latest upstream changes into a kaiko branch while preserving our work, reviewed before it touches a stable branch. The example uses `kaiko-main`; it applies identically to `kaiko-r0.5.0` (merge `r0.5.0` instead of `main`).

**Step 0 — Fast-forward the mirror (trivial, can't conflict):**

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```

**Step 1 — Create a temporary sync branch off the kaiko branch.**
Never merge upstream directly onto a stable kaiko branch — always go through a review branch:

```bash
git switch kaiko-main
git pull origin kaiko-main
git switch -c kaiko-main-sync-upstream
```

**Step 2 — Merge the mirror (upstream) into the sync branch and resolve:**

```bash
git merge main            # equivalently: git merge upstream/main
# resolve conflicts, then:
git add <resolved-file>
git commit                # completes the merge commit
```

**Step 3 — Push and open a PR** so the team can review the upstream delta:

```bash
git push -u origin kaiko-main-sync-upstream
# open PR: kaiko-main-sync-upstream -> kaiko-main
```

**Step 4 — Complete the PR with a MERGE COMMIT, not squash.**
Squashing collapses the merge base into a single commit, so the same upstream conflicts come back on the next sync.

**Verify (optional):**

```bash
git log --oneline --graph --decorate
git log main..kaiko-main --oneline     # our commits stay readable, + one merge commit per sync
```

## Testing

Upstream's unit-test job lives on NVIDIA's self-hosted runners and only fires for PRs against `NVIDIA-NeMo/Megatron-Bridge`, so it never runs on our PRs. To get signal on fork PRs we run our own lightweight job — `.github/workflows/kaiko-tests.yml`, which fires on every PR against `kaiko-*` branches and runs `tools/ci/run_tests.sh`. That same script runs locally to reproduce a result.

### How it works

- **Regression-diff, not pass/fail.** The job runs the full CPU unit suite against *two* checkouts — the PR branch and the current base — and fails only on tests that fail on the PR but pass on the base. Thus, we only care about regressions the change introduces.
- **Image source.** It pulls the NeMo image from our ghcr mirror (`ghcr.io/kaiko-ai/nvidia-nemo-mirror:<tag>`) rather than nvcr, for self-contained pulls. The fork is public, so it can't use kaiko runners or the private `kmbridge-nemo` image — but the public `nemo` image has the same torch/TE/Megatron-Core stack the unit tests need.
- **Maintaining the mirror.** `.github/workflows/mirror-nemo.yml` copies `nvcr.io/nvidia/nemo:<tag>` into ghcr. It's `workflow_dispatch`-only: **run it once whenever you introduce a new tag** (e.g. when bumping the `IMAGE` tag in `kaiko-tests.yml`). Add the new tag to that workflow's matrix and trigger it from the Actions tab.

### Running tests locally

`run_tests.sh` needs the test image and two checkouts — the base (mirror or kaiko branch) and your feature branch. From your feature branch, add a sibling checkout of the base and point the script at both:

```bash
git worktree add ../kaiko-main origin/kaiko-main    # one-time: a sibling checkout
IMAGE=nvcr.io/nvidia/nemo:26.06 tools/ci/run_tests.sh ../kaiko-main .
```

It runs the suite inside the image with each checkout overlaid via `PYTHONPATH` (so `import megatron.bridge.*` resolves to that checkout, while `megatron.core`, `torch`, and TE come from the image — no submodule init needed), CPU-only, against a read-only mount. On Apple Silicon the image runs under amd64 emulation: the "CPU does not support AVX → illegal instruction likely" boot warning is harmless — `torch` still imports (slowly; a few minutes, then the tests run in seconds).

## Repo configuration

- **Branch protection.** `main` and release-track mirrors (`r0.5.0`, etc.) require fast-forward only, no force-push, no deletion. kaiko branches (`kaiko-main`, `kaiko-r0.5.0`, etc.) require PR + 1 approval, no force-push, no deletion.
- **Team access.** `@kaiko-ai/MLLM` has `write`. The CODEOWNERS team mention resolves through this grant.
- **Labels.** `area:training`, `feature`, `bug`, `docs`, `full-test-suite` mirror upstream Bridge — apply them on our PRs the same way you would upstream.
- **Merge style.** Auto-merge and auto-delete-branch-on-merge are enabled. Feature PRs are squash-merged (PR title as subject, PR body as message). Sync PRs are merged (not squashed) to preserve the merge base.

## Cadence expectations

Upstream PRs from external contributors land in **weeks, not days** — NVIDIA's CI requires a maintainer to approve every run, and VL-touching changes need the heavy `full-test-suite` tier.

## Golden rules

**Do**

- Fast-forward mirrors from `upstream/*` only.
- Sync via a reviewed temporary branch → PR into the kaiko branch.
- Complete sync PRs with a **merge commit**.
- Squash feature PRs.

**Don't**

- Merge `upstream/*` directly onto a `kaiko-*` branch — always go through a review branch.
- Rebase or force-push any shared branch.
- Squash a sync PR.
- Commit kaiko changes onto a mirror branch.

## Why merge, and not rebase (rejected alternatives)

| discussion: https://github.com/kaiko-ai/Megatron-Bridge/issues/23

We evaluated three history-rewriting variants for ongoing syncs and rejected all of them.

**Rebase + force-push the default branch.**
Gives the cleanest, most linear history, but force-pushing a *shared* branch breaks every open PR and every local checkout, requiring cross-team coordination on every single sync. Unacceptable as a recurring cost.

**Rebase + merge the sync PR back.**
Internally inconsistent. The rebase gives our commits new SHAs, but merging the PR keeps the originals in the DAG — so the branch ends up carrying *both* copies, and the next sync replays the originals and re-hits conflicts we already resolved. Patch-id de-duplication does **not** reliably save this: it runs against the upstream target, is version/config dependent, and does not cancel duplicate commits living inside our own branch. A rebase is only consistent if the integration step is *also* a history rewrite (force-push) — which lands us back at the first option.

**Squashing sync PRs.**
Collapses the sync into a single commit, destroying the merge base, so upstream conflicts recur every time.

**What we chose — the merge workflow.** It trades a strictly linear history for safe collaboration: no force-push, nothing breaks. Our commits stay identifiable regardless — `git log <mirror>..<kaiko-branch>` lists them linearly, with a single merge commit per sync.

## References

- [kaiko-ai/kaiko-eng#33267](https://github.com/kaiko-ai/kaiko-eng/issues/33267) — fork policy and principles
- [kaiko-ai/kaiko-eng#33272](https://github.com/kaiko-ai/kaiko-eng/issues/33272) — initial setup
- Upstream: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- Upstream contributing guide: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md
