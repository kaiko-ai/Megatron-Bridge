# Fork Maintenance Guide

This document explains how we maintain the `kaiko-ai/Megatron-Bridge` fork of `NVIDIA-NeMo/Megatron-Bridge`.

## Why we fork

We need fixes and features in Bridge faster than NVIDIA's ~2-month container release cadence. The fork lets us land a change once — it ships in our image the same day, and the same diff goes upstream as a PR. See [kaiko-ai/kaiko-eng#33267](https://github.com/kaiko-ai/kaiko-eng/issues/33267) for the full rationale.

## Branching Strategy

Two long-lived branches:

- **`main`** — Pinned upstream version + our changes on top. This is what the `kmbridge-nemo` Docker image installs from.
- **`upstream-track`** — Pure mirror of `NVIDIA-NeMo/Megatron-Bridge` at the SHA we are currently based on. No kaiko commits ever land here directly.

`git log upstream-track..main --oneline` shows everything we have added on top of upstream. 

## First-time setup

Run these once per clone:

```bash
git clone --recurse-submodules https://github.com/kaiko-ai/Megatron-Bridge.git
cd Megatron-Bridge
git remote add upstream https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

# Format/lint hooks Bridge upstream uses
pip install pre-commit
pre-commit install
```

After this:

- `origin` points at our fork, `upstream` at `NVIDIA-NeMo/Megatron-Bridge`.
- Megatron-Core is vendored at `3rdparty/Megatron-LM` and is initialized by `--recurse-submodules`. We do not fork Core separately — it follows whatever Bridge's submodule pin points at. If you cloned without the flag, run `git submodule update --init`.
- Every `git commit` runs ruff format / lint and the other hooks Bridge upstream uses.

## Workflow: syncing upstream

When updating to a newer upstream version:

```bash
git checkout upstream-track
git fetch upstream
git reset --hard upstream/main     # or a specific upstream SHA
git push origin upstream-track --force
```

Then open a PR from `upstream-track` into `main` and **merge with a merge commit** (not squash, not rebase) — preserves upstream history.

Before promoting the new SHA into the `kmbridge-nemo` image, run the smoke test (see [Tests](#tests)).

## Workflow: adding a kaiko change

1. Branch off `main`: `git checkout -b feat/short-description main`.
2. Make the change. Keep it small — one logical change per PR so it can travel upstream as a single PR.
3. Commit with sign-off using Bridge's `[area] type: description` format per their [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md), e.g. `git commit -s -m "[training] feat: add step-0 validation flag"`. Pre-commit hooks run automatically from the first-time setup.
4. Open a PR targeting `main`. Apply the labels listed in [Repo configuration](#repo-configuration); for VL-touching changes, also apply `full-test-suite`.
5. After internal review and merge (squash), open the same diff as a PR against `NVIDIA-NeMo/Megatron-Bridge`. Link both PRs to each other.

When upstream merges, the commit becomes part of `upstream-track` on the next sync. Because upstream may reshape the diff during review, expect occasional conflicts on rebase — they are usually small.

### Forgot to sign off?

NVIDIA rejects unsigned commits. Fix the whole branch in one go:

```bash
git rebase --signoff main              # or upstream-track for an upstream-PR branch
git push --force-with-lease
```

This appends `Signed-off-by:` to every commit between the base and `HEAD`; the commit SHAs change but nothing else.

## Cadence expectations

Expect upstream PRs to land in **weeks, not days** — NVIDIA's CI requires a maintainer to approve every run, and most kaiko changes touch VL models, which need the heavy `full-test-suite` tier.

## Tests

Both depend on the `kmbridge-nemo` Docker image and are not wired up yet:

- **Smoke test before promoting a fork SHA** — a small real-workload run (e.g. 50-iter Qwen3.5-VL 2B) that must match the previous pin within noise before the new SHA enters the image.
- **PR CI checks** — pyright + pytest inside the image on PRs, analogous to kaiko-eng's `ci-kbridge` workflow.


## Repo configuration

- **Branch protection.** `main` requires PR + 1 approval, no force-push, no deletion. `upstream-track` allows force-push (the sync workflow needs it) but forbids deletion.
- **Team access.** `@kaiko-ai/MLLM` has `write`. The CODEOWNERS team mention resolves through this grant.
- **Labels.** `area:training`, `feature`, `bug`, `docs`, `full-test-suite` mirror upstream Bridge — apply them on our PRs the same way you would upstream.
- **Merge style.** Auto-merge and auto-delete-branch-on-merge are enabled. Squash commits take the PR title as the subject and the PR body as the message.

## See also

- [kaiko-ai/kaiko-eng#33267](https://github.com/kaiko-ai/kaiko-eng/issues/33267) — fork policy and principles
- [kaiko-ai/kaiko-eng#33272](https://github.com/kaiko-ai/kaiko-eng/issues/33272) — initial setup (this fork)
- Upstream: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- Upstream contributing guide: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md
- Precedent: https://github.com/kaiko-ai/verl/blob/main/FORK_MAINTENANCE.md
