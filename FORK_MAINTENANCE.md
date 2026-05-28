# Fork Maintenance Guide

This document explains how we maintain the `kaiko-ai/Megatron-Bridge` fork of `NVIDIA-NeMo/Megatron-Bridge`.

## Why we fork

We need fixes and features in Bridge faster than NVIDIA's ~2-month container release cadence. The fork lets us land a change once — it ships in our image the same day, and the same diff goes upstream as a PR. See [kaiko-ai/kaiko-eng#33267](https://github.com/kaiko-ai/kaiko-eng/issues/33267) for the full rationale.

## Branching Strategy

Two long-lived branches:

- **`main`** — Pinned upstream version + our changes on top. This is what the `kmbridge-nemo` Docker image installs from.
- **`upstream-track`** — Pure mirror of `NVIDIA-NeMo/Megatron-Bridge` at the SHA we are currently based on. No kaiko commits ever land here directly.

`git log upstream-track..main --oneline` shows everything we have added on top of upstream. There is no separate patches file to keep in sync.

## First-time setup

Run these once per clone:

```bash
git clone --recurse-submodules https://github.com/kaiko-ai/Megatron-Bridge.git
cd Megatron-Bridge
git remote add upstream https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

# Format/lint hooks Bridge upstream uses
pip install pre-commit
pre-commit install

# Auto-add Signed-off-by on every commit (DCO compliance)
git config format.signoff true
```

After this:

- `origin` points at our fork, `upstream` at `NVIDIA-NeMo/Megatron-Bridge`.
- The Megatron-Core submodule is initialized (see below).
- Every `git commit` runs ruff format / lint and the other hooks Bridge upstream uses.
- Every commit message automatically carries a `Signed-off-by:` line.

The DCO check in CI is a safety net — it should never fail if you ran the steps above.

## Submodule

Bridge vendors Megatron-Core (Megatron-LM) at `3rdparty/Megatron-LM`. We do not fork Core separately — it follows whatever Bridge's submodule pin points at. If you cloned without `--recurse-submodules`, run `git submodule update --init`.

## Pinned versions

The image installs from the SHA of `main` at the time of the image build. The current base is:

| Component         | SHA                                        |
| ----------------- | ------------------------------------------ |
| Megatron-Bridge   | `1ad45af6b02b1c7df9960acfda50904c7b8b40d0` |
| Megatron-Core     | `7521ecbeaf6a2dc9312303135f2572a7250723da` |

Bump these via the upstream sync workflow below; do not edit by hand.

## Workflow: syncing upstream

When updating to a newer upstream version:

```bash
git checkout upstream-track
git fetch upstream
git reset --hard upstream/main     # or a specific upstream SHA
git push origin upstream-track --force
```

Then open a PR from `upstream-track` into `main` and **merge with a merge commit** (not squash, not rebase) — preserves upstream history.

Before promoting the new SHA into the `kmbridge-nemo` image, run the smoke test (see "Smoke test" below).

## Workflow: adding a kaiko change

1. Branch off `main`: `git checkout -b feat/short-description main`.
2. Make the change. Keep it small — one logical change per PR so it can travel upstream as a single PR.
3. Commit using Bridge's `[area] type: description` format per their [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md) (e.g. `[training] feat: add step-0 validation flag`). Sign-off and pre-commit hooks run automatically if you completed the first-time setup.
4. Open a PR targeting `main`. Apply the labels listed in [Repo configuration](#repo-configuration); for VL-touching changes, also apply `full-test-suite`.
5. After internal review and merge (squash), open the same diff as a PR against `NVIDIA-NeMo/Megatron-Bridge`. Link both PRs to each other.

When upstream merges, the commit becomes part of `upstream-track` on the next sync. Because upstream may reshape the diff during review, expect occasional conflicts on rebase — they are usually small.

## Cadence expectations

External contributors cannot trigger NVIDIA's CI. A maintainer has to approve every CI run on every PR. Most kaiko changes touch VL models, which means they need the heavy `full-test-suite` tier.

In practice: "in-flight upstream → drops on next rebase" is **weeks, not days**.

## Smoke test before promoting a fork SHA

A fork SHA only becomes the pin for the `kmbridge-nemo` image after it passes a small real-workload run (e.g. 50-iter Qwen3.5-VL 2B), with loss matching the previous pin within noise. Unit tests don't catch silent training-behaviour changes.

## Internal review channel

Discuss fork PRs in the `#project-nemo-contributors` Slack channel before opening upstream.

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
