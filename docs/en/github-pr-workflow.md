# GitHub PR Workflow

[中文](../zh/github-pr-workflow.md) | [English](github-pr-workflow.md)

This document describes the correct maintainer workflow for submitting and
merging pull requests for AnyChain Benchmark Agent.

For branch protection settings and required checks, see
[GitHub PR Gates and Branch Protection](github-pr-gates.md).

## Normal Flow

1. Start from an up-to-date `main`.

   ```bash
   git switch main
   git pull --ff-only origin main
   ```

2. Create a short-lived topic branch.

   ```bash
   git switch -c docs/example-change
   ```

3. Make a focused change. Keep unrelated cleanup out of the PR.

4. Run the smallest relevant local validation.

   ```bash
   git diff --check
   python3 tools/check_public_repo_markers.py --root .
   ```

   For Agent code changes, also run:

   ```bash
   python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract
   python3 tools/check_agent_boundaries.py --root .
   python3 agent/cli.py adk-eval
   ```

5. Commit with a Conventional Commit subject.

   ```bash
   git add <changed-files>
   git commit -m "docs(readme): clarify setup paths"
   ```

6. Push the branch and open a PR.

   ```bash
   git push -u origin docs/example-change
   gh pr create --base main --head docs/example-change \
     --title "docs(readme): clarify setup paths" \
     --body "Summary and validation notes"
   ```

7. Wait for all required checks to pass.

   ```bash
   gh pr checks <number> --watch --fail-fast
   ```

8. Get the required review approval from a reviewer with write access.

9. Squash merge the PR after checks and review pass.

   ```bash
   gh pr merge <number> --squash --delete-branch
   ```

10. Sync local `main`.

    ```bash
    git switch main
    git pull --ff-only origin main
    ```

## Required Before Merge

Do not merge a PR unless all of these are true:

- Required GitHub checks are green.
- The PR has the required approval.
- User-facing config changes are documented.
- Chain template changes include validation evidence.
- Agent behavior changes include Agent contract or live-matrix evidence.
- The PR does not include secrets, local credentials, local endpoints, generated
  result archives, or public-release marker violations.

## Handling CI Failures

When CI fails:

1. Read the failing job log.
2. Reproduce the failing check locally when possible.
3. Make the smallest fix.
4. Re-run the relevant local check.
5. Commit or amend the PR branch.
6. Push and wait for GitHub checks again.

Avoid changing CI gates to make a failure disappear. If a gate is wrong, fix the
gate with a clear explanation and validation.

## Maintainer-Only Emergency Bypass

The normal process requires an approving review. Do not bypass this for ordinary
work.

If a maintainer must merge a trusted, all-green PR from the same account and no
reviewer is available, the only acceptable emergency path is:

1. Temporarily lower `required_approving_review_count` to `0`.
2. Merge the already green PR.
3. Immediately restore `required_approving_review_count` to `1`.
4. Confirm branch protection after the merge.

This is not the default workflow. It should be rare, explicit, and auditable.

## Useful Inspection Commands

```bash
gh pr view <number> --json state,mergeStateStatus,reviewDecision,statusCheckRollup,url
gh pr checks <number>
gh api repos/OWNER/REPO/branches/main/protection/required_pull_request_reviews
git status --short --branch
git log --oneline --decorate -5
```
