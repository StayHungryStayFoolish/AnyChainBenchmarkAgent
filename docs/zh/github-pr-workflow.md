# GitHub PR 提交流程

[中文](github-pr-workflow.md) | [English](../en/github-pr-workflow.md)

本文档记录 AnyChain Benchmark Agent 正确提交和合并 PR 的维护者流程。

分支保护规则和 required checks 见
[GitHub PR Gate 与分支保护](github-pr-gates.md)。

## 标准流程

1. 从最新 `main` 开始。

   ```bash
   git switch main
   git pull --ff-only origin main
   ```

2. 创建短生命周期 topic branch。

   ```bash
   git switch -c docs/example-change
   ```

3. 做聚焦修改。不要把无关清理混进同一个 PR。

4. 运行最小相关本地验证。

   ```bash
   git diff --check
   python3 tools/check_public_repo_markers.py --root .
   ```

   如果修改 Agent 代码，还需要运行：

   ```bash
   python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract
   python3 tools/check_agent_boundaries.py --root .
   python3 agent/cli.py adk-eval
   ```

5. 使用 Conventional Commit 格式提交。

   ```bash
   git add <changed-files>
   git commit -m "docs(readme): clarify setup paths"
   ```

6. 推送分支并创建 PR。

   ```bash
   git push -u origin docs/example-change
   gh pr create --base main --head docs/example-change \
     --title "docs(readme): clarify setup paths" \
     --body "Summary and validation notes"
   ```

7. 等待所有 required checks 通过。

   ```bash
   gh pr checks <number> --watch --fail-fast
   ```

8. 获得具有写权限 reviewer 的 required approval。

9. checks 和 review 都通过后 squash merge。

   ```bash
   gh pr merge <number> --squash --delete-branch
   ```

10. 同步本地 `main`。

    ```bash
    git switch main
    git pull --ff-only origin main
    ```

## 合并前必须满足

以下条件不满足时不要合并：

- GitHub required checks 全绿。
- PR 已获得 required approval。
- 用户可见配置变更已更新文档。
- chain template 变更包含验证证据。
- Agent 行为变更包含 Agent contract 或 live matrix 证据。
- PR 不包含 secrets、本地凭据、本地 endpoint、生成的结果归档或 public-release marker 违规。

## CI 失败时如何处理

CI 失败时：

1. 阅读失败 job log。
2. 尽量在本地复现失败检查。
3. 做最小修复。
4. 重新运行相关本地检查。
5. commit 或 amend 当前 PR 分支。
6. push 后等待 GitHub checks 重新运行。

不要为了让 CI 变绿而绕过 gate。如果 gate 本身有问题，需要明确说明原因并修复 gate。

## 维护者紧急绕过

标准流程要求 approving review。普通工作不要绕过这个规则。

如果维护者必须合并一个来自同一账号、所有 checks 已全绿、但暂时没有 reviewer 的可信
PR，只允许使用以下紧急流程：

1. 临时把 `required_approving_review_count` 调整为 `0`。
2. 合并已经全绿的 PR。
3. 立刻把 `required_approving_review_count` 恢复为 `1`。
4. 合并后再次确认 branch protection。

这不是默认流程，应该少用、明确、可审计。

## 常用检查命令

```bash
gh pr view <number> --json state,mergeStateStatus,reviewDecision,statusCheckRollup,url
gh pr checks <number>
gh api repos/OWNER/REPO/branches/main/protection/required_pull_request_reviews
git status --short --branch
git log --oneline --decorate -5
```
