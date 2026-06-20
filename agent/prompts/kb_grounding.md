Use Knowledge Base results as optional grounding.

The local repository remains the first source of truth for current framework capabilities. Enterprise KB results may provide private RPC samples, internal incident history, workload hints, and chain-specific operational notes.

When using KB results:
- Summarize only what the KB returned.
- Do not treat KB output as validated framework support.
- If KB suggests a custom RPC method, require chain template, param contract, proxy extraction, and fake-node fixture validation.
- If KB conflicts with local repository state, say the conflict and prefer deterministic local validation for execution.
