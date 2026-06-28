# AI Coding Guide

This file defines how an AI coding agent should work in this repository.

The goal is simple: produce code that fits the project, solves the requested
problem, and does not create avoidable cleanup work for humans.

These rules are operational instructions, not style preferences.

## 1. Read Before Writing

Before changing code, inspect the code that already exists.

- Read the files you plan to edit.
- Find nearby examples of the same kind of feature, route, component, model,
  migration, test, or utility.
- Follow the libraries and patterns already used in the project.
- Look at imports, tests, naming, error handling, and file organization before
  introducing anything new.

Do not solve the task from memory if the repository already shows how it wants
the task solved.

If no clear pattern exists, say so and explain the approach you are choosing.

## 2. Think Before Coding

Do not start implementation until the task is understood well enough to define
success.

Before coding:

- State important assumptions.
- Identify unclear requirements.
- Name meaningful tradeoffs.
- Prefer one clear recommendation when multiple approaches are possible.

If the request is ambiguous in a way that could change the implementation,
ask or state the assumption explicitly. Silent guesses create expensive rework.

## 3. Keep It Simple

Write the smallest solution that fully solves the current problem.

Avoid:

- abstractions with only one implementation
- configuration that nobody asked for
- generic frameworks around one use case
- speculative error handling for impossible states
- future-proofing that adds complexity today

Duplication is often cheaper than the wrong abstraction. Abstract only when the
second or third real use case makes the shape obvious.

## 4. Make Surgical Changes

Keep diffs narrow and intentional.

- Change only what the task requires.
- Match the style of the file you are editing.
- Do not reformat unrelated code.
- Do not rename things opportunistically.
- Remove dead code only when your change created it or the task asks for it.

Every changed line should have a clear reason connected to the request.

## 5. Verify Behavior

Code that has not been verified is only a guess.

When fixing a bug:

- Reproduce the bug first when feasible.
- Add or update a test that fails before the fix and passes after it.
- Run the smallest relevant test set, then broader tests when the risk justifies
  it.

When adding behavior:

- Test meaningful behavior, not implementation trivia.
- Cover edge cases that are likely to break in real use.
- If automated tests are impractical, explain why and perform the best available
  manual verification.

If tests were already failing before your change, report that clearly.

## 6. Work Toward a Clear Goal

Turn vague tasks into verifiable outcomes.

Examples:

- "Add validation" means defining which inputs fail, what response appears, and
  how the behavior is tested.
- "Fix the bug" means reproducing the failure, making the fix, and verifying the
  failure no longer occurs.
- "Improve performance" means measuring first, changing the bottleneck, and
  measuring again.

For multi-step work, outline the steps before making broad changes. Update the
plan as facts change.

## 7. Debug by Investigating

Do not guess at fixes.

When something fails:

- Read the complete error message and stack trace.
- Reproduce the problem.
- Change one thing at a time.
- Verify after each meaningful change.
- Understand why the bug happened before adding a workaround.

If stuck, report what was tried, what was observed, and what remains uncertain.

## 8. Be Careful With Dependencies

Do not add packages casually.

Before adding a dependency, check:

- Can the project already do this with an existing dependency?
- Can the standard library or platform API handle it?
- Is the package maintained?
- Is its size and complexity justified?
- Does it fit the project's current stack?

When adding a dependency, explain why it is needed and why simpler options are
not enough.

## 9. Communicate Precisely

Keep communication useful and specific.

- Say what changed and why.
- Call out assumptions and uncertainty.
- Flag risks or follow-up work that matter.
- Match the explanation level to the user's context.
- Use specific commit messages when asked to commit.

Avoid vague claims like "should work" when a concrete verification result can be
provided.

## 10. Watch for Common Failure Modes

Stop and reconsider if you notice any of these patterns:

- expanding a small task into a broad refactor
- inventing architecture before the need is real
- making hidden product or API decisions
- handling only the happy path
- using APIs or library features without checking they exist
- writing in a style that does not match the project
- letting one fix cascade into many unrelated changes

When a change starts spreading beyond the original task, pause and explain why.

## Definition of Done

A task is complete when:

- the requested behavior is implemented
- the change fits the existing codebase
- relevant tests or checks have been run
- unrelated files were not changed
- remaining risks are clearly reported

Good AI coding is not measured by how much code was written. It is measured by
how little human rewriting is needed afterward.
