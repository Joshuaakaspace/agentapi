name: code-review
description: Review a diff for correctness and clarity, citing file and line.
---
When reviewing code:

1. Read the file before commenting on it — never review from the diff alone.
2. Anchor every finding to `path:line` so it can be clicked.
3. Rank by consequence: a wrong result beats a style nit.
4. State the failure concretely: which input produces which wrong output.
5. If you are unsure whether something is a bug, say so rather than
   asserting it.

Use `checklist.md` in this skill for the full pass.
