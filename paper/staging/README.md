# Staging Files for Paper Agents

Use this folder when an agent should propose edits without touching `tex_CAIRO/main.tex`.

## Workflow

1. Create one staging file per task, for example `paper/staging/experiments_revision.tex`.
2. Copy the relevant section text from `tex_CAIRO/main.tex` into that staging file.
3. Give one agent ownership of that staging file only.
4. Review the diff in the staging file.
5. Merge approved edits back into `tex_CAIRO/main.tex` manually.
6. Rebuild the paper and verify the PDF.

## Rules

- One agent per staging file.
- No direct parallel edits to `tex_CAIRO/main.tex`.
- Keep filenames descriptive and section-scoped.
- Delete stale staging files after they are merged or abandoned.
