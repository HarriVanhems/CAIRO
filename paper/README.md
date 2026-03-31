# Paper Drafts

Local drafts are stored in `paper/drafts/`.

Current public draft:

- [arXiv:2602.14440](https://arxiv.org/pdf/2602.14440)

Keep source documents, submission notes, and rebuttal material here rather than at the repo root.

## Canonical Paper Build

- The canonical compile target is `tex_CAIRO/main.tex`.
- Build from VS Code with `Cmd/Ctrl+Shift+B` and choose `Build CAIRO paper`.
- The build task runs `latexmk -pdf -interaction=nonstopmode -synctex=1 main.tex` from `tex_CAIRO/`.
- The generated PDF is `tex_CAIRO/main.pdf`.

## Recommended VS Code Workflow

- Open `cairo-paper.code-workspace` instead of the broader `Research` folder when working on the paper.
- Use the CAIRO-local `.vscode/` tasks so the build always runs from the correct directory.
- Keep `tex_CAIRO/main.tex` as the canonical manuscript until the paper is intentionally split into `\input`-based section files.

## Agent Workflow

- Treat `paper/*.tex` as staging files for section rewrites, notes, or candidate replacements.
- Give each agent exactly one owned file at a time to avoid merge conflicts.
- Do not send multiple agents to edit `tex_CAIRO/main.tex` in parallel.
- After agent edits are reviewed, merge the approved text into `tex_CAIRO/main.tex` and rebuild the paper.
- Use [section_map.md](/Users/harrivanhems/Desktop/Mcgill/Research/CAIRO/paper/section_map.md) to assign reviewers by section block.
- Use [staging/README.md](/Users/harrivanhems/Desktop/Mcgill/Research/CAIRO/paper/staging/README.md) when you want an agent to propose rewrites without touching the canonical manuscript.

Use [agent_brief_template.md](/Users/harrivanhems/Desktop/Mcgill/Research/CAIRO/paper/agent_brief_template.md) as the default prompt skeleton when delegating writing work.
