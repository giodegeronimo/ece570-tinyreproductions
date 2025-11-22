# Paper Directory Structure

This folder houses everything required to build the anonymous ICLR-style paper:

```
paper/
  main.tex        # Primary ICLR manuscript (inputs section files + math_commands)
  math_commands.tex
  refs.bib
  sections/       # Per-section tex files (00_introduction.tex, 01_background.tex, ...)
  figures/        # Paper-ready figures copied from ../../figures/
  tables/         # Table snippets or CSVs converted to LaTeX
  template/       # Untouched upstream ICLR 2026 style + example files
  build/          # latexmk output (gitignored)
  .latexmkrc      # Forces latexmk aux/pdf into build/
```

## Workflow
1. Keep the official style files untouched under `template/iclr2026_template/`.
2. Edit `main.tex` (copied from the template) and include extra files via `\input{sections/...}`.
3. Place only final, size-checked figure/table assets in `figures/` and `tables/` (source plots stay under the repo-level `figures/`).
4. Update bibliographic entries in `refs.bib` and cite using natbib commands (`\citet`, `\citep`).
5. Build locally (LaTeX Workshop recipes can call `latexmk` directly thanks to `.latexmkrc`):
   ```bash
   cd report/paper
   latexmk -pdf main.tex
   ```
6. Export the resulting PDF from `report/paper/build/main.pdf` and copy it to `report/` when final.

Future additions (BetterPoster, video script) live under neighboring `report/betterposter/` and `report/video/` directories.
