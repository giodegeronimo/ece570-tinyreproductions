# Paper Directory Structure (for consuming results)

This folder only mirrors the final, paper-ready assets and LaTeX sources. Experiments are run elsewhere (see below).

```
paper/
  main.tex        # Primary ICLR manuscript (inputs section files + math_commands)
  math_commands.tex
  refs.bib
  sections/       # Per-section tex files (00_introduction.tex, 01_background.tex, ...)
  figures/        # Paper-ready figures copied from ../../figures/ (auto-generated)
  tables/         # Paper-ready CSV/TeX tables (auto-generated)
  template/       # Untouched upstream ICLR 2026 style + example files
  build/          # latexmk output (gitignored)
  .latexmkrc      # Forces latexmk aux/pdf into build/
```

## How to run experiments and regenerate figures/tables
1. From repo root, ensure data extracted under `data/singlecoil_{train,val}` (fastMRI knees, see README under data/).
2. Set up experiment configs in `notebooks/experiment_runner.ipynb`. Outputs land under `results/<timestamp>_<tag>/` with metrics CSVs, checkpoints, and qualitative panels.
3. Run `notebooks/figures_tables.ipynb` to consume the latest `results/` and regenerate:
   - `report/paper/tables/table_main_results.csv`
   - `report/paper/figures/fig_training_curve.png`
   - `report/paper/figures/fig_qualitative_idx_50_250_630.png`
4. Rebuild the paper if needed:
   ```bash
   cd report/paper
   latexmk -pdf main.tex     # PDF in build/main.pdf
   ./latexpand --empty-comments --expand-bbl build/main.bbl --output main_flat.tex main.tex  # single-file TeX
   ```

## Notes
- Do not edit `template/` (ICLR style files) to stay compliant.
- Source plots live under repo-level `figures/`; only paper-sized assets are copied into `report/paper/figures/`.
- Poster assets live under `report/betterposter/`; video link under `report/video/`.
