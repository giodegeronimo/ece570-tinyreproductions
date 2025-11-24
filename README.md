# Tiny Reproductions — Complex-Valued CNNs for MRI Reconstruction

This project reproduces the core claim of Cole et al. (complex-valued CNNs can outperform capacity-matched real-valued CNNs for accelerated MRI) on a small fastMRI single-coil knee subset. Everything is set up to be anonymous and fully reproducible.

## Final experiment snapshot
- Data: fastMRI single-coil knees; equispaced mask with acceleration **R=6** and **ACS=20**; slice-wise max-magnitude normalization.
- Split: **8,192** train slices, **1,024** val slices.
- Models: real U-Net (width × **1.42**) vs. complex U-Net (widths [16, 32, 64, 128, 256]); both ~**12.6M** params.
- Training: Adam (1e-3), batch 4, ~50k steps (~12 epochs), fixed seed/mask indices.
- Val metrics @ R=6, ACS=20 (PSNR / SSIM / ℓ1): Zero-fill 21.78 / 0.878 / 0.0584; Real U-Net 25.10 / 0.785 / 0.0462; **Complex U-Net 25.49 / 0.887 / 0.0408**.
- Paper figures: `report/paper/figures/fig_training_curve.png`, `report/paper/figures/fig_qualitative_idx_50_250_630.png`, Table 1 in `report/paper/tables/table_main_results.csv`.

## Repository layout (final)
```
data/               # fastMRI tarballs and extracted singlecoil_{train,val,test}
notebooks/          # experiment_runner.ipynb (train/log), figures_tables.ipynb (render figs/tables)
results/            # run outputs (metrics CSVs, checkpoints, qual panels)
src/                # data loaders, masking, real/complex U-Nets, utils
report/
  paper/            # LaTeX (main.tex, sections, figures, tables, build/)
  betterposter/     # Poster assets
  video/            # Demo script/notes
figures/            # Source plots copied into report/paper/figures/
```

## Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## Data
- Request access: https://fastmri.med.nyu.edu/ (single-coil knee).
- Place tarballs under `data/` (e.g., `knee_singlecoil_train.tar.xz`, etc.) and extract to `data/singlecoil_{train,val,test}/` keeping original HDF5 names.
- Do **not** commit data.

## How to reproduce the main runs
1) Ensure data is extracted to `data/singlecoil_train` and `data/singlecoil_val`.  
2) Open `notebooks/experiment_runner.ipynb` and run the real and complex configs (R=6, ACS=20, 8,192/1,024 split, width_scale=1.42 for real). Each run writes `results/<timestamp>_<tag>/` with:
   - `metrics/step_metrics.csv`, `metrics/epoch_metrics.csv`
   - `checkpoints/best.pt`, `checkpoints/latest.pt`
   - `qualitative/*.png`
   - `config.json`
3) Open `notebooks/figures_tables.ipynb` to regenerate:
   - `report/paper/tables/table_main_results.csv`
   - `report/paper/figures/fig_training_curve.png`
   - `report/paper/figures/fig_qualitative_idx_50_250_630.png`
   These files are already updated with the final runs.

## Paper, poster, video
- Paper: `report/paper/main.tex`, compiled to `report/paper/build/main.pdf`; flat file via `main_flat.tex`.
- Poster: BetterPoster content in `report/betterposter/` (48x36, PDF export).
- Video: 5-minute screencast outline in `report/video/` (describe problem → setup → how to run → results → scope/limitations).

## Original vs. adapted (high level)
- `src/models/real_unet.py`, `src/models/cx_unet.py`, `src/data/*`, `notebooks/experiment_runner.ipynb`, `notebooks/figures_tables.ipynb`: authored for this reproduction, informed by Cole et al. for architecture choice and matching capacity.
- LaTeX template under `report/paper/template/` is the official ICLR 2026 style; main.tex and sections are ours.

## Anonymity
No names or identifiers appear in code, figures, poster, or paper (per course policy). Copy checkpoints/results as needed, but keep author fields empty in commits and artifacts.
