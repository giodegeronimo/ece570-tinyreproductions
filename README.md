# Tiny Reproductions — Complex-Valued CNNs for MRI Reconstruction

This repository contains the ECE 570 Tiny Reproductions project that re-examines *Analysis of Deep Complex-Valued Convolutional Neural Networks for MRI Reconstruction* using the fastMRI single-coil knee dataset. The objective is to distill the paper’s core insight—complex-valued representations outperform capacity-matched real-valued CNNs for accelerated MRI—into a compact, fully reproducible codebase with anonymous deliverables due at the end of the week.

## Deliverables & Schedule
- **Day 1 (Mon):** finalize repo layout, baseline decisions, dataset notes, and README scaffold (this file).
- **Day 2 (Tue):** port notebook code into `src/`, add CLI/config plumbing, and script dataset prep.
- **Day 3 (Wed):** run baseline vs. complex U-Net experiments, log metrics, and generate updated figures/tables.
- **Day 4 (Thu):** draft the anonymous ICLR-style paper under `report/`, plus BetterPoster + demo script outlines.
- **Day 5 (Fri):** polish all artifacts (paper PDF, poster PDF, 5-minute demo video, code+README zip) and verify reproducibility.

## Repository Layout
```
checkpoints/          # ProjectPrimer, Checkpoint 1/2 slides (Quarto)
configs/              # YAML/JSON experiment configs (planned)
data/                 # fastMRI knee data (not tracked); includes tarballs & extracted splits
experiments/          # Named experiment folders w/ configs, logs, notes (planned)
figures/              # Exported PNG/PDF assets (cp2_* figures moved here)
notebooks/            # Exploratory notebooks (checkpoint2.ipynb, experiment_runner.ipynb)
references/           # refs.bib + citation helpers (to be populated)
report/
  paper/              # ICLR LaTeX build (main.tex + raw template + build dir)
  betterposter/       # BetterPoster layout + assets (to be filled)
  video/              # Demo video script & storyboard
  quarto/             # Legacy Quarto narrative notebook (moved from root)
results/              # Metrics/CSV/log artifacts (cp2_zero_filled_metrics.csv lives here)
scripts/              # CLI entry points (train/eval/figures/data scripts – to be added)
src/                 # Python modules: data loaders, models, training utilities
```
Additional supporting files will be added as the week progresses (e.g., BetterPoster assets under `report/` or `figures/`).

## Environment Setup
1. Use Python 3.10+ (tested locally on macOS Sonoma with Apple MPS acceleration).
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. PyTorch builds for GPU/MPS may require the official wheels: follow https://pytorch.org/get-started/locally/ if the default install lacks hardware support.
4. Set the `PYTHONPATH` root when running scripts so `src/` modules resolve, e.g., `export PYTHONPATH=$PWD`.

## Dataset Preparation (fastMRI Single-Coil Knee)
1. Request access at https://fastmri.med.nyu.edu/ and download the **single-coil knee** train/val/test tarballs.
2. Place the raw archives under `data/` (already present in this repo):
   ```
   data/
     knee_singlecoil_train.tar.xz
     knee_singlecoil_val.tar.xz
     knee_singlecoil_test.tar.xz
   ```
3. Extract each archive (train/val/test) into `data/singlecoil_{split}`. A helper script `scripts/download_data.py` will wrap this process on Day 2; for now manual extraction is acceptable. Example manual command:
   ```bash
   tar -xJf data/knee_singlecoil_train.tar.xz -C data/
   ```
4. The training and evaluation scripts will expect per-slice HDF5 files exactly as provided by fastMRI. Keep the directory names unchanged to avoid path edits in configs.
5. Document any local preprocessing (mask generation, cropping, normalization) inside your experiment README (`experiments/<run>/README.md`).

## Running Code (Current Status & Roadmap)
- **Zero-filled baseline:** computed via `notebooks/checkpoint2.ipynb`; metrics stored in `results/cp2_zero_filled_metrics.csv` and figures in `figures/`. A CLI replica (`scripts/run_zero_fill.py`) is planned for Day 2.
- **Experiment runner notebook:** `notebooks/experiment_runner.ipynb` now bootstraps run directories (`results/<timestamp>_<tag>/`), persists configs, and wires imports from `src/` so we can execute real vs. complex U-Net training before the CLI lands. Use the provided `width_scale=0.5` to match the parameter count of the complex U-Net per Cole *et al.* (2020).
- **Training pipelines:** code is migrating into `src/` (data loaders, masking, real/complex U-Nets, generic train/test loops). Tuesday’s task is to expose CLI entry points (`scripts/train.py`, `scripts/eval.py`) that wrap these modules with YAML configs so we can reproduce results headlessly.
- **Logging:** each CLI run will create `results/<timestamp>-<experiment-name>/` with (i) copied config, (ii) metrics CSV/JSON, (iii) checkpoints, and (iv) figure exports. The README will be updated once the logging helper lands.

### Results & Logging Convention
Each CLI run should write to `results/<timestamp>_<experiment-name>/` with:
```
results/2025-10-27_realunet_R4/
  config.yaml
  train.log
  metrics.csv
  metrics_summary.json
  checkpoints/
    best.ckpt
    latest.ckpt
  qualitative/
    slice000_R4_ACS24.png
    slice032_R4_ACS24.png
  tensors/
    val_predictions.pt
```
- `metrics.csv` stores per-epoch loss/metric curves (loss, PSNR, SSIM, L1).
- `metrics_summary.json` captures aggregates used directly in paper tables.
- `qualitative/` holds GT/recon/error grids for 3+ representative slices.
- `checkpoints/` always keeps both the best-validation and latest weights for reproducibility.

## Reproducibility Checklist
- Fix random seeds (Python/NumPy/PyTorch) per config.
- Record dataset split, acceleration factor, ACS width, optimizer, epochs, and compute budget in every experiment folder.
- Store plots (PSNR/SSIM vs. R, qualitative grids, method comparisons) under `figures/` with descriptive filenames referencing the experiment ID.
- Maintain anonymized references via `references/refs.bib`; cite fastMRI, PyTorch, SSIM metrics, and the reproduced paper in both the report and README.

## Original vs. Adapted Code Tracking
| Path | Status | Notes |
| --- | --- | --- |
| `notebooks/checkpoint2.ipynb` | **mixed** | Contains original masking/dataset code plus snippets adapted from fastMRI tutorials (to be cited in refs). Feeding modules under `src/` this week. |
| `notebooks/experiment_runner.ipynb` | **original** | Runs the Cole et al. reproduction baseline (real U-Net) with logging + checkpoints. |
| `src/data/*` | **original** | fastMRI dataset loader, masking utilities, reconstruction helpers, each with docstrings referencing their provenance. |
| `src/models/cx_unet.py` | **original** | Complex-valued U-Net adapted from Cole et al.; mirrors notebook prototype with docstrings describing the approach. |
| `src/models/real_unet.py` | **original** | Two-channel real-valued counterpart used for parameter-matched comparisons. |
| `report/paper/main.tex` | **original (ICLR shell)** | Working ICLR-style paper derived from the official template kept under `report/paper/template/`. |
| `report/quarto/TinyReproductionsCVCNNMRIRecon.qmd` | **original** | Legacy Quarto draft retained for reference. |
| `checkpoints/*.qmd` | **original** | Submitted checkpoint slide decks for internal milestones. |

This table will expand as new modules arrive; every adapted component will link back to its source for transparency.

## Deliverables Tracking
- **Paper:** LaTeX sources live in `report/paper/` and compile via `latexmk` to `report/paper/build/`; the rooted Quarto draft persists in `report/quarto/` for notes.
- **Checkpoints:** historical checkpoints are archived in `checkpoints/` for reference only.
- **Figures & Data:** baseline plots have been relocated under `figures/`; new experiment outputs will follow the same convention.
- **Poster & Video:** scaffolds now exist under `report/betterposter/` and `report/video/` for later in the week.

## Outstanding Tasks (Monday focus)
1. Port notebook utilities into `src/tinyrepro/` modules and write minimal unit tests/sanity scripts.
2. Stand up `scripts/run_zero_fill.py`, `scripts/train.py`, and `scripts/eval.py` with argparse/YAML config parsing.
3. Implement a repeatable data download/extract helper and document checksum verification.
4. Flesh out experiment templates under `experiments/` with README checklists (dataset, config, command, metrics, graphs).

Once these pieces are in place the README will switch from roadmap language to concrete run commands.
