# fastMRI Single-Coil Knee Dataset

## Source
- Data owner: NYU fastMRI initiative (https://fastmri.med.nyu.edu/).
- License: Requires a data use agreement; redistribution is prohibited.
- Paper reference: Zbontar et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI," arXiv:1811.08839.

## Download & Extraction
1. Request access from fastMRI and download the **single-coil knee** train/val/test archives (`knee_singlecoil_{split}.tar.tar`).
2. Place the tarballs under this `data/` directory:
   ```
   data/
     knee_singlecoil_train.tar.tar
     knee_singlecoil_val.tar.tar
     knee_singlecoil_test.tar.tar
   ```
3. Extract each archive in-place:
   ```bash
   tar -xJf knee_singlecoil_train.tar.tar -C .
   tar -xJf knee_singlecoil_val.tar.tar   -C .
   tar -xJf knee_singlecoil_test.tar.tar  -C .
   ```
   Alternatively, if supported this is faster:
   ```bash
   tar --use-compress-program="xz -T0" -xvf knee_singlecoil_val.tar.tar -C .
   ...
   ```
   This creates `singlecoil_train/`, `singlecoil_val/`, and `singlecoil_test/` with the original HDF5 files intact.

## Expected Layout
```
data/
  knee_singlecoil_train.tar.tar
  knee_singlecoil_val.tar.tar
  knee_singlecoil_test.tar.tar
  singlecoil_train/
    fileXXXX.h5
    …
  singlecoil_val/
  singlecoil_test/
```

## Provenance & Notes
- Do **not** commit raw data to git; keep this folder excluded via `.gitignore`.
- All preprocessing (mask generation, cropping, normalization) happens inside the training scripts so raw data stays untouched.
