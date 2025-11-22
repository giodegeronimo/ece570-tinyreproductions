# fastMRI Single-Coil Knee Dataset

## Source
- Data owner: NYU fastMRI initiative (https://fastmri.med.nyu.edu/).
- License: Requires a data use agreement; redistribution is prohibited.
- Paper reference: Zbontar et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI," arXiv:1811.08839.

## Download & Extraction
1. Request access from fastMRI and download the **single-coil knee** train/val/test archives (`knee_singlecoil_{split}.tar.xz`).
2. Place the tarballs under this `data/` directory:
   ```
   data/
     knee_singlecoil_train.tar.xz
     knee_singlecoil_val.tar.xz
     knee_singlecoil_test.tar.xz
   ```
3. Extract each archive in-place:
   ```bash
   tar -xJf knee_singlecoil_train.tar.xz -C .
   tar -xJf knee_singlecoil_val.tar.xz   -C .
   tar -xJf knee_singlecoil_test.tar.xz  -C .
   ```
   This creates `singlecoil_train/`, `singlecoil_val/`, and `singlecoil_test/` with the original HDF5 files intact.

## Expected Layout
```
data/
  knee_singlecoil_train.tar.xz
  knee_singlecoil_val.tar.xz
  knee_singlecoil_test.tar.xz
  singlecoil_train/
    fileXXXX.h5
    …
  singlecoil_val/
  singlecoil_test/
```

## Provenance & Notes
- Do **not** commit raw data to git; keep this folder excluded via `.gitignore`.
- Log the download date, MD5/SHA checksums, and fastMRI release version inside your experiment README whenever you use these files.
- All preprocessing (mask generation, cropping, normalization) happens inside the training scripts so raw data stays untouched.
