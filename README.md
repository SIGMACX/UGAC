# UGAC

<p align="center">
  <img src="fig_UGAC_main_7.png" width="95%" alt="Overview of UGAC">
</p>
## Overview

Consistent perturbation strategies are widely used in semi-supervised medical image segmentation, but they can struggle when unlabeled predictions are affected by distribution shifts or unstable model generalization. UGAC addresses these issues with an uncertainty-guided training framework that corrects unreliable unlabeled predictions and regularizes uncertainty propagation during optimization.

The framework is built around three ideas:

- **Dual-path uncertainty rectification**: normalized entropy is used to identify error-prone regions in unlabeled predictions, followed by bilateral correction and confidence-weighted fusion.
- **Adversarial consistency constraints**: labeled segmentation maps provide authentic structural patterns for a spectral-normalized discriminator, encouraging unlabeled predictions to remain anatomically plausible.
- **Frequency-aware segmentation design**: the full UGAC framework is designed to enhance boundary sensitivity by separating high-frequency boundary cues from low-frequency anatomical structure during decoding.


## Data Layout

Prepare the dataset outside this repository. The default configuration expects:

```text
/path/to/MMWHS/
  imagesTr/
    case_000_slice_000.png
    ...
  labelsTr/
    case_000_slice_000.png
    ...
  txt_path/
    train_imagesTr_labeled_20.txt
    train_imagesTr_unlabeled_80.txt
    val_imagesTr.txt
```

## Installation

```bash
cd UGAC
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Training

```bash
python -m mmwhs_fns.train_mmwhs \
  --config configs/mmwhs.yaml \
  --data_root /path/to/MMWHS \
  --output_dir outputs/mmwhs
```

## Citation

If you use this repository, please cite the UGAC paper when citation information becomes available.
```bash
@article{chen2025uncertainty,
  title={Uncertainty-Guided Adaptive Correction for Semi-Supervised Medical Image Segmentation},
  author={Chen, Xi and Tong, Lyuyang and Zhao, Huangxuan and Du, Bo},
  journal={IEEE Transactions on Image Processing},
  volume={34},
  pages={7975--7988},
  year={2025},
  publisher={IEEE}
}
```

