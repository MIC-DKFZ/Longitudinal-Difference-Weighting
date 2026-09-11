# [MICCAI 2024] Longitudinal Segmentation of MS Lesions via Temporal Difference Weighting

## Overview

This repository provides the implementation of the **Temporal Difference Weighting (TDW) block**, introduced in the paper **"Longitudinal Segmentation of MS Lesions via Temporal Difference Weighting"**, accepted for the [*Longitudinal Disease Tracking and Modelling with Medical Images and Data*](https://ldtm-miccai.github.io/) workshop at MICCAI 2024.

#### Read the paper: &nbsp; &nbsp;   [![arXiv](https://img.shields.io/badge/arXiv-2409.13416-B31B1B.svg)](https://arxiv.org/abs/2409.13416)

### This repository is part of LongiSeg!


<a href="https://github.com/MIC-DKFZ/LongiSeg">
    <img src="documentation/assets/LongiSeg.jpg" alt="LongiSeg" width="600"/>
</a>

#### [LongiSeg](https://github.com/MIC-DKFZ/LongiSeg) extends the powerful nnU-Net framework, specifically optimizing it for longitudinal medical image segmentation. By incorporating temporal information across multiple timepoints, LongiSeg enhances segmentation accuracy and consistency, making it a robust tool for analyzing medical imaging over time. The Difference Weighting Block inclduing the training pipeline is already included!

## Introduction

Longitudinal medical imaging plays a vital role in tracking disease progression, treatment response, and patient outcomes over time. However, traditional segmentation methods often process each time point independently, leading to inconsistencies and missed temporal patterns. To address this, we introduce **Temporal Difference Weighting (TDW)**, a novel approach that enhances segmentation by explicitly modeling temporal changes between consecutive imaging scans.

## Methodology

Our approach integrates a **Temporal Difference Weighting (TDW) block** into deep learning-based segmentation models, enabling them to leverage longitudinal information more effectively. TDW computes **normalized attention maps** from the differences between feature representations extracted at separate time points. This mechanism highlights areas of change between scans and enables improved segmentation performance.

The TDW block is designed to be modular and can be incorporated into various backbone architectures, such as the UNet. This repository provides implementations of:
- The **TDW block** as a standalone module.
- A **longitudinal UNet (LongiUNet)** without TDW.
- A **longitudinal UNet with TDW (LongiUNetDiffWeighting)** for enhanced feature merging.

## Usage

Clone and install the repository:

```bash
git clone https://github.com/MIC-DKFZ/Longitudinal-Difference-Weighting.git
cd Longitudinal-Difference-Weighting
pip install -e .
```

Import the block or the network:

```python
from difference_weighting.building_blocks.difference_weighting_block import DifferenceWeightingBlock
from difference_weighting.architectures.longi_unet import LongiUNet
from difference_weighting.architectures.longi_unet_difference_weighting import LongiUNetDiffWeighting
```

## Citation

If you use this code in your research, please cite our papers:

```bibtex
@article{kirchhoff2026exploiting,
  title={Exploiting Longitudinal Context in Clinician-Verified Interactive Lesion Tracking}, 
  author={Kirchhoff, Yannick and Rokuss, Maximilian and Mertens, Daniel Philipp and Füller, David and Hamm, Benjamin and Schreyer, Andreas and Ritter, Oliver and Maier-Hein, Klaus},
  journal={arXiv preprint arXiv:2605.23118},
  year={2026}
}
@article{rokuss2024longitudinal,
  title={Longitudinal segmentation of MS lesions via temporal Difference Weighting},
  author={Rokuss, Maximilian and Kirchhoff, Yannick and Roy, Saikat and Kovacs, Balint and Ulrich, Constantin and Wald, Tassilo and Zenk, Maximilian and Denner, Stefan and Isensee, Fabian and Vollmuth, Philipp and Kleesiek, Jens and Maier-Hein, Klaus},
  journal={arXiv preprint arXiv:2409.13416},
  year={2024}
}
```

_Copyright © German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)