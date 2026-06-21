# Multi-label Film Genre Prediction from Posters

This repository contains my solution for the Hi! PARIS / B2Ideas challenge on multi-label movie genre prediction from poster images. The goal is to predict one or more genres for each movie using visual information from its poster, including composition, color, typography, and semantic cues.

The project compares several pretrained vision backbones, including ResNet, CLIP, and DeiT, fine-tuned for multi-label classification. It includes training scripts, configuration files, evaluation utilities, and visualizations used to analyze model predictions.

## Environment
Create the conda environment with

```bash
conda env create -f environment.yml
conda activate hiparis
```

## Data preparation
To download all poster images, perform the preliminary analisys of the dataset, create train/validation/test slip files run the `prepare_data.py` script
```bash
pyton scripts/prepare_data.py
```

<p align="center">
  <img src="report/figures/genre_frequencies.png" width="45%">
  <img src="report/figures/num_genres_distribution.png" width="45%">
</p>

<p align="center">
  <em>Left: genre frequency distribution. Right: label count distribution.</em>
</p>
