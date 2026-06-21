#!/bin/bash

python scripts/train.py --config configs/resnet50_2stage.yaml

python scripts/train.py --config configs/clip_vit_b32_2stage.yaml

python scripts/train.py --config configs/timm_vit_small_patch16_224_2stage.yaml
