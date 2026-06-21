#!/bin/bash

python -m scripts.evaluate_models   --model ResNet configs/resnet50_2stage.yaml checkpoints/resnet50_2stage/best.pt   --model CLIP configs/clip_vit_b32_2stage.yaml checkpoints/clip_vit_b32_2stage/best.pt   --model DeiT configs/timm_vit_small_patch16_224_2stage.yaml checkpoints/timm_vit_small_patch16_224_2stage/best.pt   --threshold 0.5   --csv-output results/test_metrics.csv   --plot-output report/figures/test_predictions_grid.png   --num-examples 10   --top-k 5

python scripts/plot_losses.py --title "" --output report/figures/train_val_losses.png checkpoints/resnet50_2stage/history.csv checkpoints/clip_vit_b32_2stage/history.csv checkpoints/timm_vit_small_patch16_224_2stage/history.csv --labels ResNet CLIP DeiT

python scripts/plot_val_map.py --title "" --output report/figures/val_map.png checkpoints/resnet50_2stage/history.csv checkpoints/clip_vit_b32_2stage/history.csv checkpoints/timm_vit_small_patch16_224_2stage/history.csv --labels ResNet CLIP DeiT
