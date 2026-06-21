#!/bin/bash

python scripts/evaluate_models.py \
    --model ResNet configs/resnet50_2stage.yaml \
    --model CLIP configs/clip_vit_b32_2stage.yaml \
    --model DeiT configs/timm_vit_small_patch16_224_2stage.yaml \
    --threshold 0.5   \
    --csv-output results/test_metrics.csv   \
    --plot-output report/figures/test_predictions_grid.png   \
    --num-examples 10   \
    --top-k 5

python scripts/plot_losses.py \
        checkpoints/resnet50_2stage/history.csv \
	checkpoints/clip_vit_b32_2stage/history.csv \
	checkpoints/timm_vit_small_patch16_224_2stage/history.csv \
	--title "" \
	--output report/figures/train_val_losses.png \
	--labels ResNet CLIP DeiT

python scripts/plot_val_map.py \
        checkpoints/resnet50_2stage/history.csv \
	checkpoints/clip_vit_b32_2stage/history.csv \
	checkpoints/timm_vit_small_patch16_224_2stage/history.csv \
	--title "" \
	--output report/figures/val_map.png \
        --labels ResNet CLIP DeiT
