CUDA_VISIBLE_DEVICES=0 python -u test_clipreid.py --config_file configs/cloth_change/vit_clipreid.yml \
DATASETS.NAMES 'prcc' \
MODEL.STRIDE_SIZE '[12, 12]' \
MODEL.CLOTH_PROMPT True MODEL.ID_PROMPT True \
MODEL.I2T_LOSS True MODEL.I2T_CLOTH_LOSS True \
MODEL.STAGE2_I2T_LOSS True MODEL.STAGE2_I2T_CLOTH_LOSS False \
OUTPUT_DIR './logs/' \
TEST.WEIGHT '/data/test96/cloth-changing/saved_models/prcc/ViT-B-16_best.pth' \
MODEL.DES_TRANS True MODEL.DES_WEIGHT 2.0 SOLVER.SEED 2025