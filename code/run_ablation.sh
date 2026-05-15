#!/bin/bash
cd /workspace/Master-Thesis/code
source /venv/anomaly_gpt/bin/activate
CKPT="./ckpt/train_mvtec/pytorch_model_v6_best.pt"
LOG_DIR="./ckpt/train_mvtec/log/ablation"
mkdir -p $LOG_DIR

echo "=== Starting Ablation Study $(date) ==="

echo "[1/8] Baseline..."
python test_mvtec_ablation.py --k_shot 2 --anomalygpt_ckpt $CKPT > $LOG_DIR/1_baseline.log 2>&1
tail -3 $LOG_DIR/1_baseline.log

echo "[2/8] +Prompt Adapter only..."
python test_mvtec_ablation.py --k_shot 2 --anomalygpt_ckpt $CKPT --use_prompt_adapter > $LOG_DIR/2_prompt_only.log 2>&1
tail -3 $LOG_DIR/2_prompt_only.log

echo "[3/8] +Hybrid only..."
python test_mvtec_ablation.py --k_shot 2 --anomalygpt_ckpt $CKPT --use_hybrid > $LOG_DIR/3_hybrid_only.log 2>&1
tail -3 $LOG_DIR/3_hybrid_only.log

echo "[4/8] +TTA only..."
python test_mvtec_ablation.py --k_shot 2 --anomalygpt_ckpt $CKPT --use_tta > $LOG_DIR/4_tta_only.log 2>&1
tail -3 $LOG_DIR/4_tta_only.log

echo "[5/8] +Prompt Adapter + Hybrid..."
python test_mvtec_ablation.py --k_shot 2 --anomalygpt_ckpt $CKPT --use_prompt_adapter --use_hybrid > $LOG_DIR/5_prompt_hybrid.log 2>&1
tail -3 $LOG_DIR/5_prompt_hybrid.log

echo "[6/8] +Hybrid + TTA..."
python test_mvtec_ablation.py --k_shot 2 --anomalygpt_ckpt $CKPT --use_hybrid --use_tta > $LOG_DIR/6_hybrid_tta.log 2>&1
tail -3 $LOG_DIR/6_hybrid_tta.log

echo "[7/8] +Prompt Adapter + TTA..."
python test_mvtec_ablation.py --k_shot 2 --anomalygpt_ckpt $CKPT --use_prompt_adapter --use_tta > $LOG_DIR/7_prompt_tta.log 2>&1
tail -3 $LOG_DIR/7_prompt_tta.log

echo "[8/8] Full (all three)..."
python test_mvtec_ablation.py --k_shot 2 --anomalygpt_ckpt $CKPT --use_prompt_adapter --use_hybrid --use_tta > $LOG_DIR/8_full.log 2>&1
tail -3 $LOG_DIR/8_full.log

echo "=== ALL ABLATION DONE $(date) ==="
