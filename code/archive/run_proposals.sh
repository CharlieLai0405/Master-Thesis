#!/bin/bash
# Test all 3 improvement proposals independently
# Run from /workspace/Master-Thesis/code
source /venv/anomaly_gpt/bin/activate
cd /workspace/Master-Thesis/code

LOGDIR=/workspace/Master-Thesis/logs
DATE=20260425

# First, restore the dual_train checkpoint (with TextPromptAdapter, original PromptLearner)
cp ./ckpt/train_mvtec/pytorch_model_dual_train.pt ./ckpt/train_mvtec/pytorch_model.pt
echo "[!] Restored pytorch_model_dual_train.pt as active checkpoint"

echo ""
echo "=========================================="
echo "方案 1: Better Few-Shot Reference Selection"
echo "=========================================="
python test_mvtec_v1_better_refs.py --k_shot 2 > $LOGDIR/test_v1_better_refs_k2_$DATE.log 2>&1
echo "--- k=2 mixed results ---"
tail -5 $LOGDIR/test_v1_better_refs_k2_$DATE.log

echo ""
echo "=========================================="
echo "方案 4: Test-Time Augmentation"
echo "=========================================="
python test_mvtec_v4_tta.py --k_shot 2 > $LOGDIR/test_v4_tta_k2_$DATE.log 2>&1
echo "--- k=2 mixed results ---"
tail -5 $LOGDIR/test_v4_tta_k2_$DATE.log

echo ""
echo "=========================================="
echo "方案 5: Improved Similarity (top-k avg)"
echo "=========================================="
python test_mvtec_v5_topk_sim.py --k_shot 2 > $LOGDIR/test_v5_topk_sim_k2_$DATE.log 2>&1
echo "--- k=2 mixed results ---"
tail -5 $LOGDIR/test_v5_topk_sim_k2_$DATE.log

echo ""
echo "========== All proposal tests done! =========="
echo "End time: $(date)"
