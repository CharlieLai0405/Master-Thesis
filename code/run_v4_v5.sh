#!/bin/bash
# Run proposals 4 and 5 sequentially, clearing VRAM between each
source /venv/anomaly_gpt/bin/activate
cd /workspace/Master-Thesis/code
LOGDIR=/workspace/Master-Thesis/logs

echo "=========================================="
echo "方案 4: TTA + Hybrid Reference Selection"
echo "Start: $(date)"
echo "=========================================="
python test_mvtec_v4_tta.py --k_shot 2 > $LOGDIR/test_v4_tta_hybrid_k2_20260425.log 2>&1
echo "=== V4 Done ==="
tail -5 $LOGDIR/test_v4_tta_hybrid_k2_20260425.log

echo ""
echo "=========================================="
echo "方案 5: Top-k Sim + Hybrid Reference Selection"
echo "Start: $(date)"
echo "=========================================="
python test_mvtec_v5_topk_sim.py --k_shot 2 > $LOGDIR/test_v5_topk_hybrid_k2_20260425.log 2>&1
echo "=== V5 Done ==="
tail -5 $LOGDIR/test_v5_topk_hybrid_k2_20260425.log

echo ""
echo "========== All done! $(date) =========="
