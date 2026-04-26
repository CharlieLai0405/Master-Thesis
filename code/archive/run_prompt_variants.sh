#!/bin/bash
source /venv/anomaly_gpt/bin/activate
cd /workspace/Master-Thesis/code
LOGDIR=/workspace/Master-Thesis/logs

echo "=========================================="
echo "Prompt Variant 1: class-aware"
echo "Start: $(date)"
echo "=========================================="
python test_mvtec_prompt_variants.py --k_shot 2 --prompt_variant 1 > $LOGDIR/test_prompt_v1_class_aware_20260426.log 2>&1
echo "=== V1 Done ==="
tail -5 $LOGDIR/test_prompt_v1_class_aware_20260426.log

echo ""
echo "=========================================="
echo "Prompt Variant 2: explicit-instruction"
echo "Start: $(date)"
echo "=========================================="
python test_mvtec_prompt_variants.py --k_shot 2 --prompt_variant 2 > $LOGDIR/test_prompt_v2_explicit_20260426.log 2>&1
echo "=== V2 Done ==="
tail -5 $LOGDIR/test_prompt_v2_explicit_20260426.log

echo ""
echo "=========================================="
echo "Prompt Variant 3: comparison-based"
echo "Start: $(date)"
echo "=========================================="
python test_mvtec_prompt_variants.py --k_shot 2 --prompt_variant 3 > $LOGDIR/test_prompt_v3_comparison_20260426.log 2>&1
echo "=== V3 Done ==="
tail -5 $LOGDIR/test_prompt_v3_comparison_20260426.log

echo ""
echo "========== All prompt tests done! $(date) =========="
