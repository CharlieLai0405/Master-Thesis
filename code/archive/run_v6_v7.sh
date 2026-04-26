#!/bin/bash
# Run V6 (multi-scale TTA) and V7 (Gaussian post-processing) tests
source /venv/anomaly_gpt/bin/activate
cd /workspace/Master-Thesis/code

echo "=========================================="
echo "V6: Multi-scale TTA (5 augmentations)"
echo "Start: $(date)"
echo "=========================================="
python test_mvtec_v6_multiscale_tta.py --k_shot 2 > ../logs/test_v6_multiscale_tta_k2_$(date +%Y%m%d).log 2>&1
tail -3 ../logs/test_v6_multiscale_tta_k2_$(date +%Y%m%d).log
echo "=== V6 Done ==="

echo "=========================================="
echo "V7: Gaussian post-processing sigma=4"
echo "Start: $(date)"
echo "=========================================="
python test_mvtec_v7_gaussian.py --k_shot 2 --sigma 4.0 > ../logs/test_v7_gaussian_s4_k2_$(date +%Y%m%d).log 2>&1
tail -3 ../logs/test_v7_gaussian_s4_k2_$(date +%Y%m%d).log
echo "=== V7 sigma=4 Done ==="

echo "=========================================="
echo "V7: Gaussian post-processing sigma=2"
echo "Start: $(date)"
echo "=========================================="
python test_mvtec_v7_gaussian.py --k_shot 2 --sigma 2.0 > ../logs/test_v7_gaussian_s2_k2_$(date +%Y%m%d).log 2>&1
tail -3 ../logs/test_v7_gaussian_s2_k2_$(date +%Y%m%d).log
echo "=== V7 sigma=2 Done ==="

echo "=========================================="
echo "V7: Gaussian post-processing sigma=8"
echo "Start: $(date)"
echo "=========================================="
python test_mvtec_v7_gaussian.py --k_shot 2 --sigma 8.0 > ../logs/test_v7_gaussian_s8_k2_$(date +%Y%m%d).log 2>&1
tail -3 ../logs/test_v7_gaussian_s8_k2_$(date +%Y%m%d).log
echo "=== V7 sigma=8 Done ==="

echo "========== All V6/V7 tests done! $(date) =========="
