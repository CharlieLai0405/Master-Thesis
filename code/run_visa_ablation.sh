#!/bin/bash
cd /workspace/Master-Thesis/code
source /venv/anomaly_gpt/bin/activate
CKPT="./ckpt/train_visa/pytorch_model.pt"
LOG_DIR="./ckpt/train_visa/log/ablation"
mkdir -p $LOG_DIR

echo "=== Starting VisA Ablation Study $(date) ===" | tee $LOG_DIR/run_all.log

echo "[1/8] Baseline..." | tee -a $LOG_DIR/run_all.log
python test_visa_ablation.py --anomalygpt_ckpt $CKPT > $LOG_DIR/1_baseline.log 2>&1
tail -3 $LOG_DIR/1_baseline.log | tee -a $LOG_DIR/run_all.log

echo "[2/8] +Prompt Adapter only..." | tee -a $LOG_DIR/run_all.log
python test_visa_ablation.py --anomalygpt_ckpt $CKPT --use_prompt_adapter > $LOG_DIR/2_prompt_only.log 2>&1
tail -3 $LOG_DIR/2_prompt_only.log | tee -a $LOG_DIR/run_all.log

echo "[3/8] +Hybrid only..." | tee -a $LOG_DIR/run_all.log
python test_visa_ablation.py --anomalygpt_ckpt $CKPT --use_hybrid > $LOG_DIR/3_hybrid_only.log 2>&1
tail -3 $LOG_DIR/3_hybrid_only.log | tee -a $LOG_DIR/run_all.log

echo "[4/8] +TTA only..." | tee -a $LOG_DIR/run_all.log
python test_visa_ablation.py --anomalygpt_ckpt $CKPT --use_tta > $LOG_DIR/4_tta_only.log 2>&1
tail -3 $LOG_DIR/4_tta_only.log | tee -a $LOG_DIR/run_all.log

echo "[5/8] +Prompt Adapter + Hybrid..." | tee -a $LOG_DIR/run_all.log
python test_visa_ablation.py --anomalygpt_ckpt $CKPT --use_prompt_adapter --use_hybrid > $LOG_DIR/5_prompt_hybrid.log 2>&1
tail -3 $LOG_DIR/5_prompt_hybrid.log | tee -a $LOG_DIR/run_all.log

echo "[6/8] +Hybrid + TTA..." | tee -a $LOG_DIR/run_all.log
python test_visa_ablation.py --anomalygpt_ckpt $CKPT --use_hybrid --use_tta > $LOG_DIR/6_hybrid_tta.log 2>&1
tail -3 $LOG_DIR/6_hybrid_tta.log | tee -a $LOG_DIR/run_all.log

echo "[7/8] +Prompt Adapter + TTA..." | tee -a $LOG_DIR/run_all.log
python test_visa_ablation.py --anomalygpt_ckpt $CKPT --use_prompt_adapter --use_tta > $LOG_DIR/7_prompt_tta.log 2>&1
tail -3 $LOG_DIR/7_prompt_tta.log | tee -a $LOG_DIR/run_all.log

echo "[8/8] Full (all three)..." | tee -a $LOG_DIR/run_all.log
python test_visa_ablation.py --anomalygpt_ckpt $CKPT --use_prompt_adapter --use_hybrid --use_tta > $LOG_DIR/8_full.log 2>&1
tail -3 $LOG_DIR/8_full.log | tee -a $LOG_DIR/run_all.log

echo "=== ALL VISA ABLATION DONE $(date) ===" | tee -a $LOG_DIR/run_all.log
