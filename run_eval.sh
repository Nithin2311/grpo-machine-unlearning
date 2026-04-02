#!/bin/bash
# Wait for training to finish, then auto-run evaluation
echo "Waiting for training (PID 14397) to finish..."
wait 14397
echo "Training done. Starting evaluation..."
cd /root/grpo-machine-unlearning/src
python3 evaluate.py \
    --checkpoint /root/grpo-machine-unlearning/grpo_unlearning_test \
    --subject "Stephen King" \
    --n_forget 100 \
    --n_retain 100 \
    --output_dir /root/grpo-machine-unlearning/results \
    2>&1 | tee /root/grpo-machine-unlearning/eval_run.log
echo "Evaluation complete. Results in: /root/grpo-machine-unlearning/results/"
