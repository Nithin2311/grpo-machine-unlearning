#!/bin/bash
echo "Waiting for training (PID 18902) to finish..."
wait 18902
echo "Training done at: $(date)"
cd /root/grpo-machine-unlearning/src

# Use the best checkpoint available (300 if clean finish, else 200)
CKPT="/root/grpo-machine-unlearning/grpo_unlearning_test"
if [ -d "$CKPT/checkpoint-300" ]; then
    EVAL_CKPT="$CKPT/checkpoint-300"
elif [ -d "$CKPT/checkpoint-200" ]; then
    EVAL_CKPT="$CKPT/checkpoint-200"
else
    EVAL_CKPT="$CKPT/checkpoint-100"
fi

echo "Evaluating checkpoint: $EVAL_CKPT"
python3 evaluate.py \
    --checkpoint "$EVAL_CKPT" \
    --subject "Stephen King" \
    --n_forget 100 \
    --n_retain 100 \
    --output_dir /root/grpo-machine-unlearning/results \
    2>&1 | tee /root/grpo-machine-unlearning/eval_run2.log
echo "Done at: $(date)"
