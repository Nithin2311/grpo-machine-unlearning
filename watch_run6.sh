#!/bin/bash
# Auto-eval watcher for Run 6.
# Checks each checkpoint and kills training if no improvement over GDA.
#
# Thresholds (GDA baseline = FS 0.4444, ARR 0.3333):
#   ckpt-100: FS < 0.30 → STOP (clearly regressing)
#   ckpt-200: FS < 0.35 → STOP (not recovering, worse than baseline)
#   ckpt-300: FS < 0.40 → STOP (not beating GDA baseline by ckpt-300)
#   ckpt-400: FS < 0.44 → STOP (GDA already at 0.4444, no point continuing)

set -e

GDA_FS=0.4444
LOG=results/run6_watcher.log
TRAIN_LOG=train_run6.log
RESULTS_DIR=results

mkdir -p $RESULTS_DIR

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

get_train_pid() {
    pgrep -f "train_grpo_run6.py" 2>/dev/null | head -1
}

eval_checkpoint() {
    local ckpt=$1
    local label=$2
    local threshold=$3
    local out="$RESULTS_DIR/run6_${label}.json"

    log "Evaluating $ckpt ..."
    cd /root/grpo-machine-unlearning
    python3 src/eval_quick.py --checkpoint "$ckpt" --n_forget 18 --output "$out" >> $LOG 2>&1

    # Extract FS from output JSON
    FS=$(python3 -c "import json; d=json.load(open('$out')); print(d['forget_score'])")
    ARR=$(python3 -c "import json; d=json.load(open('$out')); print(d['answer_recall_rate'])")

    log "$label: FS=$FS  ARR=$ARR  (threshold=$threshold)"

    # Check threshold
    BELOW=$(python3 -c "print(1 if float('$FS') < float('$threshold') else 0)")
    if [ "$BELOW" = "1" ]; then
        log "*** FS=$FS is below threshold $threshold → STOPPING RUN 6 ***"
        TRAIN_PID=$(get_train_pid)
        if [ -n "$TRAIN_PID" ]; then
            kill "$TRAIN_PID" 2>/dev/null && log "Killed training PID $TRAIN_PID"
        fi
        log "VERDICT: PIVOT NEEDED — Run 6 is not improving enough"
        log "Best result: GDA FS=$GDA_FS. Consider NPO or simpler entity."
        return 1
    else
        log "✓ FS=$FS >= threshold $threshold, continuing..."
        return 0
    fi
}

wait_for_checkpoint() {
    local ckpt=$1
    local timeout_min=$2
    local elapsed=0
    local interval=30

    log "Waiting for checkpoint: $ckpt (timeout=${timeout_min}m)"
    while [ ! -d "$ckpt" ]; do
        sleep $interval
        elapsed=$((elapsed + interval))
        if [ $elapsed -ge $((timeout_min * 60)) ]; then
            log "Timeout waiting for $ckpt"
            return 1
        fi
    done
    # Extra 30s for checkpoint to finish writing
    sleep 30
    log "Checkpoint ready: $ckpt"
    return 0
}

cd /root/grpo-machine-unlearning

log "=== Run 6 Watcher Started ==="
log "GDA baseline: FS=$GDA_FS"
log "Early-stop thresholds: ckpt-100=0.30, ckpt-200=0.35, ckpt-300=0.40, ckpt-400=0.44"

# Checkpoint 100 — bare minimum: must not catastrophically regress
wait_for_checkpoint "grpo_unlearning_run6/checkpoint-100" 60 || exit 1
eval_checkpoint "grpo_unlearning_run6/checkpoint-100" "ckpt100" "0.30" || exit 0

# Checkpoint 200 — should be recovering toward baseline
wait_for_checkpoint "grpo_unlearning_run6/checkpoint-200" 30 || exit 1
eval_checkpoint "grpo_unlearning_run6/checkpoint-200" "ckpt200" "0.35" || exit 0

# Checkpoint 300 — should be approaching GDA
wait_for_checkpoint "grpo_unlearning_run6/checkpoint-300" 30 || exit 1
eval_checkpoint "grpo_unlearning_run6/checkpoint-300" "ckpt300" "0.40" || exit 0

# Checkpoint 400 — should be matching or beating GDA
wait_for_checkpoint "grpo_unlearning_run6/checkpoint-400" 30 || exit 1
eval_checkpoint "grpo_unlearning_run6/checkpoint-400" "ckpt400" "0.44" || exit 0

# Checkpoint 500 — full run
wait_for_checkpoint "grpo_unlearning_run6/checkpoint-500" 30 || exit 1
eval_checkpoint "grpo_unlearning_run6/checkpoint-500" "ckpt500" "0.00"  # no stop, just record

log "=== Run 6 Complete ==="
log "All checkpoints evaluated. Check results/run6_*.json"

# Print final comparison
python3 -c "
import json, glob, os
files = sorted(glob.glob('results/run6_ckpt*.json'))
print()
print('=== RUN 6 RESULTS ===')
print(f'{\"Checkpoint\":<12} | {\"FS\":>6} | {\"KLR\":>6} | {\"ARR\":>6}')
print('-'*40)
print(f'{\"GDA-final\":<12} | {\"0.4444\":>6} | {\"0.7778\":>6} | {\"0.3333\":>6}')
print(f'{\"Baseline\":<12} | {\"0.3889\":>6} | {\"0.8333\":>6} | {\"0.3889\":>6}')
for f in files:
    d = json.load(open(f))
    label = os.path.basename(f).replace('run6_','').replace('.json','')
    print(f'{label:<12} | {d[\"forget_score\"]:>6.4f} | {d[\"keyword_leak_rate\"]:>6.4f} | {d[\"answer_recall_rate\"]:>6.4f}')
" 2>/dev/null | tee -a $LOG
