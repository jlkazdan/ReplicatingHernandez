#!/bin/bash
# Run sweep sequentially — one run at a time on GPU 3
# Usage: ./run_sweep_sequential.sh <sweep_id> [max_runs]

SWEEP_ID="${1:?Usage: $0 <sweep_id> [max_runs]}"
MAX_RUNS="${2:-20}"
ENTITY="jchud-stanford-university"
PROJECT="hernandez-replication"
GPU_ID=3
LOG="/lfs/skampere1/0/jchud/ReplicatingHernandez/sweep_sequential.log"

source /lfs/skampere1/0/jchud/scaling_mem_env/bin/activate
export PYTHONPATH="/lfs/skampere1/0/jchud/ReplicatingHernandez:$PYTHONPATH"
export HF_HOME="/lfs/skampere1/0/shared_hf_cache"
export HF_DATASETS_CACHE="/lfs/skampere1/0/shared_hf_cache/datasets"

for i in $(seq 1 $MAX_RUNS); do
    echo "$(date) === Starting agent run $i/$MAX_RUNS for sweep $SWEEP_ID ===" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent --count 1 "$ENTITY/$PROJECT/$SWEEP_ID" 2>&1 | tee -a "$LOG"
    EXIT_CODE=$?
    echo "$(date) === Agent exited with code $EXIT_CODE ===" | tee -a "$LOG"

    # Check if sweep is done
    SWEEP_STATE=$(python3 -c "
import wandb
api = wandb.Api()
try:
    sweep = api.sweep('$ENTITY/$PROJECT/$SWEEP_ID')
    print(sweep.state)
except:
    print('UNKNOWN')
" 2>/dev/null)

    echo "$(date) Sweep state: $SWEEP_STATE" | tee -a "$LOG"

    if [ "$SWEEP_STATE" = "FINISHED" ]; then
        echo "$(date) Sweep complete!" | tee -a "$LOG"
        break
    fi

    # Brief pause between runs
    sleep 5
done

echo "$(date) === Sequential sweep runner done ===" | tee -a "$LOG"
