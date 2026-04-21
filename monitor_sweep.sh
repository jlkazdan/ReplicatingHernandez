#!/bin/bash
# Monitor sweep on GPU 3 in tmux session "catchall"
# Uses run_sweep_sequential.sh to ensure one run at a time

SWEEP_ID="wpc5vq1v"
ENTITY="jchud-stanford-university"
PROJECT="hernandez-replication"
TMUX_SESSION="catchall"
GPU_ID=3
LOG="/lfs/skampere1/0/jchud/ReplicatingHernandez/sweep_monitor.log"
RUNNER="/lfs/skampere1/0/jchud/ReplicatingHernandez/run_sweep_sequential.sh"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "$(timestamp) === Sweep Monitor Check ===" >> "$LOG"

# 1. Check if tmux session exists
if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "$(timestamp) ALERT: tmux session '$TMUX_SESSION' not found! Recreating..." >> "$LOG"
    tmux new-session -d -s "$TMUX_SESSION" "$RUNNER $SWEEP_ID 6; exec bash"
    echo "$(timestamp) Restarted sequential runner in tmux session '$TMUX_SESSION'" >> "$LOG"
fi

# 2. Check if sequential runner or wandb agent is running
RUNNER_PID=$(pgrep -f "run_sweep_sequential.*$SWEEP_ID" | head -1)
AGENT_PID=$(pgrep -f "wandb agent.*$SWEEP_ID" | head -1)
TRAIN_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU_ID 2>/dev/null | head -1)

if [ -n "$RUNNER_PID" ]; then
    echo "$(timestamp) Sequential runner active (PID $RUNNER_PID)" >> "$LOG"
elif [ -n "$AGENT_PID" ]; then
    echo "$(timestamp) Agent running (PID $AGENT_PID)" >> "$LOG"
elif [ -n "$TRAIN_PID" ]; then
    echo "$(timestamp) Training process on GPU $GPU_ID (PID $TRAIN_PID) but no agent/runner — may be orphaned" >> "$LOG"
else
    echo "$(timestamp) ALERT: Nothing running! Restarting sequential runner..." >> "$LOG"
    tmux send-keys -t "$TMUX_SESSION" C-c 2>/dev/null
    sleep 2
    tmux send-keys -t "$TMUX_SESSION" "$RUNNER $SWEEP_ID 6" Enter
    echo "$(timestamp) Restarted sequential runner" >> "$LOG"
fi

# 3. Check GPU 3 memory usage
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null)
echo "$(timestamp) GPU $GPU_ID memory: ${GPU_MEM} MiB" >> "$LOG"

# 4. Check sweep status via python
source /lfs/skampere1/0/jchud/scaling_mem_env/bin/activate
python3 -c "
import wandb, sys
api = wandb.Api()
sweep = api.sweep('$ENTITY/$PROJECT/$SWEEP_ID')
total = 6
done = sum(1 for r in sweep.runs if r.state == 'finished')
failed = sum(1 for r in sweep.runs if r.state in ('failed', 'crashed'))
running = sum(1 for r in sweep.runs if r.state == 'running')
print(f'Sweep: {done}/{total} done, {running} running, {failed} failed')
for r in sweep.runs:
    nr = r.config.get('data_config',{}).get('num_repeats','?')
    d = r.config.get('data_config',{}).get('direction','?')
    print(f'  {r.name} ({r.id}): {r.state} | nr={nr} dir={d}')
if done == total:
    print('ALL RUNS COMPLETE!')
" 2>&1 | tee -a "$LOG"

echo "" >> "$LOG"
