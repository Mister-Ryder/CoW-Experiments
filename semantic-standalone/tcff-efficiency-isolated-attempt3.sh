#!/usr/bin/env bash
set -uo pipefail
echo $$ > /autodl-fs/data/tcff-rebuttal-results/10_logs/efficiency_isolated_attempt3_watcher.pid
echo "$(date -Iseconds) waiting_for_selected_maps_pipeline_pid_26078"
while kill -0 26078 2>/dev/null; do sleep 30; done
while kill -0 6373 2>/dev/null; do sleep 30; done
if pgrep -f '[t]cff_efficiency.py' >/dev/null; then
  echo "unexpected_existing_efficiency_process" >&2
  exit 7
fi
if [ -e /autodl-fs/data/tcff-rebuttal-results/09_efficiency/efficiency_summary.csv ]; then
  echo "unexpected_standard_efficiency_summary_exists" >&2
  exit 8
fi
echo "$(date -Iseconds) isolated_attempt3_start"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
cd /root/autodl-tmp/tcff-49d9c2e-final
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/tcff-49d9c2e-final/SAM-Med3D:/root/autodl-tmp/tcff-49d9c2e-final
export LD_LIBRARY_PATH=/root/miniconda3/envs/cu122py39/lib:${LD_LIBRARY_PATH:-}
export NUMEXPR_MAX_THREADS=64
export OMP_NUM_THREADS=8
/root/autodl-tmp/tcff-venv/bin/python tcff_efficiency.py --project_root /root/autodl-tmp/tcff-49d9c2e-final --results_root /autodl-fs/data/tcff-rebuttal-results --device cuda:0 --warmup 10 --repeats 30
code=$?
echo "$(date -Iseconds) isolated_attempt3_exit=$code"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
exit $code
