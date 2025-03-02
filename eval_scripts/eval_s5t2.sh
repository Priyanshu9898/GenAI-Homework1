#!/bin/bash

experiment_name="LSTM_byte_embedder"

BASE_DIR="/share/csc591s25/pmalavi/tmp"
LOG_DIR="/share/csc591s25/pmalavi/logs/eval"
model_dir="/share/csc591s25/pmalavi/GenAI-for-Systems-Gym/homework-1/models/LSTM"

checkpoint_path="${BASE_DIR}/${experiment_name}/checkpoints/20000.ckpt"
config_path="${BASE_DIR}/${experiment_name}/model_config.json"

echo "Submitting evaluation job for ${experiment_name}..."

bsub <<EOF
#!/bin/bash
#BSUB -n 1
#BSUB -W 1:00
#BSUB -q short
#BSUB -J ${experiment_name}
#BSUB -o ${LOG_DIR}/${experiment_name}.out.%J
#BSUB -e ${LOG_DIR}/${experiment_name}.err.%J

source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env

cd "$model_dir"

python3 -m cache_replacement.policy_learning.cache.main --experiment_base_dir="/share/csc591s25/pmalavi/eval" --experiment_name="${experiment_name}" --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" --cache_configs="cache_replacement/policy_learning/cache/configs/eviction_policy/learned.json" --memtrace_file="cache_replacement/policy_learning/cache/traces/astar_313B_test.csv" --config_bindings="eviction_policy.scorer.checkpoint=\"${checkpoint_path}\"" --config_bindings="eviction_policy.scorer.config_path=\"${config_path}\"" --warmup_period=0
echo "Evaluation for ${experiment_name} completed."

EOF

echo "Evaluation job submitted!"
