#!/bin/bash
sizes=(32 64 128 256 512)
for size in "${sizes[@]}"; do
bsub <<EOF
#!/bin/bash
#BSUB -n 1
#BSUB -W 24:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -J mlp_width_${size}
#BSUB -o /share/csc591s25/avbhanda/logs/mlp_width_${size}.out.%J
#BSUB -e /share/csc591s25/avbhanda/logs/mlp_width_${size}.err.%J
source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env
cd "/share/csc591s25/avbhanda/models/MLP"
python3 -m cache_replacement.policy_learning.cache_model.main --experiment_base_dir=/share/csc591s25/avbhanda/tmp --experiment_name=mlp_width_${size} --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" --model_configs="cache_replacement/policy_learning/cache_model/configs/default.json" --model_bindings="lstm_hidden_size=${size}" --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_train.csv" --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv" --batch_size=32 --total_steps=40001
EOF
done
echo "All jobs submitted successfully"
