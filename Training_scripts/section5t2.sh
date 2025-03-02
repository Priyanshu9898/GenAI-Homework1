#!/bin/bash

submit_job() {
    experiment_name="LSTM_byte_embedder"
    echo "Submitting experiment: ${experiment_name}"
    bsub <<EOF

#!/bin/bash

#BSUB -n 1
#BSUB -W 24:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -J ${experiment_name}
#BSUB -o /share/csc591s25/avbhanda/logs/${experiment_name}.out.%J
#BSUB -e /share/csc591s25/avbhanda/logs/${experiment_name}.err.%J

source ~/.bashrc

conda activate /share/csc591s25/conda_env/new_env

cd "/share/csc591s25/avbhanda/models/LSTM"

python3 -m cache_replacement.policy_learning.cache_model.main --experiment_base_dir=/share/csc591s25/avbhanda/tmp --experiment_name=${experiment_name} --cache_configs=cache_replacement/policy_learning/cache/configs/default.json --model_configs=cache_replacement/policy_learning/cache_model/configs/default.json --model_configs=cache_replacement/policy_learning/cache_model/configs/byte_embedder.json --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" --model_bindings="address_embedder.max_vocab_size=5000" --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_train.csv" --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv" --batch_size=32

EOF
}

submit_job &

echo "Job submitted successfully!"