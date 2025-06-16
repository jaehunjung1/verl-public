#!/bin/bash

#SBATCH -p batch_block1
#SBATCH -A nvr_lacr_llm
#SBATCH -J qwen2.5-vl-7b-it.4nodes
#SBATCH -N 4                       # number of nodes
#SBATCH --gpus-per-node=8          # number of gpus per node
#SBATCH -t 02:00:00                # wall time
#SBATCH --ntasks-per-node=1        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --overcommit               # needed for pytorch
#SBATCH --output=slurm_logs/slurm-%j.out     # output stream
#SBATCH --error=slurm_logs/slurm-%j.out      # error stream
#SBATCH --dependency=singleton


# project settings
PROJECT_DIR="$(realpath ../../../)"
ADAPTATIONS_DIR="$PROJECT_DIR/adaptations/geo3k"
DATA_DIR="$ADAPTATIONS_DIR/data"

# enroot settings
CONTAINER_MOUNTS="/lustre:/lustre"
CONTAINER_IMAGE="$HOME/lustre/images/verl-sglang.sqsh"
export RAY_worker_register_timeout_seconds=1200  # important to preempt hanging process


# run-specific settings
#export RAY_DEBUG=legacy  # we will use vanilla RAY debugger (not VSCode extension)
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_NAME="grpo-geo3k"
EXPERIMENT_NAME="qwen2.5-vl-7b-it.4nodes"


# set logging
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."


# prepare nodes
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")  # Getting the node names
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_port=6379
export head_node_ip=$head_node:$head_port
echo "Head Node IP: $head_node_ip"


# start head node
echo "Starting head node at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" --container-name="verl-sglang" bash -c \
"ray start --head --node-ip-address=$head_node --port=$head_port --runtime-env-agent-port 9999 --metrics-export-port 9998 --dashboard-agent-grpc-port 9997 --block" &
sleep 10s


# start worker nodes
worker_num=$((SLURM_JOB_NUM_NODES))
for ((i = 1; i < worker_num; i++)); do
  worker_node=${nodes_array[$i]}
  echo "Starting worker node $i at $worker_node"
  srun --nodes=1 --ntasks=1 -w "$worker_node" --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" --container-name="verl-sglang" bash -c \
  "ray start --address $head_node_ip --runtime-env-agent-port 9999 --metrics-export-port 9998 --dashboard-agent-grpc-port 9997 --block" &
  sleep 1s
done


echo "===================================== Now Job Start ======================================="
sleep 30s


# start job
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" --container-name="verl-sglang" bash -c \
"python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$ADAPTATIONS_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_epochs=15"
