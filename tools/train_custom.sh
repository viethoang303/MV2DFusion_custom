#!/usr/bin/env bash

CONFIG=$1
GPUS=${2:-1}               # số GPU bạn muốn dùng, default=1
SEED=${3:-2015}

# Nếu chỉ chạy 1 GPU, tắt hết distributed
if [ "$GPUS" -eq 1 ]; then
  echo "Running single-GPU training on GPU 0"
  export CUDA_VISIBLE_DEVICES=0
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
  python $(dirname "$0")/train.py \
      $CONFIG \
      --launcher none \
      --gpu-id 0 \
      --seed $SEED
else
  # Dùng DDP với torch.distributed.launch
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
  PORT=${PORT:-29500}

  echo "Running DDP training on $GPUS GPUs"
  export WORLD_SIZE=$(( GPUS * NNODES ))
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS-1)))

  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
  python -m torch.distributed.launch \
      --nnodes=$NNODES \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --master_port=$PORT \
      --nproc_per_node=$GPUS \
      --use_env \
      $(dirname "$0")/train.py \
      $CONFIG \
      --seed $SEED \
      --launcher pytorch ${@:4}
fi
