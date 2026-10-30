#!/bin/bash
# Launch SPAR v2 (Qwen2.5-1.5B-Instruct, ALFWorld, 100 steps, max_steps 25, raw obs) with thread caps
cd /workspace/SPAR
export PYTHON=/workspace/SPAR/.venv/bin/python
export PATH=/workspace/SPAR/.venv/bin:$PATH
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=dlaehddn16-company
export ALFWORLD_DATA=/workspace/alfworld_data
export SPAR_SUBGOAL_DUMP_DIR=/workspace/SPAR/logs/subgoal_dump
# container pids.max=11776: keep per-process thread pools small on this 192-core box
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# ray 2.59 workers default to ~51 threads (gRPC pools); 256 env actors would exceed the limit
export RAY_core_worker_num_server_call_thread=1 RAY_worker_num_grpc_internal_threads=1 RAY_num_grpc_internal_threads=1
# speed: 1.5B fits on one GPU -> TP=1 (2 DP engines); CUDA graphs on
exec bash SPAR_v2/run_v2_alfworld.sh vllm ray_init.num_cpus=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=False
