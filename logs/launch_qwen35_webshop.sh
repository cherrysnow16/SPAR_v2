#!/bin/bash
# Launch SPAR v2 (Qwen3.5-2B, WebShop text_rich, 150 steps, max_steps 15) with thread caps
cd /workspace/SPAR
export PYTHON=/workspace/SPAR/.venv/bin/python
export PATH=/workspace/SPAR/.venv/bin:$PATH
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=dlaehddn16-company
export SPAR_SUBGOAL_DUMP_DIR=/workspace/SPAR/logs/subgoal_dump_qwen35_webshop
export MODEL_PATH=Qwen/Qwen3.5-2B
# container pids.max=11776: keep per-process thread pools small on this 192-core box
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_core_worker_num_server_call_thread=1 RAY_worker_num_grpc_internal_threads=1 RAY_num_grpc_internal_threads=1
# each WebShop env actor runs a pyserini JVM: ~316 threads by default on 192 cores, 15 with these caps
export JAVA_TOOL_OPTIONS="-XX:+UseSerialGC -XX:CICompilerCount=2 -XX:ActiveProcessorCount=1"
export EXPERIMENT_NAME=spar_v2_Qwen3.5-2B_webshop_textrich_seed0
# text-only: skip the vision tower in vLLM; TP=1 (2 DP engines); CUDA graphs on
exec bash SPAR_v2/run_v2_webshop.sh vllm ray_init.num_cpus=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.structured_outputs_config.backend=xgrammar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.structured_outputs_config.disable_any_whitespace=True
