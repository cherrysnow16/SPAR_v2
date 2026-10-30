#!/bin/bash
# Launch GraphGPO baseline (Qwen3.5-2B, WebShop text_rich, 150 steps, max_steps 15); same settings as the SPAR v2 WebShop runs
cd /workspace/SPAR
PYTHON=/workspace/SPAR/.venv/bin/python
export PATH=/workspace/SPAR/.venv/bin:$PATH
export PYTHONPATH=/workspace/SPAR:${PYTHONPATH:-}
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=dlaehddn16-company
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_core_worker_num_server_call_thread=1 RAY_worker_num_grpc_internal_threads=1 RAY_num_grpc_internal_threads=1
export JAVA_TOOL_OPTIONS="-XX:+UseSerialGC -XX:CICompilerCount=2 -XX:ActiveProcessorCount=1"
DATA_DIR=/workspace/SPAR/data
EXPERIMENT_NAME=graphgpo_Qwen3.5-2B_webshop_textrich_seed0
set -x
$PYTHON -m examples.data_preprocess.prepare --mode text --local_dir $DATA_DIR --train_data_size 16 --val_data_size 256
exec $PYTHON -m recipe.GraphGPO.main_graphgpo \
    algorithm.adv_estimator=graphgpo \
    algorithm.gamma=0.20 \
    algorithm.graphgpo.step_advantage_w=1.0 \
    algorithm.graphgpo.episode_advantage_w=1.0 \
    algorithm.graphgpo.mode=mean_std_norm \
    algorithm.graphgpo.normalize_distance=False \
    algorithm.use_kl_in_reward=False \
    data.train_files=$DATA_DIR/text/train.parquet \
    data.val_files=$DATA_DIR/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=5120 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3.5-2B \
    +actor_rollout_ref.model.text_only=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len=6144 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    env.env_name=webshop/WebAgentTextEnv \
    env.seed=0 \
    env.max_steps=15 \
    env.webshop.observation_mode=text_rich \
    +env.webshop.envs_per_actor=8 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    ray_init.num_cpus=64 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=aaai_rebuttal \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir=/workspace/SPAR/checkpoints/$EXPERIMENT_NAME \
    "$@"
