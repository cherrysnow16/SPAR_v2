#!/bin/bash
# Evaluate Qwen3.5-2B WebShop text_rich checkpoints: val_only, 128 test goals, temperature 0.4, env seeds 0/123/456/789, no wandb
# SPAR v2 step 150, GraphGPO step 150, then GraphGPO step 100 if step 150 has collapsed (mean success < 30%)
cd /workspace/SPAR
export PYTHON=/workspace/SPAR/.venv/bin/python
export PATH=/workspace/SPAR/.venv/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_core_worker_num_server_call_thread=1 RAY_worker_num_grpc_internal_threads=1 RAY_num_grpc_internal_threads=1
export JAVA_TOOL_OPTIONS="-XX:+UseSerialGC -XX:CICompilerCount=2 -XX:ActiveProcessorCount=1"
VAL=/workspace/SPAR/data_eval128/test128.parquet
SEEDS=${SEEDS:-0 123 456 789}
EVAL_ARGS=(data.val_files=$VAL data.val_batch_size=128 trainer.val_only=True trainer.resume_mode=resume_path
           trainer.logger=['console'] trainer.default_local_dir=/workspace/SPAR/checkpoints/eval_tmp)

eval_spar() {  # $1 = step
    for SEED in $SEEDS; do
        echo "===== EVAL spar_v2 global_step_$1 seed $SEED ====="
        MODEL_PATH=Qwen/Qwen3.5-2B DATA_DIR=/workspace/SPAR/data_eval128 EXPERIMENT_NAME=eval_spar_qwen35 \
        bash SPAR_v2/run_v2_webshop.sh vllm ray_init.num_cpus=64 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.enforce_eager=False \
            +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True \
            +actor_rollout_ref.rollout.engine_kwargs.vllm.structured_outputs_config.backend=xgrammar \
            +actor_rollout_ref.rollout.engine_kwargs.vllm.structured_outputs_config.disable_any_whitespace=True \
            env.seed=$SEED "${EVAL_ARGS[@]}" \
            trainer.resume_from_path=/workspace/SPAR/checkpoints/spar_v2_Qwen3.5-2B_webshop_textrich_seed0/global_step_$1
        ray stop --force >/dev/null 2>&1 || true; sleep 10
    done
}
eval_graphgpo() {  # $1 = step
    for SEED in $SEEDS; do
        echo "===== EVAL graphgpo global_step_$1 seed $SEED ====="
        bash logs/launch_graphgpo_qwen35_webshop.sh \
            env.seed=$SEED "${EVAL_ARGS[@]}" trainer.experiment_name=eval_graphgpo_qwen35 \
            trainer.resume_from_path=/workspace/SPAR/checkpoints/graphgpo_Qwen3.5-2B_webshop_textrich_seed0/global_step_$1
        ray stop --force >/dev/null 2>&1 || true; sleep 10
    done
}
mean_success() {  # mean val success over the "===== EVAL graphgpo global_step_$1" blocks of this log
    awk -v tag="EVAL graphgpo global_step_$1 " 'index($0,tag){on=1;next} /===== EVAL/{on=0}
        on && match($0,/val\/success_rate:[0-9.]+/){s+=substr($0,RSTART+17,RLENGTH-17);n++} END{print (n?s/n:0)}' "$LOG"
}

eval_spar 150
eval_graphgpo 150
m=$(mean_success 150); echo "graphgpo step 150 mean success: $m"
if awk -v m="$m" 'BEGIN{exit !(m<0.3)}'; then eval_graphgpo 100; fi
