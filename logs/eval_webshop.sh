#!/bin/bash
# Evaluate SPAR v2 WebShop text_rich checkpoints on all 500 test goals (one val batch, val_only)
cd /workspace/SPAR
export PYTHON=/workspace/SPAR/.venv/bin/python
export PATH=/workspace/SPAR/.venv/bin:$PATH
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_ENTITY=dlaehddn16-company
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_core_worker_num_server_call_thread=1 RAY_worker_num_grpc_internal_threads=1 RAY_num_grpc_internal_threads=1
export JAVA_TOOL_OPTIONS="-XX:+UseSerialGC -XX:CICompilerCount=2 -XX:ActiveProcessorCount=1"
export DATA_DIR=/workspace/SPAR/data_eval500
CKPT=checkpoints/spar_v2_Qwen2.5-1.5B-Instruct_webshop_textrich_seed0
mkdir -p $DATA_DIR
$PYTHON -m examples.data_preprocess.prepare --mode text --local_dir $DATA_DIR --train_data_size 16 --val_data_size 500
# run_v2_webshop.sh re-runs prepare with val 256 into DATA_DIR; keep the 500-goal copy separate
cp $DATA_DIR/text/test.parquet $DATA_DIR/test500.parquet
for STEP in ${STEPS:-150 100 50}; do
    echo "===== EVAL global_step_$STEP ====="
    EXPERIMENT_NAME=spar_v2_Qwen2.5-1.5B-Instruct_webshop_textrich_seed0_eval500_step$STEP \
    bash SPAR_v2/run_v2_webshop.sh vllm ray_init.num_cpus=64 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enforce_eager=False \
        data.val_files=$DATA_DIR/test500.parquet \
        data.val_batch_size=500 \
        trainer.val_only=True \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path=/workspace/SPAR/$CKPT/global_step_$STEP \
        trainer.default_local_dir=/workspace/SPAR/checkpoints/eval_tmp
    ray stop --force >/dev/null 2>&1 || true
    sleep 10
done
