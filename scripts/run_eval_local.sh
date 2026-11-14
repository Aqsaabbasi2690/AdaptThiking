#!/bin/bash
set -x
set -f   # '*' ko glob hone se rokta hai (aime*16)
export HYDRA_FULL_ERROR=1

# Default: sirf gsm8k (smoke test). Poora run: VAL_SETS="gsm8k math aime*16" bash scripts/run_eval_local.sh
VAL_SETS=${VAL_SETS:-"gsm8k"}
val_files="["
for s in $VAL_SETS; do val_files+="'./data/test/preprocessed_data/${s}.parquet',"; done
val_files="${val_files%,}]"

train_files="./data/train/preprocessed_data/deepscaler.parquet"
batch_size=128
max_response_length=16384
rollout=1
end_of_think_token_id=151649     # </think>
non_end_of_think_token_id=71486  # "Alright"

MODEL_PATH_LIST=(${MODELS:-$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B $HOME/models/AdaptThink-1.5B-delta0.05})
for MODEL_PATH in "${MODEL_PATH_LIST[@]}"; do
    NAME=${MODEL_PATH##*/}
    CKPT_DIR=$HOME/AdaptThink/outputs/$NAME
    TAG=$(echo "$VAL_SETS" | tr ' *' '__')
    RESULT_SAVE_DIR=$CKPT_DIR/eval_results_${TAG}_len${max_response_length}_n${rollout}
    EXP_NAME="eval-${TAG}-len${max_response_length}-n${rollout}-${NAME}"
    echo "EVAL MODEL: ${MODEL_PATH}"

    python3 -m src.main_ppo \
        algorithm.adv_estimator=naive \
        reward_model.reward_manager=adapt_think \
        data.train_files="$train_files" \
        data.val_files="$val_files" \
        data.train_batch_size=$batch_size \
        data.val_batch_size=128 \
        data.max_prompt_length=4096 \
        data.max_response_length=$max_response_length \
        actor_rollout_ref.adapt_think.eot_token_id=$end_of_think_token_id \
        actor_rollout_ref.adapt_think.non_eot_token_id=$non_end_of_think_token_id \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.rollout.max_num_seqs=64 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=$rollout \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.max_num_batched_tokens=24576 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        trainer.logger=['console'] \
        trainer.project_name=adapt_think_verl-eval \
        trainer.experiment_name=$EXP_NAME \
        trainer.val_before_train=True \
        +trainer.val_only=True \
        +trainer.val_result_save_dir=$RESULT_SAVE_DIR \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.default_local_dir=$CKPT_DIR \
        custom_reward_function.path="./src/adapt_think_rm.py" \
        custom_reward_function.name="adapt_think_rm" || { echo "RUN FAILED for $NAME"; exit 1; }
done