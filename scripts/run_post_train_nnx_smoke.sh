#!/usr/bin/env bash
# Post-training smoke commands from
# "MaxText NNX Migration — Smoke Test Status.md" (PT-01..PT-09).
#
# Default (FIX=0): EXACT documented flags, verbatim — no value changes.
# FIX=1: corrected for V6e-8 (8 chips). Flips the 4 values the doc gets wrong:
#   ici_data_parallelism 4->1 ; +override_model_config=True vocab_size=32000 (SFT random-init) ;
#   rollout_data_parallelism 2->8 & chips_per_vm 4->8 (RL) ; llama3.1-8b-Instruct -> llama3.1-8b.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   scripts/run_post_train_nnx_smoke.sh                 # verbatim doc (FIX=0)
#   FIX=1 scripts/run_post_train_nnx_smoke.sh           # corrected for V6e-8
#   CASES="01 04 07" scripts/run_post_train_nnx_smoke.sh
#
# Override bucket / run id if you like:
#   BASE_OUTPUT_DIR=gs://your-bucket/pt RUN_ID=myrun scripts/run_post_train_nnx_smoke.sh

set -u
cd "$(dirname "$0")/.." || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
HF_TOKEN="${HF_TOKEN:?set HF_TOKEN first}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-gs://lance-maxtext/pt_ckpt_${RUN_ID}}"
CASES="${CASES:-01 02L 02N 03 04 05 06 07 09}"
LOG_DIR="smoke_logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

LINEN_SEED="gs://lance-maxtext/pt_seed_ckpts/pt_seed_ckpts/pt_seed_ckpt_gpt352k_linen/checkpoints/9/items"
DISTILL_SEED="gs://lance-maxtext/pt_seed_ckpts/pt_seed_ckpts/pt_seed_ckpt_gpt352k_v32k_linen/checkpoints/4/items"

# --- FIX toggle: 0 = verbatim doc ; 1 = corrected for V6e-8 (8 chips) ---------
FIX="${FIX:-0}"
if [[ "${FIX}" == "1" ]]; then
  ICI_DATA=1                                              # doc 4 -> 8*4=32 != 8 chips (mesh assert)
  VOCAB_FIX="override_model_config=True vocab_size=32000" # gpt3-52k vocab 1024 vs tokenizer 32000
  CHIPS_PER_VM=8                                          # V6e-8 single host = 8 chips/VM
  ROLLOUT_DP=8                                            # tp*dp*ep must == 8 sampler devices
  LLAMA31_MODEL=llama3.1-8b                               # config is llama3.1-8b.yml (no -Instruct)
else
  ICI_DATA=4
  VOCAB_FIX=""
  CHIPS_PER_VM=4
  ROLLOUT_DP=2
  LLAMA31_MODEL=llama3.1-8b-Instruct
fi

# --- PT-01 SFT smoke (NNX) ---------------------------------------------------
pt01() {
  python3 -m maxtext.trainers.post_train.sft.train_sft \
    src/maxtext/configs/post_train/sft.yml \
    model_name=gpt3-52k \
    pure_nnx=True enable_nnx=True pure_nnx_decoder=True \
    per_device_batch_size=1 ici_fsdp_parallelism=8 ici_data_parallelism=${ICI_DATA} \
    max_target_length=1024 steps=5 eval_interval=-1 \
    gradient_accumulation_steps=1 weight_dtype=float32 \
    log_config=False enable_goodput_recording=False profiler=xplane \
    tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
    hf_access_token="$HF_TOKEN" \
    ${VOCAB_FIX} \
    base_output_directory="${BASE_OUTPUT_DIR}/" \
    run_name=pt_sft_nnx_xpk_${RUN_ID}_01_sft_smoke
}

# --- PT-02 SFT from Linen checkpoint (Linen) ---------------------------------
pt02L() {
  python3 -m maxtext.trainers.post_train.sft.train_sft \
    src/maxtext/configs/post_train/sft.yml \
    model_name=gpt3-52k \
    pure_nnx=False enable_nnx=False pure_nnx_decoder=False \
    per_device_batch_size=1 ici_fsdp_parallelism=8 ici_data_parallelism=${ICI_DATA} \
    max_target_length=1024 steps=5 eval_interval=-1 \
    gradient_accumulation_steps=1 weight_dtype=float32 \
    log_config=False enable_goodput_recording=False profiler=xplane \
    tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
    hf_access_token="$HF_TOKEN" \
    load_parameters_path="$LINEN_SEED" \
    base_output_directory="${BASE_OUTPUT_DIR}/" \
    run_name=pt_sft_nnx_xpk_${RUN_ID}_02_sft_linen_ckpt
}

# --- PT-02 SFT from NNX checkpoint (NNX) -------------------------------------
pt02N() {
  python3 -m maxtext.trainers.post_train.sft.train_sft \
    src/maxtext/configs/post_train/sft.yml \
    model_name=gpt3-52k \
    pure_nnx=True enable_nnx=True pure_nnx_decoder=True \
    per_device_batch_size=1 ici_fsdp_parallelism=8 ici_data_parallelism=${ICI_DATA} \
    max_target_length=1024 steps=5 eval_interval=-1 \
    gradient_accumulation_steps=1 weight_dtype=float32 \
    log_config=False enable_goodput_recording=False profiler=xplane \
    tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
    hf_access_token="$HF_TOKEN" \
    load_parameters_path="$LINEN_SEED" \
    base_output_directory="${BASE_OUTPUT_DIR}/" \
    run_name=pt_sft_nnx_xpk_${RUN_ID}_02_sft_nnx_ckpt
}

# --- PT-03 Cross-format SFT (Linen ckpt -> NNX model) ------------------------
pt03() {
  python3 -m maxtext.trainers.post_train.sft.train_sft \
    src/maxtext/configs/post_train/sft.yml \
    model_name=gpt3-52k \
    pure_nnx=True enable_nnx=True pure_nnx_decoder=True \
    per_device_batch_size=1 ici_fsdp_parallelism=8 ici_data_parallelism=${ICI_DATA} \
    max_target_length=1024 steps=5 eval_interval=-1 \
    gradient_accumulation_steps=1 weight_dtype=float32 \
    log_config=False enable_goodput_recording=False profiler=xplane \
    tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
    hf_access_token="$HF_TOKEN" \
    load_parameters_path="$LINEN_SEED" \
    base_output_directory="${BASE_OUTPUT_DIR}/" \
    run_name=pt_sft_nnx_xpk_${RUN_ID}_03_sft_linen_ckpt
}

# --- PT-04 RL GRPO smoke (Qwen3-0.6B) ----------------------------------------
pt04() {
  python3 -m maxtext.trainers.post_train.rl.train_rl \
    src/maxtext/configs/post_train/rl.yml \
    model_name=qwen3-0.6b \
    tokenizer_path=Qwen/Qwen3-0.6B \
    chips_per_vm=${CHIPS_PER_VM} num_batches=2 \
    rollout_data_parallelism=${ROLLOUT_DP} \
    trainer_devices_fraction=1.0 sampler_devices_fraction=1.0 \
    async_scheduling=False \
    log_config=False enable_goodput_recording=False profiler=xplane \
    debug.rl=False \
    pure_nnx=True enable_nnx=True pure_nnx_decoder=True \
    hf_access_token="$HF_TOKEN" \
    base_output_directory="${BASE_OUTPUT_DIR}/04_rl_grpo_smoke" \
    run_name=pt_rl_grpo_smoke \
    rl.loss_algo=grpo
}

# --- PT-05 RL GSPO smoke (Qwen3-0.6B) ----------------------------------------
pt05() {
  python3 -m maxtext.trainers.post_train.rl.train_rl \
    src/maxtext/configs/post_train/rl.yml \
    model_name=qwen3-0.6b \
    tokenizer_path=Qwen/Qwen3-0.6B \
    chips_per_vm=${CHIPS_PER_VM} num_batches=2 \
    rollout_data_parallelism=${ROLLOUT_DP} \
    trainer_devices_fraction=1.0 sampler_devices_fraction=1.0 \
    async_scheduling=False \
    log_config=False enable_goodput_recording=False profiler=xplane \
    debug.rl=False \
    pure_nnx=True enable_nnx=True pure_nnx_decoder=True \
    hf_access_token="$HF_TOKEN" \
    base_output_directory="${BASE_OUTPUT_DIR}/05_rl_gspo_smoke" \
    run_name=pt_rl_gspo_smoke \
    rl.loss_algo=gspo-token
}

# --- PT-06 RL GRPO functional (Llama3.1-8B) ----------------------------------
pt06() {
  python3 -m maxtext.trainers.post_train.rl.train_rl \
    src/maxtext/configs/post_train/rl.yml \
    model_name=${LLAMA31_MODEL} \
    tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
    chips_per_vm=${CHIPS_PER_VM} num_batches=2 \
    rollout_data_parallelism=${ROLLOUT_DP} \
    trainer_devices_fraction=1.0 sampler_devices_fraction=1.0 \
    async_scheduling=False \
    log_config=False enable_goodput_recording=False profiler=xplane \
    debug.rl=False \
    convert_checkpoint_if_possible=False enable_checkpointing=False \
    pure_nnx=True enable_nnx=True pure_nnx_decoder=True \
    hf_access_token="$HF_TOKEN" \
    base_output_directory="${BASE_OUTPUT_DIR}/06_rl_grpo_functional" \
    run_name=pt_rl_grpo_llama31_functional \
    rl.loss_algo=grpo
}

# --- PT-07 Distillation (GPT-3 52k student + teacher) ------------------------
pt07() {
  python3 -m maxtext.trainers.post_train.distillation.train_distill \
    src/maxtext/configs/post_train/distillation.yml \
    student_overrides.model_name=gpt3-52k student_overrides.vocab_size=32000 \
    student_overrides.override_model_config=True \
    teacher_overrides.model_name=gpt3-52k teacher_overrides.vocab_size=32000 \
    teacher_overrides.override_model_config=True \
    teacher_overrides.load_parameters_path="$DISTILL_SEED" \
    tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
    hf_access_token="$HF_TOKEN" \
    steps=5 per_device_batch_size=1 teacher_overrides.per_device_batch_size=1 \
    ici_fsdp_parallelism=8 ici_data_parallelism=${ICI_DATA} \
    weight_dtype=float32 gradient_accumulation_steps=1 \
    profiler=xplane log_config=False enable_goodput_recording=False \
    pure_nnx=True enable_nnx=True pure_nnx_decoder=True \
    base_output_directory="${BASE_OUTPUT_DIR}/" \
    run_name=pt_distill_nnx_xpk_${RUN_ID}_07_distill_smoke
}

# --- PT-09 LoRA SFT (GPT-3 52k) ----------------------------------------------
pt09() {
  python3 -m maxtext.trainers.post_train.sft.train_sft \
    src/maxtext/configs/post_train/sft.yml \
    model_name=gpt3-52k \
    pure_nnx=True enable_nnx=True pure_nnx_decoder=True \
    per_device_batch_size=1 ici_fsdp_parallelism=8 ici_data_parallelism=${ICI_DATA} \
    max_target_length=1024 steps=5 eval_interval=-1 \
    gradient_accumulation_steps=1 weight_dtype=float32 \
    log_config=False enable_goodput_recording=False profiler=xplane \
    tokenizer_path=meta-llama/Llama-2-7b-chat-hf tokenizer_type=huggingface \
    hf_access_token="$HF_TOKEN" \
    lora.enable_lora=True lora.lora_rank=8 lora.lora_alpha=16 \
    'lora.lora_module_path=decoder/layers/(self_attention/(qkv_proj|out)|mlp/(wi|wo))' \
    ${VOCAB_FIX} \
    base_output_directory="${BASE_OUTPUT_DIR}/" \
    run_name=pt_sft_nnx_xpk_${RUN_ID}_09_lora_sft
}

# --- run selected cases ------------------------------------------------------
echo "MODE: FIX=${FIX}  (0=verbatim doc, 1=corrected for V6e-8)"
for c in ${CASES}; do
  echo "===== PT-${c} ====="
  "pt${c}" 2>&1 | tee "${LOG_DIR}/${c}.log"
done
echo "logs: ${LOG_DIR}"
