# export ASCEND_LAUNCH_BLOCKING=1   # for DEBUG
# export ASCEND_RT_VISIBLE_DEVICES=3,4,5,6,7   # select used NPU

accelerate launch \
    --num_processes 7 \
    --config_file config/dpsp_z3.yaml \
    train_r1_grpo.py \
    --config config/grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
   