export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy_hil.py "$@" \
    --actor \
    --env box_placing_corner_env \
    --wandb_project box_placing_sac \
    --exp_name=sac_hil_corner_policy_evaluation \
    --eval_checkpoint_path "checkpoints 0406-18:37" \
    --eval_checkpoint_step 21000 \
    --eval_n_trajs 20 \
    --evaluation \
    # --debug
