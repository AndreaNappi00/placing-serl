export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy_hil.py "$@" \
    --actor \
    --env box_placing_vertical_env \
    --max_traj_length 200 \
    --wandb_project box_placing_vertical_sac \
    --exp_name=sac_hil_vertical_policy_evaluation \
    --eval_checkpoint_path "checkpoints 0403-15:37" \
    --eval_checkpoint_step 25000 \
    --eval_n_trajs 20 \
    --evaluation \
    # --debug
