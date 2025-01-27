export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy.py "$@" \
    --actor \
    --env box_placing_corner_env \
    --exp_name=sac_drq_policy_evaluation \
    --eval_checkpoint_path "/home/andrea/Code/voxel-serl/examples/box_placing_sac/checkpoints" \
    --eval_checkpoint_step 100000 \
    --eval_n_trajs 10 \
    --debug
