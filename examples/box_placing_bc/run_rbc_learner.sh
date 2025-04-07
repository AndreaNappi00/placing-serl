export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python bc_policy.py "$@" \
    --env box_placing_corner_env \
    --exp_name=bc_corner_policy \
    --seed 67 \
    --batch_size 256 \
    --demo_paths "demos/ur5_test_20_demos_2025-04-06_17-47-23.pkl" \
    --eval_checkpoint_step 0 \
    --checkpoint_path "/home/andrea/Code/placing-serl/examples/box_placing_bc/checkpoints" \
    # --debug # wandb is   disabled when debug
