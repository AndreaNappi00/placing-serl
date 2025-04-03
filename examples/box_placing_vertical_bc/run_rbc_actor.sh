export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
which python && \
python bc_policy.py "$@" \
    --env box_placing_vertical_env \
    --exp_name=bc_vertical_policy \
    --seed 67 \
    --batch_size 256 \
    --eval_checkpoint_step 10000 \
    --checkpoint_path "/home/andrea/Code/placing-serl/examples/box_placing_vertical_bc/checkpoints" \
    # --debug # wandb is disabled when debug
