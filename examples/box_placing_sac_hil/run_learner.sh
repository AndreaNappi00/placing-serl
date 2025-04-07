export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy_hil.py "$@" \
    --learner \
    --env box_placing_corner_env \
    --exp_name=sac_hil_corner_policy \
    --max_traj_length 150 \
    --seed 42 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --max_steps 50000 \
    --reward_scale 1 \
    --demo_paths "demos/ur5_test_20_demos_2025-04-06_18-31-19.pkl" \
    --eval_checkpoint_step 0 \
    --load_checkpoint_path "checkpoints" \
    # --debug
#    --preload_rlds_path "/home/amdrea/Code/voxel-serl/examples/box_picking_sac/rlds" \
