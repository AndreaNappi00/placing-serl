export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python sac_policy.py "$@" \
    --learner \
    --env box_placing_corner_env \
    --exp_name=sac_drq_policy \
    --max_traj_length 300 \
    --seed 42 \
    --training_starts 900 \
    --utd_ratio 8 \
    --batch_size 2048 \
    --max_steps 50000 \
    --reward_scale 1 \
    --demo_paths "demos/ur5_test_20_demos_2025-01-23_15-14-21.pkl" \
#    --preload_rlds_path "/home/amdrea/Code/voxel-serl/examples/box_picking_sac/rlds" \
#    --debug
