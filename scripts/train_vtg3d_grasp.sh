bash ./scripts/train.sh 0 \
    --_3dvtg \
    --config ./cfgs/3D_VTG_models/VTG_PPCT.yaml \
    --exp_name sanity_check \
    --num_workers 6 \
    --vtg3d_grasp_stability \