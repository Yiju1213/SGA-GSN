bash ./scripts/train.sh 0 \
    --_3dvtg \
    --config ./cfgs/3D_VTG_models/VTG_PPCT.yaml \
    --exp_name PPCT_EncDep4_FusDep6_30M --resume \
    --num_workers 6 \
    --vtg3d_grasp_stability \