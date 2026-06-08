# SGA-GSN Public Release

This is the minimal public code tree for SGA-GSN, prepared from the PoinTr and AdaPoinTr codebase.

Included code covers:

- PPCT / `SGSNet` grasp stability prediction.
- 2D baselines: `GraspStability_CNNMCA` and `GraspStability_CNN`.
- 3D baseline: `DGCNNGraspStability`.
- VTG 2D/3D dataset loaders and training runners.
- AdaPoinTr shape encoder used by the PPCT pipeline.

Not included:

- Dataset files.
- Training outputs under `experiments/`.
- Model weights in Git.
- GraspNet direct shape-completion experiments.
- Private, historical, paper, analysis, and visualization artifacts.

## Expected Layout

The code is intended to run at:

```text
/SGA-GSN
```

External data and assets should be linked or placed to match the config paths:

```text
/SGA-GSN/data/GraspNet-1B/tactile-extended
/SGA-GSN/data/graspnet-vhacd
```

The required AdaPoinTr shape checkpoint should be downloaded separately and placed at:

```text
/SGA-GSN/ckpts/ap_ps55.pth
```

`ckpts/ap_ps55.pth` is required for 3D VTG grasp-stability training and evaluation, but is not stored in Git.

## Entrypoints

3D PPCT training:

```bash
python main.py --_3dvtg --vtg3d_grasp_stability --config ./cfgs/3D_VTG_models/VTG_PPCT.yaml --exp_name public_ppct
```

2D CNNMCA baseline training:

```bash
python main.py --_2dvtg --config ./cfgs/3D_VTG_models/VTG_CNNMCA.yaml --exp_name public_cnnmca
```

DGCNN baseline training can use either `VTG_DGCNN_14M.yaml` or `VTG_DGCNN_45M.yaml` with the 3D VTG grasp-stability flags.

## Environment Notes

Install PyTorch and torchvision using the CUDA build appropriate for the target machine. Then install the Python packages listed in `requirements.txt`. PointNet2 ops should be installed from a local Pointnet2_PyTorch checkout, matching the container setup used during release preparation.
