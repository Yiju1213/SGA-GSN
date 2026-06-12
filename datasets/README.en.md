# datasets Data Loading Guide

This directory implements the VTG 2D/3D datasets. Its main job is to convert CSV files, metadata, RGB/depth/segmentation files on disk into PyTorch tensors that the runners can feed into models directly.

## Registration and Building

Datasets use a lightweight registry:

```text
datasets/__init__.py
  -> import datasets.VTG3D_Dataset
  -> import datasets.VTG2D_Dataset
datasets/build.py
  -> DATASETS = Registry('dataset')
```

`VTG3D` and `VTG2D` are registered through `@DATASETS.register_module()`. During training, `tools/builder.dataset_builder(args, config.dataset.train)` calls:

```python
build_dataset_from_cfg(config._base_, config.others)
```

Here, `config._base_` comes from `cfgs/dataset_configs/*.yaml`, while `config.others` provides `subset`, `bs`, and any fields injected by ablations. The constructed dataset is then wrapped by `torch.utils.data.DataLoader`.

## VTG3D Data Flow

`VTG3D_Dataset.py` serves 3D grasp stability and shape completion. During initialization it reads:

- `data/3DA-VTG/<subset>.csv`: sample list, usually `(obj_id, grasp_id)`.
- `data/3DA-VTG/<subset>-ids.txt`: object ids for the current split.
- `data/3DA-VTG/<obj_id>/_metadata.json`: camera matrices, poses, and the `isPositive` label for each grasp.

Per-sample loading flow:

1. Read visual depth and visual segmentation, then call `vtg3d_utils.getVisualPointCloud()` to obtain the object visual point cloud.
2. Read left/right tactile depth, then call `vtg3d_utils.getTactilePointCloud()` to obtain tactile point clouds and gel masks.
3. Sample visual/tactile point clouds to the configured sizes: `VIS_POINTS` and `TAC_POINTS`.
4. By default, align visual/tactile points and gt shape to world coordinates using camera poses and grasp/object poses; skip this when `without_spatial_align=True`.
5. Extract contact regions from tactile points using the gel masks.
6. Sample `GT_POINTS` complete-shape points from the obj mesh pointed to by `COMPLETE_MESH_PATH`.
7. Compute zero-mean from visual and contact points, then apply it to visual, tactile contact, and complete shape points.
8. Build a fixed-size `SC_INPUT_POINTS` shape-completion input; on the train split, apply scale jitter and Gaussian noise.
9. Build the grasp-stability input by concatenating left/right tactile points and appending the gel/contact dimension, producing `xyzc`.

In non-debug mode, `VTG3D.__getitem__()` returns:

```python
(
    (sc_input, gt_pc),      # shape_completion
    (tac_xyzc, label),      # grasp_stability
    zero_mean,
    sample,                # (obj_id, grasp_id)
)
```

Common tensor shapes:

- `sc_input`: `[SC_INPUT_POINTS, 3]`, shape-completion input from visual/contact points.
- `gt_pc`: `[GT_POINTS, 3]`, complete object point cloud.
- `tac_xyzc`: `[2 * TAC_POINTS, 4]`, concatenated left/right tactile points; the last channel is the gel/contact marker.
- `label`: scalar float from metadata `isPositive`.

In `tools/runner_3dvtg.py`, batches are unpacked as:

```python
sc_data, gs_data, zero_mean, sample = item
sc_input, sc_gt = sc_data[0].cuda(), sc_data[1].cuda()
gs_input, result_gs = gs_data[0].cuda(), gs_data[1].cuda()
```

Then `sc_input` goes into AdaPoinTr, while `gs_input` goes into the SGSNet/DGCNN grasp model.

## VTG2D Data Flow

`VTG2D_Dataset.py` serves the 2D CNN baselines. Initialization reads the same CSV, object ids, and metadata, and caches one tactile background image:

```text
BACKGROUND_PATH: data/3DA-VTG/bg_sim.jpg
```

Each sample reads:

- visual RGB: `VIS_RGB_PATH`
- visual segmentation: `VIS_SEG_PATH`
- left/right tactile RGB: `<grasp_id>_l` and `<grasp_id>_r` under `TAC_RGB_PATH`
- label: `isPositive` from metadata

The train split applies synchronized geometric augmentation:

- visual RGB and visual segmentation share the same resize/crop/flip/rotation parameters.
- tactile background and tactile grasp share the same geometric parameters.
- RGB/tactile streams then get light color jitter and blur.

If `ConcatTwoTactile=False`, returned fields include:

```python
{
    'visual_rgb': ...,
    'visual_seg': ...,
    'tactile_rgb_left': ...,
    'tactile_rgb_right': ...,
    'tactile_bg_left': ...,
    'tactile_bg_right': ...,
    'grasp_result': ...,
    'sample_info': ...,
}
```

If `ConcatTwoTactile=True`, left/right tactile images are concatenated horizontally before augmentation, and the dataset returns:

```python
{
    'visual_rgb': ...,
    'visual_seg': ...,
    'tactile_rgb_concat': ...,
    'tactile_bg_concat': ...,
    'grasp_result': ...,
    'sample_info': ...,
}
```

Image tensors are `[3, 224, 224]` by default and normalized with ImageNet mean/std.

## Mapping to Runners

- `GraspStability_CNNMCA` uses `visual_rgb` and `tactile_rgb_concat`.
- `GraspStability_CNN` uses `visual_rgb`, left/right tactile grasp images, and left/right backgrounds.
- In `VTG3D`, the `shape_completion` branch serves AdaPoinTr, and the `grasp_stability` branch serves SGSNet/DGCNN.

When debugging tensor shapes, trace the dataset return fields directly to the batch-unpack sections in `tools/runner_3dvtg.py` or `tools/runner_2dvtg.py`.

