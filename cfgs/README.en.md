# cfgs Configuration Guide

This directory stores experiment configurations for SGA-GSN. Most users start by selecting one model YAML from `cfgs/3D_VTG_models/`, then launch training with `main.py --config ...`.

## How Configs Are Loaded

The entry path is:

```text
main.py -> utils/parser.py -> utils/config.py
```

`utils/parser.py` parses command-line arguments and creates the experiment directory from the config path and `--exp_name`:

```text
experiments/<config_stem>/<config_parent>/<exp_name>/
```

`utils/config.py` reads YAML through `cfg_from_yaml_file()` and recursively merges fields through `merge_new_config()`. The special `_base_` key is treated as another YAML file and merged at the current location.

For example, `VTG_PPCT.yaml` contains:

```yaml
dataset : {
  train : { _base_: cfgs/dataset_configs/VTG3D.yaml,
            others: {subset: 'train'}},
  val : { _base_: cfgs/dataset_configs/VTG3D.yaml,
            others: {subset: 'test'}},
  test : { _base_: cfgs/dataset_configs/VTG3D.yaml,
            others: {subset: 'test'}}}
```

At runtime, `VTG3D.yaml` becomes `config.dataset.train._base_` and similar nodes. Fields under `others` are passed as default arguments to the dataset builder, so `VTG3D.__init__()` can read fields such as `DATA_PATH`, `GT_POINTS`, `VIS_POINTS`, and `subset` directly.

## VTG_PPCT.yaml Walkthrough

`VTG_PPCT.yaml` is the main 3D VTG grasp-stability configuration. Important fields:

- `optimizer`: currently `AdamW`; learning rate and weight decay live under `kwargs`.
- `scheduler`: currently `LambdaLR`; `decay_step`, `lr_decay`, and `lowest_decay` control LR decay.
- `bnmscheduler`: decays BatchNorm momentum.
- `dataset`: attaches `cfgs/dataset_configs/VTG3D.yaml` for train/val/test and selects splits through `others.subset`.
- `shape_model`: configures `AdaPoinTr`; in grasp-stability training it loads `ckpts/ap_ps55.pth` and stays frozen to provide shape latent features and coordinates.
- `grasp_model`: configures `SGSNet`, the PPCT grasp-stability network.
- `train_data_ratio`: fraction of the full training set used by `get_shuffled_subset()`.
- `total_bs`: single-GPU batch size; in distributed training it is divided by `world_size` and written to `config.dataset.train.others.bs`.
- `step_per_update`: gradient accumulation interval before one optimizer step.
- `max_epoch`: maximum training epoch.
- `consider_metric`: primary metric for saving the best checkpoint; grasp stability uses `AvgAcc` by default.

## PPCT Model Config

`grasp_model.NAME: SGSNet` maps to the registered model in `models/GraspStability_PPCT.py`. Key blocks:

- `use_vis_encoder`: when `True`, SGSNet re-encodes AdaPoinTr visual coordinates with its own `PointEncoder`.
- `use_vis_seq_shuf_drop` / `use_tac_seq_shuf_drop`: enable point-sequence dropout and shuffling during training.
- `tactile_encoder`: tactile input is `xyz + contact/gel`, so `input_dim` is 4.
- `visual_encoder`: visual input is usually `xyz`, so `input_dim` is 3.
- `feature_fusion`: configures `MultiResCrossAttnEntry`, including layer dimensions, heads, downsample point counts, and self/cross-attention block styles.
- `pred_head_drop_rate` / `pred_head_mid_feature`: configure the final binary logit prediction head.

The actual network flow is:

```text
VTG3D dataloader
  -> sc_input, tac_xyzc
  -> frozen AdaPoinTr(sc_input, return_latent=True)
  -> vis_f, vis_coor
  -> SGSNet visual PointEncoder re-encodes vis_coor when use_vis_encoder=True
  -> SGSNet contact PointEncoder encodes tac_xyzc
  -> MultiResCrossAttnEntry fuses tactile query with visual memory
  -> max pooling
  -> prediction head
  -> one BCE logit
```

In the current `VTG_PPCT.yaml`, `use_vis_encoder=True`, so `vis_coor` is passed through SGSNet's `visual_encoder`/`PointEncoder` again to produce fresh visual tokens. AdaPoinTr's returned `vis_f` is used directly by fusion only when `use_vis_encoder` is disabled.

## Ablation Settings

The top-level `ablation` block in `VTG_PPCT.yaml` is interpreted in `tools/runner_3dvtg.py`:

- `direct_concat`: model-phase ablation. It is copied to `config.grasp_model.direct_concat`, and SGSNet uses the direct-concat branch instead of cross-attention fusion.
- `full_shape_input`: runner-phase ablation. Training/validation sample from `sc_gt` instead of using `sc_input`, simulating complete-shape input.
- `without_spatial_align`: dataset-phase ablation. It is copied into train/val/test dataset configs, and VTG3D skips world-coordinate alignment for visual/tactile/gt points.
- `without_shape_completion`: model + runner ablation. The runner skips AdaPoinTr latent features, and SGSNet processes the shape input through its own visual encoder.

All current ablation flags are booleans. Use a new `--exp_name` after changing them to avoid mixing checkpoints and copied configs in one experiment directory.

## Common Config Files

- `cfgs/3D_VTG_models/VTG_PPCT.yaml`: main 3D PPCT/SGSNet experiment.
- `cfgs/3D_VTG_models/VTG_DGCNN_14M.yaml`: smaller 3D DGCNN baseline.
- `cfgs/3D_VTG_models/VTG_DGCNN_45M.yaml`: larger 3D DGCNN baseline.
- `cfgs/3D_VTG_models/VTG_CNNMCA.yaml`: 2D CNNMCA baseline.
- `cfgs/3D_VTG_models/VTG_CNN.yaml`: 2D CNN baseline.
- `cfgs/dataset_configs/VTG3D.yaml`: 3D VTG data paths, point counts, and noise settings.
- `cfgs/dataset_configs/VTG2D.yaml` / `VTG2D_TacCat.yaml`: 2D VTG image paths and tactile concatenation switch.
