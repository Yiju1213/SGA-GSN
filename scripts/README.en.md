# scripts Training and Resume Guide

This directory contains example training scripts. In the current public tree, all `scripts/*.sh` files call `scripts/train.sh`, but that file is not included. Prefer direct `python main.py ...` commands; the existing shell files are still useful as parameter templates.

## Recommended Entry

3D PPCT grasp-stability training:

```bash
python main.py \
  --_3dvtg \
  --vtg3d_grasp_stability \
  --config ./cfgs/3D_VTG_models/VTG_PPCT.yaml \
  --exp_name sanity_check \
  --num_workers 6
```

This command uses:

- dataset config: `cfgs/dataset_configs/VTG3D.yaml`
- shape model: `AdaPoinTr`
- grasp model: `SGSNet`
- frozen shape checkpoint: `ckpts/ap_ps55.pth`

Before launching, prepare:

```text
/SGA-GSN/data/3DA-VTG
/SGA-GSN/data/graspnet-vhacd
/SGA-GSN/ckpts/ap_ps55.pth
```

The output directory is determined by `utils/parser.py`:

```text
experiments/VTG_PPCT/3D_VTG_models/<exp_name>/
```

It stores the copied `config.yaml`, logs, `ckpt-last.pth`, `ckpt-best.pth`, and checkpoints for the final epochs.

## Resume Flow

Resume command example:

```bash
python main.py \
  --_3dvtg \
  --vtg3d_grasp_stability \
  --config ./cfgs/3D_VTG_models/VTG_PPCT.yaml \
  --exp_name sanity_check \
  --num_workers 6 \
  --resume
```

`--resume` behavior:

- `utils/config.py` switches to the copied `config.yaml` under the current experiment directory.
- `tools/builder.resume_model()` restores model weights, epoch, and best metrics from `ckpt-last.pth`.
- `tools/builder.resume_optimizer()` restores the optimizer from the same checkpoint.
- `--resume` cannot be used with `--start_ckpts`.
- `--test` cannot be used with `--resume`.

When resuming, keep `--config` and `--exp_name` pointing to the old experiment directory. Otherwise the code will look for `ckpt-last.pth` under a new experiment path.

## End-to-End Data Flow

The 3D PPCT training flow is:

```text
shell/python command
  -> utils/parser.py parses task flags and experiment path
  -> utils/config.py loads YAML and _base_ dataset config
  -> tools/builder.dataset_builder() builds VTG3D DataLoader
  -> tools/builder.model_builder() builds AdaPoinTr and SGSNet
  -> runner loads ckpts/ap_ps55.pth into AdaPoinTr and freezes it
  -> VTG3D returns sc_input, gt_pc, tac_xyzc, label
  -> AdaPoinTr(sc_input, return_latent=True) returns vis_f, vis_coor
  -> SGSNet re-encodes vis_coor and encodes tac_xyzc
  -> SGSNet fuses visual/tactile tokens and returns one logit
  -> BCE loss, validation metrics, checkpoint saving
```

Validation builds seen/unseen-style dataloader subsets and saves:

- `ckpt-best.pth`: saved when the primary metric improves.
- `ckpt-last.pth`: saved after every epoch and used for resume.
- `ckpt-epoch-XXX.pth`: extra checkpoints for the final epochs.

## Existing Shell Examples

Current files:

- `train_vtg3d_grasp.sh`: 3D PPCT training parameter template.
- `train_resume_vtg3d_grasp.sh`: 3D PPCT resume parameter template.
- `train_vtg2d_grasp.sh`: 2D CNNMCA parameter template.

They currently look like:

```bash
bash ./scripts/train.sh 0 \
  --_3dvtg \
  --config ./cfgs/3D_VTG_models/VTG_PPCT.yaml \
  --exp_name sanity_check \
  --num_workers 6 \
  --vtg3d_grasp_stability
```

Because the public tree does not include `scripts/train.sh`, use the equivalent `python main.py ...` command unless you provide your own launcher.

## 2D Baseline Notes

The 2D entry is:

```bash
python main.py \
  --_2dvtg \
  --config ./cfgs/3D_VTG_models/VTG_CNNMCA.yaml \
  --exp_name test \
  --num_workers 6
```

However, `tools/runner_2dvtg.py` currently returns immediately after `testModel2D(grasp_model, config)`. In the public tree, this means the entry profiles FLOPs/parameters/latency by default and does not reach the full training loop. To train a 2D baseline, remove or bypass that debug return first.

## Common Arguments

- `--_3dvtg` / `--_2dvtg`: choose exactly one VTG entry.
- `--vtg3d_grasp_stability`: 3D grasp-stability task; with `--_3dvtg`, choose exactly one 3D task.
- `--config`: YAML config path.
- `--exp_name`: experiment name, used in the output path.
- `--num_workers`: DataLoader worker count.
- `--start_ckpts`: initialize model weights from a checkpoint without restoring optimizer/epoch.
- `--resume`: fully resume from `ckpt-last.pth` under the current experiment directory.
- `--test --ckpts <path>`: test mode requires a checkpoint; the public 3D grasp-stability test branch is not fully implemented.
