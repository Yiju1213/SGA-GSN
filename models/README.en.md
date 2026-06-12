# models Model Structure Guide

This directory contains the shape-completion model, 3D grasp-stability models, and 2D baselines for SGA-GSN. A practical reading path is to understand the registry first, then follow the two runner-side lines: `shape_model` and `grasp_model`.

## Registration and Building

Models use the registry in `models/build.py`:

```python
MODELS = registry.Registry('models')
```

`models/__init__.py` imports model files, which triggers their `@MODELS.register_module()` decorators:

- `AdaPoinTr`
- `SGSNet`
- `GraspStability_CNNMCA`
- `GraspStability_CNN`
- `DGCNN`
- `DGCNNGraspStability`

During training, runners call:

```python
builder.model_builder(config.shape_model)
builder.model_builder(config.grasp_model)
```

`MODELS.build(cfg)` instantiates the class whose name matches `NAME` in the YAML config. When adding a new model, make sure its file is imported by `models/__init__.py` and that the YAML `NAME` matches the registered class name.

## Main Models

- `AdaPoinTr.py`: shape-completion network. During 3D grasp-stability training, it loads `ckpts/ap_ps55.pth` and stays frozen. The normal PPCT flow calls it with `return_latent=True` to get `vis_f` and `vis_coor`.
- `GraspStability_PPCT.py`: PPCT/SGSNet main model. It receives AdaPoinTr visual latent/coordinates plus tactile `xyzc`; the current default config re-encodes `vis_coor`, fuses the resulting visual tokens with tactile tokens, then outputs one grasp-stability logit.
- `dgcnn.py`: 3D DGCNN baseline. `DGCNNGraspStability` uses visual/tactile branches and a classifier for stability prediction.
- `GraspStability_CNN_MCA.py`: 2D CNNMCA baseline with visual/tactile image feature extractors and a cross-modality fusion transformer.
- `GraspStability_CNN.py`: 2D CNN baseline with camera before/during and left/right tactile before/during inputs.

## PPCT Module Hierarchy

The PPCT main path is organized inside `SGSNet`:

```text
SGSNet
  -> PointEncoder for tactile/contact points
  -> PointEncoder for visual coordinates when use_vis_encoder=True
  -> vis_mem_link / tac_mem_link
  -> MultiResCrossAttnEntry
  -> max pooling
  -> prediction head
  -> one logit
```

Related files:

- `PPCT_utils.py`: point encoders, grouping, point-sequence dropout/shuffle, and other PPCT input-processing modules.
- `Transformer_utils.py`: attention, cross-attention, graph/deformable attention blocks, and shared transformer components.
- `MultiResolutionCrossAttn.py`: multi-resolution cross-attention fusion, including tactile-query/visual-memory fusion, downsampling, and optional multi-scale aggregation.
- `GraspStability_PPCT.py`: assembles SGSNet from the above modules and provides the loss manager plus ablation forward paths.

## Training-Time Calls

3D grasp-stability training in `tools/runner_3dvtg.py` builds two models:

```python
shape_model = builder.model_builder(config.shape_model)
grasp_model = builder.model_builder(config.grasp_model)
```

Then:

1. `shape_model` loads `ckpts/ap_ps55.pth`.
2. `shape_model.parameters()` are set to `requires_grad=False`.
3. `sc_input` goes into `shape_model(sc_input, return_latent=True)`.
4. The returned `vis_f, vis_coor` and `gs_input` go into `grasp_model`.
5. With the current `use_vis_encoder=True`, SGSNet re-encodes `vis_coor` with its own `PointEncoder`; the incoming `vis_f` is used directly only when `use_vis_encoder=False`.
6. The grasp model outputs a `[B, 1]` or `[B]` logit.
7. The runner squeezes the logit to `[B]` and passes it to `LossManager`, which computes a BCE-with-logits style loss.

When `ablation.without_shape_completion=True`, the runner skips AdaPoinTr latent features and calls:

```python
grasp_model(None, sc_input, gs_input, obj_ids)
```

SGSNet then uses `forward_ablation_without_shape_completion()` and processes `sc_input` through its own visual encoder.

## Output and Metric Conventions

Grasp stability is currently a binary single-logit task:

- The target comes from metadata `isPositive`.
- The loss uses `binary_cross_entropy_with_logits`.
- Validation applies sigmoid to logits and computes metrics such as Accuracy, Precision, Recall, and F1.

Shape completion uses AdaPoinTr coarse/dense point outputs and validates with F-Score, Chamfer Distance, and optional EMD.

## Where to Look

- To change architecture: check the YAML `NAME`, then open the registered model class.
- To change PPCT fusion: start from `models/MultiResolutionCrossAttn.py`.
- To change point encoders or point-sequence dropout: inspect `models/PPCT_utils.py`.
- To change losses: grasp-stability loss is in `LossManager` in `GraspStability_PPCT.py`; shape-completion loss is in `AdaPoinTr.py`.
