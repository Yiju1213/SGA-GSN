# models 模型结构快速指引

本目录包含 SGA-GSN 的 shape completion 模型、3D grasp stability 模型和 2D baseline。阅读模型时，建议先理解 registry 构建方式，再顺着 runner 中的 `shape_model` 和 `grasp_model` 两条线定位。

## 注册与构建

模型使用 `models/build.py` 中的 registry：

```python
MODELS = registry.Registry('models')
```

`models/__init__.py` 会导入各模型文件，触发文件中的 `@MODELS.register_module()`：

- `AdaPoinTr`
- `SGSNet`
- `GraspStability_CNNMCA`
- `GraspStability_CNN`
- `DGCNN`
- `DGCNNGraspStability`

训练时，runner 调用：

```python
builder.model_builder(config.shape_model)
builder.model_builder(config.grasp_model)
```

最终 `MODELS.build(cfg)` 根据 YAML 中的 `NAME` 实例化对应类。因此新增模型时，需要确保模型文件被 `models/__init__.py` 导入，并且 YAML 的 `NAME` 与注册类名一致。

## 主要模型

- `AdaPoinTr.py`: shape completion 网络。3D grasp stability 训练中会从 `ckpts/ap_ps55.pth` 加载权重并冻结。正常 PPCT 流程使用 `return_latent=True` 得到 `vis_f` 和 `vis_coor`。
- `GraspStability_PPCT.py`: PPCT/SGSNet 主模型。接收 AdaPoinTr 的 visual latent/coordinates 和 tactile `xyzc`；当前默认配置会重新编码 `vis_coor`，再与 tactile tokens 融合并输出单个 grasp-stability logit。
- `dgcnn.py`: 3D DGCNN baseline。`DGCNNGraspStability` 使用 visual/tactile 分支和 classifier 预测稳定性。
- `GraspStability_CNN_MCA.py`: 2D CNNMCA baseline，使用 visual/tactile image feature extractor 和 cross-modality fusion transformer。
- `GraspStability_CNN.py`: 2D CNN baseline，输入 camera before/during 与左右 tactile before/during。

## PPCT 从属关系

PPCT 主链路在 `SGSNet` 中：

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

相关文件职责：

- `PPCT_utils.py`: 点云 encoder、grouping、point sequence dropout/shuffle 等 PPCT 输入处理模块。
- `Transformer_utils.py`: attention、cross-attention、graph/deformable attention block 等基础 transformer 组件。
- `MultiResolutionCrossAttn.py`: 多分辨率 cross-attention fusion，负责 tactile query 和 visual memory 的融合、下采样和多尺度聚合。
- `GraspStability_PPCT.py`: 把以上模块组织成 SGSNet，并提供 loss manager 和 ablation forward 分支。

## 训练时如何被调用

3D grasp stability 训练在 `tools/runner_3dvtg.py` 中构建两个模型：

```python
shape_model = builder.model_builder(config.shape_model)
grasp_model = builder.model_builder(config.grasp_model)
```

随后：

1. `shape_model` 加载 `ckpts/ap_ps55.pth`。
2. `shape_model.parameters()` 被设置为 `requires_grad=False`。
3. `sc_input` 进入 `shape_model(sc_input, return_latent=True)`。
4. 返回的 `vis_f, vis_coor` 与 `gs_input` 一起送入 `grasp_model`。
5. 当前 `use_vis_encoder=True` 时，SGSNet 会用自身 `PointEncoder` 重新编码 `vis_coor`；`use_vis_encoder=False` 时才直接使用传入的 `vis_f`。
6. grasp model 输出 `[B, 1]` 或 `[B]` logit。
7. runner 将 logit squeeze 成 `[B]`，并交给 `LossManager` 计算 BCE-with-logits 风格 loss。

当 `ablation.without_shape_completion=True` 时，runner 不使用 AdaPoinTr latent，而是调用：

```python
grasp_model(None, sc_input, gs_input, obj_ids)
```

SGSNet 内部会走 `forward_ablation_without_shape_completion()`，用自身 visual encoder 处理 `sc_input`。

## 输出与指标约定

Grasp stability 当前是二分类单 logit 任务：

- 训练目标来自 metadata 的 `isPositive`。
- loss 使用 `binary_cross_entropy_with_logits`。
- 验证时对 logit 做 sigmoid 后计算 Accuracy、Precision、Recall、F1 等指标。

Shape completion 使用 AdaPoinTr 输出 coarse/dense points，并在验证中计算 F-Score、Chamfer Distance 和可选 EMD。

## 定位建议

- 想改网络结构：先看对应 YAML 的 `NAME`，再进入注册模型类。
- 想改 PPCT fusion：优先看 `models/MultiResolutionCrossAttn.py`。
- 想改点云 encoder 或 point sequence dropout：看 `models/PPCT_utils.py`。
- 想改 loss：grasp stability 在 `GraspStability_PPCT.py` 的 `LossManager`，shape completion 在 `AdaPoinTr.py`。
