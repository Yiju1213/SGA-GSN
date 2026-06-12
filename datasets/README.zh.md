# datasets 数据读取快速指引

本目录实现 VTG 2D/3D 数据集。它的核心职责是把磁盘上的 CSV、metadata、RGB/depth/segmentation 文件转换成 runner 可以直接送入模型的 PyTorch tensor。

## 注册与构建

数据集使用轻量 registry：

```text
datasets/__init__.py
  -> import datasets.VTG3D_Dataset
  -> import datasets.VTG2D_Dataset
datasets/build.py
  -> DATASETS = Registry('dataset')
```

`VTG3D` 和 `VTG2D` 类通过 `@DATASETS.register_module()` 注册。训练时，`tools/builder.dataset_builder(args, config.dataset.train)` 会调用：

```python
build_dataset_from_cfg(config._base_, config.others)
```

其中 `config._base_` 来自 `cfgs/dataset_configs/*.yaml`，`config.others` 提供 `subset`、`bs` 以及 ablation 中追加的字段。构建出的 dataset 再被包装成 `torch.utils.data.DataLoader`。

## VTG3D 数据流

`VTG3D_Dataset.py` 面向 3D grasp stability 和 shape completion。初始化阶段会读取：

- `data/3DA-VTG/<subset>.csv`: 样本列表，通常是 `(obj_id, grasp_id)`。
- `data/3DA-VTG/<subset>-ids.txt`: 当前 split 的 object id 列表。
- `data/3DA-VTG/<obj_id>/_metadata.json`: 每个 grasp 的相机矩阵、位姿和 `isPositive` 标签。

单个样本的读取路径：

1. 读取 visual depth 和 visual segmentation，通过 `vtg3d_utils.getVisualPointCloud()` 得到 object visual point cloud。
2. 读取左右 tactile depth，通过 `vtg3d_utils.getTactilePointCloud()` 得到 tactile point cloud 和 gel mask。
3. 对 visual/tactile 点云采样到配置点数：`VIS_POINTS`、`TAC_POINTS`。
4. 默认用相机位姿和 grasp/object pose 对齐到 world coordinate；`without_spatial_align=True` 时跳过该对齐。
5. 从 tactile 点云中用 gel mask 提取 contact region。
6. 从 `COMPLETE_MESH_PATH` 指向的 obj mesh 采样 `GT_POINTS` 个完整形状点。
7. 用 visual 和 contact 点计算 zero-mean，并应用到 visual、tactile contact、完整形状。
8. 为 shape completion 输入构造固定大小 `SC_INPUT_POINTS`，并在 train split 加入尺度扰动和高斯噪声。
9. 为 grasp stability 输入把左右 tactile 点拼接，并给每个 tactile 点附加 gel/contact 维度，形成 `xyzc`。

非 debug 模式下，`VTG3D.__getitem__()` 返回：

```python
(
    (sc_input, gt_pc),      # shape_completion
    (tac_xyzc, label),      # grasp_stability
    zero_mean,
    sample,                # (obj_id, grasp_id)
)
```

常见 tensor 形态：

- `sc_input`: `[SC_INPUT_POINTS, 3]`，来自 visual/contact 的 shape completion 输入。
- `gt_pc`: `[GT_POINTS, 3]`，完整目标点云。
- `tac_xyzc`: `[2 * TAC_POINTS, 4]`，左右 tactile 拼接，最后一维是 gel/contact 标记。
- `label`: 标量 float，来自 metadata 的 `isPositive`。

在 `tools/runner_3dvtg.py` 中，batch 会被解包为：

```python
sc_data, gs_data, zero_mean, sample = item
sc_input, sc_gt = sc_data[0].cuda(), sc_data[1].cuda()
gs_input, result_gs = gs_data[0].cuda(), gs_data[1].cuda()
```

随后 `sc_input` 进入 AdaPoinTr，`gs_input` 进入 SGSNet/DGCNN grasp model。

## VTG2D 数据流

`VTG2D_Dataset.py` 面向 2D CNN baseline。初始化阶段读取同样的 CSV、object id、metadata，并缓存一张 tactile background：

```text
BACKGROUND_PATH: data/3DA-VTG/bg_sim.jpg
```

单个样本会读取：

- visual RGB: `VIS_RGB_PATH`
- visual segmentation: `VIS_SEG_PATH`
- left/right tactile RGB: `TAC_RGB_PATH` 中的 `<grasp_id>_l` 和 `<grasp_id>_r`
- label: metadata 中的 `isPositive`

训练 split 使用同步几何增强：

- visual RGB 和 visual seg 使用同一组 resize/crop/flip/rotation 参数。
- tactile background 和 tactile grasp 使用同一组几何参数。
- RGB/tactile 各自再应用轻量 color jitter 和 blur。

如果 `ConcatTwoTactile=False`，返回字段包括：

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

如果 `ConcatTwoTactile=True`，左右 tactile 会先水平拼接，再统一增强，返回：

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

图像 tensor 默认是 `[3, 224, 224]`，并使用 ImageNet mean/std 归一化。

## 和 runner 的对应关系

- `GraspStability_CNNMCA` 使用 `visual_rgb` 和 `tactile_rgb_concat`。
- `GraspStability_CNN` 使用 `visual_rgb`、左右 tactile grasp 和左右 background。
- `VTG3D` 的 `shape_completion` 分支服务 AdaPoinTr；`grasp_stability` 分支服务 SGSNet/DGCNN。

调试数据形态时，优先从 dataset 返回字段一路跟到 `tools/runner_3dvtg.py` 或 `tools/runner_2dvtg.py` 的 batch unpack 位置。

