# cfgs 配置快速指引

本目录保存 SGA-GSN 的实验配置。使用者通常只需要从 `cfgs/3D_VTG_models/` 选择一个模型 YAML，再通过 `main.py --config ...` 启动训练。

## 配置如何被读取

入口链路是：

```text
main.py -> utils/parser.py -> utils/config.py
```

`utils/parser.py` 解析命令行参数，并根据配置文件路径和 `--exp_name` 创建实验目录：

```text
experiments/<config_stem>/<config_parent>/<exp_name>/
```

`utils/config.py` 调用 `cfg_from_yaml_file()` 读取 YAML，并用 `merge_new_config()` 递归合并配置。特殊字段 `_base_` 会被当作另一个 YAML 文件读取，再合并到当前位置。

例如 `VTG_PPCT.yaml` 中：

```yaml
dataset : {
  train : { _base_: cfgs/dataset_configs/VTG3D.yaml,
            others: {subset: 'train'}},
  val : { _base_: cfgs/dataset_configs/VTG3D.yaml,
            others: {subset: 'test'}},
  test : { _base_: cfgs/dataset_configs/VTG3D.yaml,
            others: {subset: 'test'}}}
```

运行时，`VTG3D.yaml` 会成为 `config.dataset.train._base_` 等节点，`others` 中的字段会作为默认参数传给 dataset 构造器。因此 `VTG3D.__init__()` 最终可以直接读取 `DATA_PATH`、`GT_POINTS`、`VIS_POINTS`、`subset` 等字段。

## 以 VTG_PPCT.yaml 为例

`VTG_PPCT.yaml` 是 3D VTG grasp stability 的主配置。主要字段如下：

- `optimizer`: 当前使用 `AdamW`，学习率和 weight decay 在 `kwargs` 中设置。
- `scheduler`: 当前使用 `LambdaLR`，通过 `decay_step`、`lr_decay`、`lowest_decay` 控制学习率衰减。
- `bnmscheduler`: 对 BatchNorm momentum 做同步衰减。
- `dataset`: 为 train/val/test 分别挂载 `cfgs/dataset_configs/VTG3D.yaml`，并通过 `others.subset` 选择数据划分。
- `shape_model`: 配置 `AdaPoinTr`，在 grasp stability 训练中加载 `ckpts/ap_ps55.pth` 后冻结，用于生成 shape latent feature 和 shape coordinates。
- `grasp_model`: 配置 `SGSNet`，也就是 PPCT grasp stability 网络。
- `train_data_ratio`: 从完整训练集抽取的训练比例，代码中通过 `get_shuffled_subset()` 生效。
- `total_bs`: 单卡训练时作为 batch size；分布式训练时会按 `world_size` 均分到 `config.dataset.train.others.bs`。
- `step_per_update`: 梯度累积步数，当前 runner 在达到该步数后执行一次 optimizer step。
- `max_epoch`: 最大训练 epoch。
- `consider_metric`: 用于保存 best checkpoint 的主指标，grasp stability 默认是 `AvgAcc`。

## PPCT 模型配置

`grasp_model.NAME: SGSNet` 对应 `models/GraspStability_PPCT.py` 中注册的模型。几个关键子块：

- `use_vis_encoder`: 为 `True` 时，SGSNet 会用自己的 `PointEncoder` 重新编码 AdaPoinTr 输出的视觉坐标。
- `use_vis_seq_shuf_drop` / `use_tac_seq_shuf_drop`: 训练时对 visual/contact point sequence 做 dropout 和 shuffle。
- `tactile_encoder`: tactile 输入是 `xyz + contact/gel`，所以 `input_dim` 为 4。
- `visual_encoder`: visual 输入通常是 `xyz`，所以 `input_dim` 为 3。
- `feature_fusion`: 配置 `MultiResCrossAttnEntry`，包括每层维度、head 数、downsample 点数、self/cross attention block 风格等。
- `pred_head_drop_rate` / `pred_head_mid_feature`: 控制最终二分类 logit 预测头。

实际网络流向是：

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

当前 `VTG_PPCT.yaml` 中 `use_vis_encoder=True`，所以 `vis_coor` 会在 SGSNet 内再次经过 `visual_encoder`/`PointEncoder` 得到新的 visual tokens；只有关闭 `use_vis_encoder` 时，AdaPoinTr 返回的 `vis_f` 才会直接进入 fusion。

## Ablation 设置

`VTG_PPCT.yaml` 顶层 `ablation` 会在 `tools/runner_3dvtg.py` 中被解释：

- `direct_concat`: model-phase ablation。写入 `config.grasp_model.direct_concat`，SGSNet 使用 direct concat 分支替代 cross-attention fusion。
- `full_shape_input`: runner-phase ablation。训练/验证时用 `sc_gt` 采样替代 `sc_input`，模拟完整形状输入。
- `without_spatial_align`: dataset-phase ablation。写入 train/val/test 的 dataset config，VTG3D 不再把 visual/tactile/gt 对齐到世界坐标。
- `without_shape_completion`: model + runner ablation。runner 不调用 AdaPoinTr latent，SGSNet 用自身 visual encoder 处理 shape input。

这些 ablation 都是布尔值。修改后建议使用新的 `--exp_name`，避免和已有 checkpoint/config 混在同一个实验目录。

## 常用配置文件

- `cfgs/3D_VTG_models/VTG_PPCT.yaml`: 3D PPCT/SGSNet 主实验。
- `cfgs/3D_VTG_models/VTG_DGCNN_14M.yaml`: 3D DGCNN baseline，较小配置。
- `cfgs/3D_VTG_models/VTG_DGCNN_45M.yaml`: 3D DGCNN baseline，较大配置。
- `cfgs/3D_VTG_models/VTG_CNNMCA.yaml`: 2D CNNMCA baseline。
- `cfgs/3D_VTG_models/VTG_CNN.yaml`: 2D CNN baseline。
- `cfgs/dataset_configs/VTG3D.yaml`: 3D VTG 数据路径、点数和噪声配置。
- `cfgs/dataset_configs/VTG2D.yaml` / `VTG2D_TacCat.yaml`: 2D VTG 图像路径和 tactile 拼接开关。
