# scripts 训练与续训快速指引

本目录提供训练脚本示例。当前公开树中的 `scripts/*.sh` 都调用 `scripts/train.sh`，但该文件未包含在仓库中。因此推荐直接使用 `python main.py ...` 命令；已有 shell 文件可作为参数模板参考。

## 推荐入口

3D PPCT grasp stability 训练：

```bash
python main.py \
  --_3dvtg \
  --vtg3d_grasp_stability \
  --config ./cfgs/3D_VTG_models/VTG_PPCT.yaml \
  --exp_name sanity_check \
  --num_workers 6
```

这条命令会使用：

- dataset config: `cfgs/dataset_configs/VTG3D.yaml`
- shape model: `AdaPoinTr`
- grasp model: `SGSNet`
- frozen shape checkpoint: `ckpts/ap_ps55.pth`

启动前需要准备：

```text
/SGA-GSN/data/3DA-VTG
/SGA-GSN/data/graspnet-vhacd
/SGA-GSN/ckpts/ap_ps55.pth
```

输出目录由 `utils/parser.py` 决定：

```text
experiments/VTG_PPCT/3D_VTG_models/<exp_name>/
```

其中会保存 copied `config.yaml`、日志、`ckpt-last.pth`、`ckpt-best.pth` 和最后几个 epoch 的 checkpoint。

## 续训流程

续训命令示例：

```bash
python main.py \
  --_3dvtg \
  --vtg3d_grasp_stability \
  --config ./cfgs/3D_VTG_models/VTG_PPCT.yaml \
  --exp_name sanity_check \
  --num_workers 6 \
  --resume
```

`--resume` 的行为：

- `utils/config.py` 会改为读取当前 experiment 目录下的 `config.yaml`。
- `tools/builder.resume_model()` 从 `ckpt-last.pth` 恢复模型参数、epoch 和 best metrics。
- `tools/builder.resume_optimizer()` 从同一个 checkpoint 恢复 optimizer。
- `--resume` 不能和 `--start_ckpts` 同时使用。
- `--test` 也不能和 `--resume` 同时使用。

续训时要保持 `--config` 和 `--exp_name` 指向同一个旧实验目录，否则代码会在新的 experiment path 下寻找 `ckpt-last.pth`。

## 数据流总览

3D PPCT 训练的数据和模型流向：

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

验证阶段会分别构造 seen/unseen 风格的 dataloader 子集，并保存：

- `ckpt-best.pth`: 当前主指标优于历史 best 时保存。
- `ckpt-last.pth`: 每个 epoch 后保存，用于续训。
- `ckpt-epoch-XXX.pth`: 最后几个 epoch 额外保存。

## 现有 shell 示例

当前文件：

- `train_vtg3d_grasp.sh`: 3D PPCT 训练参数模板。
- `train_resume_vtg3d_grasp.sh`: 3D PPCT 续训参数模板。
- `train_vtg2d_grasp.sh`: 2D CNNMCA 参数模板。

这些文件目前形如：

```bash
bash ./scripts/train.sh 0 \
  --_3dvtg \
  --config ./cfgs/3D_VTG_models/VTG_PPCT.yaml \
  --exp_name sanity_check \
  --num_workers 6 \
  --vtg3d_grasp_stability
```

由于公开树没有 `scripts/train.sh`，如果不补充自己的 launcher，请改用等价的 `python main.py ...` 命令。

## 2D baseline 注意事项

2D 入口是：

```bash
python main.py \
  --_2dvtg \
  --config ./cfgs/3D_VTG_models/VTG_CNNMCA.yaml \
  --exp_name test \
  --num_workers 6
```

但当前 `tools/runner_2dvtg.py` 在 `testModel2D(grasp_model, config)` 后有一个提前 `return`，所以公开树中该入口默认只会做 FLOPs/参数/延迟 profile，不会进入完整训练循环。若要训练 2D baseline，需要先移除或绕过这个调试返回。

## 常见参数

- `--_3dvtg` / `--_2dvtg`: 二选一，选择 3D 或 2D VTG 入口。
- `--vtg3d_grasp_stability`: 3D grasp stability 任务；使用 `--_3dvtg` 时必须和 shape completion 任务二选一。
- `--config`: YAML 配置路径。
- `--exp_name`: 实验名，参与决定输出目录。
- `--num_workers`: DataLoader worker 数。
- `--start_ckpts`: 从指定 checkpoint 初始化模型，但不恢复 optimizer/epoch。
- `--resume`: 从当前实验目录的 `ckpt-last.pth` 完整续训。
- `--test --ckpts <path>`: 测试模式需要显式提供 checkpoint；当前公开树的 3D grasp stability test 分支尚未完整实现。
