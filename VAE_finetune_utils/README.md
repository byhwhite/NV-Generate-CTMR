# VAE Finetune Utils

## 训练入口

在仓库根目录运行：

```bash
python train_vae_finetune.py \
  --dataset_spec_json VAE_finetune_utils/dataset_spec_example.json \
  --run_name my_vae_finetune \
  --val_every_n_steps 200 \
  --val_vis_num_images 4
```

若环境里 `PerceptualLoss` 触发 torchvision 下载/告警，可增加：

```bash
  --disable_perceptual_loss
```

## 数据集配置

`dataset_spec_json` 是一个列表，每个元素必须包含：

- `name`: 数据集名。
- `train_json`: 训练 json 路径。
- `val_json`: 验证 json 路径。
- `samples_per_epoch`: 每个 epoch 从该数据集采样的样本数。

train/val json 内可为：

- 字符串数组：`["/abs/a.npy", "/abs/b.npy"]`
- 或对象数组（支持字段：`image/path/npy/file`）。

每个 `.npy` 需为 `(1, D, H, W)`；代码会随机/中心裁剪到 `patch_size`（默认 64³）并进行强度归一化。

## wandb 记录

- 训练迭代 loss（recon/kl/perceptual/g_adv/d_loss）
- 验证 loss 与 scale factor
- 验证可视化（原图和重建图的 3 个正交切片拼图）
