## Pre-training MAE

To pre-train ViT-Large (recommended default) with **multi-node distributed training**, run the following on 8 nodes with 8 GPUs each:
```
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
```
- Here the effective batch size is 64 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 4096. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 8 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here we use `--norm_pix_loss` as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off `--norm_pix_loss`.
- The exact same hyper-parameters and configs (initialization, augmentation, etc.) are used as our TF/TPU implementation. In our sanity checks, this PT/GPU re-implementation can reproduce the TF/TPU results within reasonable random variation. We get 85.5% [fine-tuning](FINETUNE.md) accuracy by pre-training ViT-Large for 800 epochs (85.4% in paper Table 1d with TF/TPU).
- Training time is ~42h in 64 V100 GPUs (800 epochs).

To train ViT-Base or ViT-Huge, set `--model mae_vit_base_patch16` or `--model mae_vit_huge_patch14`.

- 此处有效批次大小为 64（每个 GPU 的 `batch_size`）* 8（`nodes`）* 8（每个节点的 GPU 数量）= 4096。如果内存或 GPU 数量有限，请使用 `--accum_iter` 来维持有效批次大小，即 `batch_size`（每个 GPU 的）* `nodes` * 8（每个节点的 GPU 数量）* `accum_iter`。
- `blr` 是基础学习率。实际 `lr` 由 [线性缩放规则](https://arxiv.org/abs/1706.02677) 计算得出：`lr` = `blr` * 有效批次大小 / 256。
- 此处我们使用 `--norm_pix_loss` 作为目标参数，以实现更好的表征学习。要训练基准模型（例如，用于可视化），请使用基于像素的构建方法并关闭 `--norm_pix_loss`。
- 我们的 TF/TPU 实现使用了完全相同的超参数和配置（初始化、数据增强等）。在我们的完整性检查中，此 PT/GPU 重新实现可以在合理的随机变化范围内复现 TF/TPU 的结果。通过对 ViT-Large 进行 800 个 epoch 的预训练，我们获得了 85.5% 的 [微调](FINETUNE.md) 准确率（论文表 1d 中使用 TF/TPU 的准确率为 85.4%）。
- 在 64 块 V100 GPU（800 个 epoch）上训练时间约为 42 小时。

要训练 ViT-Base 或 ViT-Huge，请设置 `--model mae_vit_base_patch16` 或 `--model mae_vit_huge_patch14`。