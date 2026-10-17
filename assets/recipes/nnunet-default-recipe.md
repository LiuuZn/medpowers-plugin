# nnunet-default-recipe — nnU-Net v2 默认训练配方（SGD+poly / DC_and_CE+深监督 / AMP）

## 论文支撑（统一四问）

- **哪篇论文用了它**：Isensee, Jaeger, Kohl, Petersen & Maier-Hein（德国癌症研究中心 DKFZ），《nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation》，Nature Methods 2021，https://www.nature.com/articles/s41592-020-01008-z （配方论述在官方免费补充材料 MOESM1 Note 2 "Training Scheme"）。ResEnc 变体沿用同配方零改动：Isensee 等《nnU-Net Revisited》，MICCAI 2024，https://arxiv.org/abs/2404.09556 。所读版本：补充材料 PDF + 本机 nnunetv2 2.7.x 源码（`/data/lx/thymic_seg/.venv`，2026-10-15 逐行考证；版本内无 `__version__` 属性，行号以该安装为准）。
- **目的**：给"自配置"框架一套**跨任务不调参**的固定训练配方——框架的全部自由度花在预处理与网络规划上，训练循环本身保持简单且久经考验。
- **效果**：作为 nnU-Net 默认配置，23 个公开数据集上超过大多数人工特化方案（摘要）；ResEnc 论文在同配方下把 KiTS Dice 从 86.04 推到 88.17(L)/88.67(XL)，并指出多数"超越 nnU-Net"的工作败在没用好这条基线配方。
- **结论**：医学 3D 分割的事实标准配方；框架外复现 nnU-Net 基线时**每一项都应对表抄写**，任何"改进"先在原配方上站平再谈。

## 配方构成（逐项源码考证，nnUNetTrainer 默认值）

| 元素 | 精确值 | 源码出处（nnunetv2 2.7.x） |
|---|---|---|
| 优化器 | `SGD(lr=0.01, momentum=0.99, nesterov=True, weight_decay=3e-5)` | `training/nnUNetTrainer/nnUNetTrainer.py:153-154`（`initial_lr=1e-2`、`weight_decay=3e-5`）+ `configure_optimizers`（momentum/nesterov 硬编码） |
| LR 日程 | poly：`lr = lr0 · (1 − t/T)^0.9`，`T = num_epochs = 1000`，**按 epoch 步进**（每 epoch 结束 `scheduler.step()` 一次，非按 iteration） | `training/lr_scheduler/polylr.py`（`exponent: float = 0.9` 默认） |
| 训练时长 | `1000 epochs × 250 iterations`（验证 50 iterations/epoch） | `nnUNetTrainer.py:157-159` |
| 损失 | `DC_and_CE_loss(weight_ce=1, weight_dice=1, batch_dice=<plans>, smooth=1e-5, do_bg=False, dice_class=MemoryEfficientSoftDiceLoss)`；区域式任务改 `DC_and_BCE_loss(do_bg=True)` | `nnUNetTrainer.py:444-446`；损失数学见 [../objectives/dice-ce-loss.md](../objectives/dice-ce-loss.md) |
| 深监督 | scales `= 1/cumprod(pool_op_kernel_sizes)[:-1]`，而 **`pool_op_kernel_sizes ≡ architecture.arch_kwargs.strides`（含首级 [1,1,1]）**；权重 `[1/2^i for i in range(n)]`，**末位 = 0**（DDP 下 = 1e-6 防 unused-parameters 崩溃），再整体**归一化到和为 1**；`DeepSupervisionWrapper` 加权求和（零权重输出跳过不计算） | `nnUNetTrainer.py:383-387, 455-468`；恒等关系 `utilities/plans_handling/plans_handler.py:155-156`；wrapper `training/loss/deep_supervision.py` |
| 深监督 GT | 各分辨率 GT 由全分辨率 GT **最近邻** resize 下采样（`order=0`），逐轴比例可各向不同 | `training/data_augmentation/custom_transforms/deep_supervision_donwsampling.py:18`（`order: int = 0`） |
| 前景过采样 | `oversample_foreground_percent = 0.33`（每 batch 1/3 patch 中心强制落在前景） | `nnUNetTrainer.py:155`；机制见 [../augmentation/foreground-oversampling.md](../augmentation/foreground-oversampling.md) |
| AMP | `autocast(device)` 前向 + `GradScaler`；反传序列 **scale → backward → `unscale_` → `clip_grad_norm_(12)` → scaler.step → update**——**梯度裁剪 max_norm=12 是常被漏抄的隐藏默认** | `nnUNetTrainer.py:1033-1042`；CPU 设备上 autocast/GradScaler 自动不启用 |
| 权重 EMA | **无**（默认 trainer 不做权重指数滑动平均；日志 "New best EMA pseudo Dice" 的 EMA 是**监控指标的指数平滑**，不是权重 EMA，两回事） | `nnUNetTrainer.py` 训练循环 |
| checkpoint | 每 50 epoch 存 snapshot + `checkpoint_best.pth`（按上述 EMA pseudo dice 选优） | `nnUNetTrainer.py` `on_epoch_end` |

组装顺序（伪代码级）：

```python
optimizer = SGD(params, lr=0.01, momentum=0.99, nesterov=True, weight_decay=3e-5)
scheduler = PolyLRScheduler(optimizer, 0.01, num_epochs=1000)   # exponent 0.9
loss = DeepSupervisionWrapper(DC_and_CE_loss(...), weights)     # weights 见上表
for epoch in range(1000):
    for it in range(250):
        batch = sample(patches, oversample_fg=0.33)             # seg 近邻下采样成 DS 列表
        optimizer.zero_grad()
        with autocast(device):
            preds = net(x)                                      # 多分辨率输出列表
            l = loss(preds, seg_list)
        scaler.scale(l).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(net.parameters(), 12)
        scaler.step(optimizer); scaler.update()
    scheduler.step()                                            # 按 epoch，不按 iteration
```

## 量级（实例化实测，thymic_seg Dataset701_Thymus，238 train / 42 val，RTX 4090 24GB 单卡）

- 数据侧规划（M plans，CT）：patch `[96,160,160]`（z-y-x）、batch 2、spacing `[1.25, 0.7246, 0.7246]`、CTNormalization、图像三阶/标注一阶重采样、`batch_dice=True`（plans `nnUNetResEncUNetMPlans.json`，2026-10-15 从训练日志配置 dump 核对）。
- ResEnc-M（EXP-012，[../models/nnunet-resenc.md](../models/nnunet-resenc.md)）：101.93M 参数、~9.5 GB 显存、45.7 s/epoch → 1000 epoch ≈ 12.7 h。
- PlainConv 对照（EXP-011）：~7.8 GB、33.4 s/epoch → 1000 epoch ≈ 9.3 h。

## 组装关系

- 网络：[../models/nnunet-resenc.md](../models/nnunet-resenc.md)（ResEnc 预设；PlainConv 即经典 nnU-Net U-Net）
- 损失：[../objectives/dice-ce-loss.md](../objectives/dice-ce-loss.md)（DC_and_CE 数学与口径）
- 采样：[../augmentation/foreground-oversampling.md](../augmentation/foreground-oversampling.md)
- 增强：[../augmentation/nnunet-default-augmentation.md](../augmentation/nnunet-default-augmentation.md)（在线增广流水，与本配方正交叠加）

## 坑

- **poly 按 epoch 步进**：把 `scheduler.step()` 放进 iteration 循环会让 LR 在 250 倍速下衰减，前几百 epoch 就趋零——复现失败的最常见单点。
- **隐藏默认 clip 12**：`clip_grad_norm_(12)` 藏在 GradScaler 的 `unscale_` 之后，教程与多数复现都漏抄；AMP + 大 LR + 动量 0.99 组合下它承担稳定器角色。
- **负损失值正常**：DC 项是 `-DC` 无 +1 常数（见 dice-ce-loss 卡"坑"节），训练日志 train_loss 约 -0.7 是收敛良好，不是 bug。
- **"EMA pseudo dice" 不是权重 EMA 也不是全图指标**：它是 patch 级监测量（过采样 patch + 深监督加权）的指数平滑，与逐例全图验证 Dice 不可比——选 `checkpoint_best` 用它没问题，报告结论必须走独立评估。
- **深监督末位权重 = 0 但头还在**：最低分辨率输出的监督权重为 0（权重归一化分母因此不含它），但网络仍输出该分辨率、wrapper 跳过其损失计算；复刻时可保留头（对齐参数量与显存行为）。
- **weight_decay=3e-5 与 lr 耦合**：改动 lr 时保持 wd 不动（配方内 lr0×wd 是共同体；ResEnc 论文也未动）。
- **oversample 0.33 是采样层机制**：作用于 patch 中心采样，不是损失加权；与 Dice 损失同向但层次不同。

## 实验反链

- thymic_seg `EXP-011`（PlainConv 默认配置 fold 0，本配方 + 默认 plans 对照）
- thymic_seg `EXP-012`（ResEnc-M plans fold 0，本配方零改动换骨干；ep~100 快照统一口径 ccLargest 0.7060 已越既有锚点 0.7015）
