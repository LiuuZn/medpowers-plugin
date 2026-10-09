# runtime — 运行时状态与文件系统资源调度

> 本文件随插件分发，是任务仓库 `<工作区>/runtime/` 的协议文档模板，由技能 `skills/init-env` 初始化时拷入；各任务仓库以自己的副本为准。

给所有智能体提供"环境是什么样、哪些任务正在训练"的上下文，避免撞卡。设计原则：**声明式、尽力而为**——文件只是参考，实时占用以 `nvidia-smi` 为准。

## 布局

- `env.json`：**初始化结果**（由技能 `skills/init-env` 写入）：任务画像（目标、影像模态、2D/2.5D/3D 维度、任务类型——**用户必答的问卷结果**，未答复不得完成初始化）、GPU 静态清单（index/name/显存总量）、每张卡是否归本训练目录可用（用户答复或默认全部可用）、Python 训练环境状态（PyTorch + CUDA 张量是否验证通过、解释器位置）。由任务仓库自行决定是否纳入其 git 管理，是训练环境与任务画像的单一事实源。
- `data.json`：**数据管理状态**（由技能 `skills/data-mgmt` 写入）：`data_root`（数据统一根目录，默认任务仓库工作区下的 `<工作区>/data`，用户可改，答复原文随记）+ 全部数据集的登记表（路径/任务类型/维度/recipes）。任务仓库通过此表引用数据，**不复制数据**。由任务仓库自行决定是否纳入其 git 管理。
- `gpu/`：预留。动态 GPU 信息（利用率/温度/进程等）不持久化，需要时实时 `nvidia-smi`。
- `versions.json`：**版本指针**。具名版本（baseline/sota/current 等）→ EXP 编号、任务仓库、checkpoint 路径、指标、配置文档路径；「回滚到基线/切到 sota」即读此文件。
- `training/`：**训练信息**。每个进行中的训练任务一个 lease JSON，文件名用任务 slug（如 `bc3seg-abdomen-unet3.json`）。

## 初始化

环境首次使用时运行技能 `skills/init-env`（**先问任务画像，用户必须全部答复后才会继续**；随后询问可用卡、检查并按需 uv 初始化 PyTorch + CUDA 环境），结果落在 `env.json`。重新初始化前注意其中用户答复（任务画像、可用卡）不得静默丢失（见 SKILL.md 约束）。

## lease 协议（三步）

1. **训练前**：读 `env.json` 确认目标卡 `available: true`；读 `training/` 下全部 lease；并用 `nvidia-smi` 实测。三者都满足才启动；任一不满足，换卡、等待或排队。
2. **启动时**：写自己的 lease 文件（原子写：先写 `<slug>.json.tmp` 再 `mv` 为 `<slug>.json`，防止别 agent 读到半个文件）。**同时**建当轮实验记录 `EXP-*`（`status: running`；记录位置由任务仓库自选（默认 `<工作区>/experiments/`），模板见插件 `experiments/TEMPLATE.md`）。
3. **结束时**：训练结束或中断，立即删除自己的 lease；补全实验记录，`status` 改 `done` / `interrupted`。

## lease 字段（JSON）

```json
{
  "task": "任务一句话描述",
  "repo": "实际训练发生的仓库路径",
  "experiment": "对应的 EXP-* 编号（若有）",
  "agent": "哪个智能体/会话在跑",
  "pid": 12345,
  "gpus": [0],
  "gpu_mem_mb": 21000,
  "started": "2026-10-04T16:00:00+08:00",
  "eta_hours": 8,
  "log": "训练日志/曲线在任务仓库中的路径",
  "notes": "其他智能体需要知道的事（如：中途回滚不受影响）"
}
```

## stale lease 清理

文件还在但训练早已不在（机器重启、进程被杀）：核对 `pid` 与 `started` 时间，确认进程不存在后可直接删除该 lease，无需询问。清理动作在当轮实验记录（或下一条记录的 notes）里提一句。

## 边界

- 本机制不排队、不加锁、不抢占：只是共享上下文。真正互斥靠"第 1 步三确认"（env.json 可用性 + lease + nvidia-smi）。
- 只登记**训练与长时间占用 GPU 的任务**；几分钟的推理/评测不必登记。
