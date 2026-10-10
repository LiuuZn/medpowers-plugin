---
name: data-mgmt
description: 数据管理：统一规定训练数据的位置与格式，提供建立、登记、校验、预览、空间报告动作，外加 z 索引标注导入（import-planes）、软链登记源数据（scaffold --link-source）与标注审计工具（label_audit.py）。用户新建/整理/导入训练数据、询问"数据放哪/该存什么格式"、检查数据集完整性、查数据空间占用、导入既有索引式标注、对标注做一致性/上限诊断时使用。数据统一放 runtime/data.json 中 data_root 指定的根目录（默认任务仓库工作区下 <工作区>/data/<dataset>/，用户可改）；3D 默认压缩 NRRD、2D 无损 PNG；按任务类型（3D/2D 分割、2D/3D 分类、3D/2D 包围框检测、3D 关键平面检出）规定目录与标注格式；2.5D 按 3D 存；3D 张量轴序固定 z-y-x（z 轴足→头为第 0 维）；配准暂缓。
---

# data-mgmt — 数据管理

管**容器**不管**内容**：本技能规定数据放哪、以什么格式放、如何登记与校验；具体预处理方法（归一化/裁剪/重采样等）属 `assets/preprocessing` 方法卡，通过 recipe 名互链。

三个优先级，冲突时从上往下取舍：**训练高效 > 可视化方便 > 空间占用可控**。

## 数据放哪（默认值可改）

- 默认根目录 **`<工作区>/data/`**（工作区 = 从执行目录向上最近的含 `.git` 祖先，与脚本 `resolve_runtime` 的解析一致；`MTSKILLS_RUNTIME` 环境变量可显式指定 runtime 目录），一个数据集一个子目录（如 `<工作区>/data/lung_seg/`）。
- 事实源是任务仓库工作区的 `runtime/data.json`：
  - `data_root`：根目录 + 用户答复记录（`source: user/default`）。本技能首次运行发现未设定时，用 AskUserQuestion 问用户"数据统一放哪？"（说明默认 `<工作区>/data`），答复后 `set-root` 写入并记答复原文；**不得静默假设或丢失用户答复**。
  - `datasets`：全局登记表。任务仓库**永不复制数据**，凭这里的路径引用（单一正本原则）。
- 数据本体在 git 仓库之外，天然不进 git。

## 三层存储架构（每个数据集固定骨架）

```
<data_root>/<dataset>/
├── dataset.json          # 权威清单（字段见下）
├── raw/                  # ① 原始层：怎么来怎么存（DICOM/厂商格式/原始 nrrd），只读、永不修改
├── processed/<recipe>/   # ② 训练就绪层：压缩正本（.nrrd gzip / PNG）；recipe=预处理配方名
│   └── cache/            # ③ 缓存层：正本解包出的 .npy，可随时整目录删除重生成
├── splits/               # train/val/test 各一个 JSON（case_id 字符串数组），不复制文件
└── preview/              # 标注叠加 PNG（降采样，单张 ≤200KB）
```

- 三个优先级的落点：训练高效靠 **cache 层**（DataLoader 直读 npy，零解码、可 mmap）；可视化方便靠格式本身（NRRD/PNG/COCO 通用工具直接开）+ preview（不开重型工具就能翻查）；空间可控靠"**压缩的才是正本**"（cache 是易失衍生品，`du` 报告可回收量）。
- 目录只按数据组织，划分全靠 splits JSON——换划分/换 fold 不挪文件。
- **cache 用 per-case `.npy` 不用 HDF5**（决策记录）：本场景的访问模式是多 worker 随机采样（3D patch / 逐例），npy 可 mmap 零拷贝、无解码、各 worker 独立句柄、单文件损坏不伤全库；HDF5 读压缩 chunk 必须解压、多进程并发读需 swmr 或每 worker 重开句柄、单容器一损俱损，只在顺序整读超大单体时有优势。nnU-Net v2 的最终形态同样是 per-case 压缩文件 + npy 缓存。顺序流式超大规模（>10 万样本）再考虑分片容器（webdataset 等），届时 dataset.json 记 `storage: "sharded"`。

## 格式总则（所有任务）

1. **3D 默认压缩 NRRD**：`.nrrd` 单文件，写入时开 gzip（SimpleITK：`WriteImage(img, path, True)`）。头是纯文本、人类可读（文本编辑器即可查 spacing/origin/direction），ITK-SNAP/3D Slicer 直接打开；压缩效果与 NIfTI 相当。2D 默认无损 PNG（16-bit 灰度可用）。
2. **三维张量轴序固定 z-y-x**：z 轴（足→头方向）必须放第 0 维，三个维度按 z-y-x 排列，即加载为 numpy 数组后 shape 一律 `(Z, Y, X)`；NRRD 正本、cache 解包的 `.npy`、训练侧一切 3D 张量（影像/掩码/patch）都遵守本轴序，物理朝向信息仍以 header 的 space directions 为准。
3. 分割标注一律 **uint8**；框/平面等几何标注一律**世界坐标 JSON**（见下）。
4. 世界坐标约定 **RAS、物理 mm**：标注存世界坐标而非体素坐标——重采样/换 spacing 后不作废，训练时由 loader 用影像 header 的 origin/space directions 变换。
5. 几何一致性铁律：3D 标注与影像的 sizes / space directions / space origin 完全一致（`check` 强制校验）。
6. 文件名 ASCII、无空格；case_id 全数据集唯一；3D 多模态用通道后缀 `case_0000.nrrd`、`case_0001.nrrd`（nnU-Net 约定），标注文件不带后缀。
7. **2.5D 按 3D 存**：2.5D 只是建模习惯（逐切片输入/厚层各向异性），不是严谨的数据维度——数据一律存 3D NRRD（真实各向异性 spacing 进 header），切片与邻层上下文由训练侧 dataset 类负责。dataset.json 的 `dimension` 只记 `2d`/`3d`。
8. `splits/*.json` 是 case_id 的 JSON 数组，train/val/test 划分的唯一事实源（分类的 label 不带 split 列，划分只看这里）。

## 各任务类型的格式（processed/<recipe>/ 内）

| task_type | 目录与影像格式 | 标注 | 关键约定 |
|---|---|---|---|
| `seg3d` 3D 分割 | `images/case_0000.nrrd` + `labels/case.nrrd` | uint8 NRRD 掩码 | 值=classes 下标（0 背景）；标签重采样只用最近邻；几何与影像一致 |
| `seg2d` 2D 分割 | `images/<case>_<0123>.png` + `masks/` 同名 | uint8 索引 PNG（mode L，值=类别号） | 本就是单图的数据直接 `case.png` |
| `cls2d` / `cls3d` 分类 | `images/`（PNG / nrrd） | `labels.csv`（表头 case_id,label） | manifest 为权威；多标签加列 |
| `det3d` 3D 包围框检测 | `images/case_0000.nrrd` | `boxes.json`：每例 `[{label, center_world, size_world}]`（RAS mm） | 世界坐标，loader 变换到体素 |
| `det2d` 2D 包围框检测 | `images/*.png` | `annotations.json`（COCO 格式） | COCO 生态的评测/可视化工具最全 |
| `plane3d` 3D 关键平面检出 | `images/case_0000.nrrd` | `planes.json`（schema 见下） | 平面=点+法向（+面内朝向、范围） |

配准**暂缓**：其标注格式还需按细分类型（刚性/形变、单模/多模、有无合成 GT）分别设计，待细分后再补条目；此前配准数据不套用本规范。

### planes.json（3D 关键平面检出）

```json
{
  "coordinate_system": "RAS, physical mm",
  "plane_def": "center_world + normal_world + 可选 in_plane_up_world / size_world",
  "planes": {
    "case0001": [
      {
        "label": "L3_midsagittal",
        "center_world": [-12.3, 87.1, -5.0],
        "normal_world": [0.0, 0.0, 1.0],
        "in_plane_up_world": [0.0, 1.0, 0.0],
        "size_world": [120.0, 120.0]
      }
    ]
  }
}
```

- `normal_world` 单位向量定平面朝向；`in_plane_up_world`（可选，须在平面内）定面内旋转，二者共同确定按此标注从体数据重采样出的平面图像；`size_world`（可选，`[高 mm, 宽 mm]`）定平面范围，缺省由 loader 用默认约定。
- 一例可含多个平面（不同 label 或多实例）。`preview` 直接输出"按标注重采样出的平面图像"——肉眼即验标注对不对。

## dataset.json（权威清单字段）

`name` / `task_type`（上表键名）/ `dimension`（2d|3d）/ `modality` / `classes`（分割类别顺序即标签值，0 为背景）/ `n_cases` / `recipes`（recipe 名 → 预处理说明，反链资产卡与 EXP 编号）/ `coordinate`（RAS mm）/ `source`（数据来源与许可）/ `created`。scaffold 生成模板，含 TODO 的字段必须人工补全。

`annotation`（标注契约，**训练开始前必须补全**，含 TODO 视同未补）：`produced_by`（脚本/工具/人工 + 可溯源说明）、`rule`（判定规则的具体定义——含测量平面、ROI、窗宽窗位，如「轴状面上椎体边界围成面积最大的切片」）、`tolerance`（标注粒度/允差，如「±1 片」）、`verified`（是否抽样核验）。规则来自人眼主观描述时，先按 methods/label-ceiling-diagnosis.md 做神谕探针/排名检验校核，再作为 strict 指标优化目标。

## 流程

1. 读 `runtime/data.json`：无 `data_root` → AskUserQuestion 问用户（说明默认 `<工作区>/data`），`set-root` 写入并记答复原文；有 → 直接用。
2. 与用户确认：数据集名（如 `lung_seg`）、task_type（参照 env.json 的 `task_profile`，但以用户本次答复为准）、模态、recipe 名。
3. `python3 skills/data-mgmt/manage_data.py scaffold <name> --task-type <t> [--recipe r1] [--modality CT] [--root DIR]`：建骨架 + dataset.json + 标注模板 + README。
4. （如需）按 `assets/preprocessing` 方法卡做预处理，产物进 `processed/<recipe>/`；recipe 说明回写 dataset.json。
4b. （plane3d 且已有 z 索引式标注）`import-planes <mapping.json> --dataset <ds> [--seg-dir DIR] [--seg-suffix S]`：把 {case:{label:z_index}} 导入为 planes.json（LPS 影像自动转 RAS；可选分割掩码质心作面内中心）。`scaffold --link-source DIR` 可直接软链登记既有影像目录（单一正本，不复制）。
4c. **补全 dataset.json 的 `annotation` 标注契约**（produced_by / rule / tolerance / verified）。
5. `manage_data.py check <path> [--recipe r1] [--deep]`：文件配对、几何一致性、标注值域（--deep）、清单完整性、splits 引用有效性；有 FAIL 先修再往下。
6. `manage_data.py preview <path> [--recipe r1] [--limit 20]`：生成分割叠加 / 框 / 平面重采样 PNG。
7. `manage_data.py register <path>`：登记进 runtime/data.json。
8. 空间治理：`manage_data.py du [<path>|根目录]`：报告各数据集 raw/processed/cache/preview 占用，提示可回收的 cache。
9. （标注诊断）`label_audit.py profile / rank-test / review-sheet`：柱内逐片封闭面积剖面提取、标注片特征排名检验（含 z 翻转对照）、审片拼条图——模型进入平台期时按 methods/label-ceiling-diagnosis.md 对标注做上限诊断。

## 约束

- raw 只读；processed 是正本；cache 可随时删（训练时自动重建）。
- 单一正本：数据不复制进任务仓库，任务仓库只引用 runtime/data.json 的路径。
- 既有散落数据（各任务仓库内、ckd-cds-data 等）**不强制迁移**：register 登记原路径即可；新数据集一律入根目录。
- 标注类 JSON 与 dataset.json 保持人类可读是硬要求（可视化优先级的落点之一）。
- 脚本 `scaffold/register/du` 只依赖标准库；`check/preview` 的增强功能按需用 numpy / nrrd（pynrrd）/ Pillow，缺失时降级为文件级检查并明示（当前系统 python3 有 Pillow、缺 numpy 与 pynrrd，需要时 `pip install --user numpy pynrrd` 或改用任务仓库 venv 执行）。 `label_audit.py` 需要 numpy/scipy（review-sheet 另需 Pillow 与 nrrd），用任务仓库 venv 执行。
