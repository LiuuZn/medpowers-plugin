# 实验记录管理技能（exp-mgmt）设计文档

- 日期：2026-10-13
- 状态：待用户审阅
- 范围：mtskills 仓库（本仓库）新增 `skills/exp-mgmt/`，同步修订既有实验记录上下文 6 处；实验记录本体永远存放并提交在任务仓库，不进本仓库 git。

## 1. 背景与问题

现行实验记录规范（AGENTS.md「实验记录规范（强制，双时点）」、`experiments/TEMPLATE.md`、`templates/runtime/README.md` lease 协议、`hooks/session-context.sh` 红线摘要、`templates/runtime/versions.example.json`）存在三个已验证的缺口：

1. **编号非全局**：现行命名 `EXP-YYYYMMDD-<序号>-<slug>.md` 是"日期 + 当日序号"，序号每日归零，不是全局唯一标识；versions.json 里已出现 `EXP-20261005-010-ensemble` 这类长名。
2. **git 管理未强制**：现行只要求记录"归任务仓库管理"，是否提交进任务仓库 git 没有约束。实际后果：spine_locator（versions.json 引用了 EXP-010/011/012/013）仓库内**找不到任何 EXP-\* 记录文件**——编号只在引用处存在，记录本体散佚。
3. **编号检查靠自觉**：分配下一个编号、维护 INDEX.md 同步，全无工具兜底，纯文字规范已失守。

## 2. 目标与非目标

**目标**

- 每个实验在**规划时**即获得任务仓库内全局唯一的自然数编号（001 起递增）。
- 实验记录强制纳入任务仓库 git 管理，每个状态时点自动提交。
- 每次安排训练前，编号检查是强制门禁。
- 编号分配、INDEX 同步、一致性校验由脚本兜底，不靠自觉。

**非目标**

- 不改变"记录归任务仓库、绝不进 mtskills 本仓库 git"的既有红线。
- 不做跨任务仓库的编号协调（已确认：唯一性范围 = 任务仓库内）。
- 不在本设计中实现 lease 写入（lease 协议仍是 `templates/runtime/README.md` 的职责，本技能只做编号门禁与引用校验）。
- 不引入数据库/索引服务，git + markdown 是全部状态载体。

## 3. 已确认决策（与用户逐项确认）

| # | 决策 | 结论 |
|---|---|---|
| 1 | 编号唯一性范围 | **任务仓库内唯一**，各仓库序列独立从 001 递增 |
| 2 | 编号分配时机 | **规划时分配**，新增 `planned` 状态（不再等到训练启动才建记录） |
| 3 | INDEX.md | **脚本自动生成**（记录文件是唯一事实源，list 动作重生成） |
| 4 | 实现载体 | **独立技能** `skills/exp-mgmt/`（SKILL.md + exp_mgmt.py，对齐 data-mgmt 模式） |
| 5 | 取消的实验 | 新增 **`cancelled`** 状态：编号保留不复用、记录不删，理由写进记录 |
| 6 | git 提交执行者 | **脚本自动提交**（只对具体文件路径级 add，绝不动 `-A`） |
| 7 | 文件名 | **编号 + 短 slug**：`EXP-001-<slug>.md`，日期只进正文时间字段 |
| 8 | 提交信息规范 | 时点提交 `exp(EXP-001): plan / start / done / interrupted / cancel / update`；非单记录动作用 `exp: index 同步`、`exp: migrate 旧格式记录至新编号（N 条）` |
| 9 | check 兼查 lease | check 校验 `runtime/training/` lease 引用的 EXP 编号真实存在 |
| 10 | 旧格式处理 | **迁移**（migrate 动作重编号为 001..N 并改写引用），不是放任遗留 |

## 4. 编号与命名规则

- 标识符：`EXP-<NNN>`，三位零填充十进制自然数，从 001 递增；超过 999 自然进位为四位（EXP-1000），不设上限。
- 记录文件名：`EXP-<NNN>-<slug>.md`（slug 为短横线小写短语，如 `EXP-001-unet3d-baseline.md`）。脚本按**文件名前缀**解析编号，slug 不参与编号逻辑。
- 记录目录：任务仓库自选，默认 `<工作区>/experiments/`，INDEX.md 同目录（现状沿用）。
- 旧格式 `EXP-YYYYMMDD-<序号>-<slug>.md` 是迁移对象（见 §9），不是合法的新记录名。

## 5. 状态机与三时点

```
planned ──→ running ──→ done
   │            └─────→ interrupted
   └──→ cancelled
```

| 时点 | 触发 | 动作 | 记录变更 | 提交信息 |
|---|---|---|---|---|
| 规划 | 决定做某实验（下一轮决策产出即可规划） | `new` | 从模板建记录，填任务与数据 + 本轮配置决策，`status: planned`，`planned_at` | `exp(EXP-001): plan` |
| 安排训练 | 写 GPU lease 前的强制门禁 | `start` | 校验编号唯一性与 planned 状态后转 `running`，填 `started_at` | `exp(EXP-001): start` |
| 结束 | 训练结束或中断 | `finish` | 补全结果/诊断/下一轮决策/产物，转 `done` 或 `interrupted`，填 `ended_at` | `exp(EXP-001): done` / `exp(EXP-001): interrupted` |
| 取消 | 规划后决定不训 | `cancel` | 转 `cancelled`，填 `ended_at` 与取消理由；缺 `started_at` 时（planned 直接取消）同以当前时间补填——满足 check 对终态记录的双时间字段要求 | `exp(EXP-001): cancel` |

- 每次 `new / start / finish / cancel` 都会先重生成 INDEX.md 再一并提交（`index` 不单独出现于提交信息，并入各时点提交）。
- `interrupted` 必须注明原因与已产生的观察（沿用现行规范）。
- `cancelled` 的编号**永久占用不复用**：git 历史即编号分配审计日志。

### 记录文件格式（TEMPLATE.md v2 的目标形态）

```markdown
# EXP-001-<slug>

- **status**: planned
- **planned_at**: 2026-10-13T10:00:00+08:00
- **started_at**:（start 时填）
- **ended_at**:（finish/cancel 时填）

## 任务与数据【规划时填】
（沿用现模板：任务项目 / 数据 / 验收口径）

## 配置（本轮决策）【规划时填】
（沿用现模板：基础模型 / 预处理 / 损失目标 / 训练配置；lease slug 在 start 时补记）

## 结果【结束时补】
## 诊断（对照 methods/training-state-diagnosis.md）【结束时补】
## 下一轮决策（对照 methods/next-config-recommendation.md）【结束时补】
## 产物【结束时补】
## 取消理由【仅 cancelled 填】
```

时间字段是脚本解析目标（`list`/`check` 依赖），必须保持 `- **字段**: 值` 的行格式；正文各节标题沿用现模板。

## 6. git 管理纪律

- 记录**必须**提交进任务仓库 git；`experiments/` 目录不允许长期滞留未提交变更（check 提示）。
- 脚本提交的 add 范围 = **记录目录整体**（含智能体手工补填的记录内容、INDEX.md、迁移涉及的 versions.json）——仍是路径级限定，**绝不使用 `-A` / `-A .`**（呼应外层 /ssd/lx 大仓库混入无关改动的教训）。由此，`new` 之后手工填写的规划内容会在下一次脚本动作（start/finish/list）时随路径级 add 一并入库；智能体也可主动以 `exp(EXP-NNN): update` 提交。
- 任务仓库不是 git 仓库（`git rev-parse` 失败）→ 全部动作拒绝执行并报错。
- 提交者身份沿用任务仓库自身 git 配置，脚本不设置也不覆盖 user.name/email。
- mtskills 本仓库的 `experiments/` 只跟踪 TEMPLATE.md 的现状**不变**。

## 7. 技能设计（skills/exp-mgmt/）

```
skills/exp-mgmt/
├── SKILL.md        # frontmatter: name + description（触发条件），正文：流程与约束
└── exp_mgmt.py     # 全部动作的实现，零第三方依赖（stdlib only）
```

SKILL.md description 触发短语：规划实验/新建实验记录、安排训练/启动训练前、实验结束/中断/取消、查询实验状态/实验列表、实验记录迁移、实验记录校验。

### 动作

| 动作 | 命令 | 行为 |
|---|---|---|
| `new` | `exp_mgmt.py new --dir <experiments目录> --slug <slug> [--template <路径>]` | 扫描目录全部 `EXP-*.md`（含 git 已跟踪与未提交的新文件）取 max+1；从模板建 planned 记录；重生成 INDEX；`git add <两个文件> && git commit` |
| `start` | `exp_mgmt.py start --dir <目录> EXP-001` | **编号门禁**：目标存在、编号无重复、状态为 planned → 转 running、填 started_at、重生成 INDEX、提交。状态不是 planned → 拒绝并说明 |
| `finish` | `exp_mgmt.py finish --dir <目录> EXP-001 --status done\|interrupted` | running → done/interrupted、填 ended_at、重生成 INDEX、提交。记录内容（结果/诊断等）由智能体先编辑再调用本动作 |
| `cancel` | `exp_mgmt.py cancel --dir <目录> EXP-001 --reason "..."` | planned/running → cancelled、填 ended_at 与取消理由、重生成 INDEX、提交 |
| `list` | `exp_mgmt.py list --dir <目录>` | 解析全部记录 → 重生成 INDEX.md（编号排序）；有变化时按 §6 范围提交（`exp: index 同步`，顺带入库手工编辑）；同时输出人读摘要 |
| `check` | `exp_mgmt.py check --dir <目录> [--runtime <runtime目录>]` | 只读校验（见下），发现问题时非零退出 |
| `migrate` | `exp_mgmt.py migrate --dir <目录> [--runtime <runtime目录>]` | 见 §9 |

目录定位约定：`--dir` 显式指定优先；否则回落 `<工作区>/experiments/`（与现行默认一致）；支持环境变量 `MTSKILLS_EXPERIMENTS` 显式覆盖，与 `MTSKILLS_RUNTIME` 的先例对齐。`--runtime` 用于 check/migrate 定位 `runtime/training/` 与 `runtime/versions.json`。

### INDEX.md 格式（自动生成，人工不应手改）

```markdown
# 实验记录索引

> 本文件由 skills/exp-mgmt 的 list 动作自动生成，请勿手工编辑；记录文件是唯一事实源。

| 编号 | 状态 | 标题 | planned_at | started_at | ended_at |
|---|---|---|---|---|---|
| EXP-001 | done | unet3d-baseline | 2026-10-13 | 2026-10-13 | 2026-10-14 |
```

### check 校验项

1. 每个 `EXP-*.md`：文件名前缀编号解析成功且**全目录唯一**；status ∈ {planned, running, done, interrupted, cancelled}；
2. 时间字段与状态一致：planned 必无 started_at/ended_at；running 必有 started_at 必无 ended_at；done/interrupted/cancelled 必有两者；
3. INDEX.md 与记录重生成结果一致（不一致 → 提示运行 list）；
4. `--runtime` 给出时：`training/*.json` lease 的 `experiment` 字段指向存在的记录编号，且该记录状态为 running（防 lease 指向幽灵/已结束记录）；
5. 存在旧格式 `EXP-YYYYMMDD-*` 文件 → 警告并提示 migrate（不算错误）；
6. `experiments/` 下有未提交变更 → 警告（git 管理纪律）。

### 错误处理原则

- 任何动作前置校验失败（非 git 仓库、编号冲突、状态机不合法、缺时间字段）→ 拒绝执行、非零退出、逐条列出原因，**不做部分写入**（先全部校验后写盘）。
- 提交失败（如任务仓库有 pre-commit hook 拒绝）→ 如实报告，不回滚已写盘的文件，提示人工处理。

## 8. 上下文同步修改清单（本仓库内）

| 文件 | 修改 |
|---|---|
| `experiments/TEMPLATE.md` | 改写为 §5 的 v2 形态（新命名、五状态、时间字段、三时点标注） |
| `AGENTS.md` | 「实验记录规范（强制，双时点）」节改写为三时点 + git 强制 + 编号门禁；「核心方法论」第 4 步命名同步；目录结构树 skills/ 清单加 exp-mgmt 一句话 |
| `hooks/session-context.sh` | 「实验记录双时点」行改为「三时点 + 编号门禁 + git 强制」，保持 ≤14 行总约束 |
| `templates/runtime/README.md` | lease 协议第 2 步：启动时先经 exp-mgmt `start` 门禁再写 lease，lease `experiment` 字段引用新格式；`versions.json` 说明中 EXP 编号格式同步 |
| `templates/runtime/versions.example.json` | `exp` 字段示例改为 `EXP-<NNN>` |
| `README.md` | 技能清单（开头段落）加 exp-mgmt 一句话；FAQ「运行时状态」处提及实验记录 git 管理 |

发版（plugin.json version bump + tag）不在本设计范围内，按 README 版本纪律由用户另行指示。

## 9. 迁移设计（migrate）

对象：任务仓库记录目录中全部旧格式 `EXP-YYYYMMDD-<序号>-<slug>.md`。

1. 扫描旧格式文件，按（日期、当日序号）排序，依序映射为新编号 `EXP-001..EXP-N`（同目录无新格式记录时从 001 起；已有新格式记录时从 max+1 起）；
2. `git mv` 重命名（未跟踪的旧记录直接改名，随后一并入库）+ 更新文件内一级标题 + **补齐缺失的 v2 时间字段行**：`planned_at` 取文件名日期；status 为终态（done/interrupted/cancelled）且缺 `started_at`/`ended_at` 时同以文件名日期填充（文件名日期是唯一可信来源）；status 为 running 的陈旧记录留空，让 `check` 报错提示人工处置；缺 status 行的记录报错拒绝迁移，要求人工补状态。旧→新映射表写入迁移报告；
3. `--runtime` 给出时，改写 `runtime/versions.json` 中命中旧 ID 的 `exp` 字段为新 ID；lease 文件不迁移（瞬时状态，报告悬空引用即可）；
4. 重生成 INDEX.md，单次提交 `exp: migrate 旧格式记录至新编号（N 条）`；
5. 报告**悬空引用**：versions.json / lease 引用了旧编号但记录文件不存在（如 spine_locator 的 EXP-010..013）——只报告，不虚构记录；
6. 迁移后 `next`（new 的分配逻辑）从 N+1 继续；目录中仍有旧格式文件时，`new`/`start` 拒绝执行并提示先 migrate（迁移是编号连续性的前置门禁）。

## 10. 与 lease 协议的边界

- `start` 只做记录侧的门禁与状态迁移，**不写 lease**；lease 写入仍按 `templates/runtime/README.md` 由训练启动流程执行，二者顺序：`start` 通过 → 写 lease（引用 `EXP-<NNN>`）→ 三确认 → 后台分离启动。
- check 的 lease 校验是**只读**的，发现指向幽灵记录的 lease 属报警，不代删。

## 11. 验收口径

在临时 git 仓库 fixture 上验证全流程与异常路径：

1. 全流程：new → 编辑 → start → 编辑 → finish(done) 依次成功，每次 git log 恰好一条对应提交，INDEX 与记录一致；
2. 异常路径：非 git 目录拒绝；重号拒绝（手工放置同号文件后 new）；状态机违例（对 running 再 start、对 planned 直接 finish）；时间字段缺失被 check 捕获；
3. cancelled 旁路：new → cancel，编号在后续 new 中被跳过不复用；
4. 迁移：预置 3 个旧格式记录 + 含旧引用的 versions.json，migrate 后编号连续、引用改写、悬空引用出现在报告；
5. check 的 lease 校验：指向不存在编号的 lease 报警。

## 12. 决策记录

- 2026-10-13：与用户确认 §3 全部 10 项决策；方案对比（独立技能 / 纯文档 / 并入现有技能）中选定独立技能。
