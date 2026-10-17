#!/usr/bin/env bash
# SessionStart hook：每次会话向上下文注入 medpowers 方法论红线摘要（≤14 行）。
# 仅用 bash 内建 printf，不依赖任何外部命令；摘要为 AGENTS.md / templates/runtime/README.md 的忠实缩写。
set -eu

printf '%s\n' \
  '## medpowers 方法论常驻规范（强制）' \
  '- **医疗影像可视化红线**：智能体不得把自己的影像视觉判断（对影像、渲染图、切片截图"看图下结论"——判断解剖结构、确认标注或病灶位置、判定对错）作为证据或决策依据；可视化产物（对比图、审片图等）可以生成，但生成后必须交由用户查看并反馈结论，用户审片意见是唯一的视觉证据来源；对错判定只允许来自程序化测量与用户审片反馈两类渠道，报告与实验记录不得写智能体的视觉印象。' \
  '- **实验记录三时点（git 管理）**：凡经本方法论指导的实验必须记录——规划时经 exp-mgmt new 分配全局唯一编号（EXP-001 起递增）建 planned 记录并提交；训练启动与收尾由训练进程经 medpowers.training_session 自己完成（入口过 start 编号门禁转 running、门禁失败自行退出训练不开始；结束 finish 落账 done/interrupted，指标/产物自动入账；exp-mgmt CLI 的 start/finish 仅为训练代码未集成包时的遗留兜底），取消用 cancel（编号不复用）；记录必须提交进任务仓库 git 并推送远端，绝不纳入 medpowers 本仓库（模板随插件 experiments/TEMPLATE.md）；多台机器共用任务仓库时，变更动作强制拉取→确认→提交→推送（训练进程经包执行的动作同一套实现），推送成功才算完成；查询/汇报实验与训练状态（哪些在跑/可启动/进展）前，同样先拉取同步远端台账——本地副本可能滞后于他机已启动的实验（误报 planned），lease 与 nvidia-smi 只反映本机。' \
  '- **GPU lease 三步**：训练前（智能体）**三确认**——读 `runtime/env.json` 确认目标卡归本训练目录可用（available: true）、读 `runtime/training/` 全部 lease、并用 nvidia-smi 实测（快照不能替代实测），三者都满足才启动（并确认训练入口已内嵌 medpowers.training_session）；启动时训练进程经包过 start 门禁后原子写自己的 lease JSON（任务、机器名、GPU、分离后真实 pid、起止时间、日志位置）到 `runtime/training/`；结束（训练进程）finish 落账后立即删除自己的 lease。lease 是声明式参考不是硬锁，真实占用以 nvidia-smi 为准。实验记录/lease/提交信息中的机器指代一律用 notify.json 的 machine_name（禁称"本机"——多机共读无法区分）。' \
  '- **训练必须后台分离运行**：训练是长任务而智能体会话无法长期驻留，绝不能作为会话前台命令/子进程运行（智能体工具自带的后台模式若随会话终止，同样不满足）——必须用 `setsid nohup <训练命令> > 日志 2>&1 < /dev/null &`（或 tmux）做 OS 级分离，stdout/stderr 重定向到日志文件（即 lease 的 log 字段），lease 的 pid 记分离后真实训练进程；智能体只轮询日志/lease/nvidia-smi 跟踪进度，会话退出训练必须照常继续（协议正本 `templates/runtime/README.md`）。' \
  '方法论全文见插件 `methods/`（选基础模型 → 训练诊断 → 推荐下一轮配置）；其余仓库规范见任务仓库 AGENTS.md。'
