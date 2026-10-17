#!/usr/bin/env bash
# SessionStart hook（第二路）：检查本插件安装副本的"下载日期"——插件根目录的 mtime，
# 即该副本内容最近一次被平台下载/写入的时间（平台重新下载或更新会刷新它）。
# 距今超过 7 天时，向会话注入一条会话级指令：让智能体在每次回复末尾友好提醒用户
# 检查插件更新（插件模式更新走各平台插件管理器，不走 self-update 技能）。
# 本提醒尽力而为：只读本地文件、不写状态、不做网络请求；任何取值失败一律静默退出，
# 绝不阻断会话启动。
set -u

# 定位插件根：脚本位于 <plugin_root>/hooks/，不依赖平台是否导出 CLAUDE_PLUGIN_ROOT
plugin_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 0

# 下载时间 = 插件根目录 mtime（GNU stat 优先，BSD stat 兜底）
mtime="$(stat -c %Y "$plugin_root" 2>/dev/null || stat -f %m "$plugin_root" 2>/dev/null || true)"
case "$mtime" in ''|*[!0-9]*) exit 0 ;; esac

now="${EPOCHSECONDS:-}"
if [ -z "$now" ]; then
  now="$(date +%s 2>/dev/null || true)"
fi
case "$now" in ''|*[!0-9]*) exit 0 ;; esac

age_days=$(( (now - mtime) / 86400 ))
[ "$age_days" -ge 7 ] || exit 0

# 版本与日期只用于提示文本，取不到就降级为占位值
version="$(sed -n 's/^[[:space:]]*"version":[[:space:]]*"\([^"]*\)".*/\1/p' "$plugin_root/.claude-plugin/plugin.json" 2>/dev/null | head -n 1)"
if [ -n "$version" ]; then version="v${version}"; else version="版本未知"; fi
dl_date="$(date -d "@$mtime" +%F 2>/dev/null || date -r "$mtime" +%F 2>/dev/null || true)"
[ -n "$dl_date" ] || dl_date="epoch $mtime"

printf '%s\n' \
  '## medpowers 插件更新提醒（会话级指令，仅本插件下载超过一周时注入）' \
  "当前 medpowers 插件副本（${version}）下载于 ${dl_date}，已 ${age_days} 天未更新。本会话中请在**每次回复的末尾**附一行简短友好的提醒，建议用户检查 medpowers 插件更新（插件模式经所在平台的插件管理器更新，如 Claude Code /plugin、ZCode 插件市场，或对 medpowers 市场重新 add/update；见插件 README「插件安装」）。要求：一两句话即可，不干扰任务本身；用户表示已更新、已知晓或不需要后，本会话立即停止附加。"
