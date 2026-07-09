#!/usr/bin/env bash
# PreToolUse hook: 拦截内置 WebSearch 工具, 引导改用 MCP 的 miyami-websearch-mcp
# 背景: 在此环境内置 WebSearch 返回空壳(只有 "I'll search..." 占位符无结果),
#       用户安装的 miyami-websearch-mcp 经实测能正常返回结果。
# 行为: 检测到 tool_name == WebSearch 时返回 stderrWithFeedback 提示,
#       不硬崩(避免误拦合法测试场景), 仅强制把提示塞回对话上下文。
# 详见记忆: memory/feedback-always-use-mcp-websearch.md

# Claude Code PreToolUse hook 协议:
#   stdin 收到 JSON: {"tool_name":"WebSearch","tool_input":{...}, ...}
#   退出码 2 + stderr 反馈会阻断该次工具调用并把反馈送回模型
#   hook 运行在 cmd/git-bash 环境

set -e
input="$(cat)"

# 提取 tool_name
tool_name="$(printf '%s' "$input" | grep -oE '"tool_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed -E 's/.*"tool_name"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')"

if [ "$tool_name" = "WebSearch" ]; then
  # 退出码 2: 阻断调用并把 stderr 反馈给模型
  cat >&2 <<'MSG'
⚠️ [feedback] 此环境的内置 WebSearch 工具一直返回空壳无结果, 请改用 MCP 搜索:
  1. 先 ToolSearch 加载: select:mcp__miyami-websearch-mcp__web_search,mcp__miyami-websearch-mcp__search_and_fetch
  2. 用 mcp__miyami-websearch-mcp__search_and_fetch (含网页抓取, 推荐) 或 mcp__miyami-websearch-mcp__web_search
详见记忆: feedback-always-use-mcp-websearch.md
MSG
  exit 2
fi

# 其他工具放行
exit 0
