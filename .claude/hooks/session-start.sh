#!/usr/bin/env bash
# 新会话启动时，把最新一份交接文档注入上下文。
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
# 按修改时间取最新；同一天有多份时不会误选较早的那份。
latest=$(ls -1t "$root"/docs/handoff/*.md 2>/dev/null | grep -v '/README\.md$' | head -n1)

if [ -z "$latest" ]; then
    echo "[handoff] docs/handoff/ 下没有交接文档。"
    exit 0
fi

rel=${latest#"$root"/}
echo "[handoff] 新会话已自动加载最新交接文档：$rel"
echo "-----"
cat "$latest"
echo
echo "-----"
echo "如需查看其他交接文档，请阅读 docs/handoff/。"
