const CITE_RE = /\[([A-Z]{2,}-\d+[A-Z]?)\]/g;
const SESSION_KEY = "rag_session_id";
const TRANSCRIPT_KEY = "rag_transcript";

const EXAMPLES = [
  "员工满 3 年不满 5 年，年休假有几天？",
  "2026 年 9 月云资源成本的本月合计是多少元？只按明细行汇总。",
  "恒远精工为什么申请分阶段验收？三号工厂预计什么时候能联网？",
  "发票丢了还能报销吗？需要哪些材料？",
];

const STATUS_TEXT = {
  "prepare:start": "正在理解问题…",
  "retrieve:start": "正在检索资料…",
  "retrieve:done": "检索完成，正在判断路由…",
  "guard:done": "正在检查权限…",
  "route:done": "正在选择回答方式…",
  "structured:done": "表格核算完成…",
  "generate:start": "正在生成回答…",
};

const $ = (id) => document.getElementById(id);

let sessionId = readSession();
let turns = readTranscript();
let streamText = "";
let streamRenderQueued = false;

function readSession() {
  try {
    return localStorage.getItem(SESSION_KEY) || "";
  } catch {
    return "";
  }
}

function readTranscript() {
  try {
    const raw = JSON.parse(localStorage.getItem(TRANSCRIPT_KEY) || "[]");
    return Array.isArray(raw) ? raw.filter((item) => item && item.question) : [];
  } catch {
    return [];
  }
}

function persistSession(id) {
  sessionId = id || "";
  try {
    if (sessionId) localStorage.setItem(SESSION_KEY, sessionId);
    else localStorage.removeItem(SESSION_KEY);
  } catch {
    // 隐私模式下 localStorage 不可用，会话仍可在当前页面内工作
  }
}

function saveTranscript() {
  try {
    localStorage.setItem(TRANSCRIPT_KEY, JSON.stringify(turns.slice(-20)));
  } catch {
    // ignore
  }
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function setStatus(message, kind = "") {
  const el = $("status");
  el.textContent = message;
  el.className = `status ${kind}`.trim();
}

function setLoading(loading) {
  const button = $("submit");
  button.disabled = loading;
  button.textContent = loading ? "生成中…" : "提问";
}

async function loadRoles() {
  try {
    const response = await fetch("/api/roles");
    const data = await response.json();
    const select = $("role");
    select.innerHTML = "";
    for (const role of data.roles || []) {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = role;
      select.appendChild(option);
    }
    select.value = "普通员工";
  } catch {
    setStatus("角色列表加载失败，请刷新页面", "error");
  }
}

function renderExamples() {
  const container = $("examples");
  for (const question of EXAMPLES) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example";
    button.textContent = question;
    button.addEventListener("click", () => {
      $("question").value = question;
      $("question").focus();
    });
    container.appendChild(button);
  }
}

function renderAnswer(text) {
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(CITE_RE, (_, docId) =>
    `<button type="button" class="cite" data-doc="${escapeHtml(docId)}">[${escapeHtml(docId)}]</button>`
  );
  return html
    .split(/\n{2,}/)
    .map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`)
    .join("");
}

function scheduleStreamRender() {
  if (streamRenderQueued) return;
  streamRenderQueued = true;
  window.requestAnimationFrame(() => {
    streamRenderQueued = false;
    if ($("result").classList.contains("hidden")) showStreaming();
    const answer = $("answer");
    // final 已渲染后，残留的帧不能再覆盖成原始流式文本
    if (!answer.classList.contains("streaming")) return;
    answer.textContent = streamText;
  });
}

function showStreaming() {
  $("result").classList.remove("hidden");
  $("denied").classList.add("hidden");
  $("sources").innerHTML = "";
  $("debug-panel").classList.add("hidden");
  $("latency").textContent = "";
  $("usage").textContent = "";
  $("request-id").textContent = "";
  const badge = $("route-badge");
  badge.textContent = "生成中";
  badge.className = "badge route-rag";
  const answer = $("answer");
  answer.classList.add("streaming");
  answer.textContent = "";
}

function statusClass(status) {
  return ["active", "draft", "superseded", "outdated", "confidential"].includes(status)
    ? status
    : "";
}

function renderSources(sources) {
  const container = $("sources");
  if (!sources.length) {
    container.innerHTML = '<p class="muted">本次没有可展示的来源。</p>';
    return;
  }
  container.innerHTML = sources
    .map((source) => {
      const status = escapeHtml(source.status);
      return `
        <article class="source-card" data-doc="${escapeHtml(source.doc_id)}">
          <div class="source-top">
            <span class="doc-id">[${escapeHtml(source.doc_id)}]</span>
            <span class="pill ${statusClass(source.status)}">${status}</span>
          </div>
          <p class="source-title">${escapeHtml(source.title)}</p>
          <div class="source-meta">
            <span class="pill">v${escapeHtml(source.version)}</span>
            <span class="pill">生效 ${escapeHtml(source.effective_date)}</span>
            <span class="pill ${statusClass(source.classification)}">${escapeHtml(source.classification)}</span>
          </div>
        </article>`;
    })
    .join("");
}

function highlightSource(docId) {
  const cards = Array.from(document.querySelectorAll(".source-card"));
  const card = cards.find((item) => item.dataset.doc === docId);
  if (!card) {
    setStatus(`引用 [${docId}] 不在本次来源列表中`);
    return;
  }
  setStatus("");
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("flash");
  window.setTimeout(() => card.classList.remove("flash"), 1400);
}

function formatUsage(usage) {
  if (!usage || !usage.calls) return "";
  const tokens = usage.total_tokens || 0;
  const cost = Number(usage.cost_cny || 0).toFixed(4);
  const estimated = usage.estimated_calls ? "（含估算）" : "";
  return `${usage.calls} 次模型调用 · ${tokens} tokens · ¥${cost}${estimated}`;
}

function renderTrace(events) {
  const list = $("trace");
  list.innerHTML = "";
  for (const event of events || []) {
    const item = document.createElement("li");
    const node = document.createElement("span");
    node.className = "trace-node";
    node.textContent = event.node;
    item.appendChild(node);
    const detail = document.createElement("span");
    detail.className = `trace-${event.status || "ok"}`;
    detail.textContent = ` · ${event.elapsed_ms} ms · ${event.status || "ok"}`;
    item.appendChild(detail);
    if (event.summary) {
      const summary = document.createElement("span");
      summary.className = "muted";
      summary.textContent = ` ${JSON.stringify(event.summary)}`;
      item.appendChild(summary);
    }
    if (event.error) {
      const error = document.createElement("span");
      error.className = "trace-denied";
      error.textContent = ` ${event.error}`;
      item.appendChild(error);
    }
    list.appendChild(item);
  }
}

function updateSessionLine(session) {
  const el = $("session-info");
  if (!session) {
    el.textContent = sessionId ? "会话进行中" : "新会话";
    return;
  }
  const parts = [];
  if (session.turn != null) parts.push(`第 ${session.turn} 轮`);
  if (session.turn_count != null) parts.push(`已完成 ${session.turn_count} 轮`);
  if (session.remaining_turns != null) parts.push(`剩余 ${session.remaining_turns} 轮`);
  if (session.total_tokens) parts.push(`${session.total_tokens} tokens`);
  if (session.total_cost_cny != null) {
    parts.push(`¥${Number(session.total_cost_cny).toFixed(4)}`);
  }
  el.textContent = parts.length ? parts.join(" · ") : "会话进行中";
}

function renderResult(data) {
  $("result").classList.remove("hidden");
  const answer = $("answer");
  answer.classList.remove("streaming");
  answer.innerHTML = renderAnswer(data.answer);
  $("denied").classList.toggle("hidden", !data.denied);
  $("sources").classList.remove("hidden");
  renderSources(data.sources || []);

  const badge = $("route-badge");
  badge.textContent = `route: ${data.route}`;
  badge.className = `badge route-${["rag", "sql", "table"].includes(data.route) ? data.route : "error"}`;
  $("latency").textContent = `${data.latency_ms} ms`;
  $("usage").textContent = formatUsage(data.usage);
  $("request-id").textContent = `request ${data.request_id}`;
  updateSessionLine(data.session || null);

  const debugPanel = $("debug-panel");
  const hasDebug = Boolean(data.trace && data.trace.length);
  debugPanel.classList.toggle("hidden", !hasDebug);
  if (hasDebug) {
    renderTrace(data.trace);
    $("self-check").textContent = JSON.stringify(data.self_check || {}, null, 2);
  }
  setStatus("");
}

function summarize(text, limit = 140) {
  const value = String(text ?? "").replace(/\s+/g, " ").trim();
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

function renderTranscript() {
  const panel = $("transcript");
  panel.classList.toggle("hidden", turns.length === 0);
  const list = $("turns");
  list.innerHTML = "";
  for (const turn of turns) {
    const item = document.createElement("li");
    const question = document.createElement("p");
    question.className = "turn-q";
    question.textContent = `Q：${turn.question}`;
    const answer = document.createElement("p");
    answer.className = "turn-a";
    answer.textContent = `A：${summarize(turn.answer)}`;
    item.append(question, answer);
    list.appendChild(item);
  }
}

function addTurn(question, answer) {
  turns.push({ question, answer: summarize(answer, 400) });
  turns = turns.slice(-20);
  saveTranscript();
  renderTranscript();
}

function clearSession(message) {
  persistSession("");
  turns = [];
  saveTranscript();
  renderTranscript();
  updateSessionLine(null);
  $("result").classList.add("hidden");
  setStatus(message || "");
}

function errorMessage(data, status) {
  if (data && data.message) return data.message;
  if (data && typeof data.detail === "string") return data.detail;
  if (data && Array.isArray(data.detail)) {
    return data.detail.map((item) => item.msg || "").filter(Boolean).join("；");
  }
  return `请求失败（HTTP ${status}）`;
}

function statusMessage(data) {
  if (data.node === "guard" && data.status === "denied") {
    return "该问题涉及越权内容，正在整理拒答…";
  }
  if (data.node === "guard" && data.status === "not_found") {
    return "未找到可访问资料…";
  }
  if (data.node === "structured" && data.status === "fallback") {
    return "结构化工具未命中，转文本回答…";
  }
  return STATUS_TEXT[`${data.node}:${data.phase}`] || "";
}

function dispatchFrame(frame, onEvent) {
  let event = "message";
  const dataLines = [];
  for (const line of frame.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return;
  let data;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    return;
  }
  onEvent(event, data);
}

async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");
    let index;
    while ((index = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      dispatchFrame(frame, onEvent);
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) dispatchFrame(buffer.replace(/\r\n/g, "\n"), onEvent);
}

async function ask(allowRetry = true) {
  const question = $("question").value.trim();
  if (!question) {
    setStatus("请输入问题后再提交", "error");
    $("question").focus();
    return;
  }
  const sentSession = sessionId;
  setLoading(true);
  setStatus("正在检索并生成回答…");
  $("result").classList.add("hidden");
  streamText = "";
  try {
    const payload = {
      question,
      role: $("role").value,
      debug: $("debug").checked,
    };
    if (sentSession) payload.session_id = sentSession;
    const response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      if ([404, 409].includes(response.status) && sentSession && allowRetry) {
        clearSession("会话已失效或角色不一致，已自动开始新会话");
        return await ask(false);
      }
      throw new Error(errorMessage(data, response.status));
    }

    let finalData = null;
    let streamError = null;
    await readEvents(response, (event, data) => {
      if (event === "meta") {
        persistSession(data.session_id);
        updateSessionLine({ turn: data.turn });
      } else if (event === "status") {
        const message = statusMessage(data);
        if (message) setStatus(message);
      } else if (event === "delta") {
        streamText += data.text || "";
        scheduleStreamRender();
      } else if (event === "reset") {
        streamText = "";
        scheduleStreamRender();
        setStatus(
          data.reason === "truncation_retry"
            ? "回答超长，正在重新生成…"
            : "正在修正字段格式…"
        );
      } else if (event === "final") {
        finalData = data;
      } else if (event === "error") {
        streamError = data;
      }
    });

    if (streamError) throw new Error(streamError.message || "生成失败，请重试");
    if (!finalData) throw new Error("连接中断，未收到完整回答");
    persistSession(finalData.session?.session_id || sentSession);
    renderResult(finalData);
    addTurn(question, finalData.answer || "");
  } catch (error) {
    setStatus(error.message || "网络错误，请重试", "error");
  } finally {
    setLoading(false);
  }
}

$("submit").addEventListener("click", () => ask());
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    ask();
  }
});
$("answer").addEventListener("click", (event) => {
  const cite = event.target.closest(".cite");
  if (cite) highlightSource(cite.dataset.doc);
});
$("new-session").addEventListener("click", () => {
  clearSession("已开始新会话");
  $("question").value = "";
  $("question").focus();
});
$("role").addEventListener("change", () => {
  clearSession("角色已切换，已开始新会话");
});

loadRoles();
renderExamples();
renderTranscript();
updateSessionLine(null);
