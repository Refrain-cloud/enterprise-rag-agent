const CITE_RE = /\[([A-Z]{2,}-\d+[A-Z]?)\]/g;

const EXAMPLES = [
  "员工满 3 年不满 5 年，年休假有几天？",
  "2026 年 9 月云资源成本的本月合计是多少元？只按明细行汇总。",
  "恒远精工为什么申请分阶段验收？三号工厂预计什么时候能联网？",
  "发票丢了还能报销吗？需要哪些材料？",
];

const $ = (id) => document.getElementById(id);

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
  button.textContent = loading ? "检索中…" : "提问";
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
  return `${usage.calls} 次模型调用 · ${tokens} tokens · ¥${cost}`;
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

function renderResult(data) {
  $("result").classList.remove("hidden");
  $("answer").innerHTML = renderAnswer(data.answer);
  $("denied").classList.toggle("hidden", !data.denied);
  $("sources").classList.remove("hidden");
  renderSources(data.sources || []);

  const badge = $("route-badge");
  badge.textContent = `route: ${data.route}`;
  badge.className = `badge route-${["rag", "sql", "table"].includes(data.route) ? data.route : "error"}`;
  $("latency").textContent = `${data.latency_ms} ms`;
  $("usage").textContent = formatUsage(data.usage);
  $("request-id").textContent = `request ${data.request_id}`;

  const debugPanel = $("debug-panel");
  const hasDebug = Boolean(data.trace && data.trace.length);
  debugPanel.classList.toggle("hidden", !hasDebug);
  if (hasDebug) {
    renderTrace(data.trace);
    $("self-check").textContent = JSON.stringify(data.self_check || {}, null, 2);
  }
  setStatus("");
}

function errorMessage(data, status) {
  if (data && data.message) return data.message;
  if (data && typeof data.detail === "string") return data.detail;
  if (data && Array.isArray(data.detail)) {
    return data.detail.map((item) => item.msg || "").filter(Boolean).join("；");
  }
  return `请求失败（HTTP ${status}）`;
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) {
    setStatus("请输入问题后再提交", "error");
    $("question").focus();
    return;
  }
  setLoading(true);
  setStatus("正在检索并生成回答…");
  $("result").classList.add("hidden");
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        role: $("role").value,
        debug: $("debug").checked,
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(data, response.status));
    renderResult(data);
  } catch (error) {
    setStatus(error.message || "网络错误，请重试", "error");
  } finally {
    setLoading(false);
  }
}

$("submit").addEventListener("click", ask);
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

loadRoles();
renderExamples();
