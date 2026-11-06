/* 企业知识库问答前端逻辑：知识库选择、SSE 流式问答（引用/证据/降级提示）、文档管理。
 * 无框架依赖；所有请求走 /api/v1；dev 鉴权无需令牌，OIDC 模式用设置面板注入 Bearer Token。 */
"use strict";

const API = "/api/v1";
const $ = (id) => document.getElementById(id);

const state = {
  kbCode: localStorage.getItem("rag_kb") || "",
  kbs: [],
  kbMap: {},
  token: localStorage.getItem("rag_token") || "",
  // 多轮会话：done 事件回传的 conversation_id，下一轮带着它才能做追问改写
  conversationId: null,
  docPage: 1,
  docSize: 20,
  docTotal: 0,
  // 当前页文档标题：没有配置示例问题时，用它生成“《标题》主要讲了什么？”这类入口
  docTitles: [],
  busy: false,
  // 左侧文档管理可收起（问答优先），偏好保存在本浏览器
  sidebarHidden: localStorage.getItem("rag_sidebar_hidden") === "1",
};

/* ---------------- 基础请求 ---------------- */

function headers(extra = {}) {
  const h = { ...extra };
  if (state.token) h.Authorization = `Bearer ${state.token}`;
  return h;
}

async function api(path, options = {}) {
  const resp = await fetch(`${API}${path}`, { ...options, headers: headers(options.headers) });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      detail = body.message || body.detail || detail;
    } catch { /* 保留默认错误文案 */ }
    throw new Error(detail);
  }
  return resp.json();
}

/* ---------------- 健康状态 ---------------- */

async function pollHealth() {
  const dot = $("health-dot");
  try {
    const resp = await fetch("/health", { headers: headers() });
    const body = await resp.json();
    const allUp = Object.values(body.services || {}).every((v) => v === "up");
    dot.className = `dot ${allUp ? "up" : "down"}`;
    dot.title = allUp ? "全部依赖正常" : `依赖异常：${JSON.stringify(body.services)}`;
  } catch {
    dot.className = "dot down";
    dot.title = "健康检查失败";
  }
}

/* ---------------- 知识库 ---------------- */

async function loadKbs() {
  state.kbs = await api("/kbs");
  state.kbMap = {};
  const select = $("kb-select");
  select.innerHTML = "";
  for (const kb of state.kbs) {
    state.kbMap[kb.kb_code] = kb;
    const opt = document.createElement("option");
    opt.value = kb.kb_code;
    opt.textContent = `${kb.name}（${kb.kb_code}）`;
    select.appendChild(opt);
  }
  if (state.kbs.some((kb) => kb.kb_code === state.kbCode)) {
    select.value = state.kbCode;
  } else if (state.kbs.length > 0) {
    state.kbCode = state.kbs[0].kb_code;
    select.value = state.kbCode;
  }
  localStorage.setItem("rag_kb", state.kbCode);
  syncKbUi();
}

/* ---------------- 问答界面引导 ---------------- */

const GENERIC_QUESTIONS = [
  "这个知识库主要收录了哪些内容？",
  "请概括其中最关键的规定",
  "有哪些流程需要特别注意？",
];

const DOC_TITLE_LIMIT = 32;

function currentKb() {
  return state.kbMap[state.kbCode] || null;
}

/* 顶部显示当前作用范围，输入框按知识库给出占位提示 */
function syncKbUi() {
  const kb = currentKb();
  $("chat-kb-name").textContent = kb ? kb.name : "未选择知识库";
  $("chat-kb-desc").textContent = kb && kb.description ? kb.description : "";
  $("chat-input").placeholder = kb
    ? `向「${kb.name}」提问，Enter 发送，Shift+Enter 换行`
    : "请先选择知识库";
  renderSamples(kb);
}

/* 欢迎页示例问题：优先用知识库配置；没有配置时用最近文档标题生成，保证总有可点的入口 */
function renderSamples(kb) {
  const box = $("sample-questions");
  box.innerHTML = "";
  const curated = kb && Array.isArray(kb.sample_questions) ? kb.sample_questions : [];
  const questions = curated.length > 0
    ? curated
    : [
        ...state.docTitles.slice(0, 3).map((title) => {
          const short = title.length > DOC_TITLE_LIMIT ? `${title.slice(0, DOC_TITLE_LIMIT)}…` : title;
          return `《${short}》主要讲了什么？`;
        }),
        ...GENERIC_QUESTIONS,
      ];
  for (const question of questions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample-chip";
    btn.textContent = question;
    btn.addEventListener("click", () => {
      if (state.busy || !state.kbCode) return;
      sendQuestion(question);
    });
    box.appendChild(btn);
  }
}

/* 只在一轮对话开始前展示欢迎页；开始后让位给消息列表 */
function updateWelcome() {
  const hasMessages = $("messages").children.length > 0;
  $("welcome").classList.toggle("hidden", hasMessages);
}

function resetConversation() {
  state.conversationId = null;
  $("messages").innerHTML = "";
  updateWelcome();
}

/* ---------------- 文档管理 ---------------- */

async function loadDocs() {
  const status = $("doc-status-filter").value;
  const query = new URLSearchParams({ page: state.docPage, size: state.docSize });
  if (status) query.set("status", status);
  const body = await api(`/kbs/${state.kbCode}/documents?${query}`);
  state.docTotal = body.total;
  state.docTitles = body.items.filter((doc) => doc.status === "ready").map((doc) => doc.title);
  renderSamples(currentKb());
  $("doc-total").textContent = `共 ${body.total} 篇`;
  $("doc-page").textContent = String(state.docPage);
  const list = $("doc-list");
  list.innerHTML = "";
  for (const doc of body.items) {
    const li = document.createElement("li");
    li.className = "doc-item";
    const acl = doc.acl_groups.length ? doc.acl_groups.join(",") : "公开";
    li.innerHTML = `
      <div class="doc-title" title="${escapeHtml(doc.title)}">${escapeHtml(doc.title)}</div>
      <div class="doc-meta">
        <span class="badge ${escapeHtml(doc.status)}">${escapeHtml(doc.status)}</span>
        <span>v${doc.version}</span>
        <span>${doc.chunk_count} chunks</span>
        <span>${escapeHtml(acl)}</span>
      </div>
      <div class="doc-actions"><button class="ghost-btn small danger">删除</button></div>`;
    li.querySelector(".doc-actions button").addEventListener("click", () => deleteDoc(doc));
    list.appendChild(li);
  }
}

async function deleteDoc(doc) {
  if (!confirm(`确认删除文档「${doc.title}」？删除后索引与原始文件都会被清理。`)) return;
  try {
    await api(`/kbs/${state.kbCode}/documents/${doc.doc_id}`, { method: "DELETE" });
    await loadDocs();
  } catch (err) {
    alert(`删除失败：${err.message}`);
  }
}

async function submitUpload(event) {
  event.preventDefault();
  const result = $("upload-result");
  result.className = "upload-result";
  result.textContent = "上传解析中…";
  const file = $("upload-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.set("file", file);
  if ($("upload-title").value) form.set("title", $("upload-title").value);
  if ($("upload-acl").value) form.set("acl_groups", $("upload-acl").value);
  try {
    const body = await api(`/kbs/${state.kbCode}/documents`, { method: "POST", body: form });
    const verdict = body.duplicate_of
      ? "（与库内文档重复，仅登记血缘）"
      : body.reused
        ? "（内容未变化，复用既有索引）"
        : `（新建 ${body.chunk_count} 个 chunk）`;
    result.className = "upload-result ok";
    result.textContent = `入库成功 doc_id=${body.doc_id} ${verdict}`;
    $("upload-file").value = "";
    await loadDocs();
  } catch (err) {
    result.className = "upload-result err";
    result.textContent = `入库失败：${err.message}`;
  }
}

/* ---------------- 问答（SSE 流式） ---------------- */

function appendMessage(role) {
  const li = document.createElement("li");
  li.className = `msg ${role}`;
  li.innerHTML = `<div class="bubble"></div><div class="msg-meta"></div>`;
  $("messages").appendChild(li);
  $("messages").scrollTop = $("messages").scrollHeight;
  updateWelcome();
  return li;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

/* 把答案文本中的 [n] 渲染成可点击的引用角标（点击弹出片段）。
 * 先转义再处理：只做最小可读性增强（粗体/行内代码），不引入 Markdown 渲染器，
 * 避免把模型输出里的 HTML 当成结构执行。 */
function renderAnswer(bubble, text, citations) {
  const escaped = escapeHtml(text)
    .replace(/\*\*([^*\n]{1,120})\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`\n]{1,100})`/g, '<code class="inline-code">$1</code>');
  bubble.innerHTML = escaped.replace(/\[(\d+)\]/g, (match, num) => {
    const citation = citations.find((c) => c.index === Number(num));
    if (!citation) return match;
    return `<span class="cite-chip" data-index="${num}">${num}</span>`;
  });
  bubble.querySelectorAll(".cite-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const citation = citations.find((c) => c.index === Number(chip.dataset.index));
      if (!citation) return;
      showCitePop(chip, citation);
    });
  });
}

function showCitePop(anchor, citation) {
  document.querySelectorAll(".cite-pop").forEach((el) => el.remove());
  const pop = document.createElement("div");
  pop.className = "cite-pop";
  pop.innerHTML = `<div class="pop-title">[${citation.index}] ${escapeHtml(citation.title)}</div>
    <div>${escapeHtml(citation.snippet)}</div>`;
  document.body.appendChild(pop);
  const rect = anchor.getBoundingClientRect();
  pop.style.left = `${Math.min(rect.left, window.innerWidth - 440)}px`;
  pop.style.top = `${Math.min(rect.bottom + 8, window.innerHeight - 160)}px`;
  setTimeout(() => pop.addEventListener("click", () => pop.remove()), 0);
  document.addEventListener("click", function close(ev) {
    if (!pop.contains(ev.target)) { pop.remove(); document.removeEventListener("click", close); }
  });
}

function renderEvidence(metaBox, evidence) {
  if (!evidence || evidence.length === 0) return;
  const typeNames = { document: "文档", database: "数据库", api: "实时接口" };
  const box = document.createElement("div");
  box.className = "evidence-box";
  box.innerHTML = evidence
    .map(
      (item) => `
      <div class="evidence-item">
        <div class="ev-head"><span class="ev-type">[${typeNames[item.source_type] || item.source_type}]</span>
          <span>${escapeHtml(item.source_id)}</span><span>${escapeHtml(item.observed_at)}</span></div>
        <div class="ev-locator">${escapeHtml(item.locator)}</div>
        <div class="ev-snippet">${escapeHtml(item.snippet)}</div>
      </div>`
    )
    .join("");
  metaBox.appendChild(box);
}

/* 消费 SSE：POST + ReadableStream（EventSource 不支持 POST） */
async function sendQuestion(question) {
  appendMessage("user").querySelector(".bubble").textContent = question;
  const li = appendMessage("assistant");
  const bubble = li.querySelector(".bubble");
  const meta = li.querySelector(".msg-meta");
  bubble.textContent = "正在检索资料…";

  const mode = document.querySelector('input[name="mode"]:checked').value;
  let citations = [];
  let parts = [];
  state.busy = true;
  $("chat-send").disabled = true;
  try {
    const resp = await fetch(`${API}/kbs/${state.kbCode}/chat:stream`, {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      // 带上 conversation_id：后端据此加载历史做追问改写（“它/这个”才有指代对象）
      body: JSON.stringify({ question, mode, conversation_id: state.conversationId }),
    });
    if (!resp.ok || !resp.body) {
      let detail = `HTTP ${resp.status}`;
      try { detail = (await resp.json()).message || detail; } catch { /* 忽略 */ }
      throw new Error(detail);
    }
    bubble.textContent = "";
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        handleSseFrame(frame);
      }
    }
  } catch (err) {
    bubble.textContent = `请求失败：${err.message}`;
  } finally {
    state.busy = false;
    $("chat-send").disabled = false;
  }

  function handleSseFrame(frame) {
    const lines = frame.split("\n");
    let event = "message";
    let data = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (!data) return;
    const payload = JSON.parse(data);
    if (event === "citations") {
      citations = payload;
      meta.innerHTML = citations
        .map((c) => `<span class="route-badge">[${c.index}] ${escapeHtml(c.title)}</span>`)
        .join("");
    } else if (event === "token") {
      parts.push(payload.content);
      renderAnswer(bubble, parts.join(""), citations);
      $("messages").scrollTop = $("messages").scrollHeight;
    } else if (event === "answer") {
      /* 服务端最终清理后的正文（删除无效 [n] 后）：以它替换前端已累积的 token，
         否则界面会保留与后端、会话记录不一致的引用标记 */
      parts = [payload.content];
      renderAnswer(bubble, payload.content, citations);
    } else if (event === "validation") {
      /* 最终引用校验：以校验后的引用为准，重渲染角标并标注剔除情况 */
      citations = payload.citations;
      renderAnswer(bubble, parts.join(""), citations);
      const issues = payload.unverified_citations + payload.out_of_range_refs + payload.fact_conflicts;
      if (issues > 0) {
        meta.innerHTML += `<span class="route-badge">已剔除 ${issues} 条无效引用</span>`;
      }
      const verification = payload.verification;
      if (verification && verification.status !== "verified") {
        const label = verification.status === "rejected" ? "已拒答（结论无证据支持）" : "未验证";
        meta.innerHTML += `<span class="route-badge">${escapeHtml(label)}：结论覆盖 ${Math.round(verification.coverage * 100)}%</span>`;
      }
    } else if (event === "done") {
      // 保存会话 id：后续追问沿用同一会话历史；“新对话”按钮会清空
      if (payload.conversation_id) state.conversationId = payload.conversation_id;
      const routeNames = { document: "文档检索", sql: "结构化查询", api: "实时接口" };
      const routeLabel = routeNames[payload.route] || payload.route;
      meta.innerHTML += `<span class="route-badge route-${escapeHtml(payload.route)}">${escapeHtml(routeLabel)}</span>`;
      if (payload.degraded) {
        const banner = document.createElement("div");
        banner.className = "degrade-banner";
        banner.textContent = `数据降级提示：${payload.degrade_reason || "实时/结构化数据暂不可用，已停止本次作答"}`;
        li.insertBefore(banner, meta);
      }
      renderEvidence(meta, payload.evidence);
      $("messages").scrollTop = $("messages").scrollHeight;
    } else if (event === "error") {
      bubble.textContent = `出错了：${payload.message}`;
    }
  }
}

/* ---------------- 事件绑定 ---------------- */

function applySidebar() {
  $("sidebar-toggle").classList.toggle("active", !state.sidebarHidden);
  document.querySelector(".layout").classList.toggle("sidebar-hidden", state.sidebarHidden);
}

function bindEvents() {
  applySidebar();
  $("sidebar-toggle").addEventListener("click", () => {
    state.sidebarHidden = !state.sidebarHidden;
    localStorage.setItem("rag_sidebar_hidden", state.sidebarHidden ? "1" : "0");
    applySidebar();
  });
  $("kb-select").addEventListener("change", (event) => {
    state.kbCode = event.target.value;
    localStorage.setItem("rag_kb", state.kbCode);
    state.docPage = 1;
    // 换库等于换作用范围：旧会话的历史与引用不再适用，直接开新对话
    state.docTitles = [];
    resetConversation();
    syncKbUi();
    loadDocs().catch((err) => console.error(err));
  });
  $("new-chat").addEventListener("click", () => resetConversation());
  $("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("chat-input");
    const question = input.value.trim();
    if (!question || state.busy || !state.kbCode) return;
    input.value = "";
    sendQuestion(question);
  });
  $("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("chat-form").requestSubmit();
    }
  });
  $("upload-form").addEventListener("submit", submitUpload);
  $("doc-status-filter").addEventListener("change", () => { state.docPage = 1; loadDocs(); });
  $("refresh-docs").addEventListener("click", () => loadDocs());
  $("doc-prev").addEventListener("click", () => {
    if (state.docPage > 1) { state.docPage -= 1; loadDocs(); }
  });
  $("doc-next").addEventListener("click", () => {
    if (state.docPage * state.docSize < state.docTotal) { state.docPage += 1; loadDocs(); }
  });
  $("settings-btn").addEventListener("click", () => {
    $("token-input").value = state.token;
    $("settings-panel").classList.remove("hidden");
  });
  $("settings-close").addEventListener("click", () => $("settings-panel").classList.add("hidden"));
  $("token-save").addEventListener("click", () => {
    state.token = $("token-input").value.trim();
    localStorage.setItem("rag_token", state.token);
    $("settings-panel").classList.add("hidden");
    pollHealth();
  });
  $("token-clear").addEventListener("click", () => {
    state.token = "";
    $("token-input").value = "";
    localStorage.removeItem("rag_token");
  });
  $("settings-panel").addEventListener("click", (event) => {
    if (event.target === $("settings-panel")) $("settings-panel").classList.add("hidden");
  });
}

/* ---------------- 启动 ---------------- */

(async function init() {
  bindEvents();
  try {
    await loadKbs();
    await loadDocs();
  } catch (err) {
    console.error("初始化失败", err);
    const li = appendMessage("assistant");
    li.querySelector(".bubble").textContent = `服务初始化失败：${err.message}（请确认后端已启动、鉴权模式与令牌配置正确）`;
  }
  pollHealth();
  setInterval(pollHealth, 30000);
})();
