const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const chatLog = document.getElementById("chat-log");
const tableTabs = document.getElementById("table-tabs");
const tableContainer = document.getElementById("table-container");

const markedParser = window.marked;
const DOMPurify = window.DOMPurify;

if (markedParser) {
  markedParser.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false,
  });
}

const SYSTEM_PROMPT =
  "You are a data analyst assistant. Provide concise answers and return tabular data as Markdown tables when applicable.";

const state = {
  history: [],
  tables: [],
  nextTableId: 1,
  activeTabId: null,
};

function renderMarkdown(text) {
  if (!markedParser || !DOMPurify) {
    return text;
  }
  return DOMPurify.sanitize(markedParser.parse(text));
}

function updateBubbleContent(bubble, content, markdown = false) {
  if (markdown) {
    bubble.innerHTML = renderMarkdown(content);
    bubble.dataset.markdown = "true";
  } else {
    bubble.textContent = content;
    bubble.dataset.markdown = "false";
  }
}

function appendMessage(role, content, { markdown = false } = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `chat-message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "user" ? "You" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  updateBubbleContent(bubble, content, markdown);

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;

  return bubble;
}

function splitRow(row) {
  return row
    .trim()
    .replace(/^\||\|$/g, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isDividerRow(rowCells) {
  return rowCells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function parseMarkdownTable(lines) {
  if (lines.length < 2) {
    return null;
  }

  const headerCells = splitRow(lines[0]);
  const dividerCells = splitRow(lines[1]);

  if (!isDividerRow(dividerCells)) {
    return null;
  }

  const rows = [];
  for (let i = 2; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) {
      continue;
    }
    const cells = splitRow(line);
    if (cells.length !== headerCells.length) {
      return null;
    }
    rows.push(cells);
  }

  if (!rows.length) {
    return null;
  }

  return {
    headers: headerCells,
    rows,
  };
}

function extractTables(markdown) {
  const tables = [];
  const lines = markdown.split(/\r?\n/);
  let buffer = [];

  const flushBuffer = () => {
    if (buffer.length >= 2) {
      const parsed = parseMarkdownTable(buffer);
      if (parsed) {
        tables.push(parsed);
      }
    }
    buffer = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      buffer.push(line);
    } else {
      flushBuffer();
    }
  }

  flushBuffer();
  return tables;
}

function renderTableView(table) {
  const tableElement = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");

  table.headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);
  tableElement.appendChild(thead);

  const tbody = document.createElement("tbody");
  table.rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  tableElement.appendChild(tbody);
  return tableElement;
}

function setActiveTab(tabId) {
  if (!tabId) return;

  const existingActiveTab = tableTabs.querySelector(".table-tab.active");
  if (existingActiveTab) {
    existingActiveTab.classList.remove("active");
  }

  const existingActiveView = tableContainer.querySelector(".table-view.active");
  if (existingActiveView) {
    existingActiveView.classList.remove("active");
  }

  const nextTab = tableTabs.querySelector(`[data-tab-id="${tabId}"]`);
  const nextView = tableContainer.querySelector(`[data-tab-id="${tabId}"]`);

  if (nextTab && nextView) {
    nextTab.classList.add("active");
    nextView.classList.add("active");
    state.activeTabId = tabId;
  }
}

function ensureTableSurface() {
  const emptyState = tableContainer.querySelector(".empty-state");
  if (emptyState) {
    emptyState.remove();
  }
}

function addTable(parsedTable) {
  ensureTableSurface();

  const id = `table-${state.nextTableId++}`;
  const label = `Table ${state.tables.length + 1}`;

  const tabButton = document.createElement("button");
  tabButton.className = "table-tab";
  tabButton.type = "button";
  tabButton.dataset.tabId = id;
  tabButton.textContent = label;
  tabButton.addEventListener("click", () => setActiveTab(id));

  tableTabs.appendChild(tabButton);

  const tableView = document.createElement("div");
  tableView.className = "table-view";
  tableView.dataset.tabId = id;
  tableView.appendChild(renderTableView(parsedTable));

  tableContainer.appendChild(tableView);

  state.tables.push({
    id,
    label,
    ...parsedTable,
  });

  setActiveTab(id);
}

async function sendMessage(event) {
  event.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  chatInput.value = "";
  chatInput.style.height = "";

  appendMessage("user", text);
  const assistantBubble = appendMessage("assistant", "...");

  sendButton.disabled = true;
  chatInput.disabled = true;

  const payloadMessages = [
    { role: "system", content: SYSTEM_PROMPT },
    ...state.history,
    { role: "user", content: text },
  ];

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages: payloadMessages }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantText = "";
    let done = false;

    while (!done) {
      const { value, done: readerDone } = await reader.read();
      done = readerDone;
      if (value) {
        assistantText += decoder.decode(value, { stream: !done });
        updateBubbleContent(assistantBubble, assistantText, true);
        chatLog.scrollTop = chatLog.scrollHeight;
      }
    }

    assistantText = assistantText.trim();
    updateBubbleContent(assistantBubble, assistantText, true);
    state.history.push({ role: "user", content: text });
    state.history.push({ role: "assistant", content: assistantText });

    const tables = extractTables(assistantText);
    tables.forEach(addTable);
  } catch (error) {
    updateBubbleContent(assistantBubble, `⚠️ ${error.message}`, false);
    console.error(error);
  } finally {
    sendButton.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
  }
}

chatForm.addEventListener("submit", sendMessage);

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

const dividerHandle = document.getElementById("divider-handle");
const workspaceGrid = document.querySelector(".workspace-grid");
const rootStyle = document.documentElement;

let resizing = false;

if (dividerHandle && workspaceGrid) {
  const stopResizing = () => {
    if (!resizing) return;
    resizing = false;
    document.body.classList.remove("resizing");
  };

  const updateChatWidth = (clientX) => {
    const rect = workspaceGrid.getBoundingClientRect();
    const minWidth = 320;
    const maxWidth = rect.width * 0.7;
    let rawWidth = clientX - rect.left;
    rawWidth = Math.max(minWidth, Math.min(rawWidth, maxWidth));
    rootStyle.style.setProperty("--chat-panel-width", `${rawWidth}px`);
  };

  dividerHandle.addEventListener("pointerdown", (event) => {
    resizing = true;
    document.body.classList.add("resizing");
    dividerHandle.setPointerCapture(event.pointerId);
  });

  dividerHandle.addEventListener("pointermove", (event) => {
    if (!resizing) return;
    updateChatWidth(event.clientX);
  });

  const handlePointerUp = (event) => {
    if (!resizing) return;
    dividerHandle.releasePointerCapture(event.pointerId);
    stopResizing();
  };

  dividerHandle.addEventListener("pointerup", handlePointerUp);
  dividerHandle.addEventListener("pointercancel", handlePointerUp);
  window.addEventListener("pointerup", stopResizing);
  window.addEventListener("blur", stopResizing);
}

