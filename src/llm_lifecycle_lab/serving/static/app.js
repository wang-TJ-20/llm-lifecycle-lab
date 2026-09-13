"use strict";

const ui = {
  apiStatus: document.querySelector("#api-status"),
  chatMode: document.querySelector("#chat-mode"),
  completionMode: document.querySelector("#completion-mode"),
  composer: document.querySelector("#composer"),
  composerStatus: document.querySelector("#composer-status"),
  conversationTitle: document.querySelector("#conversation-title"),
  emptyState: document.querySelector("#empty-state"),
  maxTokens: document.querySelector("#max-tokens"),
  messageTemplate: document.querySelector("#message-template"),
  modeKicker: document.querySelector("#mode-kicker"),
  modelLabel: document.querySelector("#model-label"),
  prompt: document.querySelector("#prompt"),
  resetButton: document.querySelector("#reset-button"),
  seed: document.querySelector("#seed"),
  sendButton: document.querySelector("#send-button"),
  stageLabel: document.querySelector("#stage-label"),
  statusDot: document.querySelector("#status-dot"),
  systemPrompt: document.querySelector("#system-prompt"),
  systemSection: document.querySelector("#system-section"),
  temperature: document.querySelector("#temperature"),
  temperatureValue: document.querySelector("#temperature-value"),
  topP: document.querySelector("#top-p"),
  topPValue: document.querySelector("#top-p-value"),
  transcript: document.querySelector("#transcript"),
};

const state = {
  busy: false,
  messages: [],
  mode: "chat",
  model: null,
  stage: null,
};

function setConnection(status, detail) {
  ui.statusDot.classList.toggle("is-online", status === "online");
  ui.statusDot.classList.toggle("is-error", status === "error");
  ui.apiStatus.textContent = detail;
  ui.sendButton.disabled = status !== "online" || state.busy;
}

function resizeComposer() {
  ui.prompt.style.height = "auto";
  ui.prompt.style.height = `${Math.min(ui.prompt.scrollHeight, 180)}px`;
}

function resetConversation() {
  state.messages = [];
  ui.transcript.replaceChildren(ui.emptyState);
  ui.emptyState.hidden = false;
  ui.conversationTitle.textContent = "新会话";
  ui.prompt.value = "";
  resizeComposer();
  ui.prompt.focus();
}

function setMode(mode) {
  if (!["chat", "completion"].includes(mode)) return;
  const changed = state.mode !== mode;
  state.mode = mode;
  ui.chatMode.classList.toggle("is-active", mode === "chat");
  ui.completionMode.classList.toggle("is-active", mode === "completion");
  ui.chatMode.setAttribute("aria-pressed", String(mode === "chat"));
  ui.completionMode.setAttribute("aria-pressed", String(mode === "completion"));
  ui.systemSection.hidden = mode !== "chat";
  ui.modeKicker.textContent =
    mode === "chat" ? "CHAT SESSION" : "TEXT COMPLETION";
  ui.prompt.placeholder = mode === "chat" ? "输入消息" : "输入续写前缀";
  if (changed) resetConversation();
}

function appendMessage(role, text, usage = "") {
  ui.emptyState.hidden = true;
  const node = ui.messageTemplate.content.firstElementChild.cloneNode(true);
  node.dataset.role = role;
  node.querySelector(".message-role").textContent =
    role === "user" ? "YOU" : role === "assistant" ? "MODEL" : "ERROR";
  node.querySelector(".message-usage").textContent = usage;
  node.querySelector(".message-content").textContent = text;
  const copy = node.querySelector(".copy-button");
  if (role === "error") {
    copy.remove();
  } else {
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
        copy.textContent = "✓";
        window.setTimeout(() => {
          copy.textContent = "⧉";
        }, 900);
      } catch {
        copy.title = "复制失败";
      }
    });
  }
  ui.transcript.append(node);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
  return node;
}

function appendPending() {
  const node = appendMessage("assistant", "");
  node.classList.add("is-pending");
  node.querySelector(".copy-button").remove();
  return node;
}

function generationOptions() {
  const maxTokens = Number.parseInt(ui.maxTokens.value, 10);
  const seed = Number.parseInt(ui.seed.value, 10);
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new Error("Max tokens 必须是正整数");
  }
  if (!Number.isInteger(seed) || seed < 0) {
    throw new Error("Seed 必须是非负整数");
  }
  return {
    max_tokens: maxTokens,
    temperature: Number.parseFloat(ui.temperature.value),
    top_p: Number.parseFloat(ui.topP.value),
    seed,
  };
}

async function requestCompletion(prompt) {
  const body = {
    model: state.model,
    ...generationOptions(),
  };
  let endpoint;
  if (state.mode === "chat") {
    endpoint = "/v1/chat/completions";
    const system = ui.systemPrompt.value.trim();
    body.messages = [
      ...(system ? [{ role: "system", content: system }] : []),
      ...state.messages,
    ];
  } else {
    endpoint = "/v1/completions";
    body.prompt = prompt;
  }
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload.error?.message || `HTTP ${response.status}`);
    error.requestId = response.headers.get("X-Request-ID");
    throw error;
  }
  return payload;
}

async function submitPrompt(event) {
  event.preventDefault();
  if (state.busy || !state.model) return;
  const prompt = ui.prompt.value.trim();
  if (!prompt) return;

  let options;
  try {
    options = generationOptions();
  } catch (error) {
    appendMessage("error", error.message);
    return;
  }
  void options;

  state.busy = true;
  ui.sendButton.disabled = true;
  ui.resetButton.disabled = true;
  ui.composerStatus.textContent = "LOCAL · GENERATING";
  if (state.messages.length === 0) {
    ui.conversationTitle.textContent =
      prompt.length > 42 ? `${prompt.slice(0, 42)}…` : prompt;
  }
  const userMessage = { role: "user", content: prompt };
  state.messages.push(userMessage);
  appendMessage("user", prompt);
  ui.prompt.value = "";
  resizeComposer();
  const pending = appendPending();

  try {
    const response = await requestCompletion(prompt);
    const choice = response.choices[0];
    const text =
      state.mode === "chat" ? choice.message.content : choice.text;
    state.messages.push({ role: "assistant", content: text });
    pending.remove();
    appendMessage(
      "assistant",
      text,
      `${response.usage.prompt_tokens} IN · ${response.usage.completion_tokens} OUT`,
    );
    ui.composerStatus.textContent = `LOCAL · ${choice.finish_reason.toUpperCase()}`;
  } catch (error) {
    if (state.messages.at(-1) === userMessage) state.messages.pop();
    pending.remove();
    const suffix = error.requestId ? `\n${error.requestId}` : "";
    appendMessage("error", `${error.message}${suffix}`);
    ui.composerStatus.textContent = "LOCAL · ERROR";
  } finally {
    state.busy = false;
    ui.sendButton.disabled = false;
    ui.resetButton.disabled = false;
    ui.prompt.focus();
  }
}

async function boot() {
  setConnection("connecting", "连接中");
  try {
    const [modelsResponse, healthResponse] = await Promise.all([
      fetch("/v1/models"),
      fetch("/health"),
    ]);
    if (!modelsResponse.ok || !healthResponse.ok) {
      throw new Error("服务状态不可用");
    }
    const models = await modelsResponse.json();
    const health = await healthResponse.json();
    state.model = models.data[0].id;
    state.stage = health.stage;
    ui.modelLabel.textContent = state.model;
    ui.stageLabel.textContent = state.stage.toUpperCase();
    setMode(state.stage === "pretrain" ? "completion" : "chat");
    setConnection("online", "服务在线");
  } catch (error) {
    ui.stageLabel.textContent = "OFFLINE";
    ui.modelLabel.textContent = "Connection failed";
    setConnection("error", error.message);
  }
}

ui.chatMode.addEventListener("click", () => setMode("chat"));
ui.completionMode.addEventListener("click", () => setMode("completion"));
ui.resetButton.addEventListener("click", resetConversation);
ui.composer.addEventListener("submit", submitPrompt);
ui.prompt.addEventListener("input", () => {
  resizeComposer();
  ui.sendButton.disabled = state.busy || !state.model || !ui.prompt.value.trim();
});
ui.prompt.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    ui.composer.requestSubmit();
  }
});
ui.temperature.addEventListener("input", () => {
  ui.temperatureValue.textContent = Number(ui.temperature.value).toFixed(1);
});
ui.topP.addEventListener("input", () => {
  ui.topPValue.textContent = Number(ui.topP.value).toFixed(1);
});

resizeComposer();
boot();
