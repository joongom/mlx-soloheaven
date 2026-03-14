/**
 * SoloHeaven — Web Chat Client
 * Real-time thinking streaming + benchmark stats + session management
 */

const API = '';
let currentSessionId = null;
let isStreaming = false;
let selectedModel = null; // null = default (first model)

// DOM
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const sessionListEl = document.getElementById('session-list');
const newChatBtn = document.getElementById('new-chat-btn');
const searchInput = document.getElementById('search-input');
const headerTitle = document.getElementById('header-title');
const deleteChatBtn = document.getElementById('delete-chat-btn');
const cacheStatsText = document.getElementById('cache-stats-text');
const sidebarEl = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');

// Marked config
const SKIP_HIGHLIGHT = new Set(['mermaid', 'plantuml', 'diagram', 'math', 'latex', 'text', 'txt', 'plain', 'csv']);

marked.setOptions({
    highlight: (code, lang) => {
        if (lang && SKIP_HIGHLIGHT.has(lang.toLowerCase())) return code;
        if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
    gfm: true,
});

// Fix CJK bold: **text**한글 — marked fails because no word boundary after **
function fixCjkMarkdown(text) {
    return text.replace(/\*\*(.+?)\*\*(?=[\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff])/g, '<strong>$1</strong>');
}

// Fix indented code fences inside lists
function fixIndentedCodeFences(text) {
    return text.replace(
        /^([ \t]*(?:[*\-+]|\d+\.)\s+)?[ \t]*(```\w*)\n([\s\S]*?)^[ \t]*(```)\s*$/gm,
        (match, listMarker, openFence, content, closeFence) => {
            const lines = content.split('\n');
            const minIndent = lines
                .filter(l => l.trim().length > 0)
                .reduce((min, l) => {
                    const indent = l.match(/^[ \t]*/)[0].length;
                    return Math.min(min, indent);
                }, Infinity);
            const dedented = lines.map(l => l.slice(minIndent === Infinity ? 0 : minIndent)).join('\n');
            return '\n' + openFence + '\n' + dedented + closeFence + '\n';
        }
    );
}

function renderMarkdown(text) {
    const html = marked.parse(fixIndentedCodeFences(fixCjkMarkdown(text || '')));
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    tmp.querySelectorAll('pre code').forEach(el => {
        const cls = el.className || '';
        const langMatch = cls.match(/language-(\w+)/);
        if (langMatch) {
            el.setAttribute('data-lang', langMatch[1]);
            if (SKIP_HIGHLIGHT.has(langMatch[1].toLowerCase())) return;
        }
        if (!el.classList.contains('hljs')) hljs.highlightElement(el);
    });
    return tmp.innerHTML;
}

// Cursor is now pure CSS (::after on last child) — no JS injection needed

// ===== SESSIONS =====

let allSessions = [];

async function loadSessions() {
    const res = await fetch(`${API}/api/sessions`);
    allSessions = await res.json();
    renderSessionList(allSessions);
    return allSessions;
}

function renderSessionList(sessions) {
    const q = searchInput.value.toLowerCase();
    const filtered = q ? sessions.filter(s => s.title.toLowerCase().includes(q)) : sessions;

    sessionListEl.innerHTML = filtered.map(s => {
        const time = new Date(s.updated_at * 1000).toLocaleString('en-US', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        return `
            <div class="session-item ${s.id === currentSessionId ? 'active' : ''}"
                 onclick="switchSession('${s.id}')">
                <div class="session-title-row">
                    <span class="session-title">${esc(s.title || 'New Chat')}</span>
                    <button class="session-delete-btn" onclick="event.stopPropagation(); deleteSession('${s.id}')" title="Delete">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
                    </button>
                </div>
                <span class="session-time">${time}</span>
            </div>`;
    }).join('');
}

searchInput.addEventListener('input', () => renderSessionList(allSessions));

async function createSession() {
    const res = await fetch(`${API}/api/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'New Chat' }),
    });
    const session = await res.json();
    currentSessionId = session.id;
    headerTitle.textContent = 'New Chat';
    messagesEl.innerHTML = emptyState();
    await loadSessions();
    if (isMobile()) closeSidebar();
}

async function switchSession(id) {
    if (isStreaming) return;
    currentSessionId = id;
    const session = allSessions.find(s => s.id === id);
    headerTitle.textContent = session?.title || 'Chat';
    await loadMessages(id);
    renderSessionList(allSessions);
    if (isMobile()) closeSidebar();
}

async function deleteCurrentSession() {
    if (!currentSessionId || isStreaming) return;
    await deleteSession(currentSessionId);
}

async function deleteSession(id) {
    if (isStreaming) return;
    if (!confirm('Delete this chat?')) return;
    await fetch(`${API}/api/sessions/${id}`, { method: 'DELETE' });
    if (currentSessionId === id) {
        currentSessionId = null;
        messagesEl.innerHTML = emptyState();
        headerTitle.textContent = 'New Chat';
    }
    await loadSessions();
}

async function loadMessages(sessionId) {
    const res = await fetch(`${API}/api/sessions/${sessionId}/messages`);
    const messages = await res.json();
    if (messages.length === 0) {
        messagesEl.innerHTML = emptyState();
        return;
    }
    const lastIdx = messages.length - 1;
    messagesEl.innerHTML = messages.map((m, i) => {
        if (m.role === 'user') return userMsgHtml(m.content);
        if (m.role === 'assistant') return assistantMsgHtml(m.content, m.thinking, m.stats, i, i === lastIdx);
        return '';
    }).join('');
    scrollBottom();
}

// ===== SEND MESSAGE =====

let _sending = false;
async function sendMessage() {
    const content = inputEl.value.trim();
    if (!content || isStreaming || _sending) return;
    _sending = true;

    if (!currentSessionId) {
        const res = await fetch(`${API}/api/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: 'New Chat' }),
        });
        const session = await res.json();
        currentSessionId = session.id;
        headerTitle.textContent = 'New Chat';
        loadSessions();  // refresh sidebar (non-blocking)
    }

    const empty = messagesEl.querySelector('.empty-state');
    if (empty) empty.remove();

    messagesEl.insertAdjacentHTML('beforeend', userMsgHtml(content));
    inputEl.value = '';
    autoResize();
    scrollBottom();

    const msgEl = document.createElement('div');
    msgEl.className = 'msg msg-assistant streaming';
    msgEl.innerHTML = `
        <div class="msg-header">Assistant</div>
        <div class="thinking-block streaming" id="live-thinking">
            <div class="thinking-header" onclick="toggleThink(this)">
                <div class="thinking-label">
                    <span class="icon open">&#9654;</span>
                    Thinking...
                </div>
                <span class="thinking-tokens" id="live-think-count"></span>
            </div>
            <div class="thinking-body show" id="live-think-body"></div>
        </div>
        <div class="msg-body" id="live-content"></div>
    `;
    messagesEl.appendChild(msgEl);
    scrollBottom();

    const tpsEl = document.createElement('div');
    tpsEl.className = 'tps-live visible';
    tpsEl.innerHTML = '<span class="tps-num" id="live-tps">0</span> tok/s';
    document.body.appendChild(tpsEl);

    isStreaming = true;
    sendBtn.disabled = true;

    const thinkBlock = document.getElementById('live-thinking');
    const thinkBody = document.getElementById('live-think-body');
    const thinkCount = document.getElementById('live-think-count');
    const contentEl = document.getElementById('live-content');
    const liveTps = document.getElementById('live-tps');

    let fullText = '';
    let thinkingDone = false;
    let thinkText = '';
    let respText = '';
    let tokenCount = 0;
    let stats = null;

    try {
        const res = await fetch(`${API}/api/sessions/${currentSessionId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, stream: true, model: selectedModel }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const raw = line.slice(6);
                if (raw === '[DONE]') continue;

                let event;
                try { event = JSON.parse(raw); } catch { continue; }

                if (event.type === 'start') continue;

                if (event.type === 'queued') {
                    thinkBlock.querySelector('.thinking-label').innerHTML =
                        '<span class="icon open">&#9654;</span> Waiting...';
                    thinkBody.innerHTML = `<p style="color:var(--yellow)">${esc(event.message)}</p>`;
                    continue;
                }

                if (event.type === 'text') {
                    tokenCount++;
                    fullText += event.content;

                    if (event.tps > 0) liveTps.textContent = event.tps.toFixed(1);

                    if (!thinkingDone) {
                        const endIdx = fullText.indexOf('</think>');
                        if (endIdx !== -1) {
                            thinkingDone = true;
                            thinkText = fullText.substring(0, endIdx);
                            respText = fullText.substring(endIdx + 8).trimStart();

                            thinkBody.innerHTML = renderMarkdown(thinkText);
                            thinkCount.textContent = `${countTokens(thinkText)} tok`;
                            thinkBlock.classList.remove('streaming');
                            thinkBody.classList.remove('show');
                            thinkBlock.querySelector('.icon').classList.remove('open');
                            thinkBlock.querySelector('.thinking-label').innerHTML =
                                '<span class="icon">&#9654;</span> Thinking';

                            contentEl.innerHTML = renderMarkdown(respText);
                        } else {
                            thinkText = fullText;
                            thinkBody.innerHTML = renderMarkdown(thinkText);
                            thinkCount.textContent = `${countTokens(thinkText)} tok`;
                            thinkBody.scrollTop = thinkBody.scrollHeight;
                        }
                    } else {
                        respText += event.content;
                        contentEl.innerHTML = renderMarkdown(respText);
                    }
                    scrollBottom();
                }

                if (event.type === 'done') {
                    stats = event.stats;
                    if (event.content) {
                        respText = event.content;
                        contentEl.innerHTML = renderMarkdown(respText);
                    }
                    if (event.thinking) {
                        thinkText = event.thinking;
                        thinkBody.innerHTML = renderMarkdown(thinkText);
                    }
                }
            }
        }
    } catch (err) {
        contentEl.innerHTML = `<span style="color:#ef4444">Error: ${esc(err.message)}</span>`;
    }

    msgEl.classList.remove('streaming');
    tpsEl.remove();

    if (!thinkingDone && thinkText && !respText) {
        thinkBlock.classList.remove('streaming');
        thinkBody.classList.remove('show');
        thinkBlock.querySelector('.icon').classList.remove('open');
    }

    if (!thinkText) thinkBlock.style.display = 'none';

    thinkBlock.removeAttribute('id');
    thinkBody.removeAttribute('id');
    thinkCount.removeAttribute('id');
    contentEl.removeAttribute('id');

    if (stats) {
        msgEl.insertAdjacentHTML('beforeend', buildStatsBar(stats));
    }

    // Add action buttons (Branch + Regenerate + Delete)
    msgEl.insertAdjacentHTML('beforeend', `<div class="msg-actions">
        <button class="msg-action-btn" onclick="branchFromEl(this)" title="Branch from here">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="6" y1="3" x2="6" y2="15"></line><circle cx="18" cy="6" r="3"></circle>
                <circle cx="6" cy="18" r="3"></circle><path d="M18 9a9 9 0 0 1-9 9"></path>
            </svg> Branch</button>
        <button class="msg-action-btn" onclick="regenerate()" title="Regenerate response">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg> Regenerate</button>
        <button class="msg-action-btn msg-action-delete" onclick="deleteLast()" title="Delete last message">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg> Delete</button>
    </div>`);

    isStreaming = false;
    _sending = false;
    sendBtn.disabled = false;
    scrollBottom();
    loadSessions();
    loadCacheStats();
}

// ===== RENDER HELPERS =====

function emptyState() {
    return `<div class="empty-state">
        <div class="empty-logo">SoloHeaven</div>
        <p>Single-user LLM with KV Cache Optimization</p>
    </div>`;
}

function userMsgHtml(content) {
    return `<div class="msg msg-user">
        <div class="msg-header">You</div>
        <div class="msg-body">${esc(content)}</div>
    </div>`;
}

function assistantMsgHtml(content, thinking, stats, msgIndex, isLast) {
    let html = '<div class="msg msg-assistant"><div class="msg-header">Assistant</div>';
    if (thinking) {
        html += `<div class="thinking-block">
            <div class="thinking-header" onclick="toggleThink(this)">
                <div class="thinking-label"><span class="icon">&#9654;</span> Thinking</div>
                <span class="thinking-tokens">${countTokens(thinking)} tok</span>
            </div>
            <div class="thinking-body">${renderMarkdown(thinking)}</div>
        </div>`;
    }
    html += `<div class="msg-body">${renderMarkdown(content || '')}</div>`;
    if (stats) html += buildStatsBar(stats);
    if (msgIndex !== undefined) {
        html += `<div class="msg-actions">`;
        html += `<button class="msg-action-btn" onclick="branchFrom(${msgIndex})" title="Branch from here">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="6" y1="3" x2="6" y2="15"></line><circle cx="18" cy="6" r="3"></circle>
                <circle cx="6" cy="18" r="3"></circle><path d="M18 9a9 9 0 0 1-9 9"></path>
            </svg> Branch</button>`;
        if (isLast) {
            html += `<button class="msg-action-btn" onclick="regenerate()" title="Regenerate response">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                </svg> Regenerate</button>`;
            html += `<button class="msg-action-btn msg-action-delete" onclick="deleteLast()" title="Delete last message">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg> Delete</button>`;
        }
        html += `</div>`;
    }
    html += '</div>';
    return html;
}

function buildStatsBar(stats) {
    const ci = stats.cache_info || {};
    let cacheTag;
    if (stats.cache_hit) {
        const detail = ci.cached_tokens ? ` — ${ci.cached_tokens} tokens reused` : '';
        cacheTag = `<span class="stat-cache-hit">KV CACHE HIT${detail}</span>`;
    } else {
        const detail = ci.detail || 'No cache';
        cacheTag = `<span class="stat-cache-miss">MISS</span> <span class="stat-cache-detail">${esc(detail)}</span>`;
    }
    const queueTag = stats.queue_wait > 0
        ? `<span class="stat-item"><span class="stat-label">Queue</span> <span class="stat-value" style="color:var(--yellow)">${stats.queue_wait}s</span></span>`
        : '';
    const buildTag = stats.build_time > 0
        ? `<span class="stat-item"><span class="stat-label">Cache Build</span> <span class="stat-value" style="color:var(--accent)">${stats.build_time}s</span></span>`
        : '';
    return `<div class="stats-bar">
        ${cacheTag}
        ${buildTag}
        <span class="stat-item"><span class="stat-label">TTFT</span> <span class="stat-value">${stats.ttft}s</span></span>
        ${queueTag}
        <span class="stat-item"><span class="stat-label">TPS</span> <span class="stat-value">${stats.gen_tps}</span></span>
        <span class="stat-item"><span class="stat-label">Prompt</span> <span class="stat-value">${stats.prompt_tokens}</span></span>
        <span class="stat-item"><span class="stat-label">Gen</span> <span class="stat-value">${stats.completion_tokens}</span></span>
        <span class="stat-item"><span class="stat-label">Time</span> <span class="stat-value">${stats.total_time}s</span></span>
    </div>`;
}

function toggleThink(headerEl) {
    const icon = headerEl.querySelector('.icon');
    const body = headerEl.nextElementSibling;
    icon.classList.toggle('open');
    body.classList.toggle('show');
}

// ===== BRANCH & REGENERATE =====

function branchFromEl(btnEl) {
    // Find message index from DOM position
    const msgEl = btnEl.closest('.msg');
    const allMsgs = Array.from(messagesEl.querySelectorAll('.msg'));
    const msgIndex = allMsgs.indexOf(msgEl);
    if (msgIndex >= 0) branchFrom(msgIndex);
}

async function branchFrom(msgIndex) {
    if (!currentSessionId || isStreaming) return;
    if (!confirm('Create a branch from this point?')) return;
    try {
        const res = await fetch(`${API}/api/sessions/${currentSessionId}/branch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ turn: msgIndex + 1 }),
        });
        if (!res.ok) { alert('Branch failed: ' + res.statusText); return; }
        const data = await res.json();
        await loadSessions();
        await switchSession(data.session_id);
    } catch (err) { alert('Branch failed: ' + err.message); }
}

async function regenerate() {
    if (!currentSessionId || isStreaming) return;
    try {
        const msgs = await (await fetch(`${API}/api/sessions/${currentSessionId}/messages`)).json();
        const last = msgs[msgs.length - 1];
        if (!last || last.role !== 'assistant') return;
        const userMsg = msgs[msgs.length - 2];
        if (!userMsg || userMsg.role !== 'user') return;
        const userContent = userMsg.content;

        await fetch(`${API}/api/sessions/${currentSessionId}/regenerate`, { method: 'POST' });

        // Remove last assistant + user from UI
        const allMsgs = messagesEl.querySelectorAll('.msg');
        if (allMsgs.length >= 2) {
            allMsgs[allMsgs.length - 1].remove();
            allMsgs[allMsgs.length - 2].remove();
        }

        // Re-send through normal flow
        inputEl.value = userContent;
        await sendMessage();
    } catch (err) { alert('Regenerate failed: ' + err.message); }
}

async function deleteLast() {
    if (!currentSessionId || isStreaming) return;
    try {
        const res = await fetch(`${API}/api/sessions/${currentSessionId}/delete-last`, {
            method: 'POST',
        });
        if (!res.ok) { alert('Delete failed: ' + res.statusText); return; }

        // Reload messages (button state, last-message markers)
        await loadMessages(currentSessionId);
    } catch (err) { alert('Delete failed: ' + err.message); }
}

// ===== UTILITIES =====

function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function esc(text) {
    const d = document.createElement('div');
    d.textContent = text || '';
    return d.innerHTML;
}

function countTokens(text) {
    return Math.round((text || '').length / 3.5);
}

function autoResize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + 'px';
}

async function loadCacheStats() {
    try {
        const r = await fetch(`${API}/api/cache/stats`);
        const s = await r.json();
        const sessions = s.sessions || {};
        const cached = Object.keys(sessions).length;
        const totalTokens = Object.values(sessions).reduce((sum, v) => sum + (v.cache_tokens || 0), 0);
        const tokensK = (totalTokens / 1000).toFixed(1);
        cacheStatsText.innerHTML =
            `${s.active_sessions || 0} sessions &middot; ${cached} cached &middot; ${tokensK}K tokens`;
    } catch { cacheStatsText.textContent = ''; }
}

// ===== SIDEBAR (mobile) =====

function isMobile() { return window.matchMedia('(max-width: 768px)').matches; }

function openSidebar() {
    sidebarEl.classList.add('open');
    sidebarBackdrop.classList.add('visible');
    document.body.style.overflow = 'hidden';
}

function closeSidebar() {
    sidebarEl.classList.remove('open');
    sidebarBackdrop.classList.remove('visible');
    document.body.style.overflow = '';
}

// ===== EVENT LISTENERS =====

sendBtn.addEventListener('click', (e) => {
    e.preventDefault();
    sendMessage();
});
inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        sendBtn.blur();
        sendMessage();
    }
});
inputEl.addEventListener('input', autoResize);
newChatBtn.addEventListener('click', createSession);
deleteChatBtn.addEventListener('click', deleteCurrentSession);

sidebarToggle.addEventListener('click', () => {
    sidebarEl.classList.contains('open') ? closeSidebar() : openSidebar();
});
sidebarBackdrop.addEventListener('click', closeSidebar);

// Touch gestures for sidebar
let touchStartX = 0, touchStartY = 0, sidebarTouching = false;
sidebarEl.addEventListener('touchstart', (e) => {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
    sidebarTouching = true;
}, { passive: true });
sidebarEl.addEventListener('touchmove', (e) => {
    if (!sidebarTouching) return;
    const dx = e.touches[0].clientX - touchStartX;
    const dy = Math.abs(e.touches[0].clientY - touchStartY);
    if (dx < -30 && dy < 50) { closeSidebar(); sidebarTouching = false; }
}, { passive: true });
sidebarEl.addEventListener('touchend', () => { sidebarTouching = false; }, { passive: true });

document.addEventListener('touchstart', (e) => {
    if (e.touches[0].clientX < 20 && isMobile() && !sidebarEl.classList.contains('open')) {
        touchStartX = e.touches[0].clientX;
        sidebarTouching = true;
    }
}, { passive: true });
document.addEventListener('touchmove', (e) => {
    if (!sidebarTouching || !isMobile()) return;
    if (e.touches[0].clientX - touchStartX > 50) { openSidebar(); sidebarTouching = false; }
}, { passive: true });

window.addEventListener('resize', () => {
    if (!isMobile() && sidebarEl.classList.contains('open')) closeSidebar();
});

// ===== INIT =====

const modelSelect = document.getElementById('model-select');
modelSelect.addEventListener('change', () => {
    selectedModel = modelSelect.value || null;
});

(async () => {
    // Load model list into dropdown
    try {
        const r = await fetch(`${API}/v1/models`);
        const models = await r.json();
        if (models.data && models.data.length > 0) {
            modelSelect.innerHTML = models.data.map((m, i) =>
                `<option value="${esc(m.id)}">${esc(m.id)}</option>`
            ).join('');
            selectedModel = models.data[0].id;
        }
    } catch { /* ignore */ }

    const sessions = await loadSessions();
    if (sessions.length > 0) {
        await switchSession(sessions[0].id);
    } else {
        messagesEl.innerHTML = emptyState();
    }
    loadCacheStats();
})();

// Settings Sidebar Logic
const settingsSidebar = document.getElementById('settings-sidebar');
const settingsToggle = document.getElementById('settings-toggle');
const closeSettings = document.getElementById('close-settings');
const saveSettingsBtn = document.getElementById('save-settings-btn');

settingsToggle.addEventListener('click', () => {
    settingsSidebar.classList.toggle('open');
    if (settingsSidebar.classList.contains('open')) loadSessionSettings();
});
closeSettings.addEventListener('click', () => settingsSidebar.classList.remove('open'));

async function loadSessionSettings() {
    if (!currentSessionId) return;
    const res = await fetch(`${API}/api/sessions/${currentSessionId}/settings`);
    const s = await res.json();
    document.getElementById('system-prompt-editor').value = s.system_prompt || '';
    document.getElementById('temp-range').value = s.temperature || 0.6;
    document.getElementById('temp-val').innerText = s.temperature || 0.6;
    document.getElementById('thinking-budget').value = s.thinking_budget || 8192;
    document.getElementById('max-tokens').value = s.max_tokens || 32768;
    document.getElementById('context-limit').value = s.context_window_limit || 100000;
}

document.getElementById('temp-range').addEventListener('input', (e) => {
    document.getElementById('temp-val').innerText = e.target.value;
});

saveSettingsBtn.addEventListener('click', async () => {
    if (!currentSessionId) return;
    const body = {
        system_prompt: document.getElementById('system-prompt-editor').value,
        temperature: parseFloat(document.getElementById('temp-range').value),
        thinking_budget: parseInt(document.getElementById('thinking-budget').value),
        max_tokens: parseInt(document.getElementById('max-tokens').value),
        context_window_limit: parseInt(document.getElementById('context-limit').value)
    };
    const res = await fetch(`${API}/api/sessions/${currentSessionId}/settings`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (res.ok) {
        alert('Settings saved!');
        settingsSidebar.classList.remove('open');
    }
});
