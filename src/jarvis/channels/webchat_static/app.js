// ═══════════════════════════════════════════════════════════════════
// J.A.R.V.I.S. HUD — Sci-Fi WebSocket Interface
// Project Prometheus | Iron Man Inspired | Living Dashboard
// ═══════════════════════════════════════════════════════════════════
(function () {
    'use strict';

    window.JARVIS = window.JARVIS || {};

    // ── DOM References ──
    const chatMessages = document.getElementById('chatMessages');
    const chatScroll = document.getElementById('chatScroll');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const statusRing = document.getElementById('statusRing');
    const statusLabel = document.getElementById('statusLabel');
    const processingOverlay = document.getElementById('processingOverlay');
    const processingText = document.getElementById('processingText');
    const activityLog = document.getElementById('activityLog');
    const toolRegistry = document.getElementById('toolRegistry');
    const hudClock = document.getElementById('hudClock');
    const welcomeHolo = document.getElementById('welcomeHolo');
    const particleCanvas = document.getElementById('particleCanvas');
    const waveformCanvas = document.getElementById('waveformCanvas');
    const channelList = document.getElementById('channelList');
    const channelIndicator = document.getElementById('channelIndicator');
    const sessionTabs = document.getElementById('sessionTabs');
    const toastContainer = document.getElementById('toastContainer');

    // Stats elements
    const statMemory = document.getElementById('statMemory');
    const statLatency = document.getElementById('statLatency');
    const statTokens = document.getElementById('statTokens');
    const statProvider = document.getElementById('statProvider');
    const statIdentity = document.getElementById('statIdentity');
    const statChannels = document.getElementById('statChannels');

    let ws = null;
    let sessionId = null;
    let reconnectAttempts = 0;
    let totalTokens = 0;
    let isProcessing = false;
    let currentStreamDiv = null;
    let streamBuffer = '';
    const MAX_RECONNECT = 10;
    const MAX_ACTIVITY_ITEMS = 50;
    const latencyHistory = [];  // Ring buffer for heartbeat visualizer

    // ═══ PARTICLE SYSTEM ═══
    const particles = [];
    const PARTICLE_COUNT = 80;
    const CONNECTION_DISTANCE = 150;
    let pCtx = null;

    function initParticles() {
        particleCanvas.width = window.innerWidth;
        particleCanvas.height = window.innerHeight;
        pCtx = particleCanvas.getContext('2d');

        for (let i = 0; i < PARTICLE_COUNT; i++) {
            particles.push({
                x: Math.random() * particleCanvas.width,
                y: Math.random() * particleCanvas.height,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                size: Math.random() * 1.5 + 0.5,
                alpha: Math.random() * 0.3 + 0.1,
            });
        }

        window.addEventListener('resize', () => {
            particleCanvas.width = window.innerWidth;
            particleCanvas.height = window.innerHeight;
        });

        animateParticles();
    }

    function animateParticles() {
        if (!pCtx) return;
        pCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = particleCanvas.width;
            if (p.x > particleCanvas.width) p.x = 0;
            if (p.y < 0) p.y = particleCanvas.height;
            if (p.y > particleCanvas.height) p.y = 0;

            pCtx.beginPath();
            pCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            pCtx.fillStyle = `rgba(0, 240, 255, ${p.alpha})`;
            pCtx.fill();

            for (let j = i + 1; j < particles.length; j++) {
                const q = particles[j];
                const dx = p.x - q.x;
                const dy = p.y - q.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < CONNECTION_DISTANCE) {
                    const alpha = (1 - dist / CONNECTION_DISTANCE) * 0.08;
                    pCtx.beginPath();
                    pCtx.moveTo(p.x, p.y);
                    pCtx.lineTo(q.x, q.y);
                    pCtx.strokeStyle = `rgba(0, 240, 255, ${alpha})`;
                    pCtx.lineWidth = 0.5;
                    pCtx.stroke();
                }
            }
        }

        if (isProcessing) {
            for (const p of particles) {
                p.vx *= 1.001;
                p.vy *= 1.001;
                if (Math.abs(p.vx) > 2) p.vx = Math.sign(p.vx) * 0.5;
                if (Math.abs(p.vy) > 2) p.vy = Math.sign(p.vy) * 0.5;
            }
        }

        requestAnimationFrame(animateParticles);
    }

    // ═══ WAVEFORM + HEARTBEAT VISUALIZER ═══
    let wCtx = null;
    let wavePhase = 0;

    function initWaveform() {
        wCtx = waveformCanvas.getContext('2d');
        animateWaveform();
    }

    function animateWaveform() {
        if (!wCtx) return;
        const w = waveformCanvas.width;
        const h = waveformCanvas.height;
        wCtx.clearRect(0, 0, w, h);

        const amplitude = isProcessing ? 20 : 5;
        const frequency = isProcessing ? 0.05 : 0.02;
        const speed = isProcessing ? 0.08 : 0.02;

        // Base ambient wave
        for (let layer = 0; layer < 3; layer++) {
            wCtx.beginPath();
            const layerOffset = layer * 0.8;
            const layerAlpha = 0.3 - layer * 0.08;
            const layerAmp = amplitude * (1 - layer * 0.2);

            for (let x = 0; x < w; x++) {
                const y = h / 2 + Math.sin((x * frequency) + wavePhase + layerOffset) * layerAmp
                    + Math.sin((x * frequency * 2.5) + wavePhase * 1.5 + layerOffset) * (layerAmp * 0.3);
                if (x === 0) wCtx.moveTo(x, y);
                else wCtx.lineTo(x, y);
            }

            wCtx.strokeStyle = `rgba(0, 240, 255, ${layerAlpha})`;
            wCtx.lineWidth = 1;
            wCtx.stroke();
        }

        // Heartbeat overlay — plot recent response latencies
        if (latencyHistory.length > 1) {
            wCtx.beginPath();
            const step = w / Math.max(latencyHistory.length - 1, 1);
            const maxLat = Math.max(...latencyHistory, 1);
            for (let i = 0; i < latencyHistory.length; i++) {
                const x = i * step;
                const normalized = Math.min(latencyHistory[i] / maxLat, 1);
                const y = h / 2 - normalized * (h * 0.35);
                if (i === 0) wCtx.moveTo(x, y);
                else wCtx.lineTo(x, y);
            }
            wCtx.strokeStyle = 'rgba(255, 176, 0, 0.4)';
            wCtx.lineWidth = 1.5;
            wCtx.stroke();
        }

        wavePhase += speed;
        requestAnimationFrame(animateWaveform);
    }

    // ═══ CLOCK ═══
    function updateClock() {
        const now = new Date();
        const h = String(now.getHours()).padStart(2, '0');
        const m = String(now.getMinutes()).padStart(2, '0');
        const s = String(now.getSeconds()).padStart(2, '0');
        hudClock.textContent = `${h}:${m}:${s}`;
    }

    // ═══ WEBSOCKET ═══
    function connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Send stored connection_id so the server reuses our identity across reconnects
        const storedCid = localStorage.getItem('jarvis_cid') || '';
        const url = `${protocol}//${location.host}/ws${storedCid ? '?cid=' + storedCid : ''}`;

        ws = new WebSocket(url);

        ws.onopen = () => {
            setStatus('connected', 'ONLINE');
            sendButton.disabled = false;
            reconnectAttempts = 0;
            logActivity('system-boot', 'Neural link established');
            if (window.JARVIS.sounds) window.JARVIS.sounds.boot();
            // Fetch channel catalog on connect
            fetchChannels();
        };

        ws.onclose = () => {
            setStatus('error', 'OFFLINE');
            sendButton.disabled = true;
            attemptReconnect();
        };

        ws.onerror = () => {
            setStatus('error', 'ERROR');
            logActivity('error-item', 'Connection error');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleMessage(data);
            } catch (e) {
                console.error('Parse error:', e);
            }
        };
    }

    function attemptReconnect() {
        if (reconnectAttempts >= MAX_RECONNECT) {
            setStatus('error', 'FAILED');
            return;
        }
        reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 15000);
        setStatus('error', 'RECONNECTING');
        logActivity('system-boot', `Reconnecting in ${(delay / 1000).toFixed(0)}s...`);
        setTimeout(connect, delay);
    }

    function setStatus(state, text) {
        statusRing.className = 'status-ring ' + state;
        statusLabel.textContent = text;
    }

    // ═══ MESSAGE HANDLING ═══
    function handleMessage(data) {
        switch (data.type) {
            case 'system':
                sessionId = data.session_id;
                // Persist connection_id so refreshes/reconnects keep the same identity
                if (data.connection_id) {
                    localStorage.setItem('jarvis_cid', data.connection_id);
                }
                logActivity('system-boot', data.content || 'System ready');
                break;

            case 'stream':
                setProcessing(false);
                if (!currentStreamDiv) {
                    removeWelcome();
                    currentStreamDiv = createStreamMessage();
                }
                streamBuffer += data.content;
                const bubble = currentStreamDiv.querySelector('.message-content');
                bubble.innerHTML = marked.parse(streamBuffer) + '<span class="stream-cursor"></span>';
                scrollToBottom();
                break;

            case 'message':
                setProcessing(false);
                // If streaming was active, replace with final render
                if (currentStreamDiv && data.role === 'assistant') {
                    currentStreamDiv.remove();
                    currentStreamDiv = null;
                    streamBuffer = '';
                }
                if (data.role === 'assistant') {
                    addAssistantMessage(data.content, data.stats);
                    if (window.JARVIS.sounds) window.JARVIS.sounds.chime();
                } else {
                    addUserMessage(data.content);
                }
                break;

            case 'widget':
                setProcessing(false);
                renderWidgetInChat(data);
                break;

            case 'thinking':
                setProcessing(true, 'ANALYZING');
                logActivity('thinking', 'Neural processing initiated');
                if (window.JARVIS.sounds) window.JARVIS.sounds.humStart();
                break;

            case 'tool_start':
                setProcessing(true, `EXECUTING: ${data.name}`);
                logActivity('tool-start', `${data.name}()`);
                updateToolRegistry(data.name, 'active');
                if (window.JARVIS.sounds) window.JARVIS.sounds.whoosh();
                break;

            case 'tool_end':
                logActivity('tool-end', `${data.name} \u2014 ${data.duration_ms}ms`);
                updateToolRegistry(data.name, 'completed');
                if (window.JARVIS.sounds) window.JARVIS.sounds.ding();
                break;

            case 'response_stats':
                updateResponseStats(data);
                break;

            case 'stats_update':
                updateDashboardStats(data.data);
                break;

            case 'typing':
                if (data.active) setProcessing(true, 'COMPOSING');
                else setProcessing(false);
                break;

            case 'error':
                setProcessing(false);
                logActivity('error-item', data.content || 'Unknown error');
                if (window.JARVIS.sounds) window.JARVIS.sounds.error();
                break;
        }
    }

    // ═══ CHAT MESSAGES ═══
    function removeWelcome() {
        if (welcomeHolo && welcomeHolo.parentNode) {
            welcomeHolo.style.opacity = '0';
            welcomeHolo.style.transform = 'scale(0.9)';
            welcomeHolo.style.transition = 'all 0.5s ease';
            setTimeout(() => { if (welcomeHolo.parentNode) welcomeHolo.remove(); }, 500);
        }
    }

    function addUserMessage(content) {
        removeWelcome();
        const div = document.createElement('div');
        div.className = 'message user';
        div.innerHTML = `
            <div class="message-avatar">USR</div>
            <div class="message-content">${escapeHtml(content)}</div>
        `;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function addAssistantMessage(content, stats) {
        removeWelcome();
        const div = document.createElement('div');
        div.className = 'message assistant';

        const msgBubble = document.createElement('div');
        msgBubble.className = 'message-content';
        msgBubble.innerHTML = marked.parse(content);

        // Highlight code blocks
        msgBubble.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });

        // Stats footer
        if (stats) {
            const statsBar = document.createElement('div');
            statsBar.className = 'response-stats';
            const parts = [];
            if (stats.elapsed_s) parts.push(`<span>${stats.elapsed_s}s</span>`);
            if (stats.tokens) parts.push(`<span>${stats.tokens} TOK</span>`);
            if (stats.tools_used) parts.push(`<span>${stats.tools_used} TOOL${stats.tools_used > 1 ? 'S' : ''}</span>`);
            if (stats.provider) parts.push(`<span>${stats.provider.toUpperCase()}</span>`);
            statsBar.innerHTML = parts.join('<span style="opacity:0.3">|</span>');
            msgBubble.appendChild(statsBar);
        }

        div.innerHTML = '<div class="message-avatar">SYS</div>';
        div.appendChild(msgBubble);
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function createStreamMessage() {
        removeWelcome();
        const div = document.createElement('div');
        div.className = 'message assistant streaming';
        div.innerHTML = '<div class="message-avatar">SYS</div><div class="message-content"></div>';
        chatMessages.appendChild(div);
        scrollToBottom();
        return div;
    }

    function renderWidgetInChat(data) {
        removeWelcome();
        if (window.JARVIS.renderWidget) {
            const widgetEl = window.JARVIS.renderWidget(data);
            if (widgetEl) {
                const wrapper = document.createElement('div');
                wrapper.className = 'message assistant widget-message';
                wrapper.innerHTML = '<div class="message-avatar">SYS</div>';
                wrapper.appendChild(widgetEl);
                chatMessages.appendChild(wrapper);
                scrollToBottom();
            }
        }
    }

    // ═══ PROCESSING STATE ═══
    function setProcessing(active, text) {
        isProcessing = active;
        if (active) {
            document.body.classList.add('processing');
            processingOverlay.classList.add('active');
            if (text) processingText.textContent = text;
        } else {
            document.body.classList.remove('processing');
            processingOverlay.classList.remove('active');
            if (window.JARVIS.sounds) window.JARVIS.sounds.humStop();
        }
    }

    // ═══ STATS UPDATES ═══
    function updateResponseStats(data) {
        if (data.elapsed_s) {
            statLatency.textContent = `${data.elapsed_s}s`;
            latencyHistory.push(data.elapsed_s);
            if (latencyHistory.length > 30) latencyHistory.shift();
        }
        if (data.tokens) {
            totalTokens += data.tokens;
            statTokens.textContent = formatNum(totalTokens);
        }
        if (data.provider) {
            statProvider.textContent = data.provider.toUpperCase().slice(0, 12);
        }
    }

    function updateDashboardStats(s) {
        if (!s) return;
        // Memory
        if (s.memory) {
            statMemory.textContent = formatNum(s.memory.total);
        }
        // Identity
        if (s.identity) {
            statIdentity.textContent = s.identity.bootstrapped ? `DAY ${s.identity.age_days}` : 'NEW';
        }
        // Channels
        if (s.channels) {
            statChannels.textContent = `${s.channels.connected}`;
            if (channelIndicator) {
                channelIndicator.className = 'panel-indicator' + (s.channels.connected > 0 ? ' active' : '');
            }
        }
        // Provider (from live stats, overrides response_stats)
        if (s.provider && s.provider !== '--') {
            statProvider.textContent = s.provider.toUpperCase().slice(0, 12);
        }
    }

    // ═══ CHANNEL CATALOG ═══
    function fetchChannels() {
        fetch('/api/channels')
            .then(r => r.json())
            .then(data => renderChannels(data))
            .catch(() => {
                if (channelList) channelList.innerHTML = '<div class="channel-item loading">Offline</div>';
            });
    }

    function renderChannels(data) {
        if (!channelList) return;
        channelList.innerHTML = '';
        const active = new Set(data.active || []);

        (data.catalog || []).forEach(ch => {
            const isActive = active.has(ch.id);
            const item = document.createElement('div');
            item.className = `channel-item ${isActive ? 'active' : 'available'}`;

            if (isActive) {
                item.innerHTML = `<span class="channel-dot">\u25CF</span>
                    <span class="channel-name">${escapeHtml(ch.name)}</span>
                    <span class="channel-status">LIVE</span>`;
            } else if (ch.builtin) {
                item.innerHTML = `<span class="channel-dot">\u25CB</span>
                    <span class="channel-name">${escapeHtml(ch.name)}</span>
                    <span class="channel-cta" data-connect="${escapeHtml(ch.name)}">ENABLE</span>`;
            } else {
                item.innerHTML = `<span class="channel-dot dim">\u25CB</span>
                    <span class="channel-name">${escapeHtml(ch.name)}</span>
                    <span class="channel-cta" data-connect="${escapeHtml(ch.name)}">ASK JARVIS</span>`;
            }
            channelList.appendChild(item);
        });

        // Click handler for channel CTAs
        channelList.addEventListener('click', (e) => {
            const cta = e.target.closest('[data-connect]');
            if (cta) {
                const name = cta.dataset.connect;
                messageInput.value = `Connect ${name}`;
                messageInput.focus();
                sendButton.disabled = false;
            }
        });
    }

    // ═══ SESSION TABS ═══
    if (sessionTabs) {
        const newTabBtn = document.getElementById('newTabBtn');
        if (newTabBtn) {
            newTabBtn.addEventListener('click', () => {
                sendCommand('/new');
            });
        }
    }

    // ═══ ACTIVITY LOG ═══
    function logActivity(type, text) {
        const now = new Date();
        const ts = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

        const item = document.createElement('div');
        item.className = `activity-item ${type}`;
        item.innerHTML = `
            <span class="activity-time">${ts}</span>
            <span class="activity-text">${escapeHtml(text)}</span>
        `;

        activityLog.appendChild(item);

        while (activityLog.children.length > MAX_ACTIVITY_ITEMS) {
            activityLog.removeChild(activityLog.firstChild);
        }

        activityLog.scrollTop = activityLog.scrollHeight;
    }

    // ═══ TOOL REGISTRY ═══
    const seenTools = new Map();

    function updateToolRegistry(name, state) {
        if (!seenTools.has(name)) {
            if (seenTools.size === 0) {
                toolRegistry.innerHTML = '';
            }

            const item = document.createElement('div');
            item.className = `tool-item ${state}`;
            item.innerHTML = `
                <span class="tool-icon">&#x2B22;</span>
                <span class="tool-name">${escapeHtml(name)}</span>
            `;
            toolRegistry.appendChild(item);
            seenTools.set(name, item);
        } else {
            const item = seenTools.get(name);
            item.className = `tool-item ${state}`;
        }
    }

    // ═══ SEND MESSAGE ═══
    function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

        addUserMessage(text);
        ws.send(JSON.stringify({ content: text }));

        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendButton.disabled = true;
        currentStreamDiv = null;
        streamBuffer = '';

        setProcessing(true, 'TRANSMITTING');
        logActivity('system-boot', 'Query transmitted');
        if (window.JARVIS.sounds) window.JARVIS.sounds.click();
    }

    function sendCommand(cmd) {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        messageInput.value = cmd;
        sendMessage();
    }

    // ═══ SHARE — COPY CHAT AS IMAGE ═══
    const shareBtn = document.getElementById('shareBtn');
    if (shareBtn) {
        shareBtn.addEventListener('click', async () => {
            if (typeof html2canvas === 'undefined') {
                showToast('html2canvas not loaded');
                return;
            }
            try {
                const canvas = await html2canvas(chatScroll, {
                    backgroundColor: '#020810',
                    scale: 2,
                });
                canvas.toBlob(blob => {
                    if (navigator.clipboard && navigator.clipboard.write) {
                        navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
                        showToast('Copied to clipboard!');
                    } else {
                        // Fallback: download
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'jarvis-chat.png';
                        a.click();
                        URL.revokeObjectURL(url);
                        showToast('Downloaded as PNG');
                    }
                });
            } catch (e) {
                showToast('Capture failed');
                console.error(e);
            }
        });
    }

    // ═══ SOUND TOGGLE ═══
    const soundToggle = document.getElementById('soundToggle');
    const soundIcon = document.getElementById('soundIcon');
    if (soundToggle && window.JARVIS.sounds) {
        updateSoundIcon();
        soundToggle.addEventListener('click', () => {
            window.JARVIS.sounds.toggle();
            updateSoundIcon();
        });
    }
    function updateSoundIcon() {
        if (soundIcon) {
            soundIcon.textContent = window.JARVIS.sounds.enabled ? '\uD83D\uDD0A' : '\uD83D\uDD07';
        }
    }

    // ═══ STAT CLICKS ═══
    document.querySelectorAll('.stat-item[data-command]').forEach(el => {
        el.style.cursor = 'pointer';
        el.addEventListener('click', () => {
            const cmd = el.dataset.command;
            if (cmd) sendCommand(cmd);
        });
    });

    // ═══ TOAST NOTIFICATIONS ═══
    function showToast(msg, duration) {
        if (!toastContainer) return;
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = msg;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 400);
        }, duration || 2500);
    }

    // ═══ UTILITIES ═══
    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatScroll.scrollTop = chatScroll.scrollHeight;
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatNum(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return String(n);
    }

    // ═══ EVENT LISTENERS ═══
    sendButton.addEventListener('click', sendMessage);

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
        sendButton.disabled = !messageInput.value.trim();
    });

    // ═══ MARKED CONFIG ═══
    marked.setOptions({
        breaks: true,
        gfm: true,
        highlight: function (code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        }
    });

    // ═══ EXPOSE ON NAMESPACE ═══
    window.JARVIS.sendCommand = sendCommand;
    window.JARVIS.addAssistantMessage = addAssistantMessage;
    window.JARVIS.logActivity = logActivity;
    window.JARVIS.scrollToBottom = scrollToBottom;
    window.JARVIS.showToast = showToast;
    window.JARVIS.setProcessing = setProcessing;
    window.JARVIS.isProcessing = () => isProcessing;

    // ═══ BOOT SEQUENCE ═══
    function boot() {
        initParticles();
        initWaveform();
        updateClock();
        setInterval(updateClock, 1000);
        connect();

        logActivity('system-boot', 'HUD interface loaded');
        logActivity('system-boot', 'Particle renderer active');
        logActivity('system-boot', 'Establishing neural link...');
    }

    boot();
})();
