// ═══════════════════════════════════════════════════════════════════
// J.A.R.V.I.S. Widget Renderer — Generative UI for Rich Responses
// Renders structured data as interactive HUD-themed components
// ═══════════════════════════════════════════════════════════════════
(function () {
    'use strict';
    window.PREDACORE = window.PREDACORE || {};

    function el(tag, cls, html) {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (html) e.innerHTML = html;
        return e;
    }

    function esc(text) {
        const d = document.createElement('div');
        d.textContent = text;
        return d.innerHTML;
    }

    // ── Widget Renderers ──

    function renderKeyValue(data) {
        const card = el('div', 'widget-card');
        if (data.title) card.innerHTML = `<div class="widget-title">${esc(data.title)}</div>`;
        const list = el('div', 'widget-kv-list');
        (data.items || []).forEach(item => {
            const color = item.color || 'var(--hud-cyan)';
            list.innerHTML += `<div class="widget-kv-row">
                <span class="widget-kv-key">${esc(item.key)}</span>
                <span class="widget-kv-val" style="color:${color}">${esc(String(item.value))}</span>
            </div>`;
        });
        card.appendChild(list);
        return card;
    }

    function renderBarChart(data) {
        const card = el('div', 'widget-card');
        if (data.title) card.innerHTML = `<div class="widget-title">${esc(data.title)}</div>`;
        const maxVal = Math.max(...(data.bars || []).map(b => b.value), 1);
        const chart = el('div', 'widget-bar-chart');
        (data.bars || []).forEach(bar => {
            const pct = Math.round((bar.value / maxVal) * 100);
            const color = bar.color || 'var(--hud-cyan)';
            chart.innerHTML += `<div class="widget-bar-row">
                <span class="widget-bar-label">${esc(bar.label)}</span>
                <div class="widget-bar-track">
                    <div class="widget-bar-fill" style="width:${pct}%;background:${color}"></div>
                </div>
                <span class="widget-bar-value">${bar.display || bar.value}</span>
            </div>`;
        });
        card.appendChild(chart);
        return card;
    }

    function renderPipeline(data) {
        const card = el('div', 'widget-card widget-pipeline');
        card.innerHTML = '<div class="widget-title">PIPELINE EXECUTION</div>';
        const steps = el('div', 'widget-pipeline-steps');
        (data.steps || []).forEach((step, i) => {
            let icon = '\u25CB'; // empty circle
            let cls = 'pending';
            if (step.status === 'completed') { icon = '\u2713'; cls = 'completed'; }
            else if (step.status === 'running') { icon = '\u25CF'; cls = 'running'; }
            else if (step.status === 'failed') { icon = '\u2717'; cls = 'failed'; }
            else if (step.status === 'paused') { icon = '\u23F8'; cls = 'paused'; }
            const duration = step.duration_ms ? `${step.duration_ms}ms` : '';
            steps.innerHTML += `<div class="widget-pipeline-step ${cls}">
                <span class="step-icon">${icon}</span>
                <span class="step-name">${esc(step.name || `Step ${i + 1}`)}</span>
                <span class="step-time">${duration}</span>
            </div>`;
        });
        card.appendChild(steps);
        return card;
    }

    function renderChannelList(data) {
        const card = el('div', 'widget-card');
        card.innerHTML = '<div class="widget-title">CHANNEL STATUS</div>';
        const list = el('div', 'widget-channel-grid');
        (data.channels || []).forEach(ch => {
            const active = ch.active;
            list.innerHTML += `<div class="widget-channel-item ${active ? 'active' : 'inactive'}">
                <span class="widget-channel-dot">${active ? '\u25CF' : '\u25CB'}</span>
                <span class="widget-channel-name">${esc(ch.name)}</span>
            </div>`;
        });
        card.appendChild(list);
        return card;
    }

    function renderIdentityCard(data) {
        const card = el('div', 'widget-card widget-identity');
        const traits = (data.traits || []).map(t => `<span class="widget-trait">${esc(t)}</span>`).join('');
        card.innerHTML = `
            <div class="widget-title">IDENTITY</div>
            <div class="widget-identity-header">
                <span class="widget-identity-name">${esc(data.name || 'J.A.R.V.I.S.')}</span>
                <span class="widget-identity-age">DAY ${data.age_days || 0}</span>
            </div>
            ${traits ? `<div class="widget-traits">${traits}</div>` : ''}
            <div class="widget-kv-list">
                <div class="widget-kv-row">
                    <span class="widget-kv-key">Status</span>
                    <span class="widget-kv-val" style="color:var(--hud-green)">${data.bootstrapped ? 'ACTIVE' : 'BOOTSTRAP PENDING'}</span>
                </div>
                <div class="widget-kv-row">
                    <span class="widget-kv-key">Journal entries</span>
                    <span class="widget-kv-val">${data.journal_entries || 0}</span>
                </div>
            </div>`;
        return card;
    }

    function renderSessionList(data) {
        const card = el('div', 'widget-card');
        card.innerHTML = '<div class="widget-title">SESSIONS</div>';
        const list = el('div', 'widget-kv-list');
        (data.sessions || []).forEach((s, i) => {
            const active = i === 0 ? ' (ACTIVE)' : '';
            list.innerHTML += `<div class="widget-kv-row">
                <span class="widget-kv-key">${esc(s.title || 'Session ' + (i + 1))}${active}</span>
                <span class="widget-kv-val">${s.messages || 0} msgs</span>
            </div>`;
        });
        card.appendChild(list);
        return card;
    }

    function renderWorkflowCards(data) {
        const card = el('div', 'widget-card');
        card.innerHTML = '<div class="widget-title">WORKFLOWS</div>';
        const grid = el('div', 'widget-workflow-grid');
        (data.workflows || []).forEach(wf => {
            grid.innerHTML += `<div class="widget-workflow-item" data-run="${esc(wf.name)}">
                <div class="widget-workflow-name">${esc(wf.name)}</div>
                <div class="widget-workflow-meta">${wf.steps || '?'} steps</div>
            </div>`;
        });
        card.appendChild(grid);
        return card;
    }

    function renderFallback(msg) {
        const card = el('div', 'widget-card');
        card.innerHTML = `<div class="widget-title">${esc(msg.title || msg.widget_type)}</div>
            <pre class="widget-json">${esc(JSON.stringify(msg.data, null, 2))}</pre>`;
        return card;
    }

    function formatNum(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return String(n);
    }

    // ── Public API ──

    const RENDERERS = {
        key_value: renderKeyValue,
        bar_chart: renderBarChart,
        pipeline_progress: renderPipeline,
        channel_list: renderChannelList,
        identity_card: renderIdentityCard,
        session_list: renderSessionList,
        workflow_cards: renderWorkflowCards,
    };

    window.PREDACORE.renderWidget = function (msg) {
        const renderer = RENDERERS[msg.widget_type];
        return renderer ? renderer(msg.data || {}) : renderFallback(msg);
    };
})();
