// ═══════════════════════════════════════════════════════════════════
// J.A.R.V.I.S. Command Palette — Ctrl+K / "/" Quick Commands
// ═══════════════════════════════════════════════════════════════════
(function () {
    'use strict';
    window.JARVIS = window.JARVIS || {};

    const COMMANDS = [
        { cmd: '/model', desc: 'Show or switch LLM provider', section: 'System' },
        { cmd: '/sessions', desc: 'List recent sessions', section: 'Sessions' },
        { cmd: '/new', desc: 'Start a fresh session', section: 'Sessions' },
        { cmd: '/resume', desc: 'Resume a previous session', section: 'Sessions' },
        { cmd: '/link', desc: 'Link cross-channel identity', section: 'Channels' },
        { cmd: '/cancel', desc: 'Cancel current task', section: 'Control' },
    ];

    const palette = document.getElementById('commandPalette');
    const backdrop = document.getElementById('paletteBackdrop');
    const input = document.getElementById('paletteInput');
    const results = document.getElementById('paletteResults');
    let selectedIdx = 0;
    let filtered = [];

    function open() {
        palette.classList.add('open');
        input.value = '';
        selectedIdx = 0;
        renderResults('');
        setTimeout(() => input.focus(), 50);
    }

    function close() {
        palette.classList.remove('open');
        const main = document.getElementById('messageInput');
        if (main) main.focus();
    }

    function isOpen() {
        return palette.classList.contains('open');
    }

    function renderResults(query) {
        const q = query.toLowerCase().trim();
        filtered = q
            ? COMMANDS.filter(c => c.cmd.includes(q) || c.desc.toLowerCase().includes(q))
            : COMMANDS.slice();

        results.innerHTML = '';
        if (filtered.length === 0) {
            results.innerHTML = '<div class="palette-empty">No matching commands</div>';
            return;
        }

        filtered.forEach((c, i) => {
            const item = document.createElement('div');
            item.className = 'palette-item' + (i === selectedIdx ? ' selected' : '');
            item.innerHTML = `<span class="palette-cmd">${escHtml(c.cmd)}</span>
                <span class="palette-desc">${escHtml(c.desc)}</span>
                <span class="palette-section">${escHtml(c.section)}</span>`;
            item.addEventListener('click', () => selectCommand(i));
            item.addEventListener('mouseenter', () => {
                selectedIdx = i;
                updateSelection();
            });
            results.appendChild(item);
        });
    }

    function updateSelection() {
        const items = results.querySelectorAll('.palette-item');
        items.forEach((el, i) => {
            el.classList.toggle('selected', i === selectedIdx);
        });
        // Scroll selected into view
        const sel = results.querySelector('.palette-item.selected');
        if (sel) sel.scrollIntoView({ block: 'nearest' });
    }

    function selectCommand(idx) {
        const cmd = filtered[idx];
        if (!cmd) return;
        close();
        // Send the command via the chat
        if (window.JARVIS.sendCommand) {
            window.JARVIS.sendCommand(cmd.cmd);
        }
    }

    function escHtml(text) {
        const d = document.createElement('div');
        d.textContent = text;
        return d.innerHTML;
    }

    // ── Event Listeners ──

    if (input) {
        input.addEventListener('input', () => {
            selectedIdx = 0;
            renderResults(input.value);
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') { e.preventDefault(); close(); }
            else if (e.key === 'ArrowDown') {
                e.preventDefault();
                selectedIdx = Math.min(selectedIdx + 1, filtered.length - 1);
                updateSelection();
            }
            else if (e.key === 'ArrowUp') {
                e.preventDefault();
                selectedIdx = Math.max(selectedIdx - 1, 0);
                updateSelection();
            }
            else if (e.key === 'Enter') {
                e.preventDefault();
                selectCommand(selectedIdx);
            }
        });
    }

    if (backdrop) {
        backdrop.addEventListener('click', close);
    }

    // Global Ctrl+K handler
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            if (isOpen()) close();
            else open();
        }
        if (e.key === 'Escape' && isOpen()) {
            e.preventDefault();
            close();
        }
    });

    window.JARVIS.commandPalette = { open, close, isOpen };
})();
