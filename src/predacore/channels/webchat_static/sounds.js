// ═══════════════════════════════════════════════════════════════════
// J.A.R.V.I.S. Ambient Sound Engine — Web Audio API
// All sounds synthesized, no external audio files needed
// ═══════════════════════════════════════════════════════════════════
(function () {
    'use strict';

    let ctx = null;
    let enabled = localStorage.getItem('jarvis_sounds') === 'on';
    let humOsc = null;
    let humGain = null;

    function getCtx() {
        if (!ctx) {
            try {
                ctx = new (window.AudioContext || window.webkitAudioContext)();
            } catch (e) {
                return null;
            }
        }
        if (ctx.state === 'suspended') ctx.resume();
        return ctx;
    }

    function playTone(freq, type, duration, volume) {
        const c = getCtx();
        if (!c || !enabled) return;
        const osc = c.createOscillator();
        const gain = c.createGain();
        osc.type = type;
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(volume || 0.08, c.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + duration);
        osc.connect(gain).connect(c.destination);
        osc.start();
        osc.stop(c.currentTime + duration);
    }

    window.PREDACORE = window.PREDACORE || {};
    window.PREDACORE.sounds = {
        get enabled() { return enabled; },

        toggle() {
            enabled = !enabled;
            localStorage.setItem('jarvis_sounds', enabled ? 'on' : 'off');
            if (!enabled && humOsc) {
                humOsc.stop();
                humOsc = null;
            }
            return enabled;
        },

        // Soft chime — message received
        chime() {
            playTone(880, 'sine', 0.25, 0.06);
            setTimeout(() => playTone(1100, 'sine', 0.2, 0.04), 80);
        },

        // Whoosh — tool starting
        whoosh() {
            const c = getCtx();
            if (!c || !enabled) return;
            const osc = c.createOscillator();
            const gain = c.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(1500, c.currentTime);
            osc.frequency.exponentialRampToValueAtTime(200, c.currentTime + 0.15);
            gain.gain.setValueAtTime(0.04, c.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + 0.2);
            osc.connect(gain).connect(c.destination);
            osc.start();
            osc.stop(c.currentTime + 0.2);
        },

        // Ding — tool/task complete
        ding() {
            playTone(1200, 'sine', 0.15, 0.07);
        },

        // Click — message sent
        click() {
            playTone(1000, 'square', 0.03, 0.03);
        },

        // Error alert
        error() {
            playTone(300, 'square', 0.2, 0.06);
            setTimeout(() => playTone(250, 'square', 0.3, 0.04), 200);
        },

        // Low hum — continuous during processing
        humStart() {
            const c = getCtx();
            if (!c || !enabled || humOsc) return;
            humOsc = c.createOscillator();
            humGain = c.createGain();
            humOsc.type = 'sine';
            humOsc.frequency.value = 80;
            humGain.gain.setValueAtTime(0, c.currentTime);
            humGain.gain.linearRampToValueAtTime(0.03, c.currentTime + 0.5);
            humOsc.connect(humGain).connect(c.destination);
            humOsc.start();
        },

        humStop() {
            if (!humOsc || !humGain) return;
            const c = getCtx();
            if (c) humGain.gain.linearRampToValueAtTime(0, c.currentTime + 0.3);
            setTimeout(() => {
                try { if (humOsc) humOsc.stop(); } catch (e) { /* already stopped */ }
                humOsc = null;
                humGain = null;
            }, 400);
        },

        // Boot sound — ascending arpeggio
        boot() {
            playTone(440, 'sine', 0.15, 0.05);
            setTimeout(() => playTone(554, 'sine', 0.15, 0.05), 100);
            setTimeout(() => playTone(659, 'sine', 0.2, 0.06), 200);
        },
    };
})();
