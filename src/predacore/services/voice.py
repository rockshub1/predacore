"""
Voice Interface — TTS/STT abstraction layer.

Provides a unified interface for text-to-speech and speech-to-text
across multiple providers (Kokoro local, macOS system, OpenAI, edge-tts).

Usage:
    voice = VoiceInterface()

    # Text-to-Speech
    audio_bytes = await voice.synthesize("Hello, how can I help?")

    # Speech-to-Text
    transcript = await voice.transcribe(audio_bytes)
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Enums & Config ─────────────────────────────────────────────────


class VoiceProvider(str, Enum):
    """Supported voice providers."""

    KOKORO = "kokoro"        # Kokoro local neural TTS (default)
    EDGE = "edge"            # Microsoft Edge TTS (free, neural, needs internet)
    SYSTEM = "system"        # macOS `say` / espeak
    OPENAI = "openai"        # OpenAI TTS/Whisper
    GOOGLE = "google"        # Google Cloud TTS/STT
    ELEVENLABS = "elevenlabs"  # ElevenLabs TTS


# ── Kokoro Voice Presets ───────────────────────────────────────────

KOKORO_VOICES = {
    # British Male (PredaCore defaults)
    "bm_george": {"lang": "b", "desc": "British Male - George (refined, butler-like)"},
    "bm_daniel": {"lang": "b", "desc": "British Male - Daniel (warm, articulate)"},
    "bm_lewis":  {"lang": "b", "desc": "British Male - Lewis (deep, authoritative)"},
    "bm_fable":  {"lang": "b", "desc": "British Male - Fable (storyteller, smooth)"},
    # British Female
    "bf_alice":    {"lang": "b", "desc": "British Female - Alice"},
    "bf_emma":     {"lang": "b", "desc": "British Female - Emma"},
    "bf_isabella": {"lang": "b", "desc": "British Female - Isabella"},
    "bf_lily":     {"lang": "b", "desc": "British Female - Lily"},
    # American Male
    "am_onyx":    {"lang": "a", "desc": "American Male - Onyx (deep, cinematic)"},
    "am_adam":    {"lang": "a", "desc": "American Male - Adam (clean, professional)"},
    "am_echo":    {"lang": "a", "desc": "American Male - Echo (friendly, casual)"},
    "am_eric":    {"lang": "a", "desc": "American Male - Eric"},
    "am_fenrir":  {"lang": "a", "desc": "American Male - Fenrir"},
    "am_liam":    {"lang": "a", "desc": "American Male - Liam"},
    "am_michael": {"lang": "a", "desc": "American Male - Michael"},
    "am_puck":    {"lang": "a", "desc": "American Male - Puck"},
    # American Female
    "af_alloy":   {"lang": "a", "desc": "American Female - Alloy"},
    "af_bella":   {"lang": "a", "desc": "American Female - Bella"},
    "af_heart":   {"lang": "a", "desc": "American Female - Heart"},
    "af_jessica": {"lang": "a", "desc": "American Female - Jessica"},
    "af_nicole":  {"lang": "a", "desc": "American Female - Nicole"},
    "af_nova":    {"lang": "a", "desc": "American Female - Nova"},
    "af_river":   {"lang": "a", "desc": "American Female - River"},
    "af_sarah":   {"lang": "a", "desc": "American Female - Sarah"},
    "af_sky":     {"lang": "a", "desc": "American Female - Sky"},
}

# PredaCore default: bm_george (British, fastest, most PredaCore-like)
KOKORO_DEFAULT_VOICE = "bm_george"
KOKORO_ALT_VOICE = "am_onyx"  # American cinematic alternative
KOKORO_SAMPLE_RATE = 24000
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"


@dataclass
class VoiceConfig:
    """Configuration for voice interface."""

    tts_provider: VoiceProvider = VoiceProvider.KOKORO
    stt_provider: VoiceProvider = VoiceProvider.OPENAI
    voice_id: str = KOKORO_DEFAULT_VOICE
    language: str = "en"
    speed: float = 1.0
    sample_rate: int = KOKORO_SAMPLE_RATE


@dataclass
class TranscriptionResult:
    """Result of speech-to-text."""

    text: str
    language: str = "en"
    confidence: float = 1.0
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
        }


@dataclass
class SynthesisResult:
    """Result of text-to-speech."""

    audio_bytes: bytes
    format: str = "wav"
    duration_ms: int = 0
    sample_rate: int = KOKORO_SAMPLE_RATE

    @property
    def audio_base64(self) -> str:
        return base64.b64encode(self.audio_bytes).decode("ascii")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "duration_ms": self.duration_ms,
            "sample_rate": self.sample_rate,
            "size_bytes": len(self.audio_bytes),
        }


# ── Provider Abstractions ──────────────────────────────────────────


class TTSProvider(ABC):
    """Abstract TTS provider."""

    @abstractmethod
    async def synthesize(
        self, text: str, voice_id: str = "", **kwargs
    ) -> SynthesisResult:
        ...


class STTProvider(ABC):
    """Abstract STT provider."""

    @abstractmethod
    async def transcribe(
        self, audio_bytes: bytes, language: str = "en", **kwargs
    ) -> TranscriptionResult:
        ...


# ── Kokoro TTS (Local Neural TTS — Default) ───────────────────────


class KokoroTTS(TTSProvider):
    """
    Kokoro TTS — high-quality local neural TTS.

    Runs 100% offline on Apple Silicon (MPS) or CPU.
    82M parameter model, ~300MB, generates 4x faster than real-time on M1.

    Voices: 50+ voices across English (US/UK), Japanese, Chinese, Spanish,
    French, Hindi, Italian, Portuguese.

    Default PredaCore voice: bm_george (British Male, refined, fastest).
    """

    def __init__(self, device: str | None = None):
        self._pipelines: dict[str, Any] = {}  # lang_code -> KPipeline
        self._device = device  # None = auto (cpu on M1, cuda on GPU)
        self._model = None  # shared KModel instance
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-load Kokoro on first use to avoid slow startup."""
        if self._initialized:
            return

        try:
            import torch
            from kokoro import KModel

            # Auto-select device
            if self._device is None:
                # MPS has issues with some ops in Kokoro, prefer CPU for reliability
                self._device = "cpu"

            logger.info(f"Kokoro TTS: loading model on {self._device}")
            start = time.monotonic()
            self._model = KModel(repo_id=KOKORO_REPO_ID).to(self._device).eval()
            elapsed = time.monotonic() - start
            logger.info(f"Kokoro TTS: model loaded in {elapsed:.1f}s")
            self._initialized = True

        except ImportError as exc:
            raise RuntimeError(
                "Kokoro TTS not installed. Run: pip install kokoro"
            ) from exc
        except (OSError, RuntimeError) as e:
            logger.error(f"Kokoro TTS init failed: {e}")
            raise

    def _get_pipeline(self, lang_code: str) -> Any:
        """Get or create a KPipeline for the given language."""
        if lang_code not in self._pipelines:
            from kokoro import KPipeline
            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id=KOKORO_REPO_ID,
                model=self._model,  # share the loaded model
            )
        return self._pipelines[lang_code]

    async def synthesize(
        self, text: str, voice_id: str = "", **kwargs
    ) -> SynthesisResult:
        """Generate speech from text using Kokoro."""
        import numpy as np

        voice_id = voice_id or KOKORO_DEFAULT_VOICE
        speed = kwargs.get("speed", 1.0)

        # Determine language from voice prefix
        voice_info = KOKORO_VOICES.get(voice_id)
        if voice_info:
            lang_code = voice_info["lang"]
        else:
            # Infer from voice name prefix: a=American, b=British
            lang_code = voice_id[0] if voice_id and voice_id[0] in "abefhijpz" else "b"

        # Run synthesis in a thread to avoid blocking the event loop
        def _generate():
            self._ensure_initialized()
            pipe = self._get_pipeline(lang_code)

            start = time.monotonic()
            audio_chunks = []
            for result in pipe(text, voice=voice_id, speed=speed):
                if result.output is not None:
                    audio_chunks.append(result.output.audio.numpy())

            if not audio_chunks:
                return None, 0

            audio = np.concatenate(audio_chunks)
            gen_ms = int((time.monotonic() - start) * 1000)
            return audio, gen_ms

        loop = asyncio.get_running_loop()
        audio, gen_ms = await loop.run_in_executor(None, _generate)

        if audio is None:
            raise RuntimeError(f"Kokoro TTS: no audio generated for voice '{voice_id}'")

        # Convert float32 audio to WAV bytes
        audio_bytes = self._float_to_wav(audio, KOKORO_SAMPLE_RATE)
        audio_duration_ms = int(len(audio) / KOKORO_SAMPLE_RATE * 1000)

        logger.info(
            f"Kokoro TTS: voice={voice_id} | gen={gen_ms}ms | "
            f"audio={audio_duration_ms}ms | size={len(audio_bytes)}B"
        )

        return SynthesisResult(
            audio_bytes=audio_bytes,
            format="wav",
            duration_ms=gen_ms,
            sample_rate=KOKORO_SAMPLE_RATE,
        )

    @staticmethod
    def _float_to_wav(audio: Any, sample_rate: int) -> bytes:
        """Convert float32 numpy audio to WAV bytes."""
        import struct

        import numpy as np

        # Normalize and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        raw_data = audio_int16.tobytes()

        # Build WAV header (PCM, mono, 16-bit)
        num_samples = len(audio_int16)
        data_size = num_samples * 2  # 16-bit = 2 bytes per sample
        file_size = 36 + data_size

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', file_size, b'WAVE',
            b'fmt ', 16,       # PCM format chunk
            1,                 # PCM format
            1,                 # Mono
            sample_rate,       # Sample rate
            sample_rate * 2,   # Byte rate (sr * channels * bytes_per_sample)
            2,                 # Block align (channels * bytes_per_sample)
            16,                # Bits per sample
            b'data', data_size
        )

        return header + raw_data

    def get_available_voices(self) -> dict[str, str]:
        """Return available Kokoro voices with descriptions."""
        return {k: v["desc"] for k, v in KOKORO_VOICES.items()}


# ── Edge TTS (Free Neural TTS — needs internet) ───────────────────


class EdgeTTS(TTSProvider):
    """Microsoft Edge TTS — free neural voices, requires internet."""

    # Best PredaCore-like voices
    VOICE_MAP = {
        "default": "en-GB-RyanNeural",
        "ryan": "en-GB-RyanNeural",
        "guy": "en-US-GuyNeural",
        "andrew": "en-US-AndrewNeural",
    }

    async def synthesize(
        self, text: str, voice_id: str = "", **kwargs
    ) -> SynthesisResult:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("Install edge-tts: pip install edge-tts") from exc

        import tempfile

        # Map friendly names to edge-tts voice IDs
        voice = self.VOICE_MAP.get(voice_id, voice_id) or "en-GB-RyanNeural"

        start = time.monotonic()
        communicate = edge_tts.Communicate(text, voice)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name

        try:
            await communicate.save(tmp_path)
            audio_bytes = Path(tmp_path).read_bytes()
            elapsed = int((time.monotonic() - start) * 1000)

            return SynthesisResult(
                audio_bytes=audio_bytes,
                format="mp3",
                duration_ms=elapsed,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── System TTS (macOS say / Linux espeak) ──────────────────────────


class SystemTTS(TTSProvider):
    """Uses macOS `say` or Linux `espeak` for offline TTS."""

    async def synthesize(
        self, text: str, voice_id: str = "", **kwargs
    ) -> SynthesisResult:
        import platform
        import re as _re
        import tempfile

        if voice_id and not _re.match(r'^[a-zA-Z0-9._-]+$', voice_id):
            logger.warning("Invalid voice_id %r, using default", voice_id)
            voice_id = ""

        start = time.monotonic()
        system = platform.system()

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            output_path = f.name

        try:
            if system == "Darwin":
                cmd = ["say", "-o", output_path]
                if voice_id:
                    cmd.extend(["-v", voice_id])
                cmd.append(text)
            elif system == "Linux":
                cmd = ["espeak", "-w", output_path, text]
            else:
                return SynthesisResult(audio_bytes=b"", format="none")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

            audio_bytes = Path(output_path).read_bytes()
            elapsed = int((time.monotonic() - start) * 1000)

            return SynthesisResult(
                audio_bytes=audio_bytes,
                format="aiff",
                duration_ms=elapsed,
            )
        finally:
            try:
                os.unlink(output_path)
            except OSError:
                pass


# ── OpenAI TTS/STT ────────────────────────────────────────────────


class OpenAITTS(TTSProvider):
    """OpenAI TTS via direct HTTP (Prometheus SDK, no openai SDK)."""

    async def synthesize(
        self, text: str, voice_id: str = "alloy", **kwargs
    ) -> SynthesisResult:
        import os

        import httpx

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — required for OpenAI TTS")

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "tts-1",
                    "voice": voice_id,
                    "input": text,
                    "response_format": "mp3",
                },
            )
            resp.raise_for_status()
            audio_bytes = resp.content

        elapsed = int((time.monotonic() - start) * 1000)
        return SynthesisResult(
            audio_bytes=audio_bytes,
            format="mp3",
            duration_ms=elapsed,
        )


class OpenAISTT(STTProvider):
    """OpenAI Whisper STT via direct HTTP (Prometheus SDK, no openai SDK)."""

    async def transcribe(
        self, audio_bytes: bytes, language: str = "en", **kwargs
    ) -> TranscriptionResult:
        import os

        import httpx

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — required for OpenAI STT")

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"model": "whisper-1", "language": language},
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed = int((time.monotonic() - start) * 1000)
        return TranscriptionResult(
            text=data.get("text", ""),
            language=language,
            confidence=1.0,
            duration_ms=elapsed,
        )


# ── Main Voice Interface ──────────────────────────────────────────


class VoiceInterface:
    """
    Unified voice interface for PredaCore.

    Manages TTS and STT providers with automatic fallback.

    Provider priority for TTS:
        1. Kokoro (local, neural, 82M params, offline)
        2. Edge TTS (free, neural, needs internet)
        3. System (macOS `say` — robotic but instant)

    The interface automatically falls back through the chain if the
    primary provider fails.
    """

    def __init__(self, config: VoiceConfig | None = None):
        self._config = config or VoiceConfig()
        self._tts: TTSProvider | None = None
        self._tts_fallbacks: list[TTSProvider] = []
        self._stt: STTProvider | None = None
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize TTS and STT providers based on config."""
        provider = self._config.tts_provider

        # Primary TTS
        if provider == VoiceProvider.KOKORO:
            self._tts = KokoroTTS()
            self._tts_fallbacks = [EdgeTTS(), SystemTTS()]
        elif provider == VoiceProvider.EDGE:
            self._tts = EdgeTTS()
            self._tts_fallbacks = [KokoroTTS(), SystemTTS()]
        elif provider == VoiceProvider.OPENAI:
            self._tts = OpenAITTS()
            self._tts_fallbacks = [KokoroTTS(), EdgeTTS(), SystemTTS()]
        else:
            self._tts = SystemTTS()
            self._tts_fallbacks = []

        # STT
        if self._config.stt_provider == VoiceProvider.OPENAI:
            self._stt = OpenAISTT()

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        **kwargs,
    ) -> SynthesisResult:
        """Convert text to speech with automatic fallback."""
        if not self._tts:
            raise RuntimeError("No TTS provider configured")

        voice = voice_id or self._config.voice_id

        # Try primary provider
        try:
            return await self._tts.synthesize(text=text, voice_id=voice, **kwargs)
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Primary TTS ({self._tts.__class__.__name__}) failed: {e}")

        # Try fallbacks
        for fallback in self._tts_fallbacks:
            try:
                logger.info(f"Trying fallback TTS: {fallback.__class__.__name__}")
                # Map voice to appropriate format for fallback
                fallback_voice = self._map_voice_for_provider(voice, fallback)
                return await fallback.synthesize(text=text, voice_id=fallback_voice, **kwargs)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Fallback TTS ({fallback.__class__.__name__}) failed: {e}")

        raise RuntimeError("All TTS providers failed")

    async def synthesize_and_play(
        self,
        text: str,
        voice_id: str | None = None,
        **kwargs,
    ) -> SynthesisResult:
        """Synthesize speech and play it immediately."""
        result = await self.synthesize(text=text, voice_id=voice_id, **kwargs)

        if result.audio_bytes:
            await self._play_audio(result.audio_bytes, result.format)

        return result

    async def _play_audio(self, audio_bytes: bytes, fmt: str) -> None:
        """Play audio bytes using system player."""
        import tempfile

        suffix = f".{fmt}" if fmt != "none" else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "afplay", tmp_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _map_voice_for_provider(voice: str, provider: TTSProvider) -> str:
        """Map a Kokoro voice ID to the equivalent for another provider."""
        if isinstance(provider, EdgeTTS):
            # Map Kokoro British male → Edge British male
            if voice.startswith("bm_"):
                return "en-GB-RyanNeural"
            elif voice.startswith("bf_"):
                return "en-GB-SoniaNeural"
            elif voice.startswith("am_"):
                return "en-US-GuyNeural"
            elif voice.startswith("af_"):
                return "en-US-JennyNeural"
            return "en-GB-RyanNeural"
        elif isinstance(provider, SystemTTS):
            # macOS system voices
            if voice.startswith("b"):
                return "Daniel"  # British
            return ""  # Default system voice
        elif isinstance(provider, OpenAITTS):
            if "onyx" in voice:
                return "onyx"
            elif "echo" in voice:
                return "echo"
            elif "fable" in voice:
                return "fable"
            elif "nova" in voice:
                return "nova"
            return "alloy"
        return voice

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Convert speech to text."""
        if not self._stt:
            raise RuntimeError("No STT provider configured")

        return await self._stt.transcribe(
            audio_bytes=audio_bytes,
            language=language or self._config.language,
            **kwargs,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get voice interface configuration info."""
        return {
            "tts_provider": self._config.tts_provider.value,
            "tts_class": self._tts.__class__.__name__ if self._tts else None,
            "stt_provider": self._config.stt_provider.value,
            "voice_id": self._config.voice_id,
            "language": self._config.language,
            "fallback_chain": [f.__class__.__name__ for f in self._tts_fallbacks],
            "kokoro_voices": list(KOKORO_VOICES.keys()),
        }
