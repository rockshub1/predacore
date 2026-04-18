"""Voice handlers: speak, voice_note."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    missing_param,
    invalid_param,
    blocked,
    subsystem_unavailable,
    resource_not_found,
)

logger = logging.getLogger(__name__)


async def handle_speak(args: dict[str, Any], ctx: ToolContext) -> str:
    """Convert text to speech using Kokoro neural TTS (local) with fallback chain."""
    if ctx.voice is None:
        raise subsystem_unavailable("Voice interface", tool="speak")
    text = str(args.get("text") or "").strip()
    if not text:
        raise missing_param("text", tool="speak")
    voice_id = str(args.get("voice") or "").strip() or None
    try:
        result = await ctx.voice.synthesize_and_play(text, voice_id=voice_id)
        provider = ctx.voice._tts.__class__.__name__ if ctx.voice._tts else "unknown"
        voice_used = voice_id or ctx.voice._config.voice_id
        return (
            f"[Spoken via {provider} | voice={voice_used} | "
            f"gen={result.duration_ms}ms | "
            f"size={len(result.audio_bytes)}B | "
            f"text={text[:80]}{'...' if len(text) > 80 else ''}]"
        )
    except (RuntimeError, OSError, ConnectionError, ValueError) as e:
        raise ToolError(
            f"TTS error: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="speak",
        ) from e


async def handle_voice_note(args: dict[str, Any], ctx: ToolContext) -> str:
    """Record from microphone or transcribe an audio file."""
    action = str(args.get("action") or "").strip().lower()
    if action not in ("record", "transcribe"):
        raise invalid_param(
            "action", "must be 'record' or 'transcribe'", tool="voice_note"
        )
    if args.get("_depth", 0) > 1:
        raise ToolError(
            "voice_note: recursion limit reached",
            kind=ToolErrorKind.LIMIT_EXCEEDED,
            tool_name="voice_note",
        )

    if action == "transcribe":
        audio_path = str(args.get("audio_path") or "").strip()
        if not audio_path:
            raise missing_param("audio_path", tool="voice_note")
        audio_file = Path(audio_path).expanduser().resolve()
        if not audio_file.exists():
            raise resource_not_found("Audio file", audio_path, tool="voice_note")
        _AUDIO_MAX_SIZE = 25 * 1024 * 1024
        try:
            audio_size = audio_file.stat().st_size
        except OSError as e:
            raise ToolError(
                f"Error accessing audio file: {e}",
                kind=ToolErrorKind.EXECUTION,
                tool_name="voice_note",
            ) from e
        if audio_size > _AUDIO_MAX_SIZE:
            raise ToolError(
                f"Audio file too large: {audio_size / (1024*1024):.1f}MB — Whisper API limit is 25MB",
                kind=ToolErrorKind.LIMIT_EXCEEDED,
                tool_name="voice_note",
            )
        if audio_size == 0:
            raise ToolError(
                f"Audio file is empty: {audio_path}",
                kind=ToolErrorKind.INVALID_PARAM,
                tool_name="voice_note",
            )
        audio_path = str(audio_file)

        if not os.access(audio_path, os.R_OK):
            raise blocked(f"Audio file not readable: {audio_path}", tool="voice_note")

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=120) as client:
                    with open(audio_path, "rb") as f:
                        resp = await client.post(
                            "https://api.openai.com/v1/audio/transcriptions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            data={"model": "whisper-1"},
                            files={"file": (Path(audio_path).name, f, "audio/mpeg")},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                return json.dumps({
                    "text": data.get("text", ""),
                    "audio_path": audio_path,
                    "provider": "openai_whisper",
                }, indent=2)
            except (ConnectionError, TimeoutError, OSError, ValueError) as e:
                raise ToolError(
                    f"Transcription error: {e}",
                    kind=ToolErrorKind.EXECUTION,
                    tool_name="voice_note",
                ) from e

        if ctx.voice and hasattr(ctx.voice, "transcribe"):
            try:
                text = await ctx.voice.transcribe(audio_path)
                return json.dumps({
                    "text": text,
                    "audio_path": audio_path,
                    "provider": "local",
                }, indent=2)
            except (RuntimeError, OSError, ValueError) as e:
                raise ToolError(
                    f"Local transcription error: {e}",
                    kind=ToolErrorKind.EXECUTION,
                    tool_name="voice_note",
                ) from e

        raise subsystem_unavailable(
            "Transcription service (set OPENAI_API_KEY or install local STT)",
            tool="voice_note",
        )

    # action == "record"
    duration = max(1, min(int(args.get("duration_seconds") or 5), 60))
    sample_rate = int(args.get("sample_rate") or 16000)
    output_path = str(args.get("output_path") or "").strip()

    is_temp_output = not bool(output_path)
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="predacore_voice_")
        os.close(fd)

    try:
        proc = await asyncio.create_subprocess_exec(
            "sox", "-d", "-r", str(sample_rate), "-c", "1", "-b", "16",
            output_path, "trim", "0", str(duration),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=duration + 10)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            raise ToolError(
                "Recording timed out",
                kind=ToolErrorKind.TIMEOUT,
                tool_name="voice_note",
            )

        if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
            err_msg = (stderr or b"").decode(errors="replace").strip()
            raise ToolError(
                f"Recording failed: {err_msg or 'no audio captured'}",
                kind=ToolErrorKind.EXECUTION,
                tool_name="voice_note",
            )

        size = Path(output_path).stat().st_size
        result_info: dict[str, Any] = {
            "audio_path": output_path,
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "size_bytes": size,
        }

        auto_transcribe = bool(args.get("auto_transcribe", True))
        if auto_transcribe:
            transcript = await handle_voice_note(
                {"action": "transcribe", "audio_path": output_path, "_depth": args.get("_depth", 0) + 1},
                ctx,
            )
            try:
                t_data = json.loads(transcript)
                result_info["transcript"] = t_data.get("text", "")
                result_info["transcription_provider"] = t_data.get("provider", "")
            except (json.JSONDecodeError, TypeError):
                result_info["transcript_raw"] = transcript

        return json.dumps(result_info, indent=2)

    except ToolError:
        if is_temp_output:
            Path(output_path).unlink(missing_ok=True)
        raise  # Re-raise our structured errors
    except FileNotFoundError:
        if is_temp_output:
            Path(output_path).unlink(missing_ok=True)
        raise subsystem_unavailable(
            "sox (install: brew install sox on macOS, apt install sox on Linux)",
            tool="voice_note",
        )
    except OSError as e:
        if is_temp_output:
            Path(output_path).unlink(missing_ok=True)
        raise ToolError(
            f"Recording error: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="voice_note",
        ) from e
