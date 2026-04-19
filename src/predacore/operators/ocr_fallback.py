"""
PredaCore OCR Fallback — Text extraction when AX tree is unavailable.

Uses macOS Vision framework (VNRecognizeTextRequest) via PyObjC for
high-accuracy on-device OCR. Falls back to tesseract CLI if PyObjC
Vision bindings are not available.

This is the "eyes" fallback — when Accessibility API can't read UI
elements (permissions issue, non-native app, web content), OCR reads
the screenshot pixels instead.

Performance:
  - Vision framework: ~200-500ms for full screen, runs on Neural Engine
  - Tesseract fallback: ~1-3s, requires tesseract in PATH

Usage:
    from predacore.operators.ocr_fallback import OCREngine

    engine = OCREngine()
    results = await engine.extract_text("/path/to/screenshot.png")
    # [{"text": "Save", "confidence": 0.98, "bounds": [100, 200, 50, 20]}, ...]

    full_text = await engine.extract_full_text("/path/to/screenshot.png")
    # "Save    Cancel    Edit\nFile  Edit  View  ..."
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """A single OCR detection result."""
    text: str
    confidence: float  # 0.0 - 1.0
    bounds: tuple[float, float, float, float]  # x, y, w, h (normalized 0-1)
    source: str = "vision"  # "vision" or "tesseract"

    def to_dict(self) -> dict[str, Any]:
        """Serialize OCR result to dictionary."""
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "bounds": {
                "x": round(self.bounds[0], 4),
                "y": round(self.bounds[1], 4),
                "w": round(self.bounds[2], 4),
                "h": round(self.bounds[3], 4),
            },
            "source": self.source,
        }


class OCREngine:
    """Multi-backend OCR engine with automatic fallback.

    Priority:
      1. macOS Vision framework (VNRecognizeTextRequest) — fast, accurate, on-device
      2. Tesseract CLI — widely available fallback
      3. Raw screencapture text extraction — last resort
    """

    def __init__(self, prefer_backend: str = "auto"):
        """
        Args:
            prefer_backend: "vision", "tesseract", or "auto" (try vision first).
        """
        self._prefer = prefer_backend
        self._vision_available: bool | None = None
        self._tesseract_path: str | None = None

    @property
    def available(self) -> bool:
        """Check if any OCR backend is available."""
        return self._check_vision() or self._check_tesseract()

    def _check_vision(self) -> bool:
        """Check if macOS Vision framework is available via PyObjC."""
        if self._vision_available is not None:
            return self._vision_available
        try:
            import Quartz  # noqa: F401 — needed for image loading
            import Vision  # noqa: F401 — PyObjC Vision framework
            self._vision_available = True
        except ImportError:
            self._vision_available = False
        return self._vision_available

    def _check_tesseract(self) -> bool:
        """Check if tesseract CLI is available."""
        if self._tesseract_path is not None:
            return bool(self._tesseract_path)
        path = shutil.which("tesseract")
        self._tesseract_path = path or ""
        return bool(path)

    async def extract_text(
        self,
        image_path: str,
        *,
        min_confidence: float = 0.3,
        languages: list[str] | None = None,
    ) -> list[OCRResult]:
        """Extract text regions from an image with positions and confidence.

        Args:
            image_path: Path to PNG/JPEG screenshot.
            min_confidence: Minimum confidence threshold (0.0 - 1.0).
            languages: Language codes (e.g. ["en-US"]). None = auto-detect.

        Returns:
            List of OCRResult with text, confidence, and normalized bounds.
        """
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            logger.warning("OCR image not found: %s", path)
            return []

        # Try Vision framework first
        if self._prefer in ("auto", "vision") and self._check_vision():
            try:
                results = await self._extract_vision(
                    str(path), min_confidence, languages
                )
                if results:
                    return results
                logger.debug("Vision returned no results, trying tesseract")
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("Vision OCR failed: %s — falling back to tesseract", exc)

        # Tesseract fallback
        if self._prefer in ("auto", "tesseract") and self._check_tesseract():
            try:
                return await self._extract_tesseract(str(path), min_confidence)
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                logger.warning("Tesseract OCR failed: %s", exc)

        logger.warning("No OCR backend available")
        return []

    async def extract_full_text(
        self,
        image_path: str,
        *,
        min_confidence: float = 0.3,
    ) -> str:
        """Extract all text from image as a single string (no positions).

        Convenience method for when you just want the text content.
        """
        results = await self.extract_text(image_path, min_confidence=min_confidence)
        if not results:
            return ""

        # Sort by vertical position (top to bottom), then horizontal (left to right)
        sorted_results = sorted(results, key=lambda r: (r.bounds[1], r.bounds[0]))

        # Group by approximate rows (within 2% vertical distance = same line)
        lines: list[list[OCRResult]] = []
        current_line: list[OCRResult] = []
        last_y = -1.0

        for r in sorted_results:
            if last_y < 0 or abs(r.bounds[1] - last_y) < 0.02:
                current_line.append(r)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [r]
            last_y = r.bounds[1]
        if current_line:
            lines.append(current_line)

        # Build text: sort each line left-to-right, join with spaces
        text_lines = []
        for line in lines:
            line.sort(key=lambda r: r.bounds[0])
            text_lines.append("  ".join(r.text for r in line))

        return "\n".join(text_lines)

    # ── macOS Vision Framework Backend ───────────────────────────

    async def _extract_vision(
        self,
        image_path: str,
        min_confidence: float,
        languages: list[str] | None,
    ) -> list[OCRResult]:
        """Extract text using macOS Vision framework (VNRecognizeTextRequest).

        Runs in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._extract_vision_sync, image_path, min_confidence, languages
        )

    def _extract_vision_sync(
        self,
        image_path: str,
        min_confidence: float,
        languages: list[str] | None,
    ) -> list[OCRResult]:
        """Synchronous Vision framework OCR."""
        import Quartz
        import Vision

        # Load image
        url = Quartz.CFURLCreateFromFileSystemRepresentation(
            None, image_path.encode("utf-8"), len(image_path.encode("utf-8")), False
        )
        if url is None:
            raise RuntimeError(f"Failed to create URL for {image_path}")

        # Create CGImage from file
        source = Quartz.CGImageSourceCreateWithURL(url, None)
        if source is None:
            raise RuntimeError(f"Failed to load image source: {image_path}")

        cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
        if cg_image is None:
            raise RuntimeError(f"Failed to create CGImage: {image_path}")

        # Create and configure text recognition request
        results: list[OCRResult] = []
        error_ref = [None]

        def completion_handler(request, error):
            """Handle Vision framework completion callback."""
            if error:
                error_ref[0] = error
                return
            observations = request.results()
            if not observations:
                return
            for obs in observations:
                # Each observation is a VNRecognizedTextObservation
                candidates = obs.topCandidates_(1)
                if not candidates:
                    continue
                candidate = candidates[0]
                text = candidate.string()
                confidence = candidate.confidence()

                if confidence < min_confidence:
                    continue

                # Get bounding box (normalized, origin at bottom-left in Vision)
                bbox = obs.boundingBox()
                # Convert from bottom-left origin to top-left origin
                x = bbox.origin.x
                y = 1.0 - bbox.origin.y - bbox.size.height
                w = bbox.size.width
                h = bbox.size.height

                results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bounds=(x, y, w, h),
                    source="vision",
                ))

        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(
            completion_handler
        )
        request.setRecognitionLevel_(
            Vision.VNRequestTextRecognitionLevelAccurate
        )
        request.setUsesLanguageCorrection_(True)

        if languages:
            request.setRecognitionLanguages_(languages)

        # Execute request
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        success = handler.performRequests_error_([request], None)

        if error_ref[0]:
            raise RuntimeError(f"Vision OCR error: {error_ref[0]}")

        logger.debug(
            "Vision OCR: %d text regions detected (min_conf=%.2f)",
            len(results), min_confidence,
        )
        return results

    # ── Tesseract CLI Backend ────────────────────────────────────

    async def _extract_tesseract(
        self,
        image_path: str,
        min_confidence: float,
    ) -> list[OCRResult]:
        """Extract text using tesseract CLI with TSV output for positions."""
        if not self._tesseract_path:
            return []

        # Run tesseract with TSV output (includes positions + confidence)
        proc = await asyncio.create_subprocess_exec(
            self._tesseract_path, image_path, "stdout",
            "--psm", "3",  # Fully automatic page segmentation
            "-c", "tessedit_create_tsv=1",
            "tsv",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"Tesseract failed: {err_msg}")

        # Parse TSV output
        # Columns: level, page_num, block_num, par_num, line_num, word_num,
        #          left, top, width, height, conf, text
        results: list[OCRResult] = []
        lines = stdout.decode(errors="replace").strip().split("\n")

        if len(lines) < 2:
            return results

        # Get image dimensions for normalization
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except ImportError:
            # Without PIL, try to get dimensions from sips (macOS)
            try:
                sips_proc = await asyncio.create_subprocess_exec(
                    "sips", "-g", "pixelWidth", "-g", "pixelHeight", image_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                sips_out, _ = await asyncio.wait_for(sips_proc.communicate(), timeout=5)
                sips_text = sips_out.decode()
                img_w = int([l for l in sips_text.split("\n") if "pixelWidth" in l][0].split(":")[-1])
                img_h = int([l for l in sips_text.split("\n") if "pixelHeight" in l][0].split(":")[-1])
            except (OSError, ValueError, IndexError, FileNotFoundError) as exc:
                # Refusing to guess dimensions — wrong dims silently corrupt every
                # OCR coordinate downstream. Better to fail loud.
                raise RuntimeError(
                    f"Cannot determine image dimensions for OCR normalization "
                    f"(install Pillow or run on macOS with sips): {exc}"
                ) from exc

        for line in lines[1:]:  # Skip header
            parts = line.split("\t")
            if len(parts) < 12:
                continue

            text = parts[11].strip()
            if not text:
                continue

            try:
                conf = float(parts[10]) / 100.0  # Tesseract uses 0-100
            except (ValueError, IndexError):
                conf = 0.0

            if conf < min_confidence:
                continue

            try:
                left = int(parts[6])
                top = int(parts[7])
                width = int(parts[8])
                height = int(parts[9])
            except (ValueError, IndexError):
                continue

            # Normalize to 0-1 range
            results.append(OCRResult(
                text=text,
                confidence=conf,
                bounds=(
                    left / img_w,
                    top / img_h,
                    width / img_w,
                    height / img_h,
                ),
                source="tesseract",
            ))

        logger.debug(
            "Tesseract OCR: %d text regions detected (min_conf=%.2f)",
            len(results), min_confidence,
        )
        return results

    def status(self) -> dict[str, Any]:
        """Return OCR engine status."""
        return {
            "vision_available": self._check_vision(),
            "tesseract_available": self._check_tesseract(),
            "tesseract_path": self._tesseract_path or None,
            "preferred_backend": self._prefer,
            "any_available": self.available,
        }
