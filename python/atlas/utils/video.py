"""Video frame extractor for the Atlas-0 upload pipeline.

Extracts JPEG-encoded frames from MP4, MOV, WEBM, and AVI files using
PyAV (Python bindings for FFmpeg). No system FFmpeg installation required
beyond what PyAV bundles.

Install::

    pip install "atlas-0[video]"   # recommended
    pip install av>=14.0.0         # directly

Usage::

    from atlas.utils.video import extract_frames

    with open("room.mp4", "rb") as f:
        video_bytes = f.read()

    frames = extract_frames(video_bytes, max_frames=8)
    # frames is a list of JPEG bytes, one per sampled frame.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

_IMPORT_ERROR_MSG = (
    "Video support requires the 'av' package (PyAV / FFmpeg bindings).\n"
    'Install it with:  pip install "atlas-0[video]"\n'
    "Or directly:      pip install av>=14.0.0"
)

# JPEG quality for extracted frames (0-95; 85 balances size vs VLM accuracy).
_JPEG_QUALITY = 85


@dataclass(frozen=True)
class ExtractedFrame:
    """One sampled video frame with its timeline metadata."""

    index: int
    timestamp_s: float
    image_bytes: bytes


def extract_frames(
    video_bytes: bytes,
    max_frames: int = 8,
    *,
    jpeg_quality: int = _JPEG_QUALITY,
) -> list[bytes]:
    """Extract evenly-sampled JPEG frames from a video file.

    Frames are sampled uniformly across the full video duration so that a
    30-second room walkthrough produces frames from the start, middle, and end
    — not just the first few seconds.

    Args:
        video_bytes: Raw bytes of a video file (MP4, MOV, WEBM, AVI, etc.).
        max_frames: Maximum number of frames to extract. Actual count may be
            lower for very short videos.
        jpeg_quality: JPEG encoding quality (1-95). Higher = larger files +
            better VLM input. Default 85 is a good balance.

    Returns:
        List of JPEG-encoded frame bytes, in chronological order.
        Empty list if extraction fails (logged as a warning).

    Raises:
        RuntimeError: If the ``av`` package is not installed. The error message
            includes the install command.

    Example::

        frames = extract_frames(video_bytes, max_frames=6)
        for frame_bytes in frames:
            label = await vlm_engine.label_region(frame_bytes)
    """
    return [
        frame.image_bytes
        for frame in extract_frame_samples(
            video_bytes,
            max_frames=max_frames,
            jpeg_quality=jpeg_quality,
        )
    ]


def extract_frame_samples(
    video_bytes: bytes,
    max_frames: int = 8,
    *,
    jpeg_quality: int = _JPEG_QUALITY,
) -> list[ExtractedFrame]:
    """Extract evenly sampled frames with timestamps."""
    try:
        import av  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(_IMPORT_ERROR_MSG) from exc

    if not video_bytes:
        logger.warning("video_extract_empty_input")
        return []

    try:
        return _extract(video_bytes, max_frames, jpeg_quality, av)
    except Exception as exc:
        logger.warning("video_extract_failed", error=str(exc))
        return []


def _extract(
    video_bytes: bytes,
    max_frames: int,
    jpeg_quality: int,
    av: object,
) -> list[ExtractedFrame]:
    """Internal extraction logic — separated so the public function can catch all errors."""
    buf = io.BytesIO(video_bytes)

    with av.open(buf) as container:  # type: ignore[union-attr]
        video_stream = next(
            (s for s in container.streams if s.type == "video"),
            None,
        )
        if video_stream is None:
            logger.warning("video_extract_no_video_stream")
            return []

        # Total frame count — may be 0 if container doesn't report it.
        total_frames = video_stream.frames or 0
        duration_s = float(
            video_stream.duration * video_stream.time_base
            if video_stream.duration
            else container.duration / 1_000_000
            if container.duration
            else 0
        )

        logger.info(
            "video_extract_start",
            total_frames=total_frames,
            duration_s=round(duration_s, 1),
            max_frames=max_frames,
        )

        # Determine which frame indices to capture.
        target_indices = _sample_indices(total_frames, max_frames)
        # If total_frames unknown, fall back to time-based sampling.
        use_time_sampling = total_frames == 0 or not target_indices

        frames: list[ExtractedFrame] = []
        frame_index = 0

        # Time-based fallback: sample every N seconds.
        sample_interval_s = duration_s / max(max_frames, 1) if use_time_sampling else 0.0
        next_sample_time = 0.0

        for packet in container.demux(video_stream):
            for frame in packet.decode():
                pts_s = float(frame.pts * video_stream.time_base) if frame.pts else 0.0

                should_capture = (use_time_sampling and pts_s >= next_sample_time) or (
                    not use_time_sampling and frame_index in target_indices
                )

                if should_capture:
                    jpeg_bytes = _frame_to_jpeg(frame, jpeg_quality)
                    if jpeg_bytes:
                        frames.append(
                            ExtractedFrame(
                                index=frame_index,
                                timestamp_s=round(pts_s, 4),
                                image_bytes=jpeg_bytes,
                            )
                        )
                        logger.debug(
                            "video_frame_captured",
                            frame_index=frame_index,
                            pts_s=round(pts_s, 2),
                        )
                    if use_time_sampling:
                        next_sample_time += sample_interval_s
                    if len(frames) >= max_frames:
                        break

                frame_index += 1

            if len(frames) >= max_frames:
                break

    logger.info("video_extract_done", frames_extracted=len(frames))
    return frames


def _sample_indices(total_frames: int, max_frames: int) -> set[int]:
    """Compute a set of frame indices evenly distributed over the video.

    Args:
        total_frames: Total number of frames in the video.
        max_frames: Number of frames to sample.

    Returns:
        Set of integer frame indices to capture.
    """
    if total_frames <= 0 or max_frames <= 0:
        return set()
    if max_frames >= total_frames:
        return set(range(total_frames))
    step = total_frames / max_frames
    return {math.floor(i * step) for i in range(max_frames)}


def _frame_to_jpeg(frame: object, quality: int) -> bytes | None:
    """Convert a PyAV VideoFrame to JPEG bytes.

    Args:
        frame: ``av.VideoFrame`` instance.
        quality: JPEG quality 1-95.

    Returns:
        JPEG bytes, or ``None`` if conversion fails.
    """
    try:
        from PIL import Image  # type: ignore[import]

        pil_img: Image.Image = frame.to_image()  # type: ignore[union-attr]
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("video_frame_to_jpeg_failed", error=str(exc))
        return None


def is_video_available() -> bool:
    """Return ``True`` if PyAV is installed and video extraction is available.

    Use this to give users a clear capability check without raising exceptions.

    Example::

        if not is_video_available():
            print("Install av: pip install 'atlas-0[video]'")
    """
    try:
        import av  # type: ignore[import]  # noqa: F401

        return True
    except ImportError:
        return False
