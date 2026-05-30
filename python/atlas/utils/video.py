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


def _video_stream_pixels(video_stream: object) -> int:
    """Return ``width * height`` for a video stream, or 0 when unknown.

    Reads the coded dimensions from the container header without decoding any
    frame.  Non-integer/missing values (e.g. test doubles) are treated as
    unknown so callers skip the resolution check rather than misfire.
    """
    width = getattr(video_stream, "width", 0)
    height = getattr(video_stream, "height", 0)
    if isinstance(width, int) and isinstance(height, int):
        return max(0, width) * max(0, height)
    return 0


@dataclass(frozen=True)
class ExtractedFrame:
    """One sampled video frame with its timeline metadata."""

    index: int
    timestamp_s: float
    image_bytes: bytes


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata probed from a video container."""

    duration_s: float
    frame_count: int
    width: int = 0
    height: int = 0


def extract_frames(
    video_bytes: bytes,
    max_frames: int = 8,
    *,
    jpeg_quality: int = _JPEG_QUALITY,
    max_pixels: int | None = None,
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
        max_pixels: Optional cap on the video's coded ``width * height``.
            Videos above this are rejected before any frame is decoded, to
            block resolution-bomb DoS. ``None`` disables the check.

    Returns:
        List of JPEG-encoded frame bytes, in chronological order.
        Empty list if extraction fails or is rejected (logged as a warning).

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
            max_pixels=max_pixels,
        )
    ]


def extract_frame_samples(
    video_bytes: bytes,
    max_frames: int = 8,
    *,
    jpeg_quality: int = _JPEG_QUALITY,
    max_pixels: int | None = None,
) -> list[ExtractedFrame]:
    """Extract evenly sampled frames with timestamps.

    When *max_pixels* is set, videos whose coded resolution exceeds it are
    rejected before any frame is decoded (resolution-bomb guard).
    """
    try:
        import av  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(_IMPORT_ERROR_MSG) from exc

    if not video_bytes:
        logger.warning("video_extract_empty_input")
        return []

    try:
        return _extract(video_bytes, max_frames, jpeg_quality, av, max_pixels)
    except Exception as exc:
        logger.warning("video_extract_failed", error=str(exc))
        return []


def probe_video_metadata(video_bytes: bytes) -> VideoMetadata | None:
    """Return basic duration/frame metadata for a video, if available."""
    try:
        import av  # type: ignore[import]
    except ImportError:
        return None

    if not video_bytes:
        return None

    try:
        return _probe_metadata(video_bytes, av)
    except Exception as exc:
        logger.warning("video_probe_failed", error=str(exc))
        return None


def _extract(
    video_bytes: bytes,
    max_frames: int,
    jpeg_quality: int,
    av: object,
    max_pixels: int | None = None,
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

        # Resolution-bomb guard: reject before decoding any frame, since a
        # single oversized frame can allocate gigabytes during decode.
        if max_pixels is not None and max_pixels > 0:
            pixels = _video_stream_pixels(video_stream)
            if pixels > max_pixels:
                logger.warning(
                    "video_extract_resolution_rejected",
                    pixels=pixels,
                    max_pixels=max_pixels,
                )
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


def _probe_metadata(video_bytes: bytes, av: object) -> VideoMetadata | None:
    """Probe video duration and frame count without extracting frames."""
    buf = io.BytesIO(video_bytes)

    with av.open(buf) as container:  # type: ignore[union-attr]
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is None:
            return None

        duration_s = float(
            video_stream.duration * video_stream.time_base
            if video_stream.duration
            else container.duration / 1_000_000
            if container.duration
            else 0
        )
        width = video_stream.width if isinstance(video_stream.width, int) else 0
        height = video_stream.height if isinstance(video_stream.height, int) else 0
        return VideoMetadata(
            duration_s=max(0.0, round(duration_s, 4)),
            frame_count=int(video_stream.frames or 0),
            width=max(0, width),
            height=max(0, height),
        )


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
