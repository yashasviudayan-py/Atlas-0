#!/usr/bin/env python3
"""Generate a deterministic sample walkthrough fixture for upload reports."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def _draw_frame(index: int, total: int) -> Image.Image:
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), (241, 232, 219))
    draw = ImageDraw.Draw(img)

    pan = int((index - (total - 1) / 2) * 28)

    # Background walls / floor.
    draw.rectangle((0, int(height * 0.62), width, height), fill=(215, 198, 176))
    draw.rectangle((0, 0, width, int(height * 0.62)), fill=(246, 239, 228))
    draw.rectangle((width * 0.08 + pan, height * 0.18, width * 0.42 + pan, height * 0.56), fill=(230, 220, 205))

    # Left book stack on a low console.
    console_x = 180 + pan
    draw.rectangle((console_x - 80, 430, console_x + 120, 455), fill=(120, 97, 74))
    draw.rectangle((console_x - 46, 356, console_x + 6, 430), fill=(120, 74, 44))
    draw.rectangle((console_x - 10, 338, console_x + 46, 430), fill=(157, 104, 63))
    draw.rectangle((console_x + 28, 320, console_x + 86, 430), fill=(189, 126, 70))

    # Center table with a blue vase near the front edge.
    table_x = 650 + pan
    draw.rectangle((table_x - 170, 400, table_x + 170, 425), fill=(146, 108, 75))
    draw.rectangle((table_x - 150, 425, table_x - 120, 520), fill=(120, 86, 60))
    draw.rectangle((table_x + 120, 425, table_x + 150, 520), fill=(120, 86, 60))
    draw.ellipse((table_x - 20, 310, table_x + 40, 390), fill=(94, 176, 242))
    draw.rectangle((table_x - 6, 284, table_x + 10, 325), fill=(94, 176, 242))
    draw.ellipse((table_x - 16, 278, table_x + 20, 300), fill=(94, 176, 242))

    # Right tall lamp.
    lamp_x = 1010 + pan
    draw.rectangle((lamp_x - 12, 182, lamp_x + 10, 560), fill=(86, 92, 101))
    draw.polygon(
        [
            (lamp_x - 64, 154),
            (lamp_x + 50, 154),
            (lamp_x + 28, 254),
            (lamp_x - 42, 254),
        ],
        fill=(171, 177, 186),
    )
    draw.ellipse((lamp_x - 62, 548, lamp_x + 54, 604), fill=(74, 80, 88))

    # Mild shadow accents.
    draw.ellipse((table_x - 38, 388, table_x + 58, 436), fill=(124, 119, 118))
    draw.ellipse((lamp_x - 70, 592, lamp_x + 70, 642), fill=(150, 140, 132))

    return img


def main() -> None:
    out_dir = Path(__file__).parent.parent / "data" / "sample_walkthrough" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 5
    for index in range(total):
        frame = _draw_frame(index, total)
        frame.save(out_dir / f"frame_{index + 1:02d}.jpg", format="JPEG", quality=92, optimize=True)


if __name__ == "__main__":
    main()
