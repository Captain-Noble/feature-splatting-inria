#!/usr/bin/env python
import argparse
import re
from pathlib import Path
from typing import List, Tuple

import cv2


STEP_PATTERN = re.compile(r"^(?P<prefix>.+)_step(?P<step>\d+)(?P<tail>(?:_.+)?)\.(png|jpg|jpeg)$", re.IGNORECASE)


def collect_frames(directory: Path, mode: str, recursive: bool) -> List[Path]:
    if recursive:
        candidates = [p for p in directory.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in directory.iterdir() if p.is_file()]

    image_candidates = [p for p in candidates if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    frames: List[Path] = []
    for p in image_candidates:
        m = STEP_PATTERN.match(p.name)
        if not m:
            continue
        tail = m.group("tail")
        if mode == "color" and tail:
            # Keep only main rendered RGB frames like: 0_1_step00001.png
            continue
        frames.append(p)
    return sorted(frames, key=sort_key)


def sort_key(path: Path) -> Tuple[str, int, str]:
    m = STEP_PATTERN.match(path.name)
    if not m:
        return (path.name.lower(), -1, "")
    prefix = m.group("prefix").lower()
    step = int(m.group("step"))
    tail = m.group("tail").lower()
    return (prefix, step, tail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play rendered image sequence in order.")
    parser.add_argument("dir", type=Path, help="Frame directory, e.g. output/.../renders")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS (default: 30)")
    parser.add_argument(
        "--mode",
        choices=["color", "all"],
        default="color",
        help="color: only main RGB frames; all: include _depth/_heatmap etc.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search frames recursively")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--dry-run", action="store_true", help="Print ordered files without opening a window")
    return parser.parse_args()


def play_frames(frames: List[Path], fps: float, loop: bool) -> None:
    delay_ms = max(1, int(1000.0 / max(fps, 0.1)))
    window = "Image Sequence (Space pause/resume, Esc/q quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    paused = False
    idx = 0
    n = len(frames)
    while True:
        frame_path = frames[idx]
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[warn] failed to read: {frame_path}")
        else:
            cv2.imshow(window, image)
            cv2.setWindowTitle(window, f"{frame_path.name} ({idx + 1}/{n})")

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("a"):
            idx = max(0, idx - 1)
            continue
        if key == ord("d"):
            idx = min(n - 1, idx + 1)
            continue
        if paused:
            continue

        idx += 1
        if idx >= n:
            if loop:
                idx = 0
            else:
                break

    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    if not args.dir.exists() or not args.dir.is_dir():
        print(f"[error] directory not found: {args.dir}")
        return 1

    frames = collect_frames(args.dir, args.mode, args.recursive)
    if not frames:
        print("[error] no matched frames found")
        return 2

    print(f"matched frames: {len(frames)}")
    print(f"first: {frames[0].name}")
    print(f"last : {frames[-1].name}")

    if args.dry_run:
        for p in frames:
            print(p)
        return 0

    play_frames(frames, fps=args.fps, loop=args.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
