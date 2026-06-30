#!/usr/bin/env python3
"""Convert Genie instruction_segments into GR00T episodes.jsonl sub_tasks.

Place this file in a dataset root, for example:

    /root/gpufree-data/Isaac-GR00T/dataset/task_2609/convert_instruction_segments_to_subtasks.py

Then run:

    python convert_instruction_segments_to_subtasks.py

It reads:

    meta/info.json["instruction_segments"]

and writes:

    meta/episodes.jsonl[*]["sub_tasks"]

A timestamped backup of episodes.jsonl is created before writing.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert info.json instruction_segments to episodes.jsonl sub_tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset root. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing sub_tasks if episodes.jsonl already contains them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a summary without writing files.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_dataset(dataset_root: Path, *, force: bool, dry_run: bool) -> None:
    dataset_root = dataset_root.resolve()
    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing {episodes_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if "instruction_segments" not in info:
        raise KeyError(f"{info_path} has no instruction_segments field")

    instruction_segments = info["instruction_segments"]
    episodes = read_jsonl(episodes_path)

    if not force and any("sub_tasks" in ep for ep in episodes):
        raise RuntimeError(
            f"{episodes_path} already contains sub_tasks. "
            "Use --force if you intentionally want to overwrite them."
        )

    converted = 0
    total_subtasks = 0
    clipped = 0

    for episode in episodes:
        ep_idx = str(episode["episode_index"])
        if ep_idx not in instruction_segments:
            raise KeyError(f"Missing instruction_segments for episode {ep_idx}")

        length = int(episode["length"])
        sub_tasks = []

        for segment in instruction_segments[ep_idx]:
            start = int(segment["start_frame_index"])
            raw_end = int(segment["end_frame_index"])
            end = min(raw_end, length)
            text = str(segment["instruction"]).strip()

            if raw_end != end:
                clipped += 1
            if start < end and text:
                sub_tasks.append({"start": start, "end": end, "text": text})

        episode["sub_tasks"] = sub_tasks
        converted += 1
        total_subtasks += len(sub_tasks)

    print(f"dataset: {dataset_root}")
    print(f"episodes: {converted}")
    print(f"sub_tasks: {total_subtasks}")
    print(f"clipped end_frame_index values: {clipped}")
    if episodes:
        print("first episode sub_tasks:")
        print(json.dumps(episodes[0].get("sub_tasks", []), ensure_ascii=False, indent=2))

    if dry_run:
        print("dry-run: no files written")
        return

    backup_path = episodes_path.with_suffix(
        ".jsonl.bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    shutil.copy2(episodes_path, backup_path)
    write_jsonl(episodes_path, episodes)

    print(f"updated: {episodes_path}")
    print(f"backup:  {backup_path}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).expanduser() if args.dataset else Path(__file__).parent
    convert_dataset(dataset_root, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
