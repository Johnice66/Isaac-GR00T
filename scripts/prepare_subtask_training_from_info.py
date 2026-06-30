#!/usr/bin/env python
"""Prepare a GR00T LeRobot dataset for subtask-conditioned training.

This script converts stage/subtask annotations stored in ``meta/info.json`` into
the format already supported by ``LeRobotEpisodeLoader``:

    meta/episodes.jsonl:
        {"episode_index": 0, ..., "sub_tasks": [
            {"start": 0, "end": 80, "text": "approach the handle"},
            ...
        ]}

and writes a sibling modality config where the language key is ``sub_task``.

The script is intentionally conservative. If it cannot identify the annotation
schema in ``info.json`` it prints candidate keys and exits without writing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any


SEGMENT_CONTAINER_KEYS = (
    "sub_tasks",
    "subtasks",
    "sub_tasks_info",
    "subtask_info",
    "stages",
    "stage_annotations",
    "phases",
    "phase_annotations",
    "segments",
)
EPISODE_CONTAINER_KEYS = (
    "episodes",
    "episode_annotations",
    "episode_infos",
    "episode_info",
    "episode_metadata",
)
START_KEYS = ("start", "start_frame", "start_index", "start_idx", "frame_start", "begin")
END_KEYS = ("end", "end_frame", "end_index", "end_idx", "frame_end", "stop")
TEXT_KEYS = (
    "text",
    "task",
    "label",
    "name",
    "description",
    "stage",
    "phase",
    "subtask",
    "sub_task",
)
EPISODE_KEYS = ("episode_index", "episode_id", "episode", "episode_idx", "episode_no")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert info.json stage annotations to GR00T sub_task language training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        help="Dataset root path(s), e.g. /root/.../dataset/task_2609 /root/.../dataset/task_2178",
    )
    parser.add_argument(
        "--source-config",
        default="meta/config.py",
        help="Config file relative to each dataset root to use as the template.",
    )
    parser.add_argument(
        "--output-config",
        default="meta/config_subtask.py",
        help="Output config file relative to each dataset root.",
    )
    parser.add_argument(
        "--end-inclusive",
        action="store_true",
        help="Treat annotation end frame as inclusive and convert it to exclusive end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect and print what would be changed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output config and backup files.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def int_like(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value.strip())
    return None


def first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def parse_segment(record: Any, *, end_inclusive: bool) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None

    start_raw = first_value(record, START_KEYS)
    end_raw = first_value(record, END_KEYS)
    text_raw = first_value(record, TEXT_KEYS)

    start = int_like(start_raw)
    end = int_like(end_raw)
    if start is None or end is None or text_raw is None:
        return None

    text = str(text_raw).strip()
    if not text:
        return None

    if end_inclusive:
        end += 1

    return {"start": start, "end": end, "text": text}


def episode_index_from_record(record: dict[str, Any]) -> int | None:
    for key in EPISODE_KEYS:
        if key in record:
            idx = int_like(record[key])
            if idx is not None:
                return idx
    return None


def parse_segment_list(
    value: Any,
    *,
    episode_index: int | None,
    end_inclusive: bool,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(value, list):
        return result

    for item in value:
        if not isinstance(item, dict):
            continue

        item_episode_index = episode_index_from_record(item)

        direct_segment = parse_segment(item, end_inclusive=end_inclusive)
        if direct_segment is not None:
            target_episode = item_episode_index if item_episode_index is not None else episode_index
            if target_episode is not None:
                result[target_episode].append(direct_segment)
            continue

        nested_episode = item_episode_index if item_episode_index is not None else episode_index
        for key in SEGMENT_CONTAINER_KEYS:
            if key in item:
                nested = parse_segment_list(
                    item[key],
                    episode_index=nested_episode,
                    end_inclusive=end_inclusive,
                )
                for ep_idx, segments in nested.items():
                    result[ep_idx].extend(segments)

    return result


def merge_segment_maps(
    left: dict[int, list[dict[str, Any]]],
    right: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    merged: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in (left, right):
        for ep_idx, segments in source.items():
            merged[ep_idx].extend(segments)
    return merged


def extract_from_episode_container(
    value: Any,
    *,
    end_inclusive: bool,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)

    if isinstance(value, dict):
        iterable = []
        for key, item in value.items():
            key_episode_index = int_like(key)
            iterable.append((key_episode_index, item))
    elif isinstance(value, list):
        iterable = [(None, item) for item in value]
    else:
        return result

    for key_episode_index, item in iterable:
        if not isinstance(item, dict):
            continue
        record_episode_index = episode_index_from_record(item)
        episode_index = record_episode_index if record_episode_index is not None else key_episode_index
        if episode_index is None:
            continue

        for key in SEGMENT_CONTAINER_KEYS:
            if key not in item:
                continue
            nested = parse_segment_list(
                item[key],
                episode_index=episode_index,
                end_inclusive=end_inclusive,
            )
            result = merge_segment_maps(result, nested)

    return result


def extract_from_top_level_container(
    value: Any,
    *,
    end_inclusive: bool,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)

    if isinstance(value, dict):
        for key, item in value.items():
            key_episode_index = int_like(key)
            if key_episode_index is not None:
                nested = parse_segment_list(
                    item,
                    episode_index=key_episode_index,
                    end_inclusive=end_inclusive,
                )
                result = merge_segment_maps(result, nested)
                continue

            if isinstance(item, dict):
                episode_index = episode_index_from_record(item)
                for nested_key in SEGMENT_CONTAINER_KEYS:
                    if nested_key in item:
                        nested = parse_segment_list(
                            item[nested_key],
                            episode_index=episode_index,
                            end_inclusive=end_inclusive,
                        )
                        result = merge_segment_maps(result, nested)
            elif isinstance(item, list):
                nested = parse_segment_list(
                    item,
                    episode_index=None,
                    end_inclusive=end_inclusive,
                )
                result = merge_segment_maps(result, nested)

    elif isinstance(value, list):
        nested = parse_segment_list(value, episode_index=None, end_inclusive=end_inclusive)
        result = merge_segment_maps(result, nested)

    return result


def extract_subtasks_from_info(
    info: dict[str, Any],
    *,
    end_inclusive: bool,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for key in EPISODE_CONTAINER_KEYS:
        if key in info:
            nested = extract_from_episode_container(info[key], end_inclusive=end_inclusive)
            result = merge_segment_maps(result, nested)

    for key in SEGMENT_CONTAINER_KEYS:
        if key in info:
            nested = extract_from_top_level_container(info[key], end_inclusive=end_inclusive)
            result = merge_segment_maps(result, nested)

    normalized: dict[int, list[dict[str, Any]]] = {}
    for ep_idx, segments in result.items():
        unique = []
        seen = set()
        for segment in sorted(segments, key=lambda x: (x["start"], x["end"], x["text"])):
            signature = (segment["start"], segment["end"], segment["text"])
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(segment)
        normalized[ep_idx] = unique
    return normalized


def summarize_info_shape(info: Any, prefix: str = "", depth: int = 0) -> list[str]:
    if depth > 3:
        return []
    rows: list[str] = []
    if isinstance(info, dict):
        for key, value in list(info.items())[:80]:
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                rows.append(f"{path}: dict({len(value)})")
                rows.extend(summarize_info_shape(value, path, depth + 1))
            elif isinstance(value, list):
                rows.append(f"{path}: list({len(value)})")
                if value:
                    rows.extend(summarize_info_shape(value[0], path + "[0]", depth + 1))
            else:
                rows.append(f"{path}: {type(value).__name__}")
    return rows


def validate_segments(
    dataset_path: Path,
    episodes: list[dict[str, Any]],
    segments_by_episode: dict[int, list[dict[str, Any]]],
) -> list[str]:
    warnings: list[str] = []
    episode_lengths = {int(ep["episode_index"]): int(ep["length"]) for ep in episodes}
    known_episodes = set(episode_lengths)
    annotated_episodes = set(segments_by_episode)

    missing = sorted(known_episodes - annotated_episodes)
    extra = sorted(annotated_episodes - known_episodes)
    if missing:
        warnings.append(
            f"{dataset_path}: {len(missing)} episodes have no sub_tasks; first missing: {missing[:10]}"
        )
    if extra:
        warnings.append(
            f"{dataset_path}: {len(extra)} subtask episode ids are not in episodes.jsonl: {extra[:10]}"
        )

    for ep_idx, segments in segments_by_episode.items():
        length = episode_lengths.get(ep_idx)
        if length is None:
            continue
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            if start < 0 or end <= start or end > length:
                warnings.append(
                    f"{dataset_path}: episode {ep_idx} has out-of-range subtask "
                    f"{segment} for length {length}"
                )
    return warnings


def with_backup(path: Path, *, force: bool) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{timestamp}")
    if backup.exists() and not force:
        raise FileExistsError(f"Backup already exists: {backup}")
    shutil.copy2(path, backup)
    return backup


def write_subtask_config(source_config: Path, output_config: Path, *, force: bool) -> None:
    if output_config.exists() and not force:
        raise FileExistsError(
            f"{output_config} already exists. Pass --force to overwrite, or remove it manually."
        )

    text = source_config.read_text(encoding="utf-8")
    updated = text
    updated = updated.replace('"annotation.human.task_description"', '"sub_task"')
    updated = updated.replace("'annotation.human.task_description'", "'sub_task'")

    if updated == text and '"sub_task"' not in text and "'sub_task'" not in text:
        raise ValueError(
            f"Could not update language key in {source_config}. "
            "Expected annotation.human.task_description. Please edit language.modality_keys "
            "to ['sub_task'] manually."
        )

    output_config.write_text(updated, encoding="utf-8")


def process_dataset(dataset_path: Path, args: argparse.Namespace) -> None:
    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    source_config = dataset_path / args.source_config
    output_config = dataset_path / args.output_config

    if not info_path.exists():
        raise FileNotFoundError(info_path)
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)
    if not source_config.exists():
        raise FileNotFoundError(source_config)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = read_jsonl(episodes_path)
    segments_by_episode = extract_subtasks_from_info(info, end_inclusive=args.end_inclusive)

    if not segments_by_episode:
        print(f"\n[FAILED] {dataset_path}")
        print("No usable subtask annotations were detected in meta/info.json.")
        print("Recognized segment fields:")
        print(f"  start: {START_KEYS}")
        print(f"  end:   {END_KEYS}")
        print(f"  text:  {TEXT_KEYS}")
        print("\nTop-level info.json shape:")
        for row in summarize_info_shape(info)[:120]:
            print(" ", row)
        raise SystemExit(2)

    warnings = validate_segments(dataset_path, episodes, segments_by_episode)
    print(f"\n[OK] {dataset_path}")
    print(f"Detected subtask annotations for {len(segments_by_episode)} episodes.")
    total_segments = sum(len(v) for v in segments_by_episode.values())
    print(f"Detected {total_segments} total subtask segments.")
    for ep_idx in sorted(segments_by_episode)[:3]:
        print(f"  episode {ep_idx}: {segments_by_episode[ep_idx][:5]}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings[:40]:
            print(" ", warning)
        if len(warnings) > 40:
            print(f"  ... {len(warnings) - 40} more warnings")
        if any("out-of-range" in warning for warning in warnings):
            raise SystemExit("Refusing to write because at least one segment is out of range.")

    if args.dry_run:
        print("Dry run only; no files written.")
        return

    backup_path = with_backup(episodes_path, force=args.force)
    updated_episodes = []
    for episode in episodes:
        updated = deepcopy(episode)
        ep_idx = int(updated["episode_index"])
        if ep_idx in segments_by_episode:
            updated["sub_tasks"] = segments_by_episode[ep_idx]
        updated_episodes.append(updated)
    write_jsonl(episodes_path, updated_episodes)
    write_subtask_config(source_config, output_config, force=args.force)

    print(f"Wrote: {episodes_path}")
    print(f"Backup: {backup_path}")
    print(f"Wrote: {output_config}")
    print("Use this config for training:")
    print(f"  --modality-config-path {output_config}")


def main() -> None:
    args = parse_args()
    for dataset_path_raw in args.dataset_paths:
        process_dataset(Path(dataset_path_raw).expanduser().resolve(), args)


if __name__ == "__main__":
    main()
