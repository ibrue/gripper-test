"""umi dataset — episode-based recording layout.

A dataset is a directory. Each recording session is one *episode* —
a subdirectory holding everything captured during that take:

    ~/umi-data/
      20260510-143022-pick-block/
        samples.csv      synchronized signals (servo state + VO pose + IMU)
        video.mp4        camera feed (optional)
        meta.json        name, task, notes, duration, sample/frame counts,
                         gripper config, slam backend, software version
      20260510-150114-pour-cup/
        ...

The studio's "Dataset" tab uses ``DatasetManager`` to list, tag, delete,
and reveal episodes; the recorder writes new ones into the same root.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Episode:
    path: str
    name: str
    task: str = ""
    notes: str = ""
    started_at: float = 0.0
    duration_s: float = 0.0
    n_samples: int = 0
    n_frames: int = 0
    gripper: dict = field(default_factory=dict)
    slam_backend: str = ""
    software: str = "umi"

    @property
    def samples_path(self) -> str:
        return os.path.join(self.path, "samples.csv")

    @property
    def video_path(self) -> str:
        return os.path.join(self.path, "video.mp4")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.path, "meta.json")

    @property
    def has_video(self) -> bool:
        return os.path.exists(self.video_path)

    def save_meta(self) -> None:
        with open(self.meta_path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Episode":
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                data["path"] = path
                fields = cls.__dataclass_fields__
                return cls(**{k: v for k, v in data.items() if k in fields})
            except (OSError, json.JSONDecodeError):
                pass
        return cls(path=path, name=os.path.basename(path), started_at=os.path.getmtime(path))


class DatasetManager:
    def __init__(self, root: str):
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)

    def list_episodes(self) -> list[Episode]:
        if not os.path.isdir(self.root):
            return []
        items = []
        for entry in os.listdir(self.root):
            full = os.path.join(self.root, entry)
            if not os.path.isdir(full):
                continue
            try:
                items.append(Episode.load(full))
            except Exception:
                continue
        # Most recent first.
        items.sort(key=lambda ep: ep.started_at or 0, reverse=True)
        return items

    def new_episode(self, task: str = "", suffix: str = "") -> Episode:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dirname = stamp
        if suffix:
            safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in suffix.strip())
            if safe:
                dirname = f"{stamp}-{safe}"
        path = os.path.join(self.root, dirname)
        os.makedirs(path, exist_ok=True)
        ep = Episode(
            path=path,
            name=dirname,
            task=task,
            started_at=time.time(),
        )
        ep.save_meta()
        return ep

    def delete(self, episode: Episode) -> None:
        if os.path.isdir(episode.path):
            shutil.rmtree(episode.path)

    def update_notes(self, episode: Episode, notes: str, task: Optional[str] = None) -> None:
        episode.notes = notes
        if task is not None:
            episode.task = task
        episode.save_meta()
