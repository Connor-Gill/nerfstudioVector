#!/usr/bin/env python3
"""
Vector dataparser helper: loads a transforms.json that points to vector files and
provides a small API to return colors for pixel coordinates.

This class is a small wrapper around VectorImage for integration/experimentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from nerfstudio.data.vector.vector_image import VectorImage


class VectorDataparser:
    def __init__(self, transforms_path: Path, input_dir: Optional[Path] = None, samples_per_path: int = 200):
        self.transforms_path = Path(transforms_path)
        self.input_dir = Path(input_dir) if input_dir is not None else self.transforms_path.parent
        self.samples_per_path = int(samples_per_path)
        with open(self.transforms_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.frames = self.data.get("frames", [])
        self._vector_cache: Dict[str, VectorImage] = {}

    def _resolve_image_path(self, file_path_value: str) -> Path:
        p = Path(file_path_value)
        if not p.is_absolute():
            p = (self.input_dir / p).resolve()
        return p

    def _get_vector_for_frame(self, frame_idx: int) -> VectorImage:
        frame = self.frames[frame_idx]
        fp = frame.get("file_path")
        if fp is None:
            raise RuntimeError("frame missing file_path")
        resolved = self._resolve_image_path(fp)
        key = str(resolved)
        if key not in self._vector_cache:
            w = frame.get("w", None)
            h = frame.get("h", None)
            self._vector_cache[key] = VectorImage(resolved, target_w=w, target_h=h, samples_per_path=self.samples_per_path)
        return self._vector_cache[key]

    def get_frame_size(self, frame_idx: int) -> Tuple[int, int]:
        f = self.frames[frame_idx]
        w = int(f.get("w", 1024))
        h = int(f.get("h", 1024))
        return (w, h)

    def get_frame_colors(self, frame_idx: int, pixel_xy: np.ndarray) -> np.ndarray:
        """
        pixel_xy: (N,2) array with x,y pixel coordinates (origin top-left).
        Returns (N,3) float32 in [0,1].
        """
        vec = self._get_vector_for_frame(frame_idx)
        return vec.sample(pixel_xy)
