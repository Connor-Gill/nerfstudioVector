#!/usr/bin/env python3
"""
Simple vector image loader that samples SVG path fills at continuous coordinates.

Prototype approach:
- Uses svgpathtools to parse path outlines and cssutils to read style attributes (style parsing optional).
- Samples each path into a polyline (N points).
- For a query set of (x,y) pixel coordinates, performs point-in-polygon tests (using matplotlib.path.Path)
  in painter's order to determine fill color at each point.

Limitations:
- Not a full vector renderer (no strokes, gradients, masks beyond simple paint order).
- CPU-based point-in-polygon tests; consider moving to vectorized PyTorch for speed if needed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Optional third-party libs; raise an explicit error if missing.
try:
    import svgpathtools as spt
except Exception as e:  # pragma: no cover - dependency errors should be explicit
    raise RuntimeError("svgpathtools is required for vector parsing. Install via `pip install svgpathtools`.") from e

try:
    import cssutils  # type: ignore
except Exception:
    cssutils = None  # optional; we'll fallback to basic parsing

try:
    from matplotlib.path import Path as MplPath
except Exception as e:
    raise RuntimeError("matplotlib is required for point-in-polygon operations. Install via `pip install matplotlib`.") from e


_SUPPORTED_SVG_EXT = {".svg"}
_SUPPORTED_PDF_EXT = {".pdf"}


def _hex_to_rgb_norm(hexstr: Optional[str]) -> Tuple[float, float, float]:
    """Convert #RRGGBB / #RGB / rgb(...) to normalized (0..1) rgb tuple."""
    if hexstr is None:
        return (0.0, 0.0, 0.0)
    s = str(hexstr).strip()
    if s.startswith("rgb"):
        nums = re.findall(r"[\d\.]+", s)
        if len(nums) >= 3:
            return (float(nums[0]) / 255.0, float(nums[1]) / 255.0, float(nums[2]) / 255.0)
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        r = int(s[0] * 2, 16)
        g = int(s[1] * 2, 16)
        b = int(s[2] * 2, 16)
    elif len(s) == 6:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    else:
        # Unknown color format -> black
        return (0.0, 0.0, 0.0)
    return (r / 255.0, g / 255.0, b / 255.0)


class VectorPrimitive:
    """
    Holds a single vector primitive as a polygon (Nx2 numpy array) and a fill color.
    painter_order: int, lower numbers drawn first.
    """
    def __init__(self, polygon: np.ndarray, fill_rgb: Tuple[float, float, float], painter_order: int):
        self.polygon = polygon.astype(np.float32)  # shape (N,2) in pixel coordinates [x, y] (origin top-left)
        self.fill_rgb = np.array(fill_rgb, dtype=np.float32)
        self.painter_order = painter_order
        self._mpl_path = MplPath(self.polygon) if self.polygon.shape[0] >= 3 else None

    def contains_points(self, pts: np.ndarray) -> np.ndarray:
        """pts: (M,2) numpy array of x,y. returns boolean mask (M,)"""
        if self._mpl_path is None:
            return np.zeros(pts.shape[0], dtype=bool)
        return self._mpl_path.contains_points(pts)


class VectorImage:
    """
    Load an SVG file (or a PDF converted to an SVG externally) and produce a set of VectorPrimitive objects in pixel coordinates.

    Usage:
        vi = VectorImage(svg_path, target_w=1024, target_h=768, samples_per_path=200)
        colors = vi.sample(np.array([[10.0, 20.0], [100.0, 200.0]]))  # shape (2,3) (float32)
    """
    def __init__(self, path: Path, target_w: Optional[int] = None, target_h: Optional[int] = None, samples_per_path: int = 200):
        self.path = Path(path)
        self.samples_per_path = int(samples_per_path)
        self._primitives: List[VectorPrimitive] = []
        self._parse_svg(target_w=target_w, target_h=target_h)

    def _parse_svg(self, target_w: Optional[int], target_h: Optional[int]) -> None:
        # Use svgpathtools to read paths and attributes
        try:
            doc = spt.Document(str(self.path))
        except Exception:
            # fallback to svg2paths
            paths, attributes = spt.svg2paths(str(self.path))
            # Build a minimal Document-like object for required fields
            class _D:
                elements = list(zip(paths, attributes))
                width = None
                height = None
                viewbox = None
            doc = _D()

        svg_w = getattr(doc, "width", None)
        svg_h = getattr(doc, "height", None)

        if target_w is not None and target_h is not None:
            w = int(target_w)
            h = int(target_h)
        else:
            w = int(svg_w) if svg_w is not None else 1024
            h = int(svg_h) if svg_h is not None else 1024

        # gather drawable elements
        elements = []
        try:
            # doc.elements may already be (element...) for svgpathtools.Document
            elements = list(getattr(doc, "elements", []))
        except Exception:
            elements = []

        painter_order = 0
        for el in elements:
            # el may be (path, attrs) tuple or element-like object
            path_obj = None
            attrs = {}
            if isinstance(el, tuple) and len(el) >= 2:
                path_obj, attrs = el[0], el[1]
            else:
                # element-like object
                path_obj = getattr(el, "path", el)
                attrs = getattr(el, "attributes", {}) or getattr(el, "properties", {}) or {}

            # determine fill
            fill = attrs.get("fill", None)
            style = attrs.get("style", None)
            if style:
                if cssutils is not None:
                    decls = cssutils.parseStyle(style)
                    val = decls.getPropertyValue("fill")
                    if val:
                        fill = val
                else:
                    m = re.search(r"fill\s*:\s*([^;]+)", style)
                    if m:
                        fill = m.group(1).strip()

            if fill is None or str(fill).lower() == "none":
                # skip shapes without fill for now
                continue

            fill_rgb = _hex_to_rgb_norm(fill)

            # sample path into points
            try:
                length = path_obj.length()
            except Exception:
                try:
                    length = sum(seg.length() for seg in path_obj)
                except Exception:
                    length = 1.0

            sample_count = max(3, min(self.samples_per_path, int(length) + 3))
            pts = []
            for i in range(sample_count):
                t = i / max(1, sample_count - 1)
                try:
                    p = path_obj.point(t)
                    x, y = float(p.real), float(p.imag)
                except Exception:
                    continue
                pts.append((x, y))
            if len(pts) < 3:
                continue

            # Attempt to map using viewBox if available
            try:
                vb = getattr(doc, "viewbox", None)
                if vb is not None and len(vb) >= 4:
                    minx, miny, vbw, vbh = vb
                    scale_x = w / float(vbw)
                    scale_y = h / float(vbh)
                    pts_px = np.array([((x - minx) * scale_x, (y - miny) * scale_y) for (x, y) in pts], dtype=np.float32)
                else:
                    if svg_w is not None and svg_h is not None:
                        sx = w / float(svg_w)
                        sy = h / float(svg_h)
                        pts_px = np.array([(x * sx, y * sy) for (x, y) in pts], dtype=np.float32)
                    else:
                        pts_px = np.array(pts, dtype=np.float32)
            except Exception:
                pts_px = np.array(pts, dtype=np.float32)

            prim = VectorPrimitive(pts_px, fill_rgb, painter_order)
            self._primitives.append(prim)
            painter_order += 1

        # sort painter order ascending (first drawn at bottom)
        self._primitives.sort(key=lambda p: p.painter_order)

    def sample(self, points_xy: np.ndarray, default_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """
        Sample colors at given points.
        points_xy: (M,2) array of x,y pixel coordinates. Origin top-left.
        Returns (M,3) float32 array in 0..1.
        Painter's algorithm: later primitives override earlier ones.
        """
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            raise ValueError("points_xy must be (M,2)")

        M = points_xy.shape[0]
        colors = np.tile(np.array(default_color, dtype=np.float32)[None, :], (M, 1))

        for prim in self._primitives:
            mask = prim.contains_points(points_xy)
            if not np.any(mask):
                continue
            colors[mask, :] = prim.fill_rgb[None, :]

        return colors
