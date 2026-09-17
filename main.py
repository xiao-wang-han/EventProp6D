

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_code_dir = os.path.dirname(os.path.realpath(__file__))
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

import cv2
import imageio
import logging
import nvdiffrast.torch as dr
import numpy as np
import torch
import trimesh
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import (
    draw_posed_3d_box,
    draw_xyz_axis,
    glcam_in_cvcam,
    projection_matrix_from_intrinsics,
    set_logging_format,
    set_seed,
)
from offscreen_renderer import ModelRendererOffscreen
from scipy.spatial.transform import Rotation

try:
    import yaml
except Exception as exc:  # pragma: no cover - 用户环境的导入阶段保护
    raise RuntimeError("PyYAML is required to read Event6D camchain calibration.") from exc


def parse_frame_list(spec: str | None, n_frames: int) -> list[int]:
    if not spec:
        return list(range(n_frames))
    selected: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            selected.extend(range(int(a), int(b) + 1))
        else:
            selected.append(int(part))
    return [i for i in selected if 0 <= i < n_frames]


def parse_float_list(spec: str) -> list[float]:
    values = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def parse_frame_number_set(spec: str | None) -> set[int]:
    """Parse comma-separated frame numbers and inclusive ranges."""
    selected: set[int] = set()
    if not spec:
        return selected
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_text, b_text = part.split("-", 1)
            a = int(a_text.strip())
            b = int(b_text.strip())
            lo, hi = sorted((a, b))
            selected.update(range(lo, hi + 1))
        else:
            selected.add(int(part))
    return selected


def parse_cam_from_kalibr(camchain_path: str, cam_key: str) -> tuple[np.ndarray, tuple[int, int]]:
    with open(camchain_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    node = data[cam_key]
    fx, fy, cx, cy = map(float, node["intrinsics"])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    width, height = map(int, node["resolution"])
    return K, (height, width)


def load_rgb_to_event(camchain_path: str) -> np.ndarray:
    with open(camchain_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.asarray(data["cam1"]["T_cn_cnm1"], dtype=np.float64)


def invert_transform(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def relative_pose(a_in_cam: np.ndarray, b_in_cam: np.ndarray) -> np.ndarray:
    return np.linalg.inv(a_in_cam) @ b_in_cam


def rotation_error_deg(a_in_cam: np.ndarray, b_in_cam: np.ndarray) -> float:
    rel = relative_pose(a_in_cam, b_in_cam)
    return float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rel[:3, :3]).as_rotvec())))


def first_existing_path(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0]


def default_event6d_root() -> str:
    if os.name == "nt":
        return first_existing_path(
            r"D:\data\Event6D",
            r"/home/wushiqing/dataset/Event6D",
            r"/home/wushiqing/dataset/Event6d",
        )
    return first_existing_path(
        r"/home/wushiqing/dataset/Event6D",
        r"/home/wushiqing/dataset/Event6d",
        r"D:\data\Event6D",
    )


class Event6DSequenceReader:
    """供 FoundationPose-main 入口使用的轻量 Event6D 读取器。"""

    def __init__(
        self,
        sequence_dir: str,
        dataset_root: str,
        camchain_path: str | None = None,
        depth_scale: float = 0.001,
        zfar: float = np.inf,
        use_startend: bool = False,
        initial_mask_file: str | None = None,
    ):
        self.sequence_dir = os.path.abspath(sequence_dir)
        self.dataset_root = os.path.abspath(dataset_root)
        self.camchain_path = camchain_path or os.path.join(self.dataset_root, "0001-camchain.yaml")
        self.depth_scale = float(depth_scale)
        self.zfar = float(zfar)
        self.use_startend = bool(use_startend)

        rgb_dir = Path(self.sequence_dir, "rgb")
        valid_rgb_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.color_files_all = sorted(
            (p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_rgb_extensions),
            key=lambda p: int(p.stem),
        ) if rgb_dir.exists() else []
        if not self.color_files_all:
            raise RuntimeError(f"No numbered RGB frames found under {self.sequence_dir}/rgb")

        if os.path.exists(self.camchain_path):
            self.K, (self.H, self.W) = parse_cam_from_kalibr(self.camchain_path, "cam0")
            self.K_event, (self.H_event, self.W_event) = parse_cam_from_kalibr(self.camchain_path, "cam1")
            self.T_rgb_to_event = load_rgb_to_event(self.camchain_path)
            logging.info(f"Using Event6D camera calibration: {self.camchain_path}")
        else:
            intrinsics_path = Path(self.sequence_dir, "cam_K.txt")
            if not intrinsics_path.exists():
                raise RuntimeError(
                    f"Camera calibration not found: {self.camchain_path}; "
                    f"RGB-only event data require {intrinsics_path}"
                )
            self.K = np.loadtxt(str(intrinsics_path), dtype=np.float64).reshape(3, 3)
            first_color = cv2.imread(str(self.color_files_all[0]), cv2.IMREAD_COLOR)
            if first_color is None:
                raise RuntimeError(f"Unable to read RGB frame: {self.color_files_all[0]}")
            self.H, self.W = first_color.shape[:2]
            # Synthetic events generated from these RGB frames share the RGB
            # optical center, resolution, and camera coordinate system.
            self.K_event = self.K.copy()
            self.H_event, self.W_event = self.H, self.W
            self.T_rgb_to_event = np.eye(4, dtype=np.float64)
            logging.info(
                f"Using co-located synthetic event calibration from {intrinsics_path} "
                f"at {self.W}x{self.H}"
            )
        self.T_event_to_rgb = invert_transform(self.T_rgb_to_event)
        self.last_event_path: Path | None = None
        self.last_event_frame_num: int | None = None
        self.last_event_read_reason = ""

        obj_path = os.path.join(self.sequence_dir, "obj.txt")
        if os.path.exists(obj_path):
            with open(obj_path, "r", encoding="utf-8") as f:
                self.obj_id = f.readline().strip()
        else:
            self.obj_id = ""

        self.startend_start = None
        self.startend_end = None
        startend_path = os.path.join(self.sequence_dir, "startend.txt")
        if os.path.exists(startend_path):
            with open(startend_path, "r", encoding="utf-8") as f:
                start_s, end_s = f.readline().strip().split()[:2]
            self.startend_start = int(start_s)
            self.startend_end = int(end_s)

        if self.use_startend and self.startend_start is not None and self.startend_end is not None:
            self.color_files = [
                p for p in self.color_files_all
                if self.startend_start <= int(p.stem) <= self.startend_end
            ]
        else:
            if self.use_startend:
                logging.warning(f"{startend_path} is missing; using every RGB frame")
            self.color_files = list(self.color_files_all)
        if not self.color_files:
            if self.use_startend:
                raise RuntimeError(f"No frames in start/end range {self.startend_start}-{self.startend_end}")
            raise RuntimeError(f"No RGB frames available under {self.sequence_dir}/rgb")
        self.id_strs = [p.stem for p in self.color_files]
        self.start = int(self.color_files[0].stem)
        self.end = int(self.color_files[-1].stem)

        mask_files = []
        for mask_dir_name in ("mask", "masks"):
            mask_dir = Path(self.sequence_dir, mask_dir_name)
            if mask_dir.exists():
                mask_files.extend(
                    p for p in mask_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".npy", ".png", ".jpg", ".jpeg"}
                )
        mask_files = sorted(mask_files, key=lambda p: int(p.stem))
        if not mask_files:
            raise RuntimeError(f"No initial mask found under {self.sequence_dir}/mask or masks")
        if initial_mask_file:
            self.initial_mask_path = Path(initial_mask_file)
            if not self.initial_mask_path.exists():
                raise RuntimeError(f"--initial_mask_file does not exist: {self.initial_mask_path}")
        else:
            matching_masks = [p for p in mask_files if p.stem == self.id_strs[0]]
            self.initial_mask_path = matching_masks[-1] if matching_masks else mask_files[0]
        if self.initial_mask_path.stem != self.id_strs[0]:
            logging.warning(
                f"Initial mask name {self.initial_mask_path.name} does not match first RGB frame {self.id_strs[0]}. "
                "If --use_startend is enabled this usually means the first registration mask is not aligned."
            )
        logging.info(
            f"Using initial mask {self.initial_mask_path} for first RGB frame {self.id_strs[0]} "
            f"(use_startend={self.use_startend})"
        )
        if self.initial_mask_path.suffix.lower() == ".npy":
            self.initial_mask = np.load(str(self.initial_mask_path)).astype(bool)
        else:
            initial_mask_image = cv2.imread(str(self.initial_mask_path), cv2.IMREAD_GRAYSCALE)
            if initial_mask_image is None:
                raise RuntimeError(f"Unable to read initial mask: {self.initial_mask_path}")
            self.initial_mask = initial_mask_image > 0
        if self.initial_mask.shape[:2] != (self.H, self.W):
            self.initial_mask = cv2.resize(
                self.initial_mask.astype(np.uint8), (self.W, self.H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    def __len__(self) -> int:
        return len(self.color_files)

    def frame_number(self, i: int) -> int:
        return int(self.color_files[i].stem)

    def get_color(self, i: int) -> np.ndarray:
        color = cv2.cvtColor(cv2.imread(str(self.color_files[i])), cv2.COLOR_BGR2RGB)
        if color.shape[:2] != (self.H, self.W):
            color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return color

    def get_depth(self, i: int) -> np.ndarray:
        depth_name = f"{self.color_files[i].stem}.png"
        depth_candidates = [
            Path(self.sequence_dir, "depth_aligned_to_color", depth_name),
            Path(self.sequence_dir, "depth", depth_name),
        ]
        depth_path = next((p for p in depth_candidates if p.exists()), depth_candidates[0])
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise RuntimeError(f"Missing depth: {depth_path}")
        depth = depth.astype(np.float32) * self.depth_scale
        if depth.shape[:2] != (self.H, self.W):
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0.0
        return depth

    def get_initial_mask(self) -> np.ndarray:
        return self.initial_mask.copy()

    def get_mesh_path(self) -> str:
        if self.obj_id:
            event6d_mesh = os.path.join(self.dataset_root, "simple_mesh", self.obj_id, "textured.obj")
            if os.path.exists(event6d_mesh):
                return event6d_mesh
        local_meshes = sorted(Path(self.sequence_dir, "mesh").glob("*.obj"))
        if local_meshes:
            return str(local_meshes[0])
        raise RuntimeError("No mesh found; pass --mesh_file or provide mesh/*.obj")

    def get_raw_events_for_frame(
        self,
        i: int,
        event_frame_offset: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 跟踪 prev_rgb -> current_rgb 时使用偏移 0，即 parsed_events/current_rgb.npz。
        # Event6D 在线读取器在生成 rgb_idx 之后的事件时使用 rgb_idx+1。
        frame_num = self.frame_number(i)
        requested_event_num = frame_num + int(event_frame_offset)
        event_path = Path(self.sequence_dir, "parsed_events", f"{requested_event_num:06d}.npz")
        self.last_event_path = event_path
        self.last_event_frame_num = requested_event_num
        self.last_event_read_reason = "requested"
        if not event_path.exists():
            fallback_num = requested_event_num + 1
            event_path = Path(self.sequence_dir, "parsed_events", f"{fallback_num:06d}.npz")
            self.last_event_path = event_path
            self.last_event_frame_num = fallback_num
            self.last_event_read_reason = "fallback+1"
        if not event_path.exists():
            self.last_event_read_reason = "missing"
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float32),
            )
        data = np.load(str(event_path), allow_pickle=True)["data"]
        return data["x"], data["y"], data["t"], data["p"]


class ProjectionMaskRenderer:
    def __init__(self, mesh: trimesh.Trimesh, K: np.ndarray, H: int, W: int, zfar: float):
        self.mesh = mesh
        self.K = K
        self.H = H
        self.W = W
        self.renderer = None
        try:
            self.renderer = ModelRendererOffscreen(cam_K=K, H=H, W=W, zfar=zfar)
        except Exception as exc:
            logging.warning(f"Offscreen renderer unavailable, using hull fallback: {exc}")

    def render_mask(self, ob_in_cam: np.ndarray) -> np.ndarray:
        depth = self.render_depth(ob_in_cam)
        return (depth > 0).astype(np.uint8)

    def render_depth(self, ob_in_cam: np.ndarray) -> np.ndarray:
        if self.renderer is not None:
            try:
                _, depth = self.renderer.render(mesh=self.mesh, ob_in_cvcam=ob_in_cam)
                return depth.astype(np.float32)
            except Exception as exc:
                logging.warning(f"Render failed, using hull fallback: {exc}")
                self.renderer = None
        return self._project_hull_depth(ob_in_cam)

    def _project_hull_depth(self, ob_in_cam: np.ndarray) -> np.ndarray:
        depth = np.zeros((self.H, self.W), dtype=np.float32)
        verts = np.asarray(self.mesh.vertices)
        verts_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
        pts = (ob_in_cam @ verts_h.T).T[:, :3]
        valid = pts[:, 2] > 1e-6
        if valid.sum() < 3:
            return depth
        uvw = (self.K @ pts[valid].T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        inside = (
            (uv[:, 0] >= 0) & (uv[:, 0] < self.W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
        )
        uv_inside = uv[inside]
        if len(uv_inside) < 3:
            return depth
        hull = cv2.convexHull(uv_inside.astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(depth, hull, float(np.median(pts[valid][inside, 2])))
        return depth

    def _project_hull_mask(self, ob_in_cam: np.ndarray) -> np.ndarray:
        verts = np.asarray(self.mesh.vertices)
        verts_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
        pts = (ob_in_cam @ verts_h.T).T[:, :3]
        valid = pts[:, 2] > 1e-6
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        if valid.sum() < 3:
            return mask
        uvw = (self.K @ pts[valid].T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        inside = (
            (uv[:, 0] >= 0) & (uv[:, 0] < self.W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
        )
        uv = uv[inside]
        if len(uv) < 3:
            return mask
        hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 1)
        return mask


def build_lopet_model_lines(
    mesh: trimesh.Trimesh,
    max_lines: int,
    sharp_angle_deg: float,
    min_length_m: float,
) -> np.ndarray:
    """从 CAD 网格中提取紧凑的三维线段端点，用于 LOPET 匹配。"""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    edge_blocks: list[np.ndarray] = []

    if hasattr(mesh, "face_adjacency_edges") and hasattr(mesh, "face_adjacency_angles"):
        adjacency_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
        adjacency_angles = np.abs(np.asarray(mesh.face_adjacency_angles, dtype=np.float64))
        if len(adjacency_edges) > 0 and len(adjacency_angles) == len(adjacency_edges):
            sharp = adjacency_angles >= np.deg2rad(float(sharp_angle_deg))
            if np.any(sharp):
                edge_blocks.append(adjacency_edges[sharp])

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size > 0:
        all_edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        all_edges = np.sort(all_edges, axis=1)
        unique_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        if len(boundary_edges) > 0:
            edge_blocks.append(boundary_edges)

    if not edge_blocks:
        edge_blocks.append(np.asarray(mesh.edges_unique, dtype=np.int64))

    edges = np.vstack(edge_blocks)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    keep = lengths >= max(float(min_length_m), 0.0)
    edges = edges[keep]
    lengths = lengths[keep]
    if len(edges) == 0:
        raise RuntimeError("No CAD model lines could be extracted from mesh.")

    order = np.argsort(lengths)[::-1]
    max_lines = int(max_lines)
    if max_lines > 0:
        order = order[:max_lines]
    edges = edges[order]
    return verts[edges].astype(np.float64)


def project_lopet_model_lines(
    model_lines: np.ndarray,
    pose_event: np.ndarray,
    K_event: np.ndarray,
    H: int,
    W: int,
    min_projected_length_px: float,
    fov_margin_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    pts = model_lines.reshape(-1, 3)
    pts_cam = (pose_event[:3, :3] @ pts.T).T + pose_event[:3, 3]
    pts_cam = pts_cam.reshape(-1, 2, 3)
    z_ok = np.all(pts_cam[:, :, 2] > 1e-6, axis=1)

    uv = np.zeros((len(model_lines), 2, 2), dtype=np.float64)
    valid_z = pts_cam[:, :, 2] > 1e-6
    uvw = (K_event @ pts_cam.reshape(-1, 3).T).T.reshape(-1, 2, 3)
    uv[valid_z] = uvw[:, :, :2][valid_z] / uvw[:, :, 2:3][valid_z]

    line_len = np.linalg.norm(uv[:, 0] - uv[:, 1], axis=1)
    x_min = np.min(uv[:, :, 0], axis=1)
    x_max = np.max(uv[:, :, 0], axis=1)
    y_min = np.min(uv[:, :, 1], axis=1)
    y_max = np.max(uv[:, :, 1], axis=1)
    margin = float(fov_margin_px)
    overlaps_fov = (x_max >= -margin) & (x_min < W + margin) & (y_max >= -margin) & (y_min < H + margin)
    valid = z_ok & overlaps_fov & (line_len >= float(min_projected_length_px))
    valid_idx = np.nonzero(valid)[0]
    return uv[valid], valid_idx


def split_events(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    sub_windows: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if len(t) == 0 or sub_windows <= 1:
        return [(x, y, t, p)]
    t0 = float(t[0])
    t1 = float(t[-1])
    if t1 <= t0:
        return [(x, y, t, p)]
    windows = []
    for j in range(sub_windows):
        a = t0 + (t1 - t0) * j / sub_windows
        b = t0 + (t1 - t0) * (j + 1) / sub_windows
        m = (t >= a) & (t <= b if j == sub_windows - 1 else t < b)
        windows.append((x[m], y[m], t[m], p[m]))
    return windows


def filter_events_by_mask(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    mask: np.ndarray,
    dilation_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x) == 0:
        return x, y, t, p
    m = mask.astype(np.uint8)
    if dilation_px > 0:
        k = 2 * int(dilation_px) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        m = cv2.dilate(m, kernel, iterations=1)
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    inside = (xi >= 0) & (xi < m.shape[1]) & (yi >= 0) & (yi < m.shape[0])
    keep = np.zeros(len(x), dtype=bool)
    keep[inside] = m[yi[inside], xi[inside]] > 0
    return x[keep], y[keep], t[keep], p[keep]


def event_pixels_to_rgb_pixels(
    x_event: np.ndarray,
    y_event: np.ndarray,
    z_event: np.ndarray,
    p_event: np.ndarray,
    K_event: np.ndarray,
    K_rgb: np.ndarray,
    T_event_to_rgb: np.ndarray,
    H_rgb: int,
    W_rgb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将事件相机像素转换到 RGB 相机像素坐标系。"""
    valid = np.isfinite(z_event) & (z_event > 1e-6)
    if valid.sum() == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=p_event.dtype),
        )

    z = z_event[valid].astype(np.float64)
    xe = (x_event[valid].astype(np.float64) - K_event[0, 2]) * z / K_event[0, 0]
    ye = (y_event[valid].astype(np.float64) - K_event[1, 2]) * z / K_event[1, 1]
    pts_event = np.stack([xe, ye, z], axis=0)
    pts_rgb = T_event_to_rgb[:3, :3] @ pts_event + T_event_to_rgb[:3, 3:4]

    in_front = pts_rgb[2] > 1e-6
    if in_front.sum() == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=p_event.dtype),
        )
    pts_rgb = pts_rgb[:, in_front]
    pol = p_event[valid][in_front]
    uvw = K_rgb @ pts_rgb
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    inside = (u >= 0) & (u < W_rgb) & (v >= 0) & (v < H_rgb)
    return (
        np.rint(u[inside]).astype(np.int32),
        np.rint(v[inside]).astype(np.int32),
        pol[inside],
    )


def bbox_corners_from_bounds(bbox: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bbox, dtype=np.float64).reshape(2, 3)
    mn, mx = bounds[0], bounds[1]
    return np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )


def project_bbox_xyxy(
    K: np.ndarray,
    pose_center_in_cam: np.ndarray,
    bbox: np.ndarray,
    H: int,
    W: int,
) -> tuple[float, float, float, float] | None:
    corners = bbox_corners_from_bounds(bbox)
    pts = (pose_center_in_cam[:3, :3] @ corners.T).T + pose_center_in_cam[:3, 3]
    valid = pts[:, 2] > 1e-6
    if valid.sum() < 2:
        return None
    uvw = (K @ pts[valid].T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    finite = np.isfinite(uv).all(axis=1)
    if finite.sum() < 2:
        return None
    uv = uv[finite]
    x0, y0 = np.min(uv, axis=0)
    x1, y1 = np.max(uv, axis=0)
    if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def mask_to_xyxy(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def square_roi_from_xyxy(
    xyxy: tuple[float, float, float, float] | None,
    H: int,
    W: int,
    context_scale: float,
    min_size: float,
) -> tuple[float, float, float, float]:
    if xyxy is None:
        side = float(max(H, W))
        cx = 0.5 * float(W)
        cy = 0.5 * float(H)
    else:
        x0, y0, x1, y1 = xyxy
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        side = max(float(x1 - x0), float(y1 - y0))
    side = max(side * max(float(context_scale), 1.0), float(min_size), 1.0)
    return cx - 0.5 * side, cy - 0.5 * side, cx + 0.5 * side, cy + 0.5 * side


def roi_camera_matrix(
    K: np.ndarray,
    roi: tuple[float, float, float, float],
    out_h: int,
    out_w: int,
) -> np.ndarray:
    x0, y0, x1, y1 = roi
    sx = float(out_w) / max(float(x1 - x0), 1e-6)
    sy = float(out_h) / max(float(y1 - y0), 1e-6)
    K_roi = np.asarray(K, dtype=np.float64).copy()
    K_roi[0, 0] *= sx
    K_roi[1, 1] *= sy
    K_roi[0, 2] = (K_roi[0, 2] - x0) * sx
    K_roi[1, 2] = (K_roi[1, 2] - y0) * sy
    return K_roi


def crop_resize_with_padding(
    img: np.ndarray,
    roi: tuple[float, float, float, float],
    out_h: int,
    out_w: int,
    interpolation: int,
) -> np.ndarray:
    x0, y0, x1, y1 = roi
    xi0 = int(np.floor(x0))
    yi0 = int(np.floor(y0))
    xi1 = int(np.ceil(x1))
    yi1 = int(np.ceil(y1))
    crop_w = max(xi1 - xi0, 1)
    crop_h = max(yi1 - yi0, 1)
    crop_shape = (crop_h, crop_w) + (() if img.ndim == 2 else (img.shape[2],))
    crop = np.zeros(crop_shape, dtype=img.dtype)

    src_x0 = max(xi0, 0)
    src_y0 = max(yi0, 0)
    src_x1 = min(xi1, img.shape[1])
    src_y1 = min(yi1, img.shape[0])
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - xi0
        dst_y0 = src_y0 - yi0
        crop[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = img[src_y0:src_y1, src_x0:src_x1]
    return cv2.resize(crop, (int(out_w), int(out_h)), interpolation=interpolation)


def make_rgb_roi_from_mask(
    mask: np.ndarray,
    args: argparse.Namespace,
    H: int,
    W: int,
) -> tuple[float, float, float, float]:
    return square_roi_from_xyxy(
        mask_to_xyxy(mask),
        H,
        W,
        context_scale=args.roi_context_scale,
        min_size=args.roi_min_size,
    )


def make_rgb_roi_from_pose(
    pose_rgb: np.ndarray,
    inv_to_origin: np.ndarray,
    bbox: np.ndarray,
    K: np.ndarray,
    args: argparse.Namespace,
    H: int,
    W: int,
) -> tuple[float, float, float, float]:
    xyxy = project_bbox_xyxy(K, pose_rgb @ inv_to_origin, bbox, H, W)
    return square_roi_from_xyxy(
        xyxy,
        H,
        W,
        context_scale=args.roi_context_scale,
        min_size=args.roi_min_size,
    )


def make_event_roi_from_pose(
    pose_rgb: np.ndarray,
    inv_to_origin: np.ndarray,
    bbox: np.ndarray,
    K_event: np.ndarray,
    T_rgb_to_event: np.ndarray,
    args: argparse.Namespace,
    H_event: int,
    W_event: int,
) -> tuple[float, float, float, float]:
    pose_center_event = T_rgb_to_event @ (pose_rgb @ inv_to_origin)
    xyxy = project_bbox_xyxy(K_event, pose_center_event, bbox, H_event, W_event)
    return square_roi_from_xyxy(
        xyxy,
        H_event,
        W_event,
        context_scale=args.roi_context_scale,
        min_size=args.roi_min_size,
    )


def make_roi_rgbd_inputs(
    color: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    roi: tuple[float, float, float, float],
    roi_size: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    size = int(roi_size)
    color_roi = crop_resize_with_padding(color, roi, size, size, cv2.INTER_LINEAR)
    depth_roi = crop_resize_with_padding(depth, roi, size, size, cv2.INTER_NEAREST)
    K_roi = roi_camera_matrix(K, roi, size, size)
    mask_roi = None
    if mask is not None:
        mask_roi = crop_resize_with_padding(mask.astype(np.uint8), roi, size, size, cv2.INTER_NEAREST)
        mask_roi = (mask_roi > 0).astype(np.uint8)
    return color_roi, depth_roi, K_roi, mask_roi


def render_event_filter_mask(
    args: argparse.Namespace,
    event_mask_renderer: ProjectionMaskRenderer,
    T_rgb_to_event: np.ndarray,
    prev_pose_rgb: np.ndarray,
    current_pose_rgb: np.ndarray,
) -> np.ndarray:
    """渲染用于剔除非物体事件的事件相机支持掩码。"""
    if args.event_mask_pose in ("kf", "current"):
        return event_mask_renderer.render_mask(T_rgb_to_event @ current_pose_rgb)
    if args.event_mask_pose == "union":
        prev_event_mask = event_mask_renderer.render_mask(T_rgb_to_event @ prev_pose_rgb)
        current_event_mask = event_mask_renderer.render_mask(T_rgb_to_event @ current_pose_rgb)
        return np.logical_or(prev_event_mask > 0, current_event_mask > 0).astype(np.uint8)
    return event_mask_renderer.render_mask(T_rgb_to_event @ prev_pose_rgb)


def translation_error_m(a_in_cam: np.ndarray, b_in_cam: np.ndarray) -> float:
    return float(np.linalg.norm(a_in_cam[:3, 3] - b_in_cam[:3, 3]))


def score_pose_depth_consistency(
    renderer: ProjectionMaskRenderer,
    ob_in_cam: np.ndarray,
    observed_depth: np.ndarray,
    depth_threshold: float,
    occlusion_threshold: float,
    min_render_pixels: int,
) -> dict[str, float]:
    """评估渲染 CAD 深度与当前观测深度的一致程度。

    前景遮挡物（例如比渲染物体更近的手）不会被视为硬失败；评分仍偏向可见
    CAD 像素具有匹配 RGB-D 支持的姿态。
    """
    rendered_depth = renderer.render_depth(ob_in_cam)
    rendered_mask = rendered_depth > 1e-6
    mask_pixels = int(rendered_mask.sum())
    if mask_pixels < int(min_render_pixels):
        return {
            "score": -1.0,
            "mask_pixels": float(mask_pixels),
            "support_ratio": 0.0,
            "visible_inlier_ratio": 0.0,
            "median_abs_depth": np.inf,
            "occlusion_ratio": 0.0,
        }

    obs_valid = np.isfinite(observed_depth) & (observed_depth > 1e-6)
    valid = rendered_mask & obs_valid
    if valid.sum() == 0:
        return {
            "score": -0.5,
            "mask_pixels": float(mask_pixels),
            "support_ratio": 0.0,
            "visible_inlier_ratio": 0.0,
            "median_abs_depth": np.inf,
            "occlusion_ratio": 0.0,
        }

    diff = observed_depth[valid].astype(np.float32) - rendered_depth[valid].astype(np.float32)
    occluded = diff < -abs(float(occlusion_threshold))
    comparable = ~occluded
    occlusion_ratio = float(occluded.sum() / max(mask_pixels, 1))
    support_ratio = float(valid.sum() / max(mask_pixels, 1))

    if comparable.sum() == 0:
        return {
            "score": -0.25 - 0.1 * occlusion_ratio,
            "mask_pixels": float(mask_pixels),
            "support_ratio": support_ratio,
            "visible_inlier_ratio": 0.0,
            "median_abs_depth": np.inf,
            "occlusion_ratio": occlusion_ratio,
        }

    abs_diff = np.abs(diff[comparable])
    inliers = abs_diff <= abs(float(depth_threshold))
    visible_inlier_ratio = float(inliers.sum() / max(mask_pixels, 1))
    comparable_inlier_ratio = float(inliers.sum() / max(comparable.sum(), 1))
    median_abs_depth = float(np.median(abs_diff)) if len(abs_diff) else np.inf

    score = (
        visible_inlier_ratio
        + 0.25 * comparable_inlier_ratio
        + 0.10 * support_ratio
        - 0.10 * occlusion_ratio
        - 0.50 * min(median_abs_depth, 0.20)
    )
    return {
        "score": float(score),
        "mask_pixels": float(mask_pixels),
        "support_ratio": support_ratio,
        "visible_inlier_ratio": visible_inlier_ratio,
        "median_abs_depth": median_abs_depth,
        "occlusion_ratio": occlusion_ratio,
    }


def make_event_motion_band(mask: np.ndarray, dilation: int) -> np.ndarray:
    """保留狭窄轮廓带，以抑制内部手部/背景事件。"""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not np.any(binary):
        return binary
    edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    radius = max(int(dilation), 0)
    if radius <= 0:
        return edge
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(edge, kernel)


def event_cell_centers(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    cell_size: int,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将事件聚合为极性感知网格单元，以实现稳定的 RANSAC 匹配。"""
    if len(x) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int8)
    size = max(int(cell_size), 1)
    cells: dict[tuple[int, int, int], list[float]] = {}
    for xi, yi, pi in zip(x, y, p):
        key = (int(np.floor(float(xi) / size)), int(np.floor(float(yi) / size)), 1 if float(pi) > 0 else -1)
        item = cells.setdefault(key, [0.0, 0.0, 0.0])
        item[0] += float(xi)
        item[1] += float(yi)
        item[2] += 1.0
    points = []
    polarities = []
    threshold = max(int(min_count), 1)
    for (gx, gy, polarity), (sx, sy, count) in cells.items():
        if count >= threshold:
            points.append([sx / count, sy / count])
            polarities.append(polarity)
    if not points:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int8)
    return np.asarray(points, dtype=np.float32), np.asarray(polarities, dtype=np.int8)


def camera_frame_rotation_increment(
    newer_pose: np.ndarray,
    older_pose: np.ndarray,
) -> np.ndarray:
    """返回左乘形式的相机坐标系旋转向量。"""
    delta = newer_pose[:3, :3] @ older_pose[:3, :3].T
    return Rotation.from_matrix(delta).as_rotvec()


def stable_rotation_motion_cue(
    args: argparse.Namespace,
    pose_previous: np.ndarray | None,
    pose_older: np.ndarray | None,
    pose_older2: np.ndarray | None,
) -> tuple[bool, np.ndarray, float, float]:
    """在事件更新前检测可重复的相机坐标系角度增量。"""
    if pose_previous is None or pose_older is None or pose_older2 is None:
        return False, np.zeros(3, dtype=np.float64), 0.0, 0.0
    latest = camera_frame_rotation_increment(pose_previous, pose_older)
    before = camera_frame_rotation_increment(pose_older, pose_older2)
    latest_norm = float(np.linalg.norm(latest))
    before_norm = float(np.linalg.norm(before))
    latest_deg = float(np.rad2deg(latest_norm))
    before_deg = float(np.rad2deg(before_norm))
    if latest_norm < 1e-9 or before_norm < 1e-9:
        return False, latest, latest_deg, 0.0
    axis_cosine = float(np.dot(latest, before) / (latest_norm * before_norm))
    active = bool(
        args.v20_enable_rotation_compensation
        and latest_deg >= float(args.v20_rotation_compensation_min_deg)
        and before_deg >= float(args.v20_rotation_compensation_min_deg)
        and axis_cosine >= float(args.v20_rotation_compensation_axis_cosine)
    )
    return active, latest, latest_deg, axis_cosine


def event_count_trend_cue(
    args: argparse.Namespace,
    subwindow_counts: list[int] | np.ndarray,
) -> tuple[bool, float, float, float]:
    """测量跨子窗口的归一化单调事件数量趋势。

    该数量只用于触发双向搜索，不作为独立的旋转方向测量，因为相反旋转可能
    产生相近的无符号事件数量。
    """
    counts = np.asarray(subwindow_counts, dtype=np.float64).reshape(-1)
    if len(counts) < 3:
        return False, 0.0, 0.0, 0.0
    counts = counts[np.isfinite(counts) & (counts >= 0.0)]
    if len(counts) < 3:
        return False, 0.0, 0.0, 0.0
    median_count = float(np.median(counts))
    if median_count < max(float(args.v21_event_count_min_subwindow), 1.0):
        return False, 0.0, 0.0, median_count
    normalized = counts / max(median_count, 1e-9)
    coordinate = np.linspace(-0.5, 0.5, len(normalized), dtype=np.float64)
    denominator = float(np.dot(coordinate, coordinate))
    slope = float(np.dot(coordinate, normalized - np.mean(normalized)) / max(denominator, 1e-9))
    differences = np.diff(normalized)
    signs = np.sign(differences[np.abs(differences) > 1e-6])
    if len(signs) == 0:
        consistency = 0.0
    else:
        consistency = float(abs(np.sum(signs)) / len(signs))
    active = bool(
        abs(slope) >= float(args.v21_event_count_trend_min)
        and consistency >= float(args.v21_event_count_trend_consistency)
    )
    return active, slope, consistency, median_count


def apply_affine_to_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.asarray(points, dtype=np.float32).reshape(-1, 2)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return ((affine[:, :2] @ pts.T).T + affine[:, 2]).astype(np.float32)


def cad_rotation_compensation_affine(
    args: argparse.Namespace,
    mesh: trimesh.Trimesh,
    pose_rgb: np.ndarray,
    K_event: np.ndarray,
    T_rgb_to_event: np.ndarray,
    event_roi: tuple[float, float, float, float],
    processing_size: int,
    frame_rotvec_rgb: np.ndarray,
    sub_window_count: int,
) -> tuple[np.ndarray | None, float, int]:
    """用从早期到后期的局部仿射变换近似 CAD 旋转流。"""
    rotvec_rgb = np.asarray(frame_rotvec_rgb, dtype=np.float64).reshape(3)
    if np.linalg.norm(rotvec_rgb) < 1e-9:
        return None, 0.0, 0
    # 相位相关比较前、后半窗口事件纹理的中心，随后
    # v14_event_temporal_extrapolation 将该运动扩展到完整子窗口。因此只补偿
    # 相同的中心到中心时间间隔。
    fraction = 1.0 / (
        max(int(sub_window_count), 1)
        * max(float(args.v14_event_temporal_extrapolation), 1.0)
    )
    rotvec_event = T_rgb_to_event[:3, :3] @ rotvec_rgb
    rotvec_event *= fraction * float(args.v20_rotation_compensation_scale)
    predicted_deg = float(np.rad2deg(np.linalg.norm(rotvec_event)))
    if predicted_deg < 1e-4:
        return None, predicted_deg, 0

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    max_vertices = max(int(args.v20_rotation_compensation_max_vertices), 32)
    if len(vertices) > max_vertices:
        indices = np.linspace(0, len(vertices) - 1, max_vertices).astype(np.int64)
        vertices = vertices[indices]

    pose_event = T_rgb_to_event @ pose_rgb
    rotation_after = Rotation.from_rotvec(rotvec_event).as_matrix() @ pose_event[:3, :3]
    before = (pose_event[:3, :3] @ vertices.T).T + pose_event[:3, 3]
    after = (rotation_after @ vertices.T).T + pose_event[:3, 3]
    valid = (
        np.isfinite(before).all(axis=1)
        & np.isfinite(after).all(axis=1)
        & (before[:, 2] > 1e-5)
        & (after[:, 2] > 1e-5)
    )
    if np.count_nonzero(valid) < 12:
        return None, predicted_deg, int(np.count_nonzero(valid))
    before = before[valid]
    after = after[valid]
    uv0_h = (K_event @ before.T).T
    uv1_h = (K_event @ after.T).T
    uv0 = uv0_h[:, :2] / uv0_h[:, 2:3]
    uv1 = uv1_h[:, :2] / uv1_h[:, 2:3]

    x0, y0, x1, y1 = map(float, event_roi)
    size = max(int(processing_size), 8)
    sx = float(size) / max(x1 - x0, 1e-6)
    sy = float(size) / max(y1 - y0, 1e-6)
    uv0[:, 0] = (uv0[:, 0] - x0) * sx
    uv0[:, 1] = (uv0[:, 1] - y0) * sy
    uv1[:, 0] = (uv1[:, 0] - x0) * sx
    uv1[:, 1] = (uv1[:, 1] - y0) * sy
    margin = float(size) * 0.35
    inside = (
        np.isfinite(uv0).all(axis=1)
        & np.isfinite(uv1).all(axis=1)
        & (uv0[:, 0] >= -margin) & (uv0[:, 0] < size + margin)
        & (uv0[:, 1] >= -margin) & (uv0[:, 1] < size + margin)
        & (uv1[:, 0] >= -margin) & (uv1[:, 0] < size + margin)
        & (uv1[:, 1] >= -margin) & (uv1[:, 1] < size + margin)
    )
    uv0 = uv0[inside].astype(np.float32)
    uv1 = uv1[inside].astype(np.float32)
    if len(uv0) < 12:
        return None, predicted_deg, len(uv0)

    affine, _ = cv2.estimateAffine2D(uv0, uv1, method=cv2.LMEDS)
    if affine is None or not np.isfinite(affine).all():
        affine, _ = cv2.estimateAffinePartial2D(uv0, uv1, method=cv2.LMEDS)
    if affine is None or not np.isfinite(affine).all():
        return None, predicted_deg, len(uv0)
    return np.asarray(affine, dtype=np.float64), predicted_deg, len(uv0)


def estimate_event_2d_motion(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    image_width: int,
    image_height: int,
    motion_center_xy: tuple[float, float] | None = None,
    rotation_compensation_affine: np.ndarray | None = None,
    predicted_rotation_deg: float = 0.0,
) -> tuple[dict[str, float | str | bool], dict[str, np.ndarray]]:
    """在可选 CAD 旋转补偿后估计残余平移。"""
    stats: dict[str, float | str | bool] = {
        "event_count": float(len(x)),
        "early_count": 0.0,
        "late_count": 0.0,
        "match_count": 0.0,
        "inlier_count": 0.0,
        "inlier_ratio": 0.0,
        "spatial_sectors": 0.0,
        "rms_px": np.inf,
        "dx": 0.0,
        "dy": 0.0,
        "phase_dx": 0.0,
        "phase_dy": 0.0,
        "affine_center_dx": 0.0,
        "affine_center_dy": 0.0,
        "rotation_flow_weight": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "phase_response": 0.0,
        "rotation_compensated": False,
        "rotation_compensation_rejected": False,
        "predicted_rotation_deg": float(predicted_rotation_deg),
        "raw_phase_dx": 0.0,
        "raw_phase_dy": 0.0,
        "raw_phase_response": 0.0,
        "translation_step_m": 0.0,
        "confidence": 0.0,
        "accepted": False,
        "reason": "not_evaluated",
    }
    debug = {
        "early": np.empty((0, 2), dtype=np.float32),
        "late": np.empty((0, 2), dtype=np.float32),
        "late_raw": np.empty((0, 2), dtype=np.float32),
        "src": np.empty((0, 2), dtype=np.float32),
        "dst": np.empty((0, 2), dtype=np.float32),
        "inliers": np.empty(0, dtype=bool),
    }
    if len(x) < int(args.v14_event_min_events):
        stats["reason"] = "too_few_events"
        return stats, debug
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & np.isfinite(p)
    x, y, t, p = x[finite], y[finite], t[finite], p[finite]
    if len(x) < int(args.v14_event_min_events):
        stats["reason"] = "too_few_finite_events"
        return stats, debug
    split_t = float(np.quantile(t, 0.5))
    early_keep = t <= split_t
    late_keep = t > split_t
    early, early_pol = event_cell_centers(
        x[early_keep], y[early_keep], p[early_keep], args.v14_event_cell_size, args.v14_event_min_cell_count
    )
    late, late_pol = event_cell_centers(
        x[late_keep], y[late_keep], p[late_keep], args.v14_event_cell_size, args.v14_event_min_cell_count
    )
    debug["early"], debug["late_raw"] = early, late
    stats["early_count"], stats["late_count"] = float(len(early)), float(len(late))
    if len(early) < 3 or len(late) < 3:
        stats["reason"] = "too_few_temporal_cells"
        return stats, debug

    # 相位相关使用完整的过滤事件纹理，相比仅靠最近网格匹配能更可靠地恢复
    # 大图像位移。下方网格/RANSAC 模型仍作为独立的空间一致性检查。
    def make_phase_image(keep: np.ndarray) -> np.ndarray:
        image = np.zeros((int(image_height), int(image_width)), dtype=np.float32)
        xi = np.rint(x[keep]).astype(np.int32)
        yi = np.rint(y[keep]).astype(np.int32)
        inside = (xi >= 0) & (xi < int(image_width)) & (yi >= 0) & (yi < int(image_height))
        if np.any(inside):
            np.add.at(image, (yi[inside], xi[inside]), 1.0)
        image = cv2.GaussianBlur(image, (0, 0), 2.0)
        peak = float(np.max(image))
        if peak > 1e-6:
            image /= peak
        return image

    early_image = make_phase_image(early_keep)
    late_image = make_phase_image(late_keep)
    late_image_uncompensated = late_image.copy()
    try:
        raw_shift, raw_response = cv2.phaseCorrelate(early_image, late_image)
        stats["raw_phase_dx"] = float(raw_shift[0]) * float(args.v14_event_temporal_extrapolation)
        stats["raw_phase_dy"] = float(raw_shift[1]) * float(args.v14_event_temporal_extrapolation)
        stats["raw_phase_response"] = float(raw_response)
    except cv2.error:
        pass

    late_for_motion = late
    if rotation_compensation_affine is not None:
        affine_rotation = np.asarray(rotation_compensation_affine, dtype=np.float64).reshape(2, 3)
        inverse_rotation = cv2.invertAffineTransform(affine_rotation)
        late_for_motion = apply_affine_to_points(late, inverse_rotation)
        late_image = cv2.warpAffine(
            late_image,
            inverse_rotation,
            (int(image_width), int(image_height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        peak = float(np.max(late_image))
        if peak > 1e-6:
            late_image /= peak
        stats["rotation_compensated"] = True
    debug["late"] = late_for_motion
    try:
        phase_shift, phase_response = cv2.phaseCorrelate(early_image, late_image)
        phase_dx = float(phase_shift[0]) * float(args.v14_event_temporal_extrapolation)
        phase_dy = float(phase_shift[1]) * float(args.v14_event_temporal_extrapolation)
        phase_response = float(phase_response)
    except cv2.error:
        phase_dx, phase_dy, phase_response = 0.0, 0.0, 0.0
    if (
        bool(stats["rotation_compensated"])
        and float(phase_response)
        < float(stats["raw_phase_response"])
        * float(args.v20_rotation_compensation_min_response_ratio)
    ):
        late_for_motion = late
        late_image = late_image_uncompensated
        phase_dx = float(stats["raw_phase_dx"])
        phase_dy = float(stats["raw_phase_dy"])
        phase_response = float(stats["raw_phase_response"])
        stats["rotation_compensated"] = False
        stats["rotation_compensation_rejected"] = True
        debug["late"] = late_for_motion
    stats["phase_response"] = phase_response

    max_dist = float(args.v14_event_max_match_px)
    pairs = []
    for i, point in enumerate(early):
        distances = np.linalg.norm(late_for_motion - point[None, :], axis=1)
        same_pol = late_pol == early_pol[i]
        distances = np.where(same_pol, distances, np.inf)
        j = int(np.argmin(distances))
        distance = float(distances[j])
        if not np.isfinite(distance):
            j = int(np.argmin(np.linalg.norm(late_for_motion - point[None, :], axis=1)))
            distance = float(np.linalg.norm(late_for_motion[j] - point))
        if distance <= max_dist:
            pairs.append((distance, i, j))
    pairs.sort(key=lambda item: item[0])
    used_late = set()
    src, dst = [], []
    for distance, i, j in pairs:
        if j in used_late:
            continue
        used_late.add(j)
        src.append(early[i])
        dst.append(late_for_motion[j])
    src = np.asarray(src, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
    debug["src"], debug["dst"] = src, dst
    stats["match_count"] = float(len(src))
    if len(src) < int(args.v14_event_min_matches):
        stats["reason"] = "too_few_matches"
        return stats, debug

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(args.v14_event_ransac_threshold_px),
        maxIters=int(args.v14_event_ransac_iters),
        confidence=float(args.v14_event_ransac_confidence),
        refineIters=10,
    )
    if affine is None or inlier_mask is None:
        stats["reason"] = "ransac_failed"
        return stats, debug
    inliers = inlier_mask.reshape(-1).astype(bool)
    debug["inliers"] = inliers
    inlier_src, inlier_dst = src[inliers], dst[inliers]
    if len(inlier_src) == 0:
        stats["reason"] = "no_ransac_inliers"
        return stats, debug
    predicted = (affine[:, :2] @ inlier_src.T).T + affine[:, 2]
    residual = np.linalg.norm(predicted - inlier_dst, axis=1)
    linear = affine[:, :2]
    scale = float(np.sqrt(max(abs(np.linalg.det(linear)), 1e-9)))
    rotation_deg = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    if motion_center_xy is None:
        motion_center = np.median(inlier_src, axis=0).astype(np.float64)
    else:
        motion_center = np.asarray(motion_center_xy, dtype=np.float64)
    affine_center = linear @ motion_center + affine[:, 2]
    affine_center_motion = (
        affine_center - motion_center
    ) * float(args.v14_event_temporal_extrapolation)
    affine_center_dx = float(affine_center_motion[0])
    affine_center_dy = float(affine_center_motion[1])

    # 相位相关最擅长平移，但物体大旋转会移动整个事件纹理，使旋转流泄漏到该
    # 平移中。此时在渲染物体中心计算 RANSAC 相似变换，并向这一旋转补偿位移
    # 融合，而不是使用仿射矩阵原点处的平移。
    rotation_ratio = abs(rotation_deg) / max(float(args.v16_motion_rotation_blend_deg), 1e-6)
    scale_ratio = abs(scale - 1.0) / max(float(args.v16_motion_scale_blend), 1e-6)
    rotation_flow_weight = float(
        np.clip(max(rotation_ratio, scale_ratio), 0.0, 1.0)
        * np.clip(float(args.v16_motion_affine_blend), 0.0, 1.0)
    )
    dx = (1.0 - rotation_flow_weight) * phase_dx + rotation_flow_weight * affine_center_dx
    dy = (1.0 - rotation_flow_weight) * phase_dy + rotation_flow_weight * affine_center_dy
    center = np.array([0.5 * image_width, 0.5 * image_height], dtype=np.float32)
    angles = np.arctan2(inlier_src[:, 1] - center[1], inlier_src[:, 0] - center[0])
    sectors = len(np.unique(np.floor((angles + np.pi) / (2 * np.pi) * 8).astype(np.int32)))
    inlier_ratio = float(len(inlier_src) / max(len(src), 1))
    rms_px = float(np.sqrt(np.mean(residual ** 2)))
    motion_px = float(np.hypot(dx, dy))
    phase_quality = float(np.clip(phase_response / 0.5, 0.0, 1.0))
    confidence = float(
        inlier_ratio
        * min(sectors / max(float(args.v14_event_min_sectors), 1.0), 1.0)
        * np.exp(-rms_px / max(float(args.v14_event_rms_scale_px), 1e-6))
        * (0.5 + 0.5 * phase_quality)
    )
    stats.update({
        "inlier_count": float(len(inlier_src)),
        "inlier_ratio": inlier_ratio,
        "spatial_sectors": float(sectors),
        "rms_px": rms_px,
        "dx": dx,
        "dy": dy,
        "phase_dx": phase_dx,
        "phase_dy": phase_dy,
        "affine_center_dx": affine_center_dx,
        "affine_center_dy": affine_center_dy,
        "rotation_flow_weight": rotation_flow_weight,
        "rotation_deg": rotation_deg,
        "scale": scale,
        "confidence": confidence,
    })
    accepted = (
        len(inlier_src) >= int(args.v14_event_min_inliers)
        and inlier_ratio >= float(args.v14_event_min_inlier_ratio)
        and sectors >= int(args.v14_event_min_sectors)
        and rms_px <= float(args.v14_event_max_rms_px)
        and motion_px >= float(args.v14_event_min_motion_px)
        and motion_px <= float(args.v14_event_max_motion_px)
        and abs(rotation_deg) <= float(args.v14_event_max_rotation_deg)
        and abs(scale - 1.0) <= float(args.v14_event_max_scale_change)
        and confidence >= float(args.v14_event_min_confidence)
        and phase_response >= float(args.v14_event_min_phase_response)
    )
    stats["accepted"] = bool(accepted)
    stats["reason"] = "accepted" if accepted else "motion_quality_gate"
    return stats, debug


def pose_from_event_2d_motion(
    base_pose_rgb: np.ndarray,
    dx: float,
    dy: float,
    K_event: np.ndarray,
    T_rgb_to_event: np.ndarray,
    T_event_to_rgb: np.ndarray,
    max_translation_m: float,
) -> tuple[np.ndarray, float]:
    """只反投影事件 XY 运动，把 Z 和三维旋转留给 RGB-D 细化。"""
    pose_event = T_rgb_to_event @ base_pose_rgb
    z = max(float(pose_event[2, 3]), 1e-4)
    delta = np.array([
        float(dx) * z / max(float(K_event[0, 0]), 1e-6),
        float(dy) * z / max(float(K_event[1, 1]), 1e-6),
        0.0,
    ], dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if max_translation_m > 0 and length > max_translation_m:
        delta *= float(max_translation_m) / max(length, 1e-9)
        length = float(max_translation_m)
    pose_event[:3, 3] += delta
    return T_event_to_rgb @ pose_event, length


def binary_mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments((np.asarray(mask) > 0).astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0:
        return None
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def expand_roi(
    roi: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = map(float, roi)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    size = max(x1 - x0, y1 - y0, 1.0) * max(float(scale), 1.0)
    half = 0.5 * size
    return cx - half, cy - half, cx + half, cy + half


def fixed_event_window_from_pose(
    args: argparse.Namespace,
    pose_rgb: np.ndarray,
    inv_to_origin: np.ndarray,
    bbox: np.ndarray,
    K_event: np.ndarray,
    T_rgb_to_event: np.ndarray,
    H_event: int,
    W_event: int,
) -> tuple[float, float, float, float]:
    """以投影 CAD 包围框为中心放置固定原始分辨率事件窗口。"""
    projected_roi = make_event_roi_from_pose(
        pose_rgb,
        inv_to_origin,
        bbox,
        K_event,
        T_rgb_to_event,
        args,
        H_event,
        W_event,
    )
    x0, y0, x1, y1 = projected_roi
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    size = max(float(args.v17_event_window_size_px), 1.0)
    half = 0.5 * size
    return cx - half, cy - half, cx + half, cy + half


def crop_events_to_processing_window(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    roi: tuple[float, float, float, float],
    processing_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """在完整事件坐标中裁剪，并把保留事件映射到紧凑局部网格。"""
    x0, y0, x1, y1 = map(float, roi)
    keep = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & np.isfinite(p)
        & (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    )
    x_abs = np.asarray(x[keep], dtype=np.float32)
    y_abs = np.asarray(y[keep], dtype=np.float32)
    t_out = np.asarray(t[keep])
    p_out = np.asarray(p[keep])
    size = max(int(processing_size), 8)
    scale_x = float(size) / max(x1 - x0, 1e-6)
    scale_y = float(size) / max(y1 - y0, 1e-6)
    x_local = ((x_abs - x0) * scale_x).astype(np.float32)
    y_local = ((y_abs - y0) * scale_y).astype(np.float32)
    return x_local, y_local, t_out, p_out, x_abs, y_abs, scale_x, scale_y


def restore_motion_to_event_pixels(
    stats: dict[str, float | str | bool],
    scale_x: float,
    scale_y: float,
) -> None:
    """把局部窗口位移字段转换回完整事件相机像素。"""
    for key in ("dx", "phase_dx", "affine_center_dx", "raw_phase_dx"):
        stats[key] = float(stats.get(key, 0.0)) / max(float(scale_x), 1e-9)
    for key in ("dy", "phase_dy", "affine_center_dy", "raw_phase_dy"):
        stats[key] = float(stats.get(key, 0.0)) / max(float(scale_y), 1e-9)


def expanded_cad_support_in_event_window(
    mesh: trimesh.Trimesh,
    pose_event: np.ndarray,
    K_event: np.ndarray,
    roi: tuple[float, float, float, float],
    processing_size: int,
    dilation_px_full: float,
) -> np.ndarray:
    """将 CAD 凸包投影到局部事件窗口，并为旋转后新显露的边缘扩张。"""
    size = max(int(processing_size), 8)
    mask = np.zeros((size, size), dtype=np.uint8)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    points = (pose_event[:3, :3] @ vertices.T).T + pose_event[:3, 3]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    if np.count_nonzero(valid) < 3:
        return mask
    projected = (K_event @ points[valid].T).T
    uv = projected[:, :2] / projected[:, 2:3]
    x0, y0, x1, y1 = map(float, roi)
    scale_x = float(size) / max(x1 - x0, 1e-6)
    scale_y = float(size) / max(y1 - y0, 1e-6)
    uv[:, 0] = (uv[:, 0] - x0) * scale_x
    uv[:, 1] = (uv[:, 1] - y0) * scale_y
    finite = np.isfinite(uv).all(axis=1)
    if np.count_nonzero(finite) < 3:
        return mask
    hull = cv2.convexHull(uv[finite].astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 1)
    dilation_local = int(np.ceil(
        max(float(dilation_px_full), 0.0) * 0.5 * (scale_x + scale_y)
    ))
    if dilation_local > 0:
        kernel_size = 2 * dilation_local + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)
    return mask


def filter_local_events_by_support(
    x_local: np.ndarray,
    y_local: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    x_abs: np.ndarray,
    y_abs: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi = np.rint(x_local).astype(np.int32)
    yi = np.rint(y_local).astype(np.int32)
    inside = (
        (xi >= 0) & (xi < support.shape[1])
        & (yi >= 0) & (yi < support.shape[0])
    )
    keep = np.zeros(len(xi), dtype=bool)
    keep[inside] = support[yi[inside], xi[inside]] > 0
    return x_local[keep], y_local[keep], t[keep], p[keep], x_abs[keep], y_abs[keep]


def rotation_reversal_cue(
    args: argparse.Namespace,
    pose_previous: np.ndarray,
    pose_older: np.ndarray | None,
    pose_baseline: np.ndarray,
) -> tuple[bool, float, float, float]:
    """检测方向与近期历史相反的基线角度增量。"""
    if not args.v20_enable_reversal_hypotheses or pose_older is None:
        return False, 0.0, 0.0, 1.0
    baseline_vec = camera_frame_rotation_increment(pose_baseline, pose_previous)
    history_vec = camera_frame_rotation_increment(pose_previous, pose_older)
    baseline_norm = float(np.linalg.norm(baseline_vec))
    history_norm = float(np.linalg.norm(history_vec))
    baseline_deg = float(np.rad2deg(baseline_norm))
    history_deg = float(np.rad2deg(history_norm))
    if baseline_norm < 1e-9 or history_norm < 1e-9:
        return False, baseline_deg, history_deg, 1.0
    axis_cosine = float(np.dot(baseline_vec, history_vec) / (baseline_norm * history_norm))
    active = bool(
        baseline_deg >= float(args.v20_reversal_min_baseline_deg)
        and history_deg >= float(args.v20_reversal_min_history_deg)
        and axis_cosine <= float(args.v20_reversal_axis_max_cosine)
    )
    return active, baseline_deg, history_deg, axis_cosine


def build_v16_event_rotation_hypotheses(
    args: argparse.Namespace,
    pose_event_prior: np.ndarray,
    pose_previous: np.ndarray,
    pose_older: np.ndarray | None,
    pose_baseline: np.ndarray,
    force_expanded: bool | None = None,
    reversal_active: bool = False,
    bidirectional_active: bool = False,
) -> tuple[list[np.ndarray], list[str], bool, float]:
    """不依赖卡尔曼状态模型构建以事件为中心的 SO(3) 假设。"""
    hypotheses: list[np.ndarray] = []
    labels: list[str] = []

    def append(rotation: np.ndarray, label: str):
        pose = pose_event_prior.copy()
        pose[:3, :3] = Rotation.from_matrix(rotation).as_matrix()
        for existing in hypotheses:
            if rotation_error_deg(existing, pose) < float(args.v16_rotation_dedup_deg):
                return
        hypotheses.append(pose)
        labels.append(label)

    append(pose_event_prior[:3, :3], "event_xy")
    baseline_step = rotation_error_deg(pose_previous, pose_baseline)
    append(pose_baseline[:3, :3], "baseline_R_event_t")

    history_step = 0.0
    history_rotvec = None
    if pose_older is not None:
        delta_camera = pose_previous[:3, :3] @ pose_older[:3, :3].T
        history_rotvec = Rotation.from_matrix(delta_camera).as_rotvec()
        history_step = float(np.rad2deg(np.linalg.norm(history_rotvec)))

    cue_deg = max(float(baseline_step), float(history_step))
    if force_expanded is None:
        triggered = bool(
            args.v16_enable_rotation_recovery
            and cue_deg >= float(args.v16_rotation_trigger_deg)
        )
    else:
        triggered = bool(args.v16_enable_rotation_recovery and force_expanded)
    if (
        triggered
        and (reversal_active or bidirectional_active)
        and history_rotvec is not None
        and np.linalg.norm(history_rotvec) > 1e-8
    ):
        reverse_scales = (
            parse_float_list(args.v21_bidirectional_rotation_scales)
            if bidirectional_active
            else parse_float_list(args.v20_reversal_history_scales)
        )
        for scale in reverse_scales:
            scaled = -history_rotvec * abs(float(scale))
            max_rad = np.deg2rad(float(args.v16_rotation_max_step_deg))
            length = float(np.linalg.norm(scaled))
            if length > max_rad > 0:
                scaled *= max_rad / length
            predicted = Rotation.from_rotvec(scaled).as_matrix() @ pose_previous[:3, :3]
            append(predicted, f"reverse_history_{abs(float(scale)):g}x")

    if triggered and history_rotvec is not None and np.linalg.norm(history_rotvec) > 1e-8:
        for scale in parse_float_list(args.v16_rotation_history_scales):
            scaled = history_rotvec * float(scale)
            max_rad = np.deg2rad(float(args.v16_rotation_max_step_deg))
            length = float(np.linalg.norm(scaled))
            if length > max_rad > 0:
                scaled *= max_rad / length
            predicted = Rotation.from_rotvec(scaled).as_matrix() @ pose_previous[:3, :3]
            append(predicted, f"history_{scale:g}x")

    if triggered and args.v16_use_camera_axis_hypotheses:
        angle_deg = float(np.clip(
            cue_deg * float(args.v16_rotation_axis_scale),
            float(args.v16_rotation_axis_min_deg),
            float(args.v16_rotation_max_step_deg),
        ))
        # 同时扰动 RGB-D 基线朝向和上一已接受朝向。前者通常包含正确的粗旋转，
        # 但平移/ROI 可能错误；后者则保留为受保护后备方案。
        for anchor_name, anchor_rotation in (
            ("baseline", pose_baseline[:3, :3]),
            ("previous", pose_previous[:3, :3]),
        ):
            for axis_name, axis in (
                ("cam_x", np.array([1.0, 0.0, 0.0])),
                ("cam_y", np.array([0.0, 1.0, 0.0])),
                ("cam_z", np.array([0.0, 0.0, 1.0])),
            ):
                for sign in (-1.0, 1.0):
                    delta = Rotation.from_rotvec(axis * np.deg2rad(sign * angle_deg)).as_matrix()
                    append(
                        delta @ anchor_rotation,
                        f"{anchor_name}_{axis_name}_{sign * angle_deg:+.1f}deg",
                    )

    max_hypotheses = max(int(args.v16_rotation_max_hypotheses), 1)
    return hypotheses[:max_hypotheses], labels[:max_hypotheses], triggered, cue_deg


def build_v19_stable_axis_hypotheses(
    args: argparse.Namespace,
    pose_event_prior: np.ndarray,
    pose_previous: np.ndarray,
    pose_older: np.ndarray | None,
    pose_older2: np.ndarray | None,
) -> tuple[list[np.ndarray], list[str], bool, float, float]:
    """仅在连续两个相机坐标系旋转轴一致后传播角速度。"""
    if (
        not args.v19_enable_stable_axis_protection
        or pose_older is None
        or pose_older2 is None
    ):
        return [], [], False, 0.0, 0.0
    delta_latest = pose_previous[:3, :3] @ pose_older[:3, :3].T
    delta_before = pose_older[:3, :3] @ pose_older2[:3, :3].T
    rotvec_latest = Rotation.from_matrix(delta_latest).as_rotvec()
    rotvec_before = Rotation.from_matrix(delta_before).as_rotvec()
    latest_deg = float(np.rad2deg(np.linalg.norm(rotvec_latest)))
    before_deg = float(np.rad2deg(np.linalg.norm(rotvec_before)))
    if latest_deg < 1e-6 or before_deg < 1e-6:
        return [], [], False, latest_deg, 0.0
    axis_cosine = float(
        np.dot(rotvec_latest, rotvec_before)
        / max(np.linalg.norm(rotvec_latest) * np.linalg.norm(rotvec_before), 1e-9)
    )
    active = bool(
        latest_deg >= float(args.v19_stable_rotation_min_deg)
        and before_deg >= float(args.v19_stable_rotation_min_deg)
        and axis_cosine >= float(args.v19_stable_axis_min_cosine)
    )
    if not active:
        return [], [], False, latest_deg, axis_cosine
    poses = []
    labels = []
    for scale in parse_float_list(args.v19_stable_rotation_scales):
        pose = pose_event_prior.copy()
        pose[:3, :3] = (
            Rotation.from_rotvec(rotvec_latest * float(scale)).as_matrix()
            @ pose_previous[:3, :3]
        )
        poses.append(pose)
        labels.append(f"stable_axis_{float(scale):g}x")
    return poses, labels, True, latest_deg, axis_cosine


def make_event_motion_visualization(
    args: argparse.Namespace,
    debug: dict[str, np.ndarray],
    H: int,
    W: int,
    frame_label: str,
    stats: dict[str, float | str | bool],
) -> np.ndarray:
    """在事件坐标中绘制早期/后期事件网格及 RANSAC 内点匹配。"""
    canvas = np.zeros((int(H), int(W), 3), dtype=np.uint8)
    early = debug.get("early", np.empty((0, 2)))
    late = debug.get("late", np.empty((0, 2)))
    src = debug.get("src", np.empty((0, 2)))
    dst = debug.get("dst", np.empty((0, 2)))
    inliers = debug.get("inliers", np.zeros(len(src), dtype=bool))
    for point in early:
        u, v = np.rint(point).astype(int)
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(canvas, (u, v), 2, (40, 80, 255), -1)
    for point in late:
        u, v = np.rint(point).astype(int)
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(canvas, (u, v), 2, (255, 120, 40), -1)
    for i, (point_a, point_b) in enumerate(zip(src, dst)):
        a = tuple(np.rint(point_a).astype(int))
        b = tuple(np.rint(point_b).astype(int))
        color = (40, 255, 80) if i < len(inliers) and inliers[i] else (0, 150, 255)
        if all(0 <= value for value in (*a, *b)):
            cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
    # cv2.putText(
    #     canvas,
    #     f"{frame_label} | {'ACCEPT' if stats.get('accepted') else 'REJECT'} | "
    #     f"dx={float(stats.get('dx', 0.0)):.1f} dy={float(stats.get('dy', 0.0)):.1f} "
    #     f"rot2d={float(stats.get('rotation_deg', 0.0)):.1f} "
    #     f"blend={float(stats.get('rotation_flow_weight', 0.0)):.2f} "
    #     f"inlier={float(stats.get('inlier_ratio', 0.0)):.2f} "
    #     f"rms={float(stats.get('rms_px', 0.0)):.2f}",
    #     (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    # )
    # cv2.putText(
    #     canvas,
    #     f"rot-comp={int(bool(stats.get('rotation_compensated')))} "
    #     f"pred={float(stats.get('predicted_rotation_deg', 0.0)):.1f}deg "
    #     f"raw=({float(stats.get('raw_phase_dx', 0.0)):.1f},"
    #     f"{float(stats.get('raw_phase_dy', 0.0)):.1f})",
    #     (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 255), 2, cv2.LINE_AA,
    # )
    out_w = max(int(args.contour_vis_width), 1)
    out_h = max(int(args.contour_vis_height), 1)
    return cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def score_pose_candidates_with_foundationpose(
    est: FoundationPose,
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> dict[str, float]:
    """通过一次批量 FoundationPose 评分器调用排序原始网格姿态。"""
    if not candidates or est.scorer is None:
        return {}
    tf_to_center = est.get_tf_to_centered_mesh().data.cpu().numpy()
    inv_tf_to_center = np.linalg.inv(tf_to_center)
    names = list(candidates)
    centered = np.asarray([
        np.asarray(candidates[name]).reshape(4, 4) @ inv_tf_to_center
        for name in names
    ])
    scores, _ = est.scorer.predict(
        mesh=est.mesh,
        rgb=rgb,
        depth=depth,
        K=K,
        ob_in_cams=centered,
        normal_map=None,
        mesh_tensors=est.mesh_tensors,
        glctx=est.glctx,
        mesh_diameter=est.diameter,
        get_vis=False,
    )
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    return {name: float(values[i]) for i, name in enumerate(names)}


def select_v21_pose_candidates(
    args: argparse.Namespace,
    renderer: ProjectionMaskRenderer,
    observed_depth: np.ndarray,
    pose_previous: np.ndarray,
    pose_baseline: np.ndarray,
    pose_event_prior: np.ndarray,
    pose_event_refined: np.ndarray | None,
    event_hypothesis_labels: list[str] | None,
    extra_event_candidates: dict[str, np.ndarray] | None,
    event_used: bool,
    event_confidence: float,
    rotation_recovery_triggered: bool,
    large_event_motion_gate: bool,
    stable_rotation_active: bool,
    stable_rotation_step_deg: float,
    rotation_compensation_active: bool,
    learned_scores: dict[str, float] | None = None,
) -> tuple[np.ndarray, str, dict[str, float], str]:
    """利用事件方向和防回退门控选择细化后的假设。"""
    candidates = {
        "previous": pose_previous,
        "baseline": pose_baseline,
        "event_prior": pose_event_prior,
    }
    refined_event_poses = [] if pose_event_refined is None else list(np.asarray(pose_event_refined).reshape(-1, 4, 4))
    labels = list(event_hypothesis_labels or [])
    if len(labels) < len(refined_event_poses):
        labels.extend(f"hyp_{i:02d}" for i in range(len(labels), len(refined_event_poses)))
    event_keys = []
    for i, pose in enumerate(refined_event_poses):
        key = f"event::{labels[i]}"
        candidates[key] = pose
        event_keys.append(key)
    for key, pose in (extra_event_candidates or {}).items():
        candidate_key = key if key.startswith("event::") else f"event::{key}"
        candidates[candidate_key] = pose
        event_keys.append(candidate_key)
    learned_scores = learned_scores or {}
    scores = {}
    for name, pose in candidates.items():
        if name in learned_scores:
            scores[name] = float(learned_scores[name])
        else:
            scores[name] = float(score_pose_depth_consistency(
                renderer,
                pose,
                observed_depth,
                depth_threshold=args.pose_select_depth_threshold,
                occlusion_threshold=args.pose_select_occlusion_threshold,
                min_render_pixels=args.pose_select_min_render_pixels,
            )["score"])
    previous_score = scores["previous"]
    baseline_score = scores["baseline"]
    protected = pose_previous
    source = "previous"
    protected_score = previous_score
    baseline_step_t = translation_error_m(pose_previous, pose_baseline)
    baseline_step_r = rotation_error_deg(pose_previous, pose_baseline)
    baseline_plausible = (
        baseline_step_t <= float(args.v14_baseline_max_translation_m)
        and baseline_step_r <= float(args.v14_baseline_max_rotation_deg)
    )
    if baseline_plausible and baseline_score >= previous_score - float(args.v14_baseline_score_drop):
        protected = pose_baseline
        source = "baseline"
        protected_score = baseline_score

    event_prior_score = scores["event_prior"]
    event_prior_step_t = translation_error_m(pose_previous, pose_event_prior)
    event_prior_step_r = rotation_error_deg(pose_previous, pose_event_prior)
    large_event_motion_gate = bool(
        large_event_motion_gate
        and (
            event_prior_step_t > float(args.v18_static_candidate_max_prior_residual_m)
            or translation_error_m(pose_baseline, pose_event_prior)
            > float(args.v18_static_candidate_max_prior_residual_m)
        )
    )
    event_prior_plausible = (
        bool(event_used)
        and float(event_confidence) >= float(args.v14_event_min_confidence)
        and event_prior_step_t <= float(args.v15_event_prior_max_translation_m)
        and event_prior_step_r <= float(args.v15_event_prior_max_rotation_deg)
        and (
            bool(large_event_motion_gate)
            or event_prior_score >= protected_score + float(args.v15_event_prior_score_margin)
        )
    )
    if event_prior_plausible:
        protected = pose_event_prior
        source = "event_prior"
        protected_score = event_prior_score

    best_event_key = None
    best_event_pose = None
    best_event_score = -np.inf
    best_event_step_t = np.inf
    best_event_step_r = np.inf
    event_max_rotation = (
        float(args.v16_rotation_candidate_max_step_deg)
        if rotation_recovery_triggered
        else float(args.v14_event_branch_max_rotation_deg)
    )
    for key in event_keys:
        candidate = candidates[key]
        step_t = translation_error_m(pose_previous, candidate)
        step_r = rotation_error_deg(pose_previous, candidate)
        plausible = (
            step_t <= float(args.v14_event_branch_max_translation_m)
            and step_r <= event_max_rotation
        )
        candidate_score = float(scores[key])
        if plausible and candidate_score > best_event_score:
            best_event_key = key
            best_event_pose = candidate
            best_event_score = scores[key]
            best_event_step_t = step_t
            best_event_step_r = step_r

    event_margin = (
        float(args.v16_rotation_event_score_margin)
        if rotation_recovery_triggered
        else float(args.v14_event_score_margin)
    )
    if large_event_motion_gate:
        event_margin = -float(args.v18_large_motion_refined_score_tolerance)
    event_plausible = (
        bool(event_used)
        and best_event_pose is not None
        and float(event_confidence) >= float(args.v14_event_min_confidence)
        and best_event_score >= protected_score + event_margin
    )
    if event_plausible:
        protected = best_event_pose
        source = best_event_key.replace("event::", "event_rot:", 1)
        protected_score = best_event_score

    stable_override = False
    stable_contracted_rejected = 0
    stable_score_drop = 0.0
    if stable_rotation_active and event_used:
        stable_ranked = []
        for key in event_keys:
            if "stable_axis_" not in key:
                continue
            candidate = candidates[key]
            step_t = translation_error_m(pose_previous, candidate)
            step_r = rotation_error_deg(pose_previous, candidate)
            if (
                step_t > float(args.v14_event_branch_max_translation_m)
                or step_r > float(args.v16_rotation_candidate_max_step_deg)
            ):
                continue
            scale = 1.0
            tail = key.rsplit("_", 1)[-1]
            if tail.endswith("x"):
                try:
                    scale = float(tail[:-1])
                except ValueError:
                    scale = 1.0
            expected_step = abs(float(stable_rotation_step_deg) * scale)
            is_raw_or_hybrid = "raw_stable_axis_" in key or "hybrid_stable_axis_" in key
            if (
                not is_raw_or_hybrid
                and expected_step > 1e-6
                and step_r
                < expected_step * float(args.v20_stable_min_angular_preservation)
            ):
                stable_contracted_rejected += 1
                continue
            adjusted = (
                float(scores[key])
                - float(args.v19_stable_scale_penalty) * abs(scale - 1.0)
            )
            stable_ranked.append((adjusted, float(scores[key]), key, candidate))
        if stable_ranked:
            stable_ranked.sort(key=lambda item: item[0], reverse=True)
            _, stable_score, stable_key, stable_pose = stable_ranked[0]
            stable_score_drop = float(protected_score - stable_score)
            stable_motion_preserved = (
                "raw_stable_axis_" in stable_key
                or "hybrid_stable_axis_" in stable_key
            )
            allowed_drop = float(args.v19_stable_score_tolerance)
            if rotation_compensation_active and stable_motion_preserved:
                allowed_drop = max(
                    allowed_drop,
                    float(args.v20_stable_score_drop_tolerance),
                )
            if stable_score >= protected_score - allowed_drop:
                protected = stable_pose
                source = stable_key.replace("event::", "event_rot:", 1)
                protected_score = stable_score
                stable_override = True
    scores.update({
        "raw": baseline_score,
        "previous": previous_score,
        "baseline": baseline_score,
        "event_prior": scores["event_prior"],
        "event": float(best_event_score if np.isfinite(best_event_score) else 0.0),
        "event_used": float(bool(event_used)),
        "event_confidence": float(event_confidence),
        "baseline_step_t": float(baseline_step_t),
        "baseline_step_r": float(baseline_step_r),
        "event_step_t": float(best_event_step_t if np.isfinite(best_event_step_t) else 0.0),
        "event_step_r": float(best_event_step_r if np.isfinite(best_event_step_r) else 0.0),
        "event_prior_step_t": float(event_prior_step_t),
        "event_prior_step_r": float(event_prior_step_r),
        "event_prior_plausible": float(event_prior_plausible),
        "event_plausible": float(event_plausible),
        "rotation_recovery_triggered": float(rotation_recovery_triggered),
        "large_event_motion_gate": float(large_event_motion_gate),
        "stable_rotation_active": float(stable_rotation_active),
        "stable_override": float(stable_override),
        "stable_contracted_rejected": float(stable_contracted_rejected),
        "stable_score_drop": float(stable_score_drop),
        "event_hypothesis_count": float(len(refined_event_poses)),
    })
    reason = (
        f"selected={source}; scores previous/baseline/prior/event="
        f"{previous_score:.3f}/{baseline_score:.3f}/{event_prior_score:.3f}/{scores['event']:.3f}; "
        f"event_conf={event_confidence:.3f}; prior_plausible={int(event_prior_plausible)}; "
        f"event_plausible={int(event_plausible)}; rotation_recovery={int(rotation_recovery_triggered)}; "
        f"large_event_gate={int(large_event_motion_gate)}; "
        f"stable_axis={int(stable_rotation_active)}; stable_override={int(stable_override)}; "
        f"stable_score_drop={stable_score_drop:.3f}; "
        f"stable_contracted_rejected={stable_contracted_rejected}; "
        f"hypotheses={len(refined_event_poses)}; scorer="
        f"{'foundationpose_batch' if learned_scores else 'rendered_depth'}"
    )
    return protected, source, scores, reason


def draw_lopet_projected_lines(
    img: np.ndarray,
    model_lines: np.ndarray,
    pose_cam: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    color: tuple[int, int, int],
    max_lines: int,
    thickness: int,
) -> np.ndarray:
    uv_lines, _ = project_lopet_model_lines(
        model_lines,
        pose_cam,
        K,
        H,
        W,
        min_projected_length_px=3.0,
        fov_margin_px=20.0,
    )
    if len(uv_lines) == 0:
        return img
    if max_lines > 0 and len(uv_lines) > max_lines:
        idx = np.linspace(0, len(uv_lines) - 1, int(max_lines)).astype(np.int64)
        uv_lines = uv_lines[idx]
    out = img
    for line in uv_lines:
        p1 = tuple(np.rint(line[0]).astype(int))
        p2 = tuple(np.rint(line[1]).astype(int))
        cv2.line(out, p1, p2, color, max(int(thickness), 1), cv2.LINE_AA)
    return out


def make_event_track_visualization(
    args: argparse.Namespace,
    reader: Event6DSequenceReader,
    event_points_for_vis: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    model_lines: np.ndarray,
    bbox: np.ndarray,
    inv_to_origin: np.ndarray,
    prev_pose_rgb: np.ndarray | None,
    prior_pose_rgb: np.ndarray,
    final_pose_rgb: np.ndarray,
    frame_name: str,
    event_used: bool,
    event_update_count: int,
    sub_window_count: int,
    line_score: float,
    pose_select_source: str,
    rel_t: float,
    rel_r: float,
) -> np.ndarray:
    vis = np.zeros((reader.H, reader.W, 3), dtype=np.uint8)
    if event_points_for_vis:
        x = np.concatenate([item[0] for item in event_points_for_vis]).astype(np.int32)
        y = np.concatenate([item[1] for item in event_points_for_vis]).astype(np.int32)
        p = np.concatenate([item[2] for item in event_points_for_vis])
        inside = (x >= 0) & (x < reader.W_event) & (y >= 0) & (y < reader.H_event)
        x, y, p = x[inside], y[inside], p[inside]
        max_points = int(args.event_track_max_points)
        if max_points > 0 and len(x) > max_points:
            sel = np.linspace(0, len(x) - 1, max_points).astype(np.int64)
            x, y, p = x[sel], y[sel], p[sel]
        z_event = float((reader.T_rgb_to_event @ final_pose_rgb)[2, 3])
        z = np.full(len(x), z_event, dtype=np.float32)
        u_rgb, v_rgb, p_rgb = event_pixels_to_rgb_pixels(
            x,
            y,
            z,
            p,
            K_event=reader.K_event,
            K_rgb=reader.K,
            T_event_to_rgb=reader.T_event_to_rgb,
            H_rgb=reader.H,
            W_rgb=reader.W,
        )
        if len(u_rgb) > 0:
            pos = p_rgb > 0
            vis[v_rgb[pos], u_rgb[pos]] = [255, 80, 60]
            vis[v_rgb[~pos], u_rgb[~pos]] = [80, 160, 255]

    final_rgb_pose = final_pose_rgb @ inv_to_origin
    if prev_pose_rgb is not None:
        prev_rgb_pose = prev_pose_rgb @ inv_to_origin
        vis = draw_posed_3d_box(
            reader.K, img=vis, ob_in_cam=prev_rgb_pose, bbox=bbox,
            line_color=(0, 255, 255), linewidth=2,
        )
    if bool(args.event_track_draw_model_lines):
        vis = draw_lopet_projected_lines(
            vis,
            model_lines,
            prior_pose_rgb,
            reader.K,
            reader.H,
            reader.W,
            color=(255, 120, 0),
            max_lines=int(args.event_track_max_model_lines),
            thickness=1,
        )
    vis = draw_posed_3d_box(
        reader.K, img=vis, ob_in_cam=final_rgb_pose, bbox=bbox,
        line_color=(0, 255, 0), linewidth=3,
    )
    vis = draw_xyz_axis(
        vis, ob_in_cam=final_rgb_pose, scale=0.1, K=reader.K,
        thickness=3, transparency=0, is_input_rgb=True,
    )
    status = "USED" if event_used else "NOT_USED"
    cv2.putText(
        vis,
        f"EVENT->RGB frame={frame_name} event={status} updates={event_update_count}/{sub_window_count} score={line_score:.3f} sel={pose_select_source}",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"green=final | cyan=previous | rel={rel_t:.3f}m/{rel_r:.1f}deg",
        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return vis


def stack_visual_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        new_w = max(1, int(right.shape[1] * left.shape[0] / max(right.shape[0], 1)))
        right = cv2.resize(right, (new_w, left.shape[0]), interpolation=cv2.INTER_AREA)
    return np.concatenate([left, right], axis=1)


def resize_for_window(img: np.ndarray, max_width: int, max_height: int, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    factor = min(float(scale), max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if factor >= 0.999:
        return img
    size = (max(1, int(w * factor)), max(1, int(h * factor)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def elapsed_ms(t_start: float) -> float:
    return (time.perf_counter() - t_start) * 1000.0


def summarize_timing_rows(rows: list[dict], path: str) -> None:
    if not rows:
        return
    stage_fields = [k for k in rows[0].keys() if k.endswith("_ms")]
    mean_total = float(np.mean([float(r.get("total_ms", 0.0)) for r in rows]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "mean_ms", "max_ms", "sum_ms", "percent_of_mean_total"],
        )
        writer.writeheader()
        for stage in stage_fields:
            values = np.array([float(r.get(stage, 0.0)) for r in rows], dtype=np.float64)
            writer.writerow({
                "stage": stage,
                "mean_ms": float(values.mean()),
                "max_ms": float(values.max()),
                "sum_ms": float(values.sum()),
                "percent_of_mean_total": float(values.mean() / (mean_total + 1e-9) * 100.0),
            })



V38_BUILD_ID = "v38_complete_forward_only_gpu_batch_2window_20260717"
_V25_BUILD_STABLE = build_v19_stable_axis_hypotheses
_V25_SELECT_POSE = select_v21_pose_candidates
_V25_CROP_EVENTS = crop_events_to_processing_window
_V25_FIXED_EVENT_WINDOW = fixed_event_window_from_pose
_V25_EXPANDED_CAD_SUPPORT = expanded_cad_support_in_event_window
_V38_EVENT_WINDOWS: list[dict] = []
_V38_EVENT_CONTEXT: dict = {}


def capture_v38_fixed_event_window(
    args,
    pose_rgb,
    inv_to_origin,
    bbox,
    K_event,
    T_rgb_to_event,
    H_event,
    W_event,
):
    roi = _V25_FIXED_EVENT_WINDOW(
        args,
        pose_rgb,
        inv_to_origin,
        bbox,
        K_event,
        T_rgb_to_event,
        H_event,
        W_event,
    )
    _V38_EVENT_CONTEXT.update({
        "K_event": np.asarray(K_event, dtype=np.float64).copy(),
        "T_rgb_to_event": np.asarray(T_rgb_to_event, dtype=np.float64).copy(),
        "H_event": int(H_event),
        "W_event": int(W_event),
        "event_window_count": max(int(args.event_sub_windows), 1),
    })
    return roi


def capture_v38_event_window(x, y, t, p, roi, processing_size):
    result = _V25_CROP_EVENTS(x, y, t, p, roi, processing_size)
    _, _, t_local, p_local, x_abs, y_abs, _, _ = result
    _V38_EVENT_WINDOWS.append({
        "x": np.asarray(x_abs, dtype=np.float32).copy(),
        "y": np.asarray(y_abs, dtype=np.float32).copy(),
        "t": np.asarray(t_local, dtype=np.float64).copy(),
        "p": np.asarray(p_local).copy(),
        "roi": tuple(map(float, roi)),
        "size": max(int(processing_size), 8),
    })
    keep_count = max(int(_V38_EVENT_CONTEXT.get("event_window_count", 2)), 1)
    del _V38_EVENT_WINDOWS[:-keep_count]
    return result


def capture_v38_cad_support(
    mesh,
    pose_event,
    K_event,
    roi,
    processing_size,
    dilation_px_full,
):
    _V38_EVENT_CONTEXT["mesh"] = mesh
    _V38_EVENT_CONTEXT["K_event"] = np.asarray(K_event, dtype=np.float64).copy()
    return _V25_EXPANDED_CAD_SUPPORT(
        mesh,
        pose_event,
        K_event,
        roi,
        processing_size,
        dilation_px_full,
    )


def interpolate_v38_pose(
    pose_start: np.ndarray,
    pose_end: np.ndarray,
    fraction: float,
) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    result = np.asarray(pose_start, dtype=np.float64).copy()
    delta_rotation = (
        np.asarray(pose_end, dtype=np.float64)[:3, :3]
        @ result[:3, :3].T
    )
    delta_rotvec = Rotation.from_matrix(delta_rotation).as_rotvec()
    result[:3, :3] = (
        Rotation.from_rotvec(delta_rotvec * fraction).as_matrix()
        @ result[:3, :3]
    )
    result[:3, 3] = (
        (1.0 - fraction) * result[:3, 3]
        + fraction * np.asarray(pose_end, dtype=np.float64)[:3, 3]
    )
    return result


def render_v38_model_edges(
    pose_rgb: np.ndarray,
    record: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    mesh = _V38_EVENT_CONTEXT.get("mesh")
    K_event = _V38_EVENT_CONTEXT.get("K_event")
    T_rgb_to_event = _V38_EVENT_CONTEXT.get("T_rgb_to_event")
    if mesh is None or K_event is None or T_rgb_to_event is None:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    pose_event = np.asarray(T_rgb_to_event) @ np.asarray(pose_rgb, dtype=np.float64)
    points = (pose_event[:3, :3] @ vertices.T).T + pose_event[:3, 3]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    if np.count_nonzero(valid) < 3:
        return None
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    uvw = (np.asarray(K_event) @ points[valid].T).T
    projected[valid] = uvw[:, :2] / uvw[:, 2:3]
    x0, y0, x1, y1 = record["roi"]
    size = int(record["size"])
    projected[:, 0] = (projected[:, 0] - x0) * size / max(x1 - x0, 1e-6)
    projected[:, 1] = (projected[:, 1] - y0) * size / max(y1 - y0, 1e-6)

    mask = np.zeros((size, size), dtype=np.uint8)
    if len(faces) > 0:
        face_valid = valid[faces].all(axis=1)
        triangles = projected[faces[face_valid]]
        finite_triangles = np.isfinite(triangles).all(axis=(1, 2))
        triangles = triangles[finite_triangles]
        if len(triangles) > 0:
            margin = float(size)
            intersects = (
                (np.max(triangles[:, :, 0], axis=1) >= -margin)
                & (np.min(triangles[:, :, 0], axis=1) < size + margin)
                & (np.max(triangles[:, :, 1], axis=1) >= -margin)
                & (np.min(triangles[:, :, 1], axis=1) < size + margin)
            )
            polygons = [
                np.rint(item).astype(np.int32)
                for item in triangles[intersects]
            ]
            if polygons:
                cv2.fillPoly(mask, polygons, 1)
    if np.count_nonzero(mask) < 20:
        uv = projected[valid]
        finite = np.isfinite(uv).all(axis=1)
        if np.count_nonzero(finite) < 3:
            return None
        hull = cv2.convexHull(uv[finite].astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 1)
    edge = cv2.morphologyEx(
        mask,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return mask, edge


def score_v38_rendered_event_window(
    model_mask: np.ndarray,
    model_edge: np.ndarray,
    record: dict,
) -> tuple[float, int]:
    x = np.asarray(record["x"], dtype=np.float64)
    y = np.asarray(record["y"], dtype=np.float64)
    t = np.asarray(record["t"], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x, y, t = x[finite], y[finite], t[finite]
    if len(t) < 40 or np.count_nonzero(model_edge) < 10:
        return 0.0, 0
    # The last 40% of each sub-window is a compact time surface near that
    # sub-window's target pose instead of a motion-blurred event frame.
    keep = t >= float(np.quantile(t, 0.60))
    x, y, t = x[keep], y[keep], t[keep]
    x0, y0, x1, y1 = record["roi"]
    size = int(record["size"])
    xi = np.rint((x - x0) * size / max(x1 - x0, 1e-6)).astype(np.int32)
    yi = np.rint((y - y0) * size / max(y1 - y0, 1e-6)).astype(np.int32)
    inside = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    xi, yi, t = xi[inside], yi[inside], t[inside]
    if len(xi) < 30:
        return 0.0, 0
    event_binary = np.zeros((size, size), dtype=np.uint8)
    event_binary[yi, xi] = 1
    event_binary = cv2.dilate(
        event_binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    distance_to_event = cv2.distanceTransform(
        1 - event_binary, cv2.DIST_L2, 3
    )
    edge_y, edge_x = np.nonzero(model_edge)
    coverage = float(np.mean(np.exp(-distance_to_event[edge_y, edge_x] / 3.0)))

    distance_to_model = cv2.distanceTransform(
        1 - (model_edge > 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    support = cv2.dilate(
        model_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    supported = support[yi, xi] > 0
    if np.count_nonzero(supported) >= 20:
        precision = float(np.mean(np.exp(
            -distance_to_model[yi[supported], xi[supported]] / 3.0
        )))
    else:
        precision = 0.0
    support_ratio = float(np.count_nonzero(supported) / max(len(xi), 1))
    score = 0.65 * coverage + 0.25 * precision + 0.10 * support_ratio
    return float(score), int(len(xi))


def score_v38_pose_on_event_window(
    pose_rgb: np.ndarray,
    record: dict,
) -> tuple[float, int]:
    """CPU fallback for environments where CUDA rasterization is unavailable."""
    rendered = render_v38_model_edges(pose_rgb, record)
    if rendered is None:
        return 0.0, 0
    return score_v38_rendered_event_window(rendered[0], rendered[1], record)


def render_v38_model_masks_gpu(
    pose_rgb_batch: np.ndarray,
    records: list[dict],
) -> np.ndarray | None:
    """Rasterize all candidate/window CAD silhouettes in one CUDA batch."""
    glctx = _V38_EVENT_CONTEXT.get("glctx")
    mesh_tensors = _V38_EVENT_CONTEXT.get("mesh_tensors")
    K_event = _V38_EVENT_CONTEXT.get("K_event")
    T_rgb_to_event = _V38_EVENT_CONTEXT.get("T_rgb_to_event")
    H_event = _V38_EVENT_CONTEXT.get("H_event")
    W_event = _V38_EVENT_CONTEXT.get("W_event")
    if (
        glctx is None
        or mesh_tensors is None
        or K_event is None
        or T_rgb_to_event is None
        or H_event is None
        or W_event is None
        or not records
    ):
        return None
    if _V38_EVENT_CONTEXT.get("gpu_batch_disabled", False):
        return None

    sizes = {int(record["size"]) for record in records}
    if len(sizes) != 1:
        logging.warning(
            f"V38 GPU trajectory fallback: mixed processing sizes={sorted(sizes)}"
        )
        return None
    size = sizes.pop()

    try:
        pos = mesh_tensors["pos"]
        faces = mesh_tensors["faces"]
        device = pos.device
        poses_rgb = np.asarray(pose_rgb_batch, dtype=np.float32).reshape(-1, 4, 4)
        T_event = np.asarray(T_rgb_to_event, dtype=np.float32).reshape(4, 4)
        poses_event = T_event[None] @ poses_rgb
        ob_in_cams = torch.as_tensor(poses_event, device=device, dtype=torch.float32)

        cv_to_gl = torch.as_tensor(
            glcam_in_cvcam, device=device, dtype=torch.float32
        ).reshape(1, 4, 4)
        projection = projection_matrix_from_intrinsics(
            np.asarray(K_event),
            height=int(H_event),
            width=int(W_event),
            znear=0.001,
            zfar=10.0,
        )
        projection = torch.as_tensor(
            projection, device=device, dtype=torch.float32
        ).reshape(1, 4, 4)
        matrices = projection @ (cv_to_gl @ ob_in_cams)

        ones = torch.ones((len(pos), 1), device=device, dtype=pos.dtype)
        pos_homo = torch.cat([pos, ones], dim=1)
        pos_clip = (
            matrices[:, None] @ pos_homo[None, :, :, None]
        )[..., 0]

        boxes = torch.as_tensor(
            [record["roi"] for record in records],
            device=device,
            dtype=torch.float32,
        )
        left = boxes[:, 0]
        top = float(H_event) - boxes[:, 1]
        right = boxes[:, 2]
        bottom = float(H_event) - boxes[:, 3]
        width = torch.clamp(right - left, min=1e-6)
        height = torch.clamp(top - bottom, min=1e-6)
        crop_tf = torch.eye(
            4, device=device, dtype=torch.float32
        ).reshape(1, 4, 4).repeat(len(records), 1, 1)
        crop_tf[:, 0, 0] = float(W_event) / width
        crop_tf[:, 1, 1] = float(H_event) / height
        crop_tf[:, 3, 0] = (float(W_event) - right - left) / width
        crop_tf[:, 3, 1] = (float(H_event) - top - bottom) / height
        pos_clip = (pos_clip @ crop_tf).contiguous()

        rast_out, _ = dr.rasterize(
            glctx,
            pos_clip,
            faces,
            resolution=np.asarray([size, size]),
        )
        masks = torch.flip(rast_out[..., 3] > 0, dims=[1])
        return masks.to(dtype=torch.uint8).cpu().numpy()
    except Exception as exc:
        _V38_EVENT_CONTEXT["gpu_batch_disabled"] = True
        logging.exception(
            f"V38 GPU trajectory rasterization failed; using CPU fallback: {exc}"
        )
        return None


def aggregate_v38_window_scores(
    scores: list[float],
    counts: list[int],
) -> tuple[float, int, list[float]]:
    valid = np.asarray(counts) >= 30
    required_valid = max(2, int(np.ceil(0.75 * len(counts))))
    if np.count_nonzero(valid) < required_valid:
        return 0.0, int(np.sum(counts)), scores
    score_array = np.asarray(scores, dtype=np.float64)
    weight_array = np.arange(1, len(scores) + 1, dtype=np.float64)
    trajectory_score = float(np.sum(
        score_array[valid] * weight_array[valid]
    ) / np.sum(weight_array[valid]))
    return trajectory_score, int(np.sum(counts)), scores


def score_v38_pose_trajectories_gpu(
    pose_start: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> dict[str, tuple[float, int, list[float]]] | None:
    """Score every candidate across the actual 2/4 windows with one raster call."""
    records = list(_V38_EVENT_WINDOWS)
    if len(records) < 2 or not candidates:
        return None

    names = list(candidates)
    trajectory_poses = []
    raster_records = []
    for name in names:
        pose_end = np.asarray(candidates[name])
        for index, record in enumerate(records):
            fraction = float(index + 1) / float(len(records))
            trajectory_poses.append(
                interpolate_v38_pose(pose_start, pose_end, fraction)
            )
            raster_records.append(record)

    total_start = time.perf_counter()
    render_start = time.perf_counter()
    masks = render_v38_model_masks_gpu(
        np.asarray(trajectory_poses), raster_records
    )
    render_ms = (time.perf_counter() - render_start) * 1000.0
    if masks is None or len(masks) != len(trajectory_poses):
        return None

    results = {}
    cursor = 0
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for name in names:
        scores = []
        counts = []
        for record in records:
            model_mask = np.asarray(masks[cursor], dtype=np.uint8)
            cursor += 1
            model_edge = cv2.morphologyEx(
                model_mask, cv2.MORPH_GRADIENT, edge_kernel
            )
            score, count = score_v38_rendered_event_window(
                model_mask, model_edge, record
            )
            scores.append(score)
            counts.append(count)
        results[name] = aggregate_v38_window_scores(scores, counts)

    total_ms = (time.perf_counter() - total_start) * 1000.0
    logging.warning(
        "V38 GPU batched event trajectory: "
        f"candidates={len(names)}, windows={len(records)}, "
        f"masks={len(trajectory_poses)}, render_ms={render_ms:.1f}, "
        f"total_ms={total_ms:.1f}"
    )
    return results


def score_v38_pose_trajectory(
    pose_start: np.ndarray,
    pose_end: np.ndarray,
) -> tuple[float, int, list[float]]:
    records = list(_V38_EVENT_WINDOWS)
    if len(records) < 2:
        return 0.0, 0, []
    scores = []
    counts = []
    weights = []
    for index, record in enumerate(records):
        fraction = float(index + 1) / float(len(records))
        pose = interpolate_v38_pose(pose_start, pose_end, fraction)
        score, count = score_v38_pose_on_event_window(pose, record)
        scores.append(score)
        counts.append(count)
        weights.append(float(index + 1))
    return aggregate_v38_window_scores(scores, counts)


def build_v38_stable_axis_hypotheses(
    args,
    pose_event_prior: np.ndarray,
    pose_previous: np.ndarray,
    pose_older: np.ndarray | None,
    pose_older2: np.ndarray | None,
):
    """Use normal V25 stability first, then a guarded latest-step fallback."""
    args._v38_single_step_rotation_active = False
    # This sequence has no large inter-frame rotation.  More importantly, the
    # V38 fallback must honor the public stable-axis switch: previously the
    # normal V25 branch was disabled, but the fallback below still generated
    # rotation hypotheses from tracking noise.
    if not bool(args.v19_enable_stable_axis_protection):
        return [], [], False, 0.0, 0.0
    poses, labels, active, latest_deg, axis_cosine = _V25_BUILD_STABLE(
        args,
        pose_event_prior=pose_event_prior,
        pose_previous=pose_previous,
        pose_older=pose_older,
        pose_older2=pose_older2,
    )
    if active or pose_older is None:
        return poses, labels, active, latest_deg, axis_cosine

    delta_latest = pose_previous[:3, :3] @ pose_older[:3, :3].T
    rotvec_latest = Rotation.from_matrix(delta_latest).as_rotvec()
    latest_norm = float(np.linalg.norm(rotvec_latest))
    latest_deg = float(np.rad2deg(latest_norm))
    if latest_norm < 1e-8 or latest_deg < 12.0 or latest_deg > 30.0:
        return [], [], False, latest_deg, axis_cosine

    # A weak older step may be the tracking failure itself.  It is not required
    # to have the same magnitude, but an actual direction reversal is rejected.
    axis_cosine = 1.0
    if pose_older2 is not None:
        delta_before = pose_older[:3, :3] @ pose_older2[:3, :3].T
        rotvec_before = Rotation.from_matrix(delta_before).as_rotvec()
        before_norm = float(np.linalg.norm(rotvec_before))
        if before_norm > np.deg2rad(2.0):
            axis_cosine = float(
                np.dot(rotvec_latest, rotvec_before)
                / max(latest_norm * before_norm, 1e-9)
            )
            if axis_cosine < float(args.v19_stable_axis_min_cosine):
                return [], [], False, latest_deg, axis_cosine

    recovery_poses = []
    recovery_labels = []
    for scale in (0.85, 1.0, 1.15):
        candidate = pose_event_prior.copy()
        candidate[:3, :3] = (
            Rotation.from_rotvec(rotvec_latest * scale).as_matrix()
            @ pose_previous[:3, :3]
        )
        recovery_poses.append(candidate)
        recovery_labels.append(f"stable_axis_v38_single_{scale:g}x")

    args._v38_single_step_rotation_active = True
    logging.warning(
        "V38 single-step rapid-rotation recovery: "
        f"latest={latest_deg:.2f}deg, older-axis-cos={axis_cosine:.3f}, "
        f"scales={[0.85, 1.0, 1.15]}"
    )
    return recovery_poses, recovery_labels, True, latest_deg, axis_cosine


def select_v38_pose_candidates(args, *call_args, **call_kwargs):
    """Use the configured event windows to rank the positive V38 candidates."""
    single_step_active = bool(
        getattr(args, "_v38_single_step_rotation_active", False)
    )
    old_tolerance = float(args.v19_stable_score_tolerance)
    if single_step_active:
        args.v19_stable_score_tolerance = max(old_tolerance, 0.40)
    try:
        selected = _V25_SELECT_POSE(args, *call_args, **call_kwargs)
    finally:
        args.v19_stable_score_tolerance = old_tolerance
    if not single_step_active:
        return selected
    pose, source, scores, reason = selected

    def supplied(name, positional_index):
        if name in call_kwargs:
            return call_kwargs[name]
        return call_args[positional_index] if positional_index < len(call_args) else None

    pose_previous = supplied("pose_previous", 2)
    pose_baseline = supplied("pose_baseline", 3)
    pose_event_prior = supplied("pose_event_prior", 4)
    pose_event_refined = supplied("pose_event_refined", 5)
    event_hypothesis_labels = supplied("event_hypothesis_labels", 6) or []
    extra_event_candidates = supplied("extra_event_candidates", 7) or {}
    if pose_previous is None:
        reason = f"{reason}; v38_event_distance_field=missing_previous"
        return pose, source, scores, reason

    candidates = {
        "previous": np.asarray(pose_previous),
        "baseline": np.asarray(pose_baseline),
        "event_prior": np.asarray(pose_event_prior),
    }
    refined = (
        []
        if pose_event_refined is None
        else list(np.asarray(pose_event_refined).reshape(-1, 4, 4))
    )
    labels = list(event_hypothesis_labels)
    if len(labels) < len(refined):
        labels.extend(f"hyp_{i:02d}" for i in range(len(labels), len(refined)))
    for label, candidate in zip(labels, refined):
        if "v38_single" in label:
            candidates[f"event::{label}"] = candidate
    for key, candidate in extra_event_candidates.items():
        candidate_key = key if str(key).startswith("event::") else f"event::{key}"
        if "v38_single" in candidate_key:
            candidates[candidate_key] = np.asarray(candidate)

    event_scores = {}
    event_counts = {}
    event_window_scores = {}
    batch_results = score_v38_pose_trajectories_gpu(
        np.asarray(pose_previous), candidates
    )
    if batch_results is None:
        logging.warning(
            "V38 event trajectory uses slow CPU fallback; CUDA batch unavailable"
        )
    for key, candidate in candidates.items():
        if batch_results is not None and key in batch_results:
            trajectory_score, event_count, window_scores = batch_results[key]
        else:
            trajectory_score, event_count, window_scores = score_v38_pose_trajectory(
                np.asarray(pose_previous), np.asarray(candidate)
            )
        event_scores[key] = float(trajectory_score)
        event_counts[key] = int(event_count)
        event_window_scores[key] = window_scores

    event_values = np.asarray(list(event_scores.values()), dtype=np.float64)
    total_event_count = max(event_counts.values(), default=0)
    event_range = (
        float(np.max(event_values) - np.min(event_values))
        if len(event_values) else 0.0
    )
    if total_event_count < 120 or event_range < 0.008:
        reason = (
            f"{reason}; v38_event_distance_field=insufficient/"
            f"events={total_event_count}/range={event_range:.4f}"
        )
        logging.warning(f"V38 event-score fallback: source={source}; {reason}")
        return pose, source, scores, reason

    event_min = float(np.min(event_values))
    event_normalized = {
        key: (value - event_min) / max(event_range, 1e-9)
        for key, value in event_scores.items()
    }
    learned_available = {
        key: float(scores[key])
        for key in candidates
        if key in scores and np.isfinite(float(scores[key]))
    }
    if learned_available:
        learned_max = max(learned_available.values())
        learned_normalized = {
            key: float(np.exp((value - learned_max) / 0.20))
            for key, value in learned_available.items()
        }
    else:
        learned_normalized = {key: 1.0 for key in candidates}
    combined = {
        key: 0.80 * event_normalized[key]
        + 0.20 * learned_normalized.get(key, 0.0)
        for key in candidates
    }
    best_key = max(combined, key=combined.get)
    current_key = (
        source.replace("event_rot:", "event::", 1)
        if str(source).startswith("event_rot:")
        else str(source)
    )
    current_event_score = float(event_scores.get(current_key, 0.0))
    best_event_score = float(event_scores[best_key])
    should_override = bool(
        "v38_single" in best_key
        and best_event_score >= current_event_score + 0.006
        and best_event_score > 0.03
    )
    compact_log = ", ".join(
        f"{key}={event_scores[key]:.4f}/{combined[key]:.3f}/"
        f"{[round(x, 4) for x in event_window_scores[key]]}"
        for key in sorted(candidates)
    )
    if should_override:
        pose = np.asarray(candidates[best_key]).copy()
        source = best_key.replace("event::", "event_rot:", 1)
        scores["event"] = float(scores.get(best_key, scores.get("event", 0.0)))
        reason = (
            f"event-distance trajectory selected={source}; "
            f"score={best_event_score:.4f}, previous_selection={current_key}/"
            f"{current_event_score:.4f}; candidates[{compact_log}]"
        )
    else:
        reason = (
            f"{reason}; v38_event_distance_kept={source}; "
            f"best={best_key}/{best_event_score:.4f}; candidates[{compact_log}]"
        )
    logging.warning(f"V38 positive-direction event selection: {reason}")
    return pose, source, scores, reason


def main(
    default_save_debug_images: bool = True,
    save_pose_outputs: bool = True,
    foundationpose_debug_override: int | None = None,
    save_csv_outputs: bool = True,
):
    global build_v19_stable_axis_hypotheses
    global select_v21_pose_candidates
    global crop_events_to_processing_window
    global fixed_event_window_from_pose
    global expanded_cad_support_in_event_window

    # 在单文件内部安装 V38 钩子；_V25_* 已保存上方定义的原始实现。
    build_v19_stable_axis_hypotheses = build_v38_stable_axis_hypotheses
    select_v21_pose_candidates = select_v38_pose_candidates
    crop_events_to_processing_window = capture_v38_event_window
    fixed_event_window_from_pose = capture_v38_fixed_event_window
    expanded_cad_support_in_event_window = capture_v38_cad_support

    parser = argparse.ArgumentParser(description="160x160 ROI 事件引导跟踪 V38")

    # =========================
    # 数据集/路径设置
    # =========================
    # Event6D 数据集根目录。默认值使脚本无需传入 --dataset_root 即可运行。
    parser.add_argument("--dataset_root", type=str, default=default_event6d_root(),
                        help="Event6D dataset root directory.")
    # dataset_root 下的序列。默认值使脚本无需传入 --sequence 即可运行。
    parser.add_argument("--sequence", type=str, default="mustard_1101/0001",
                        help="Sequence relative to dataset_root, e.g. banana_1101/0003.")
    # 序列的绝对目录。使用 dataset_root + sequence 时保持为 None。
    parser.add_argument("--test_scene_dir", type=str, default=None,
                        help="Optional absolute sequence dir. Overrides --dataset_root/--sequence.")
    # 网格路径。保持为 None 时自动读取 obj.txt，并使用 simple_mesh/<obj_id>/textured.obj。
   # 网格路径。保持为 None 时自动读取 obj.txt，并使用 simple_mesh/<obj_id>/textured.obj。
    parser.add_argument("--mesh_file", type=str, default=None,
                        help="Optional mesh path. Defaults to simple_mesh/<obj.txt>/textured.obj.")
    # 标定文件。保持为 None 时使用 <dataset_root>/0001-camchain.yaml。
    parser.add_argument("--camchain", type=str, default=None,
                        help="Optional camchain yaml. Defaults to <dataset_root>/0001-camchain.yaml.")
    # 所有可视化、姿态和 CSV 指标均写入此目录。
    parser.add_argument("--debug_dir", type=str, default=f"{_code_dir}/debug_event_contour_foundationpose_event6d_v38mustard")

    # =========================
    # FoundationPose 设置
    # =========================
    # 首帧配准的细化迭代次数。
    parser.add_argument("--est_refine_iter", type=int, default=5)
    # 后续帧跟踪的细化迭代次数。
    parser.add_argument("--track_refine_iter", type=int, default=2)
    # FoundationPose 调试级别。0 为静默；1/2 保存更多可视化调试结果。
    parser.add_argument("--debug", type=int, default=2)
    # RGB 帧率。Event6D RGB 为 30 FPS；解析后的事件位于相邻 RGB 帧之间。
    parser.add_argument("--fps", type=float, default=30.0)
    # 每隔 N 帧取一帧。增大该值可对大帧间运动进行压力测试。
    parser.add_argument("--frame_stride", type=int, default=1)
    # 可选的显式帧编号/范围，例如 "0,5,10-20"。None 表示全部选中帧。
    parser.add_argument("--frame_indices", type=str, default=None)
    # 默认启用：将 RGB 帧限制在 startend.txt 记录的区间内。
    parser.add_argument("--use_startend", action="store_true", default=True,
                        help="Use startend.txt to crop frames. Enabled by default in V12.")
    # 可选的显式首帧掩码。保持为 None 时优先使用与首个 RGB 帧匹配的掩码，否则使用第一个掩码文件。
    parser.add_argument("--initial_mask_file", type=str, default=None,
                        help="Optional explicit initial mask .npy path. Overrides automatic mask selection.")
    # 相对于当前 RGB 帧编号的事件帧偏移。
    # 跟踪上一 RGB 帧 -> 当前 RGB 帧时，默认值 0 读取 parsed_events/current_rgb.npz。
    # 仅在有意采用 Event6D 在线读取器的“当前 RGB 帧之后事件”行为时设为 1。
    parser.add_argument("--event_frame_offset", type=int, default=0,
                        help="Offset added to current RGB frame number when reading parsed_events/%06d.npz.")

    # =========================
    # ROI 加速设置
    # =========================
    # 默认启用：在以姿态为中心的 ROI 图块上执行首帧配准和 RGB-D 跟踪。
    parser.add_argument("--use_roi_input", action=argparse.BooleanOptionalAction, default=True,
                        help="Run RGB-D registration/tracking on cropped ROI patches while keeping poses in full-camera coordinates.")
    # 方形 ROI 输入大小。保存的可视化仍保持原始 RGB 分辨率。
    parser.add_argument("--roi_size", type=int, default=160,
                        help="Square ROI patch size for RGB-D FoundationPose inputs; V14 event motion uses full event coordinates.")
    # 应用于投影 CAD 包围框或初始掩码包围框周围的上下文缩放系数。
    parser.add_argument("--roi_context_scale", type=float, default=1.8,
                        help="Scale factor around the projected object bbox before resizing to --roi_size.")
    # 缩放前、以原图像素计的未裁剪 ROI 最小边长。
    parser.add_argument("--roi_min_size", type=float, default=96.0,
                        help="Minimum ROI side length in original pixels before resizing.")
    # Event6D 保留随姿态调整的原图 ROI，仅把所得图块缩放到 160 x 160。
    # V17 对事件采用相同的分离方式。
    parser.add_argument("--v17_use_fixed_event_window", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a moving fixed-size event window centered on the projected CAD bbox.")
    parser.add_argument("--v17_event_window_size_px", type=float, default=480.0,
                        help="Event search-window side length in original event-camera pixels.")
    parser.add_argument("--v17_event_processing_size", type=int, default=160,
                        help="Local event correlation resolution after cropping the original event window.")
    parser.add_argument("--v17_use_foundationpose_batch_scorer", action=argparse.BooleanOptionalAction, default=True,
                        help="Rank all pose candidates in one learned FoundationPose scorer call.")
    parser.add_argument("--v17_use_adaptive_rotation_bank", action=argparse.BooleanOptionalAction, default=False,
                        help="Start with two rotation candidates and expand the bank only on difficult frames.")
    parser.add_argument("--v17_rotation_hard_trigger_deg", type=float, default=18.0,
                        help="Baseline angular jump that forces expanded rotation recovery despite a static compact winner.")
    parser.add_argument("--v18_use_expanded_cad_support", action=argparse.BooleanOptionalAction, default=True,
                        help="Filter the broad event window with an expanded CPU-projected CAD hull.")
    parser.add_argument("--v18_cad_support_dilation_px", type=float, default=100.0,
                        help="CAD support expansion in original event-camera pixels.")
    parser.add_argument("--v18_cad_support_min_events", type=int, default=500,
                        help="Fallback to the full event window when expanded CAD support contains fewer events.")
    parser.add_argument("--v18_low_response_rescue", action=argparse.BooleanOptionalAction, default=True,
                        help="Rescue low-positive-response motion that agrees with the previous reliable event direction.")
    parser.add_argument("--v18_low_response_min", type=float, default=0.02,
                        help="Minimum positive phase response for temporal-direction rescue.")
    parser.add_argument("--v18_direction_rescue_cosine", type=float, default=0.65,
                        help="Minimum direction cosine with the previous reliable frame motion.")
    parser.add_argument("--v18_large_motion_threshold_px", type=float, default=45.0,
                        help="Frame event displacement that activates translation scales and static-pose rejection.")
    parser.add_argument("--v18_large_motion_min_consistency", type=float, default=0.72,
                        help="Minimum directional concentration for treating event displacement as reliable large motion.")
    parser.add_argument("--v18_translation_scales", type=str, default="0.6,1.0,1.5",
                        help="Event translation scales refined on reliable large-motion frames.")
    parser.add_argument("--v18_static_candidate_max_prior_residual_m", type=float, default=0.025,
                        help="Maximum distance from event prior for a baseline pose under reliable large event motion.")
    parser.add_argument("--v18_compact_rotation_score_tolerance", type=float, default=0.12,
                        help="Expand rotation bank when a rotating compact candidate is this close to the best learned score.")
    parser.add_argument("--v18_large_motion_refined_score_tolerance", type=float, default=0.08,
                        help="Allowed learned-score drop for an event-refined candidate under reliable large motion.")
    parser.add_argument("--v19_enable_stable_axis_protection", action=argparse.BooleanOptionalAction, default=False,
                        help="Protect continuous high-speed rotation when consecutive axes agree.")
    parser.add_argument("--v19_stable_rotation_min_deg", type=float, default=8.0,
                        help="Minimum angular increment in both preceding frames for stable-axis detection.")
    parser.add_argument("--v19_stable_axis_min_cosine", type=float, default=0.85,
                        help="Minimum cosine between the two preceding camera-frame rotation axes.")
    parser.add_argument("--v19_stable_rotation_scales", type=str, default="0.8,1.0,1.2",
                        help="Angular-speed scales generated around the latest stable rotation increment.")
    parser.add_argument("--v19_stable_score_tolerance", type=float, default=0.25,
                        help="Allowed learned-score drop when preserving a stable-axis candidate.")
    parser.add_argument("--v19_stable_scale_penalty", type=float, default=0.05,
                        help="Preference penalty for stable angular scales farther from constant velocity.")
    parser.add_argument("--v19_keep_raw_stable_priors", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep unrefined stable-axis priors as final selectable candidates.")
    # V20：在估计事件平移前移除 CAD 预测的旋转流。
    parser.add_argument("--v20_enable_rotation_compensation", action=argparse.BooleanOptionalAction, default=False,
                        help="Inverse-warp late events by CAD-predicted stable rotation before translation estimation.")
    parser.add_argument("--v20_rotation_compensation_min_deg", type=float, default=8.0,
                        help="Minimum angle in both preceding frames before rotation compensation is enabled.")
    parser.add_argument("--v20_rotation_compensation_axis_cosine", type=float, default=0.85,
                        help="Minimum preceding-axis cosine for CAD rotation compensation.")
    parser.add_argument("--v20_rotation_compensation_scale", type=float, default=1.0,
                        help="Scale applied to the historical frame rotation before splitting it across event windows.")
    parser.add_argument("--v20_rotation_compensation_max_vertices", type=int, default=3000,
                        help="Maximum CAD vertices used to fit each local rotation-flow affine warp.")
    parser.add_argument("--v20_rotation_compensation_min_response_ratio", type=float, default=0.45,
                        help="Reject compensation if phase response falls below this fraction of the raw response.")
    # 旋转补偿会降低残余平移的可信度；接受完整位移前先对分数倍事件平移评分。
    parser.add_argument("--v20_rotation_translation_scales", type=str, default="0.25,0.5,1.0",
                        help="Translation scales scored when CAD rotation compensation is active.")
    # V20：显式覆盖角运动方向变化，同时避免每帧都扩展候选库。
    parser.add_argument("--v20_enable_reversal_hypotheses", action=argparse.BooleanOptionalAction, default=True,
                        help="Force signed historical rotation hypotheses when RGB-D angular motion reverses history.")
    parser.add_argument("--v20_reversal_min_baseline_deg", type=float, default=4.0,
                        help="Minimum current baseline rotation for reversal detection.")
    parser.add_argument("--v20_reversal_min_history_deg", type=float, default=8.0,
                        help="Minimum preceding rotation for reversal detection.")
    parser.add_argument("--v20_reversal_axis_max_cosine", type=float, default=0.0,
                        help="Maximum baseline/history axis cosine counted as a direction reversal.")
    parser.add_argument("--v20_reversal_history_scales", type=str, default="0.4,0.7,1.0",
                        help="Magnitudes of negative historical increments generated after reversal detection.")
    # V20：保留角度幅值，同时仍允许 RGB-D 细化平移。
    parser.add_argument("--v20_keep_stable_hybrid_candidates", action=argparse.BooleanOptionalAction, default=True,
                        help="Combine each raw stable rotation with its RGB-D-refined translation.")
    parser.add_argument("--v20_stable_min_angular_preservation", type=float, default=0.80,
                        help="Reject refined stable candidates retaining less than this fraction of their raw angle.")
    parser.add_argument("--v20_stable_score_drop_tolerance", type=float, default=2.0,
                        help="Maximum RGB-D score drop allowed for a motion-consistent stable candidate under rotation compensation.")
    # V21：根据单调事件数量趋势触发双向旋转候选库。
    parser.add_argument("--v21_enable_bidirectional_rotation", action=argparse.BooleanOptionalAction, default=True,
                        help="Refine forward and reverse historical rotation hypotheses on ambiguous motion frames.")
    parser.add_argument("--v21_rotation_trigger_deg", type=float, default=8.0,
                        help="Minimum RGB-D angular cue required before an event-count trend can trigger bidirectional search.")
    parser.add_argument("--v21_bidirectional_rotation_scales", type=str, default="0.6,0.8,1.0",
                        help="Absolute historical rotation scales used for both forward and reverse hypotheses.")
    parser.add_argument("--v21_event_count_min_subwindow", type=int, default=80,
                        help="Minimum filtered events per sub-window before event-count trend analysis is trusted.")
    parser.add_argument("--v21_event_count_trend_min", type=float, default=0.08,
                        help="Minimum normalized event-count slope that activates bidirectional rotation search.")
    parser.add_argument("--v21_event_count_trend_consistency", type=float, default=0.66,
                        help="Minimum monotonic sign consistency of adjacent event-count changes.")

    # =========================
    # 事件位移设置
    # =========================
    # V38 默认使用两个子时间步；可显式传入 4 复现实验中的四窗口模式。
    parser.add_argument("--event_sub_windows", type=int, default=5,
                        help="Split each inter-frame event packet into N windows. V38 defaults to 2; pass 4 for the previous mode.")
    # 用于渲染事件过滤掩码的姿态。KF 预测漂移时 previous 更安全。
    parser.add_argument("--event_mask_pose", choices=["previous", "current", "kf", "union"], default="current",
                        help="Render event support from the current propagated pose; kf is a legacy alias.")
    # 每个子窗口重新渲染事件支持掩码。默认 False 以保持 V9 速度。
    parser.add_argument("--event_mask_update_each_subwindow", action="store_true", default=False,
                        help="Re-render event mask after each sub-window event-line update. Slower, sometimes more accurate.")

    # =========================
    # LOPET 风格点到线优化设置
    # =========================
    # 提取锐边/边界边后保留的最大 CAD 网格线数量。
    parser.add_argument("--lopet_max_model_lines", type=int, default=600,
                        help="Maximum sharp/boundary CAD edges used as LOPET 3D model lines.")
    # 网格面邻接角阈值。值越大，仅保留越显著的 CAD 边。
    parser.add_argument("--lopet_sharp_edge_angle_deg", type=float, default=30.0,
                        help="Minimum dihedral angle in degrees for CAD sharp-edge extraction.")
    # 线匹配前丢弃很短的三维网格边。
    parser.add_argument("--lopet_min_model_line_length_m", type=float, default=0.001,
                        help="Discard CAD model lines shorter than this length in meters.")
    # 保存的轮廓调试图像大小。计算仍按 roi_size 进行。
    parser.add_argument("--contour_vis_width", type=int, default=960)
    parser.add_argument("--contour_vis_height", type=int, default=720)

    # =========================
    # V14 鲁棒二维事件运动传播
    # =========================
    # 从狭窄的渲染轮廓带而非完整膨胀物体掩码中取事件，以减少手部/内部纹理污染。
    parser.add_argument("--v14_event_motion_band_dilation", type=int, default=8,
                        help="Silhouette-band dilation in event pixels for motion filtering.")
    parser.add_argument("--v14_event_min_events", type=int, default=80,
                        help="Minimum filtered events in one sub-window before motion estimation.")
    parser.add_argument("--v14_event_cell_size", type=int, default=4,
                        help="Event-cell size in pixels for temporal centroid matching.")
    parser.add_argument("--v14_event_min_cell_count", type=int, default=2,
                        help="Minimum events in a polarity-aware cell.")
    parser.add_argument("--v14_event_max_match_px", type=float, default=24.0,
                        help="Maximum early-to-late cell match distance in pixels.")
    parser.add_argument("--v14_event_min_matches", type=int, default=6,
                        help="Minimum temporal cell matches before RANSAC.")
    parser.add_argument("--v14_event_ransac_threshold_px", type=float, default=3.0,
                        help="RANSAC reprojection threshold in event pixels.")
    parser.add_argument("--v14_event_ransac_iters", type=int, default=200,
                        help="Maximum RANSAC iterations for event affine motion.")
    parser.add_argument("--v14_event_ransac_confidence", type=float, default=0.99,
                        help="RANSAC confidence for event motion estimation.")
    parser.add_argument("--v14_event_min_inliers", type=int, default=5,
                        help="Minimum RANSAC inlier count for an event update.")
    parser.add_argument("--v14_event_min_inlier_ratio", type=float, default=0.40,
                        help="Minimum RANSAC inlier ratio for an event update.")
    parser.add_argument("--v14_event_min_sectors", type=int, default=2,
                        help="Minimum occupied angular sectors among event inliers.")
    parser.add_argument("--v14_event_min_phase_response", type=float, default=0.12,
                        help="Minimum phase-correlation response for an event motion update.")
    parser.add_argument("--v14_event_temporal_extrapolation", type=float, default=2.0,
                        help="Scale early-to-late half-window motion to the full event sub-window.")
    parser.add_argument("--v14_event_max_rms_px", type=float, default=4.0,
                        help="Maximum RANSAC inlier RMS residual in pixels.")
    parser.add_argument("--v14_event_rms_scale_px", type=float, default=4.0,
                        help="RMS scale used to convert event motion quality to confidence.")
    parser.add_argument("--v14_event_min_motion_px", type=float, default=0.5,
                        help="Minimum early-to-late image motion in pixels.")
    parser.add_argument("--v14_event_max_motion_px", type=float, default=32.0,
                        help="Maximum early-to-late image motion in pixels.")
    parser.add_argument("--v14_event_max_rotation_deg", type=float, default=8.0,
                        help="Maximum fitted 2D affine rotation in one event sub-window.")
    parser.add_argument("--v14_event_max_scale_change", type=float, default=0.15,
                        help="Maximum fitted affine scale change in one event sub-window.")
    parser.add_argument("--v14_event_min_confidence", type=float, default=0.35,
                        help="Minimum event motion confidence for pose propagation.")
    parser.add_argument("--v14_event_max_translation_step_m", type=float, default=0.025,
                        help="Maximum XY pose translation generated by one event sub-window.")
    # 仅当事件分支移动量足以具有意义时才考虑该分支。
    parser.add_argument("--v14_event_min_pose_shift_m", type=float, default=0.001,
                        help="Minimum event prior shift before running the second FoundationPose branch.")
    parser.add_argument("--v14_baseline_score_drop", type=float, default=0.05,
                        help="Allowed RGB-D score drop for a temporally plausible baseline pose.")
    parser.add_argument("--v14_baseline_max_translation_m", type=float, default=0.08,
                        help="Maximum baseline FoundationPose frame translation considered plausible.")
    parser.add_argument("--v14_baseline_max_rotation_deg", type=float, default=20.0,
                        help="Maximum baseline FoundationPose frame rotation considered plausible.")
    parser.add_argument("--v14_event_branch_max_translation_m", type=float, default=0.12,
                        help="Maximum event-branch refined translation from the previous pose.")
    parser.add_argument("--v14_event_branch_max_rotation_deg", type=float, default=20.0,
                        help="Maximum event-branch refined rotation from the previous pose.")
    parser.add_argument("--v14_event_score_margin", type=float, default=0.02,
                        help="Required RGB-D score advantage for the event branch over baseline.")
    # V15：当第二次 FoundationPose 细化使已验证的二维事件先验变差时，允许该
    # 事件先验本身继续保留。
    parser.add_argument("--v15_event_prior_score_margin", type=float, default=0.0,
                        help="Minimum RGB-D score margin for selecting event_prior over the protected pose.")
    parser.add_argument("--v15_event_prior_max_translation_m", type=float, default=0.12,
                        help="Maximum event-prior translation from the previous accepted pose.")
    parser.add_argument("--v15_event_prior_max_rotation_deg", type=float, default=50,
                        help="Maximum event-prior rotation from the previous accepted pose.")

    # V16：补偿系统性的事件位移低估，而不依赖恒速或卡尔曼运动状态。
    parser.add_argument("--v16_event_translation_gain", type=float, default=1.12,
                        help="Gain applied to accepted event-center displacement before XY back-projection.")
    parser.add_argument("--v16_translation_gain_max_affine_rotation_deg", type=float, default=3.0,
                        help="Disable translation gain when fitted event rotation exceeds this value.")
    parser.add_argument("--v16_motion_affine_blend", type=float, default=0.75,
                        help="Maximum blend from phase motion to rotation-compensated affine-center motion.")
    parser.add_argument("--v16_motion_rotation_blend_deg", type=float, default=3.0,
                        help="Fitted 2D rotation at which affine-center motion receives its maximum blend.")
    parser.add_argument("--v16_motion_scale_blend", type=float, default=0.08,
                        help="Fitted scale change at which affine-center motion receives its maximum blend.")

    # V16：事件流使目标居中；紧凑 SO(3) 候选库通过一次批量 FoundationPose
    # 细化调用恢复快速旋转。
    parser.add_argument("--v16_enable_rotation_recovery", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable event-centered rotation hypotheses during rapid rotation.")
    parser.add_argument("--v16_rotation_trigger_deg", type=float, default=6.0,
                        help="Baseline/history angular step that activates the expanded rotation bank.")
    parser.add_argument("--v16_rotation_history_scales", type=str, default="1.0",
                        help="Scales of the previous accepted angular increment used as rotation hypotheses.")
    parser.add_argument("--v16_use_camera_axis_hypotheses", action=argparse.BooleanOptionalAction, default=True,
                        help="Add positive/negative camera X/Y/Z rotation hypotheses after triggering.")
    parser.add_argument("--v16_rotation_axis_scale", type=float, default=1.0,
                        help="Scale from the detected angular cue to camera-axis hypothesis angle.")
    parser.add_argument("--v16_rotation_axis_min_deg", type=float, default=6.0,
                        help="Minimum camera-axis hypothesis angle after rotation recovery triggers.")
    parser.add_argument("--v16_rotation_max_step_deg", type=float, default=25.0,
                        help="Maximum angular step represented by one generated hypothesis.")
    parser.add_argument("--v16_rotation_candidate_max_step_deg", type=float, default=20.0,
                        help="Maximum refined event-hypothesis rotation from the previous accepted pose.")
    parser.add_argument("--v16_rotation_max_hypotheses", type=int, default=10,
                        help="Maximum event-centered hypotheses refined in one FoundationPose batch.")
    parser.add_argument("--v16_rotation_dedup_deg", type=float, default=0.75,
                        help="Angular distance below which generated rotation hypotheses are deduplicated.")
    parser.add_argument("--v16_rotation_roi_scale", type=float, default=1.35,
                        help="ROI expansion used while the rapid-rotation hypothesis bank is active.")
    parser.add_argument("--v16_rotation_event_score_margin", type=float, default=0.01,
                        help="RGB-D score margin required by a triggered rotation hypothesis.")

    # =========================
    # 姿态候选选择设置
    # =========================
    # 默认启用：在 FoundationPose 细化后保护事件候选和上一帧候选。
    parser.add_argument("--disable_pose_candidate_selection", action="store_true", default=False,
                        help="Disable RGB-D scoring among raw, event-prior, and previous poses.")
    parser.add_argument("--pose_select_abrupt_translation_m", type=float, default=0.05,
                        help="Raw translation above this value is treated as an abrupt pose jump.")
    parser.add_argument("--pose_select_abrupt_rotation_deg", type=float, default=20.0,
                        help="Raw rotation above this value is treated as an abrupt pose jump.")
    parser.add_argument(
        "--v38_lock_large_rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the previous accepted rotation when the selected pose exceeds "
            "--v38_max_frame_rotation_deg; translation and depth are retained."
        ),
    )
    parser.add_argument(
        "--v38_max_frame_rotation_deg",
        type=float,
        default=20.0,
        help="Maximum accepted inter-frame rotation for the no-large-rotation mustard sequence.",
    )
    parser.add_argument(
        "--v38_rotation_limit_frames",
        type=str,
        default="",
        help=(
            "RGB frame numbers that use --v38_rotation_limit_deg instead of the global "
            "--v38_max_frame_rotation_deg, e.g. 257-259 or 257,258,259."
        ),
    )
    parser.add_argument(
        "--v38_rotation_limit_deg",
        type=float,
        default=12.0,
        help="Per-frame rotation limit for frames selected by --v38_rotation_limit_frames.",
    )
    parser.add_argument(
        "--v38_translation_scale_frames",
        type=str,
        default="",
        help=(
            "RGB frame numbers that use --v38_frame_translation_scales instead of "
            "the normal large-motion translation scales."
        ),
    )
    parser.add_argument(
        "--v38_frame_translation_scales",
        type=str,
        default="1.0",
        help="Translation scales used on frames selected by --v38_translation_scale_frames.",
    )
    parser.add_argument(
        "--v38_depth_drift_gate_frames",
        type=str,
        default="",
        help=(
            "RGB frame numbers on which excessive refined-candidate Z drift is reset "
            "to the unrefined event-prior Z while keeping refined X/Y and rotation."
        ),
    )
    parser.add_argument(
        "--v38_max_refined_prior_z_drift_m",
        type=float,
        default=0.02,
        help=(
            "Maximum absolute refined-candidate Z drift from the event prior on "
            "frames selected by --v38_depth_drift_gate_frames."
        ),
    )
    # 姿态选择中渲染深度与观测深度的残差阈值，单位为米。
    parser.add_argument("--pose_select_depth_threshold", type=float, default=0.035,
                        help="Depth inlier threshold in meters for rendered CAD vs observed depth.")
    # 观测深度比渲染深度近超过该值时，视为前景遮挡。
    parser.add_argument("--pose_select_occlusion_threshold", type=float, default=0.02,
                        help="Foreground occlusion threshold in meters during pose selection.")
    # 剔除 RGB 像素过少的渲染候选。
    parser.add_argument("--pose_select_min_render_pixels", type=int, default=200,
                        help="Minimum rendered CAD pixels required for candidate depth scoring.")

    # =========================
    # 可视化设置
    # =========================
    # 默认 False：跟踪时显示实时 OpenCV 窗口。
    # 在窗口中按 q 或 Esc 可提前停止。无界面服务器上使用 --no_show_window。
    parser.add_argument("--no_show_window", action="store_true", default=False,
                        help="Disable real-time OpenCV window. Default is False/show window.")
    # 启用 show_window 时 OpenCV waitKey 的延迟，单位为毫秒。
    parser.add_argument("--window_delay", type=int, default=1)
    # 实时可视化面板的显示缩放比例。
    parser.add_argument("--window_scale", type=float, default=0.55)
    # 可视化窗口最大宽度。
    parser.add_argument("--window_max_width", type=int, default=1280)
    # 可视化窗口最大高度。
    parser.add_argument("--window_max_height", type=int, default=720)
    # 保存 PNG 调试图像。event_vis 在 RGB 相机像素坐标系中绘制。
    parser.add_argument(
        "--save_debug_images",
        action=argparse.BooleanOptionalAction,
        default=bool(default_save_debug_images),
                        help="Save track_vis/event_vis PNGs in addition to realtime display.")
    # 默认启用：并排显示 RGB 跟踪和 RGB 坐标事件跟踪。
    parser.add_argument("--event_track_in_window", action=argparse.BooleanOptionalAction, default=True,
                        help="Show RGB-coordinate event tracking visualization next to RGB tracking in the realtime window.")
    # event_track_vis 上绘制的过滤事件点最大数量。<=0 时全部绘制。
    parser.add_argument("--event_track_max_points", type=int, default=50000,
                        help="Maximum filtered event points drawn in RGB-coordinate event tracking visualization.")
    # 默认禁用：旧版轮廓专用叠加图单独保存。
    parser.add_argument("--event_track_draw_model_lines", action=argparse.BooleanOptionalAction, default=False,
                        help="Draw legacy mesh-edge overlays on RGB-coordinate event visualization.")
    # event_track_vis 上绘制的投影 CAD 线最大数量，以避免画面杂乱。
    parser.add_argument("--event_track_max_model_lines", type=int, default=180,
                        help="Maximum projected CAD model lines drawn in RGB-coordinate event visualization.")
    # 默认启用：为每个帧间事件子窗口保存 RGB 坐标事件可视化。
    parser.add_argument("--save_event_substep_vis", action=argparse.BooleanOptionalAction, default=True,
                        help="Save event_track_vis images for each inter-frame event sub-window.")
    # 默认启用：为每个子窗口保存 V14 早期/后期事件网格运动叠加图。
    parser.add_argument("--save_event_contour_vis", action=argparse.BooleanOptionalAction, default=True,
                        help="Save V14 event-cell motion alignment images for each event sub-window.")
    args = parser.parse_args()
    args._v38_rotation_limit_frames = parse_frame_number_set(args.v38_rotation_limit_frames)
    args._v38_translation_scale_frames = parse_frame_number_set(args.v38_translation_scale_frames)
    args._v38_depth_drift_gate_frames = parse_frame_number_set(args.v38_depth_drift_gate_frames)

    # V38 只允许正方向历史恢复，命令行也不能重新开启反向候选。
    args.v20_enable_reversal_hypotheses = False
    args.v21_enable_bidirectional_rotation = False

    set_logging_format()
    logging.warning(
        f"Starting {V38_BUILD_ID}; complete single-file V38, "
        "forward-only recovery with normal saving"
    )
    set_seed(0)

    sequence_dir = args.test_scene_dir or os.path.join(args.dataset_root, args.sequence)
    reader = Event6DSequenceReader(
        sequence_dir=sequence_dir,
        dataset_root=args.dataset_root,
        camchain_path=args.camchain,
        use_startend=args.use_startend,
        initial_mask_file=args.initial_mask_file,
    )
    mesh_file = args.mesh_file or reader.get_mesh_path()

    os.makedirs(args.debug_dir, exist_ok=True)
    for sub in [
        "track_vis", "event_vis", "event_track_vis", "event_contour_vis", "ob_in_cam", "ob_in_cam_filtered",
        "ob_in_cam_prior", "relative_pose",
    ]:
        os.makedirs(os.path.join(args.debug_dir, sub), exist_ok=True)

    mesh = trimesh.load(mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    inv_to_origin = np.linalg.inv(to_origin)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    lopet_model_lines = build_lopet_model_lines(
        mesh,
        max_lines=args.lopet_max_model_lines,
        sharp_angle_deg=args.lopet_sharp_edge_angle_deg,
        min_length_m=args.lopet_min_model_line_length_m,
    )
    logging.info(
        f"V14 event-motion model initialized from {len(mesh.vertices)} CAD vertices; "
        f"{len(lopet_model_lines)} legacy edges are retained only for optional visualization."
    )

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=(
            args.debug
            if foundationpose_debug_override is None
            else int(foundationpose_debug_override)
        ),
        glctx=glctx,
    )
    _V38_EVENT_CONTEXT.update({
        "glctx": est.glctx,
        "mesh_tensors": est.mesh_tensors,
        "gpu_batch_disabled": False,
    })

    event_mask_renderer = ProjectionMaskRenderer(mesh, reader.K_event, reader.H_event, reader.W_event, zfar=10.0)
    rgb_pose_renderer = ProjectionMaskRenderer(mesh, reader.K, reader.H, reader.W, zfar=10.0)

    frame_ids = parse_frame_list(args.frame_indices, len(reader))
    frame_ids = frame_ids[::max(args.frame_stride, 1)]
    if len(frame_ids) < 2:
        raise RuntimeError("Need at least two frames for tracking.")

    metrics_path = os.path.join(args.debug_dir, "event_contour_metrics.csv")
    timing_path = os.path.join(args.debug_dir, "event_contour_timing.csv")
    timing_summary_path = os.path.join(args.debug_dir, "event_contour_timing_summary.csv")
    corr_stats_path = os.path.join(args.debug_dir, "event_contour_stats.csv")
    if not save_csv_outputs:
        metrics_path = os.devnull
        timing_path = os.devnull
        timing_summary_path = os.devnull
        corr_stats_path = os.devnull
    timing_stage_fields = [
        "frame_meta_ms",
        "load_color_ms",
        "load_depth_ms",
        "initial_mask_ms",
        "roi_prepare_ms",
        "initial_registration_ms",
        "render_pred_event_mask_ms",
        "load_events_ms",
        "split_events_ms",
        "event_filter_ms",
        "event_center_ms",
        "event_motion_total_ms",
        "event_contour_align_ms",
        "make_prior_ms",
        "foundationpose_track_ms",
        "foundationpose_baseline_ms",
        "foundationpose_event_ms",
        "pose_select_ms",
        "state_update_ms",
        "relative_pose_ms",
        "event_vis_ms",
        "event_track_vis_ms",
        "event_contour_vis_ms",
        "save_pose_txt_ms",
        "track_vis_draw_ms",
        "track_vis_save_ms",
        "show_window_ms",
        "metrics_write_ms",
        "total_ms",
    ]
    timing_rows = []
    with open(metrics_path, "w", newline="", encoding="utf-8") as f, \
            open(timing_path, "w", newline="", encoding="utf-8") as timing_f, \
            open(corr_stats_path, "w", newline="", encoding="utf-8") as corr_f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index", "frame_name", "timestamp", "dt",
                "event_count", "du", "dv", "event_confidence",
                "dominant_vote_count", "local_vector_count", "dominant_ratio",
                "event_large_motion_gate", "event_count_trend", "event_count_trend_consistency",
                "event_count_median", "bidirectional_rotation_active",
                "prior_base_source",
                "raw_pose_selected", "state_update_source", "state_select_reason",
                "prior_residual_trans_m", "prior_residual_rot_deg",
                "event_used", "event_contour_update_count", "event_contour_skip_count",
                "event_contour_mean_score", "delta_rgb_m", "z_used",
                "pose_select_source", "pose_raw_score", "pose_baseline_score", "pose_event_score",
                "pose_event_prior_score", "pose_previous_score",
                "pose_raw_abrupt", "pose_event_supports_raw", "pose_event_prior_plausible", "pose_select_reason",
                "rotation_recovery_triggered", "rotation_hypothesis_count",
                "rotation_cue_deg", "selected_event_hypothesis",
                "stable_rotation_active", "stable_rotation_step_deg",
                "stable_rotation_axis_cosine", "stable_rotation_override",
                "stable_contracted_rejected", "stable_score_drop", "reversal_rotation_active",
                "rotation_compensation_active", "rotation_compensation_step_deg",
                "rotation_compensation_axis_cosine",
                "relative_trans_m", "relative_rot_deg",
            ],
        )
        writer.writeheader()
        timing_writer = csv.DictWriter(
            timing_f,
            fieldnames=[
                "frame_index", "frame_name", "step_i", "is_init",
                "raw_event_count", "mask_event_count", "rgb_vis_event_count",
                "sub_window_count", "debug", "save_debug_images",
                "dominant_vote_count", "local_vector_count", "dominant_ratio",
                "event_large_motion_gate", "event_count_trend", "event_count_trend_consistency",
                "event_count_median", "bidirectional_rotation_active",
                "event_contour_update_count", "event_contour_skip_count",
                "event_contour_mean_score",
                "prior_base_source",
                "pose_select_source", "pose_raw_score", "pose_baseline_score", "pose_event_score",
                "pose_event_prior_score", "pose_previous_score",
                "pose_raw_abrupt", "pose_event_supports_raw", "pose_event_prior_plausible",
                "rotation_recovery_triggered", "rotation_hypothesis_count",
                "rotation_cue_deg", "selected_event_hypothesis",
                "stable_rotation_active", "stable_rotation_step_deg",
                "stable_rotation_axis_cosine", "stable_rotation_override",
                "stable_contracted_rejected", "stable_score_drop", "reversal_rotation_active",
                "rotation_compensation_active", "rotation_compensation_step_deg",
                "rotation_compensation_axis_cosine",
                "raw_pose_selected", "state_update_source",
                "prior_residual_trans_m", "prior_residual_rot_deg",
                "track_refine_iter", *timing_stage_fields,
            ],
        )
        timing_writer.writeheader()
        corr_writer = csv.DictWriter(
            corr_f,
            fieldnames=[
                "frame_name", "sub_window", "event_count", "recent_event_count",
                "early_count", "late_count", "match_count", "inlier_count", "inlier_ratio",
                "spatial_sectors", "rms_px", "phase_response", "motion_confidence",
                "rotation_compensated", "rotation_compensation_rejected",
                "predicted_rotation_deg", "rotation_projection_count",
                "raw_phase_dx", "raw_phase_dy", "raw_phase_response",
                "affine_rotation_deg", "affine_scale", "phase_dx", "phase_dy",
                "affine_center_dx", "affine_center_dy", "rotation_flow_weight", "translation_gain",
                "contour_point_count", "contour_edge_count",
                "initial_cost", "coarse_cost", "final_cost",
                "initial_coverage", "final_coverage", "event_coverage",
                "coarse_du", "coarse_dv", "translation_step_m", "rotation_step_deg",
                "contour_score", "accepted_event_update", "reason", "image",
            ],
        )
        corr_writer.writeheader()

        prev_pose = None
        prev_prev_pose = None
        prev_prev_prev_pose = None
        prev_reliable_event_motion_px = np.zeros(2, dtype=np.float64)
        prev_frame_num = None
        prev_event_z = None
        fps_times = []
        stop_requested = False

        for step_i, frame_i in enumerate(frame_ids):
            frame_t0 = time.perf_counter()
            timing = {name: 0.0 for name in timing_stage_fields}
            raw_event_count = 0
            rgb_vis_event_count = 0
            sub_window_count = 0
            dominant_vote_count = 0
            local_vector_count = 0
            dominant_ratio = 0.0
            event_large_motion_gate = False
            event_kf_update_count = 0
            event_kf_predict_only_count = 0
            event_kf_mean_confidence = 0.0
            pose_select_source = "init"
            pose_raw_score = 0.0
            pose_prior_score = 0.0
            pose_baseline_score = 0.0
            pose_event_score = 0.0
            pose_previous_score = 0.0
            pose_raw_abrupt = 0
            pose_event_supports_raw = 0
            pose_event_prior_plausible = 0
            pose_scores = {}
            rotation_recovery_triggered = False
            rotation_hypothesis_count = 0
            rotation_cue_deg = 0.0
            selected_event_hypothesis = ""
            stable_rotation_active = False
            stable_rotation_step_deg = 0.0
            stable_rotation_axis_cosine = 0.0
            stable_rotation_override = 0
            stable_contracted_rejected = 0
            rotation_compensation_active = False
            rotation_compensation_step_deg = 0.0
            rotation_compensation_axis_cosine = 0.0
            reversal_rotation_active = False
            reversal_baseline_deg = 0.0
            reversal_history_deg = 0.0
            reversal_axis_cosine = 1.0
            raw_stable_candidates = {}
            subwindow_event_counts = []
            event_count_trend_active = False
            event_count_trend = 0.0
            event_count_trend_consistency = 0.0
            event_count_median = 0.0
            bidirectional_rotation_active = False
            pose_select_reason = "init"
            prior_base_source = "init"
            kf_update_accepted = True
            kf_update_source = "init"
            kf_gate_reason = "init"
            prior_residual_trans_m = 0.0
            prior_residual_rot_deg = 0.0
            event_points_for_vis = []

            t_stage = time.perf_counter()
            frame_name = reader.id_strs[frame_i]
            frame_num = reader.frame_number(frame_i)
            timestamp = frame_num / args.fps
            timing["frame_meta_ms"] = elapsed_ms(t_stage)

            t_stage = time.perf_counter()
            color = reader.get_color(frame_i)
            timing["load_color_ms"] = elapsed_ms(t_stage)

            t_stage = time.perf_counter()
            depth = reader.get_depth(frame_i)
            timing["load_depth_ms"] = elapsed_ms(t_stage)
            logging.info(f"--- frame {frame_name} ({step_i + 1}/{len(frame_ids)}) ---")

            if step_i == 0:
                t_stage = time.perf_counter()
                init_mask = reader.get_initial_mask()
                if init_mask.sum() < 10:
                    raise RuntimeError(f"Initial Event6D mask too small on frame {frame_name}")
                timing["initial_mask_ms"] = elapsed_ms(t_stage)

                t_stage = time.perf_counter()
                if args.use_roi_input:
                    rgb_roi = make_rgb_roi_from_mask(init_mask, args, reader.H, reader.W)
                    color_input, depth_input, K_input, mask_input = make_roi_rgbd_inputs(
                        color,
                        depth,
                        reader.K,
                        rgb_roi,
                        args.roi_size,
                        mask=init_mask,
                    )
                else:
                    color_input, depth_input, K_input, mask_input = color, depth, reader.K, init_mask
                timing["roi_prepare_ms"] += elapsed_ms(t_stage)

                t_stage = time.perf_counter()
                pose_raw = est.register(
                    K=K_input,
                    rgb=color_input,
                    depth=depth_input,
                    ob_mask=mask_input,
                    iteration=args.est_refine_iter,
                )
                timing["initial_registration_ms"] = elapsed_ms(t_stage)
                pose_filtered = pose_raw.copy()
                pose_prior = pose_raw.copy()
                prev_event_z = float((reader.T_rgb_to_event @ pose_raw)[2, 3])
                event_count = 0
                du = dv = event_confidence = 0.0
                event_used = False
                delta_rgb = np.zeros(3, dtype=np.float64)
                z_used = prev_event_z
                rel_t = rel_r = 0.0
                pose_for_next = pose_raw.copy()
            else:
                # 只保留当前帧间包的事件窗口，避免两窗口模式混入上一帧窗口。
                _V38_EVENT_WINDOWS.clear()
                dt = max((frame_num - prev_frame_num) / args.fps, 1.0 / args.fps)

                t_stage = time.perf_counter()
                pose_event_prior = prev_pose.copy()
                rendered_event_mask = None
                event_window_roi = None
                if args.v17_use_fixed_event_window:
                    event_window_roi = fixed_event_window_from_pose(
                        args,
                        pose_event_prior,
                        inv_to_origin,
                        bbox,
                        reader.K_event,
                        reader.T_rgb_to_event,
                        reader.H_event,
                        reader.W_event,
                    )
                else:
                    rendered_event_mask = render_event_filter_mask(
                        args,
                        event_mask_renderer,
                        reader.T_rgb_to_event,
                        prev_pose,
                        pose_event_prior,
                    )
                timing["render_pred_event_mask_ms"] = elapsed_ms(t_stage)

                t_stage = time.perf_counter()
                x, y, t, p = reader.get_raw_events_for_frame(frame_i, event_frame_offset=args.event_frame_offset)
                raw_event_count = len(x)
                logging.info(
                    f"Loaded events {reader.last_event_path} for transition "
                    f"{prev_frame_num:06d}->{frame_num:06d}, offset={args.event_frame_offset}, "
                    f"reason={reader.last_event_read_reason}, count={raw_event_count}"
                )
                timing["load_events_ms"] = elapsed_ms(t_stage)

                t_stage = time.perf_counter()
                sub_windows = split_events(x, y, t, p, args.event_sub_windows)
                sub_window_count = len(sub_windows)
                timing["split_events_ms"] = elapsed_ms(t_stage)

                (
                    rotation_compensation_active,
                    rotation_compensation_rotvec,
                    rotation_compensation_step_deg,
                    rotation_compensation_axis_cosine,
                ) = stable_rotation_motion_cue(
                    args,
                    pose_previous=prev_pose,
                    pose_older=prev_prev_pose,
                    pose_older2=prev_prev_prev_pose,
                )
                logging.info(
                    f"V21 pre-event rotation compensation frame={frame_name}: "
                    f"active={rotation_compensation_active}, "
                    f"step={rotation_compensation_step_deg:.2f}deg, "
                    f"axis_cos={rotation_compensation_axis_cosine:.3f}"
                )

                du_total = 0.0
                dv_total = 0.0
                event_confidence_last = 0.0
                event_count = 0
                event_used = False
                event_points_for_vis = []
                event_line_scores = []
                event_motion_vectors = []
                event_motion_t0 = time.perf_counter()

                for window_i, (xw, yw, tw, pw) in enumerate(sub_windows):
                    mask_for_window = rendered_event_mask
                    if args.event_mask_update_each_subwindow and args.v17_use_fixed_event_window:
                        event_window_roi = fixed_event_window_from_pose(
                            args,
                            pose_event_prior,
                            inv_to_origin,
                            bbox,
                            reader.K_event,
                            reader.T_rgb_to_event,
                            reader.H_event,
                            reader.W_event,
                        )
                    elif args.event_mask_update_each_subwindow:
                        t_stage = time.perf_counter()
                        mask_for_window = render_event_filter_mask(
                            args,
                            event_mask_renderer,
                            reader.T_rgb_to_event,
                            prev_pose,
                            pose_event_prior,
                        )
                        timing["render_pred_event_mask_ms"] += elapsed_ms(t_stage)

                    t_stage = time.perf_counter()
                    if args.v17_use_fixed_event_window:
                        (
                            xf, yf, tf, pf,
                            xf_vis, yf_vis,
                            motion_scale_x, motion_scale_y,
                        ) = crop_events_to_processing_window(
                            xw,
                            yw,
                            tw,
                            pw,
                            event_window_roi,
                            args.v17_event_processing_size,
                        )
                        if args.v18_use_expanded_cad_support and len(xf) > 0:
                            broad_events = (xf, yf, tf, pf, xf_vis, yf_vis)
                            cad_support = expanded_cad_support_in_event_window(
                                mesh,
                                reader.T_rgb_to_event @ pose_event_prior,
                                reader.K_event,
                                event_window_roi,
                                args.v17_event_processing_size,
                                args.v18_cad_support_dilation_px,
                            )
                            supported_events = filter_local_events_by_support(
                                xf, yf, tf, pf, xf_vis, yf_vis, cad_support
                            )
                            if len(supported_events[0]) >= int(args.v18_cad_support_min_events):
                                xf, yf, tf, pf, xf_vis, yf_vis = supported_events
                            else:
                                xf, yf, tf, pf, xf_vis, yf_vis = broad_events
                        motion_width = motion_height = max(int(args.v17_event_processing_size), 8)
                        motion_center_xy = (0.5 * motion_width, 0.5 * motion_height)
                    else:
                        motion_mask = make_event_motion_band(
                            mask_for_window,
                            args.v14_event_motion_band_dilation,
                        )
                        xf, yf, tf, pf = filter_events_by_mask(
                            xw, yw, tw, pw, motion_mask, 0
                        )
                        xf_vis, yf_vis = xf, yf
                        motion_scale_x = motion_scale_y = 1.0
                        motion_width, motion_height = reader.W_event, reader.H_event
                        motion_center_xy = binary_mask_centroid(mask_for_window)
                    timing["event_filter_ms"] += elapsed_ms(t_stage)
                    event_count += len(xf)
                    subwindow_event_counts.append(int(len(xf)))
                    if args.save_debug_images and len(xf) > 0:
                        event_points_for_vis.append((xf_vis.copy(), yf_vis.copy(), pf.copy()))

                    t_stage = time.perf_counter()
                    rotation_affine = None
                    predicted_sub_rotation_deg = 0.0
                    rotation_projection_count = 0
                    if rotation_compensation_active and args.v17_use_fixed_event_window:
                        rotation_pose = pose_event_prior.copy()
                        completed_fraction = float(window_i) / max(float(sub_window_count), 1.0)
                        rotation_pose[:3, :3] = (
                            Rotation.from_rotvec(
                                rotation_compensation_rotvec * completed_fraction
                            ).as_matrix()
                            @ prev_pose[:3, :3]
                        )
                        (
                            rotation_affine,
                            predicted_sub_rotation_deg,
                            rotation_projection_count,
                        ) = cad_rotation_compensation_affine(
                            args,
                            mesh,
                            rotation_pose,
                            reader.K_event,
                            reader.T_rgb_to_event,
                            event_window_roi,
                            args.v17_event_processing_size,
                            rotation_compensation_rotvec,
                            sub_window_count,
                        )
                    motion_stats, motion_debug = estimate_event_2d_motion(
                        args,
                        xf,
                        yf,
                        tf,
                        pf,
                        motion_width,
                        motion_height,
                        motion_center_xy=motion_center_xy,
                        rotation_compensation_affine=rotation_affine,
                        predicted_rotation_deg=predicted_sub_rotation_deg,
                    )
                    motion_stats["rotation_projection_count"] = float(rotation_projection_count)
                    if args.v17_use_fixed_event_window:
                        restore_motion_to_event_pixels(
                            motion_stats,
                            motion_scale_x,
                            motion_scale_y,
                        )
                    timing["event_center_ms"] += elapsed_ms(t_stage)

                    substep_before_pose = pose_event_prior.copy()
                    contour_accepted = bool(motion_stats["accepted"])
                    contour_score = float(motion_stats["confidence"])
                    if (
                        not contour_accepted
                        and args.v18_low_response_rescue
                        and not bool(motion_stats.get("rotation_compensated", False))
                        and motion_stats.get("reason") == "motion_quality_gate"
                        and float(motion_stats["phase_response"]) >= float(args.v18_low_response_min)
                        and contour_score >= float(args.v14_event_min_confidence)
                        and np.linalg.norm(prev_reliable_event_motion_px) >= float(args.v14_event_min_motion_px)
                    ):
                        rescue_vector = np.asarray(
                            [float(motion_stats["dx"]), float(motion_stats["dy"])],
                            dtype=np.float64,
                        )
                        rescue_norm = float(np.linalg.norm(rescue_vector))
                        previous_norm = float(np.linalg.norm(prev_reliable_event_motion_px))
                        direction_cosine = float(
                            np.dot(rescue_vector, prev_reliable_event_motion_px)
                            / max(rescue_norm * previous_norm, 1e-9)
                        )
                        max_motion_full = float(args.v14_event_max_motion_px) / max(
                            min(float(motion_scale_x), float(motion_scale_y)), 1e-9
                        )
                        if (
                            rescue_norm >= float(args.v14_event_min_motion_px)
                            and rescue_norm <= max_motion_full
                            and direction_cosine >= float(args.v18_direction_rescue_cosine)
                        ):
                            contour_accepted = True
                            motion_stats["accepted"] = True
                            motion_stats["reason"] = "temporal_direction_rescue"
                    translation_gain = 1.0
                    if contour_accepted:
                        if (
                            not bool(motion_stats.get("rotation_compensated", False))
                            and abs(float(motion_stats["rotation_deg"])) <= float(
                                args.v16_translation_gain_max_affine_rotation_deg
                            )
                        ):
                            translation_gain = max(float(args.v16_event_translation_gain), 0.0)
                        pose_event_prior, _ = pose_from_event_2d_motion(
                            pose_event_prior,
                            float(motion_stats["dx"]) * translation_gain,
                            float(motion_stats["dy"]) * translation_gain,
                            reader.K_event,
                            reader.T_rgb_to_event,
                            reader.T_event_to_rgb,
                            args.v14_event_max_translation_step_m,
                        )
                        event_used = True
                        event_kf_update_count += 1
                        event_line_scores.append(contour_score)
                        accepted_vector = np.asarray([
                            float(motion_stats["dx"]) * translation_gain,
                            float(motion_stats["dy"]) * translation_gain,
                        ], dtype=np.float64)
                        event_motion_vectors.append(accepted_vector)
                        du_total += float(accepted_vector[0])
                        dv_total += float(accepted_vector[1])
                        event_confidence_last = max(event_confidence_last, contour_score)
                        if not args.v17_use_fixed_event_window:
                            rendered_event_mask = event_mask_renderer._project_hull_mask(
                                reader.T_rgb_to_event @ pose_event_prior
                            )
                    else:
                        event_kf_predict_only_count += 1

                    translation_step_m = translation_error_m(substep_before_pose, pose_event_prior)
                    rotation_step_deg = rotation_error_deg(substep_before_pose, pose_event_prior)
                    motion_stats["translation_step_m"] = translation_step_m
                    motion_stats["rotation_step_deg"] = rotation_step_deg
                    motion_stats["translation_gain"] = translation_gain
                    timing["event_contour_align_ms"] += elapsed_ms(t_stage)
                    logging.info(
                        f"V38 event motion {frame_name} sub{window_i + 1}/{sub_window_count}: "
                        f"accepted={contour_accepted}, reason={motion_stats['reason']}, "
                        f"events={int(motion_stats['event_count'])}, matches={int(motion_stats['match_count'])}, "
                        f"inliers={int(motion_stats['inlier_count'])}/{int(motion_stats['match_count'])}, "
                        f"flow=({float(motion_stats['dx']):.1f},{float(motion_stats['dy']):.1f}) px, "
                        f"raw_phase=({float(motion_stats['raw_phase_dx']):.1f},"
                        f"{float(motion_stats['raw_phase_dy']):.1f}), "
                        f"rot_comp={int(bool(motion_stats['rotation_compensated']))}/"
                        f"rej={int(bool(motion_stats['rotation_compensation_rejected']))}/"
                        f"{float(motion_stats['predicted_rotation_deg']):.1f}deg, "
                        f"affine_rot={float(motion_stats['rotation_deg']):.1f}deg, "
                        f"blend={float(motion_stats['rotation_flow_weight']):.2f}, gain={translation_gain:.2f}, "
                        f"phase={float(motion_stats['phase_response']):.3f}, "
                        f"rms={float(motion_stats['rms_px']):.2f}, conf={contour_score:.3f}"
                    )

                    contour_image_name = ""
                    if args.debug >= 1 and args.save_debug_images and args.save_event_contour_vis:
                        t_corr = time.perf_counter()
                        contour_image = make_event_motion_visualization(
                            args,
                            motion_debug,
                            motion_height,
                            motion_width,
                            frame_label=f"{frame_name} sub{window_i + 1}/{sub_window_count}",
                            stats=motion_stats,
                        )
                        contour_image_name = f"{frame_name}_sub{window_i + 1:02d}.png"
                        imageio.imwrite(
                            os.path.join(args.debug_dir, "event_contour_vis", contour_image_name),
                            contour_image,
                        )
                        timing["event_contour_vis_ms"] += elapsed_ms(t_corr)

                    corr_writer.writerow({
                        "frame_name": frame_name,
                        "sub_window": window_i + 1,
                        "event_count": int(motion_stats["event_count"]),
                        "recent_event_count": int(motion_stats["early_count"] + motion_stats["late_count"]),
                        "early_count": int(motion_stats["early_count"]),
                        "late_count": int(motion_stats["late_count"]),
                        "match_count": int(motion_stats["match_count"]),
                        "inlier_count": int(motion_stats["inlier_count"]),
                        "inlier_ratio": motion_stats["inlier_ratio"],
                        "spatial_sectors": int(motion_stats["spatial_sectors"]),
                        "rms_px": motion_stats["rms_px"],
                        "phase_response": motion_stats["phase_response"],
                        "rotation_compensated": int(bool(motion_stats["rotation_compensated"])),
                        "rotation_compensation_rejected": int(bool(motion_stats["rotation_compensation_rejected"])),
                        "predicted_rotation_deg": motion_stats["predicted_rotation_deg"],
                        "rotation_projection_count": int(motion_stats["rotation_projection_count"]),
                        "raw_phase_dx": motion_stats["raw_phase_dx"],
                        "raw_phase_dy": motion_stats["raw_phase_dy"],
                        "raw_phase_response": motion_stats["raw_phase_response"],
                        "affine_rotation_deg": motion_stats["rotation_deg"],
                        "affine_scale": motion_stats["scale"],
                        "phase_dx": motion_stats["phase_dx"],
                        "phase_dy": motion_stats["phase_dy"],
                        "affine_center_dx": motion_stats["affine_center_dx"],
                        "affine_center_dy": motion_stats["affine_center_dy"],
                        "rotation_flow_weight": motion_stats["rotation_flow_weight"],
                        "translation_gain": motion_stats["translation_gain"],
                        "motion_confidence": motion_stats["confidence"],
                        "contour_point_count": int(motion_stats["match_count"]),
                        "contour_edge_count": int(motion_stats["inlier_count"]),
                        "initial_cost": motion_stats["rms_px"],
                        "coarse_cost": motion_stats["rms_px"],
                        "final_cost": motion_stats["rms_px"],
                        "initial_coverage": motion_stats["inlier_ratio"],
                        "final_coverage": motion_stats["inlier_ratio"],
                        "event_coverage": motion_stats["confidence"],
                        "coarse_du": motion_stats["dx"],
                        "coarse_dv": motion_stats["dy"],
                        "translation_step_m": motion_stats["translation_step_m"],
                        "rotation_step_deg": motion_stats["rotation_step_deg"],
                        "contour_score": contour_score,
                        "accepted_event_update": int(contour_accepted),
                        "reason": motion_stats["reason"],
                        "image": contour_image_name,
                    })
                    corr_f.flush()

                    if args.debug >= 1 and args.save_debug_images and args.save_event_substep_vis:
                        t_vis_sub = time.perf_counter()
                        substep_used = bool(contour_accepted)
                        sub_rel_t = translation_error_m(substep_before_pose, pose_event_prior)
                        sub_rel_r = rotation_error_deg(substep_before_pose, pose_event_prior)
                        substep_vis = make_event_track_visualization(
                            args=args,
                            reader=reader,
                            event_points_for_vis=[(xf_vis.copy(), yf_vis.copy(), pf.copy())] if len(xf) > 0 else [],
                            model_lines=lopet_model_lines,
                            bbox=bbox,
                            inv_to_origin=inv_to_origin,
                            prev_pose_rgb=prev_pose,
                            prior_pose_rgb=substep_before_pose,
                            final_pose_rgb=pose_event_prior,
                            frame_name=f"{frame_name} sub{window_i + 1}/{sub_window_count}",
                            event_used=substep_used,
                            event_update_count=event_kf_update_count,
                            sub_window_count=sub_window_count,
                            line_score=float(contour_score),
                            pose_select_source="event_substep",
                            rel_t=sub_rel_t,
                            rel_r=sub_rel_r,
                        )
                        imageio.imwrite(
                            os.path.join(args.debug_dir, "event_track_vis", f"{frame_name}_sub{window_i + 1:02d}.png"),
                            substep_vis,
                        )
                        timing["event_track_vis_ms"] += elapsed_ms(t_vis_sub)

                timing["event_motion_total_ms"] = elapsed_ms(event_motion_t0)
                du, dv, event_confidence = du_total, dv_total, event_confidence_last
                event_kf_mean_confidence = float(np.mean(event_line_scores)) if event_line_scores else 0.0
                if event_motion_vectors:
                    vectors = np.asarray(event_motion_vectors, dtype=np.float64)
                    norms = np.linalg.norm(vectors, axis=1)
                    valid_vectors = norms > 1e-6
                    if np.any(valid_vectors):
                        unit_vectors = vectors[valid_vectors] / norms[valid_vectors, None]
                        dominant_ratio = float(np.linalg.norm(np.sum(unit_vectors, axis=0)) / len(unit_vectors))
                        local_vector_count = int(len(unit_vectors))
                        dominant_vote_count = int(round(dominant_ratio * local_vector_count))
                event_large_motion_gate = bool(
                    event_used
                    and np.hypot(du, dv) >= float(args.v18_large_motion_threshold_px)
                    and dominant_ratio >= float(args.v18_large_motion_min_consistency)
                    and event_kf_update_count >= 2
                )
                if event_used and dominant_ratio >= float(args.v18_large_motion_min_consistency):
                    prev_reliable_event_motion_px = np.asarray([du, dv], dtype=np.float64)
                (
                    event_count_trend_active,
                    event_count_trend,
                    event_count_trend_consistency,
                    event_count_median,
                ) = event_count_trend_cue(args, subwindow_event_counts)
                logging.info(
                    f"V21 event-count trend frame={frame_name}: "
                    f"counts={subwindow_event_counts}, active={event_count_trend_active}, "
                    f"slope={event_count_trend:.3f}, consistency={event_count_trend_consistency:.2f}"
                )

                t_stage = time.perf_counter()
                pose_prior = pose_event_prior.copy()
                delta_rgb = pose_prior[:3, 3] - prev_pose[:3, 3]
                prior_base_source = "event_2d" if event_used else "previous_pose"
                z_used = float((reader.T_rgb_to_event @ pose_prior)[2, 3])
                timing["make_prior_ms"] = elapsed_ms(t_stage)

                t_stage = time.perf_counter()
                # 基线分支：始终围绕上一已接受姿态裁剪。绝不允许事件提议定义
                # 唯一的 RGB-D ROI。
                if args.use_roi_input:
                    rgb_roi = make_rgb_roi_from_pose(
                        prev_pose,
                        inv_to_origin,
                        bbox,
                        reader.K,
                        args,
                        reader.H,
                        reader.W,
                    )
                    color_input, depth_input, K_input, _ = make_roi_rgbd_inputs(
                        color,
                        depth,
                        reader.K,
                        rgb_roi,
                        args.roi_size,
                        mask=None,
                    )
                else:
                    color_input, depth_input, K_input = color, depth, reader.K
                timing["roi_prepare_ms"] += elapsed_ms(t_stage)

                t_stage = time.perf_counter()
                est.set_pose_last_from_ob_in_cam(prev_pose)
                pose_baseline = est.track_one(
                    rgb=color_input,
                    depth=depth_input,
                    K=K_input,
                    iteration=args.track_refine_iter,
                )
                timing["foundationpose_baseline_ms"] = elapsed_ms(t_stage)
                timing["foundationpose_track_ms"] = timing["foundationpose_baseline_ms"]
                (
                    reversal_rotation_active,
                    reversal_baseline_deg,
                    reversal_history_deg,
                    reversal_axis_cosine,
                ) = rotation_reversal_cue(
                    args,
                    pose_previous=prev_pose,
                    pose_older=prev_prev_pose,
                    pose_baseline=pose_baseline,
                )

                # 事件分支：先仅使用事件 XY 和基线旋转。只有紧凑学习式评分支持
                # 真实角度更新，或基线跳变超过硬恢复阈值时，才评估更大的旋转候选库。
                pose_event_refined = None
                event_hypothesis_labels = []
                compact_scores = {}
                event_pose_support = bool(event_used or rotation_compensation_active)
                selection_color_input, selection_depth_input, selection_K_input = (
                    color_input,
                    depth_input,
                    K_input,
                )
                if (
                    event_pose_support
                    and (
                        translation_error_m(prev_pose, pose_prior)
                        >= float(args.v14_event_min_pose_shift_m)
                        or rotation_compensation_active
                    )
                ):
                    t_event_fp = time.perf_counter()
                    baseline_rotation_step = rotation_error_deg(prev_pose, pose_baseline)
                    rotation_cue_deg = baseline_rotation_step
                    translation_scales = (
                        parse_float_list(args.v18_translation_scales)
                        if event_large_motion_gate
                        else (
                            parse_float_list(args.v20_rotation_translation_scales)
                            if rotation_compensation_active
                            else [1.0]
                        )
                    )
                    if frame_num in args._v38_translation_scale_frames:
                        translation_scales = parse_float_list(args.v38_frame_translation_scales)
                        if not translation_scales:
                            raise ValueError(
                                "--v38_frame_translation_scales must contain at least one scale"
                            )
                        logging.info(
                            f"V38 frame-local translation scales frame={frame_name}: "
                            f"{translation_scales}"
                        )
                    compact_hypotheses = []
                    compact_labels = []
                    event_delta_translation = pose_prior[:3, 3] - prev_pose[:3, 3]
                    for scale in translation_scales:
                        scaled_prior = pose_prior.copy()
                        scaled_prior[:3, 3] = (
                            prev_pose[:3, 3] + float(scale) * event_delta_translation
                        )
                        label = "event_xy" if abs(float(scale) - 1.0) < 1e-6 else f"event_t_{float(scale):g}x"
                        if not any(
                            translation_error_m(existing, scaled_prior) < 1e-4
                            and rotation_error_deg(existing, scaled_prior) < float(args.v16_rotation_dedup_deg)
                            for existing in compact_hypotheses
                        ):
                            compact_hypotheses.append(scaled_prior)
                            compact_labels.append(label)
                    (
                        stable_poses,
                        stable_labels,
                        stable_rotation_active,
                        stable_rotation_step_deg,
                        stable_rotation_axis_cosine,
                    ) = build_v19_stable_axis_hypotheses(
                        args,
                        pose_event_prior=pose_prior,
                        pose_previous=prev_pose,
                        pose_older=prev_prev_pose,
                        pose_older2=prev_prev_prev_pose,
                    )
                    bidirectional_rotation_active = bool(
                        args.v21_enable_bidirectional_rotation
                        and event_count_trend_active
                        and (
                            rotation_compensation_active
                            or stable_rotation_active
                            or baseline_rotation_step >= float(args.v21_rotation_trigger_deg)
                        )
                    )
                    for stable_pose, stable_label in zip(stable_poses, stable_labels):
                        compact_hypotheses.append(stable_pose)
                        compact_labels.append(stable_label)
                        if args.v19_keep_raw_stable_priors:
                            raw_stable_candidates[f"event::raw_{stable_label}"] = stable_pose.copy()
                    baseline_rotation_prior = pose_prior.copy()
                    baseline_rotation_prior[:3, :3] = pose_baseline[:3, :3]
                    if rotation_error_deg(pose_prior, baseline_rotation_prior) >= float(args.v16_rotation_dedup_deg):
                        compact_hypotheses.append(baseline_rotation_prior)
                        compact_labels.append("baseline_R_event_t")
                    if args.use_roi_input:
                        event_rgb_roi = make_rgb_roi_from_pose(
                            pose_prior,
                            inv_to_origin,
                            bbox,
                            reader.K,
                            args,
                            reader.H,
                            reader.W,
                        )
                        if stable_rotation_active or baseline_rotation_step >= float(args.v16_rotation_trigger_deg):
                            event_rgb_roi = expand_roi(event_rgb_roi, args.v16_rotation_roi_scale)
                        event_color_input, event_depth_input, event_K_input, _ = make_roi_rgbd_inputs(
                            color,
                            depth,
                            reader.K,
                            event_rgb_roi,
                            args.roi_size,
                            mask=None,
                        )
                    else:
                        event_color_input, event_depth_input, event_K_input = color, depth, reader.K
                    selection_color_input, selection_depth_input, selection_K_input = (
                        event_color_input,
                        event_depth_input,
                        event_K_input,
                    )
                    compact_refined, _ = est.track_hypotheses(
                        rgb=event_color_input,
                        depth=event_depth_input,
                        K=event_K_input,
                        ob_in_cam_hypotheses=compact_hypotheses,
                        iteration=args.track_refine_iter,
                    )
                    if frame_num in args._v38_depth_drift_gate_frames:
                        max_z_drift_m = max(
                            float(args.v38_max_refined_prior_z_drift_m),
                            0.0,
                        )
                        depth_guarded_refined = []
                        for label, refined_pose in zip(
                            compact_labels,
                            np.asarray(compact_refined).reshape(-1, 4, 4),
                        ):
                            guarded_pose = np.asarray(refined_pose).copy()
                            z_drift_m = abs(
                                float(guarded_pose[2, 3]) - float(pose_prior[2, 3])
                            )
                            if z_drift_m > max_z_drift_m:
                                refined_z_m = float(guarded_pose[2, 3])
                                guarded_pose[2, 3] = float(pose_prior[2, 3])
                                logging.warning(
                                    f"V38 reset refined depth drift frame={frame_name}: "
                                    f"label={label}, refined_z={refined_z_m:.5f}m, "
                                    f"prior_z={float(pose_prior[2, 3]):.5f}m, "
                                    f"drift={z_drift_m:.5f}m>{max_z_drift_m:.5f}m; "
                                    "kept refined X/Y and rotation"
                                )
                            depth_guarded_refined.append(guarded_pose)
                        compact_refined = np.asarray(
                            depth_guarded_refined,
                            dtype=np.float64,
                        ).reshape(-1, 4, 4)
                    pose_event_refined = compact_refined
                    event_hypothesis_labels = list(compact_labels)

                    if args.v20_keep_stable_hybrid_candidates and stable_poses:
                        stable_pose_by_label = dict(zip(stable_labels, stable_poses))
                        for label, refined_pose in zip(compact_labels, compact_refined):
                            raw_stable = stable_pose_by_label.get(label)
                            if raw_stable is None:
                                continue
                            hybrid = refined_pose.copy()
                            hybrid[:3, :3] = raw_stable[:3, :3]
                            raw_stable_candidates[f"event::hybrid_{label}"] = hybrid
                        # 旋转补偿下，残余事件平移仍可能包含弯曲旋转流。保留两个
                        # 运动一致的平移，使评分器可拒绝完整残差而不丢弃稳定旋转本身。
                        if rotation_compensation_active:
                            for stable_pose, stable_label in zip(stable_poses, stable_labels):
                                stable_scale_label = stable_label.rsplit("_", 1)[-1]
                                for translation_label, translation_pose in (
                                    ("prev_t", prev_pose),
                                    ("baseline_t", pose_baseline),
                                ):
                                    alternative = stable_pose.copy()
                                    alternative[:3, 3] = translation_pose[:3, 3]
                                    raw_stable_candidates[
                                        f"event::raw_stable_axis_{translation_label}_{stable_scale_label}"
                                    ] = alternative

                    compact_candidates = {
                        "previous": prev_pose,
                        "baseline": pose_baseline,
                        "event_prior": pose_prior,
                    }
                    for label, pose in zip(compact_labels, compact_refined):
                        compact_candidates[f"event::{label}"] = pose
                    compact_candidates.update(raw_stable_candidates)
                    if args.v17_use_foundationpose_batch_scorer:
                        compact_scores = score_pose_candidates_with_foundationpose(
                            est,
                            event_color_input,
                            event_depth_input,
                            event_K_input,
                            compact_candidates,
                        )
                    compact_best_key = max(compact_scores, key=compact_scores.get) if compact_scores else "baseline"
                    compact_best_rotation = rotation_error_deg(prev_pose, compact_candidates[compact_best_key])
                    compact_best_score = float(compact_scores.get(compact_best_key, 0.0))
                    compact_supports_rotation = any(
                        rotation_error_deg(prev_pose, candidate) >= float(args.v16_rotation_trigger_deg)
                        and float(compact_scores.get(name, -np.inf))
                        >= compact_best_score - float(args.v18_compact_rotation_score_tolerance)
                        for name, candidate in compact_candidates.items()
                    ) if compact_scores else (
                        compact_best_rotation >= float(args.v16_rotation_trigger_deg)
                    )
                    rotation_recovery_triggered = bool(
                        stable_rotation_active
                        or reversal_rotation_active
                        or bidirectional_rotation_active
                        or (
                            args.v17_use_adaptive_rotation_bank
                            and baseline_rotation_step >= float(args.v16_rotation_trigger_deg)
                            and (
                                compact_supports_rotation
                                or baseline_rotation_step >= float(args.v17_rotation_hard_trigger_deg)
                            )
                        )
                    )

                    if rotation_recovery_triggered:
                        full_hypotheses, full_labels, _, rotation_cue_deg = build_v16_event_rotation_hypotheses(
                            args,
                            pose_event_prior=pose_prior,
                            pose_previous=prev_pose,
                            pose_older=prev_prev_pose,
                            pose_baseline=pose_baseline,
                            force_expanded=True,
                            reversal_active=reversal_rotation_active,
                            bidirectional_active=bidirectional_rotation_active,
                        )
                        extra = [
                            (pose, label)
                            for pose, label in zip(full_hypotheses, full_labels)
                            if label not in compact_labels
                        ]
                        extra = extra[:max(
                            int(args.v16_rotation_max_hypotheses) - len(compact_hypotheses),
                            0,
                        )]
                        if extra:
                            extra_refined, _ = est.track_hypotheses(
                                rgb=event_color_input,
                                depth=event_depth_input,
                                K=event_K_input,
                                ob_in_cam_hypotheses=[item[0] for item in extra],
                                iteration=args.track_refine_iter,
                            )
                            pose_event_refined = np.concatenate([compact_refined, extra_refined], axis=0)
                            event_hypothesis_labels.extend(item[1] for item in extra)

                    rotation_hypothesis_count = len(event_hypothesis_labels)
                    timing["foundationpose_event_ms"] = elapsed_ms(t_event_fp)
                    timing["foundationpose_track_ms"] += timing["foundationpose_event_ms"]
                    logging.info(
                        f"V21 adaptive rotation frame={frame_name}: expanded={rotation_recovery_triggered}, "
                        f"baseline_step={baseline_rotation_step:.2f}deg, compact_best={compact_best_key}/"
                        f"{compact_best_rotation:.2f}deg, stable={stable_rotation_active}/"
                        f"{stable_rotation_step_deg:.1f}deg/cos={stable_rotation_axis_cosine:.2f}, "
                        f"reversal={reversal_rotation_active}/base={reversal_baseline_deg:.1f}deg/"
                        f"hist={reversal_history_deg:.1f}deg/cos={reversal_axis_cosine:.2f}, "
                        f"count_trend={int(bidirectional_rotation_active)}/{event_count_trend:.3f}, "
                        f"hypotheses={event_hypothesis_labels}"
                    )

                t_stage = time.perf_counter()
                pose_selection_enabled = not bool(args.disable_pose_candidate_selection)
                if not pose_selection_enabled:
                    pose_select_source = "baseline" if pose_baseline is not None else "previous"
                    pose_raw = pose_baseline.copy()
                    pose_raw_score = pose_baseline_score = 0.0
                    pose_event_score = pose_prior_score = pose_previous_score = 0.0
                    pose_raw_abrupt = 0
                    pose_event_supports_raw = int(event_used)
                    pose_event_prior_plausible = 0
                    pose_select_reason = "candidate selection disabled; use baseline FoundationPose"
                else:
                    candidate_map = {
                        "previous": prev_pose,
                        "baseline": pose_baseline,
                        "event_prior": pose_prior,
                    }
                    if pose_event_refined is not None:
                        for label, pose in zip(event_hypothesis_labels, pose_event_refined):
                            candidate_map[f"event::{label}"] = pose
                    candidate_map.update(raw_stable_candidates)
                    learned_pose_scores = {}
                    if args.v17_use_foundationpose_batch_scorer:
                        if compact_scores and not rotation_recovery_triggered:
                            learned_pose_scores = compact_scores
                        else:
                            learned_pose_scores = score_pose_candidates_with_foundationpose(
                                est,
                                selection_color_input,
                                selection_depth_input,
                                selection_K_input,
                                candidate_map,
                            )
                    selected_pose, pose_select_source, pose_scores, pose_select_reason = select_v21_pose_candidates(
                        args,
                        rgb_pose_renderer,
                        depth,
                        pose_previous=prev_pose,
                        pose_baseline=pose_baseline,
                        pose_event_prior=pose_prior,
                        pose_event_refined=pose_event_refined,
                        event_hypothesis_labels=event_hypothesis_labels,
                        extra_event_candidates=raw_stable_candidates,
                        event_used=event_pose_support,
                        event_confidence=event_confidence,
                        rotation_recovery_triggered=rotation_recovery_triggered,
                        large_event_motion_gate=event_large_motion_gate,
                        stable_rotation_active=stable_rotation_active,
                        stable_rotation_step_deg=stable_rotation_step_deg,
                        rotation_compensation_active=rotation_compensation_active,
                        learned_scores=learned_pose_scores,
                    )
                    pose_raw_score = float(pose_scores.get("raw", 0.0))
                    pose_baseline_score = float(pose_scores.get("baseline", 0.0))
                    pose_event_score = float(pose_scores.get("event", 0.0))
                    pose_prior_score = float(pose_scores.get("event_prior", 0.0))
                    pose_previous_score = float(pose_scores.get("previous", 0.0))
                    pose_raw_abrupt = int(
                        pose_scores.get("baseline_step_t", 0.0) > float(args.pose_select_abrupt_translation_m)
                        or pose_scores.get("baseline_step_r", 0.0) > float(args.pose_select_abrupt_rotation_deg)
                    )
                    pose_event_supports_raw = int(
                        pose_scores.get("event_plausible", 0.0) > 0.5
                        or pose_scores.get("event_prior_plausible", 0.0) > 0.5
                    )
                    pose_event_prior_plausible = int(pose_scores.get("event_prior_plausible", 0.0) > 0.5)
                    stable_rotation_override = int(pose_scores.get("stable_override", 0.0) > 0.5)
                    stable_contracted_rejected = int(pose_scores.get("stable_contracted_rejected", 0.0))
                    pose_raw = selected_pose.copy()

                # Mustard/0001 contains no large inter-frame rotation.  Event
                # translation candidates are refined in full SE(3), so they can
                # still acquire a spurious rotation even when all explicit
                # rotation-recovery branches are disabled.  Apply the domain
                # constraint at the final state boundary while retaining the
                # selected translation and depth.
                selected_rotation_step_deg = rotation_error_deg(prev_pose, pose_raw)
                max_frame_rotation_deg = max(float(args.v38_max_frame_rotation_deg), 0.0)
                rotation_limit_scope = "global"
                if frame_num in args._v38_rotation_limit_frames:
                    max_frame_rotation_deg = max(float(args.v38_rotation_limit_deg), 0.0)
                    rotation_limit_scope = "local"
                if (
                    bool(args.v38_lock_large_rotation)
                    and selected_rotation_step_deg > max_frame_rotation_deg
                ):
                    rejected_rotation_source = pose_select_source
                    pose_raw[:3, :3] = prev_pose[:3, :3]
                    pose_select_reason = (
                        f"{pose_select_reason}; v38_rotation_locked="
                        f"{selected_rotation_step_deg:.3f}deg>"
                        f"{max_frame_rotation_deg:.3f}deg; "
                        f"scope={rotation_limit_scope}; source={rejected_rotation_source}"
                    )
                    logging.warning(
                        f"V38 lock large rotation frame={frame_name}: "
                        f"source={rejected_rotation_source}, "
                        f"step={selected_rotation_step_deg:.3f}deg, "
                        f"limit={max_frame_rotation_deg:.3f}deg, "
                        f"scope={rotation_limit_scope}; "
                        "kept translation/depth and reused previous rotation"
                    )
                if pose_select_source.startswith("event_rot:"):
                    selected_event_hypothesis = pose_select_source.split(":", 1)[1]
                # track_hypotheses 不会修改 FoundationPose 的循环姿态状态；
                # 始终将其与选定姿态同步。
                est.set_pose_last_from_ob_in_cam(pose_raw)
                timing["pose_select_ms"] = elapsed_ms(t_stage)

                t_stage = time.perf_counter()
                prior_residual_trans_m = translation_error_m(pose_prior, pose_raw)
                prior_residual_rot_deg = rotation_error_deg(pose_prior, pose_raw)
                kf_gate_reason = pose_select_reason
                kf_update_accepted = (
                    pose_select_source in ("baseline", "event_prior")
                    or pose_select_source.startswith("event_rot:")
                )
                kf_update_source = pose_select_source
                if pose_select_source in ("previous", "event_prior"):
                    logging.warning(
                        f"Use protected {pose_select_source} pose on frame {frame_name}: {pose_select_reason}"
                    )
                pose_filtered = pose_raw.copy()
                pose_for_next = pose_raw.copy()
                est.set_pose_last_from_ob_in_cam(pose_for_next)
                timing["state_update_ms"] = elapsed_ms(t_stage)

                prev_event_z = float((reader.T_rgb_to_event @ pose_for_next)[2, 3])

                t_stage = time.perf_counter()
                rel = relative_pose(prev_pose, pose_for_next)
                rel_t = float(np.linalg.norm(rel[:3, 3]))
                rel_r = rotation_error_deg(prev_pose, pose_for_next)
                timing["relative_pose_ms"] = elapsed_ms(t_stage)

                if args.debug >= 1 and args.save_debug_images:
                    t_stage = time.perf_counter()
                    event_vis = np.zeros((reader.H, reader.W, 3), dtype=np.uint8)
                    n_rgb_events = 0
                    n_event_events = 0
                    if event_points_for_vis:
                        xvis = np.concatenate([item[0] for item in event_points_for_vis])
                        yvis = np.concatenate([item[1] for item in event_points_for_vis])
                        pvis = np.concatenate([item[2] for item in event_points_for_vis])
                        n_event_events = len(xvis)
                        zvis = np.full(len(xvis), prev_event_z, dtype=np.float32)
                        u_rgb, v_rgb, p_rgb = event_pixels_to_rgb_pixels(
                            xvis, yvis, zvis, pvis,
                            K_event=reader.K_event,
                            K_rgb=reader.K,
                            T_event_to_rgb=reader.T_event_to_rgb,
                            H_rgb=reader.H,
                            W_rgb=reader.W,
                        )
                        n_rgb_events = len(u_rgb)
                        rgb_vis_event_count = n_rgb_events
                        if n_rgb_events > 0:
                            pos = p_rgb > 0
                            event_vis[v_rgb[pos], u_rgb[pos]] = [255, 80, 60]
                            event_vis[v_rgb[~pos], u_rgb[~pos]] = [80, 160, 255]
                    if args.v17_use_fixed_event_window and event_window_roi is not None:
                        x0, y0, x1, y1 = event_window_roi
                        corners_x = np.asarray([x0, x1, x1, x0], dtype=np.float32)
                        corners_y = np.asarray([y0, y0, y1, y1], dtype=np.float32)
                        corners_z = np.full(4, prev_event_z, dtype=np.float32)
                        window_u, window_v, _ = event_pixels_to_rgb_pixels(
                            corners_x,
                            corners_y,
                            corners_z,
                            np.ones(4, dtype=np.float32),
                            K_event=reader.K_event,
                            K_rgb=reader.K,
                            T_event_to_rgb=reader.T_event_to_rgb,
                            H_rgb=reader.H,
                            W_rgb=reader.W,
                        )
                        if len(window_u) == 4:
                            window_polygon = np.stack([window_u, window_v], axis=1).astype(np.int32)
                            cv2.polylines(event_vis, [window_polygon], True, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(
                        event_vis,
                        f"RGB-coordinate event_vis {n_rgb_events}/{n_event_events} z={prev_event_z:.3f} du={du:.1f} dv={dv:.1f} event_updates={event_kf_update_count}/{sub_window_count} window={int(args.v17_event_window_size_px)}->{int(args.v17_event_processing_size)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                    imageio.imwrite(os.path.join(args.debug_dir, "event_vis", f"{frame_name}.png"), event_vis)
                    timing["event_vis_ms"] = elapsed_ms(t_stage)

            if step_i == 0:
                dt = 0.0

            if save_pose_outputs:
                t_stage = time.perf_counter()
                np.savetxt(os.path.join(args.debug_dir, "ob_in_cam", f"{frame_name}.txt"), pose_raw.reshape(4, 4))
                np.savetxt(os.path.join(args.debug_dir, "ob_in_cam_filtered", f"{frame_name}.txt"), pose_filtered.reshape(4, 4))
                np.savetxt(os.path.join(args.debug_dir, "ob_in_cam_prior", f"{frame_name}.txt"), pose_prior.reshape(4, 4))
                if prev_pose is not None:
                    np.savetxt(
                        os.path.join(args.debug_dir, "relative_pose", f"{frame_name}.txt"),
                        relative_pose(prev_pose, pose_for_next),
                    )
                timing["save_pose_txt_ms"] = elapsed_ms(t_stage)

            if args.debug >= 1:
                t_stage = time.perf_counter()
                event_track_vis = make_event_track_visualization(
                    args=args,
                    reader=reader,
                    event_points_for_vis=event_points_for_vis,
                    model_lines=lopet_model_lines,
                    bbox=bbox,
                    inv_to_origin=inv_to_origin,
                    prev_pose_rgb=prev_pose,
                    prior_pose_rgb=pose_prior,
                    final_pose_rgb=pose_filtered,
                    frame_name=frame_name,
                    event_used=event_used,
                    event_update_count=event_kf_update_count,
                    sub_window_count=sub_window_count,
                    line_score=event_kf_mean_confidence,
                    pose_select_source=pose_select_source,
                    rel_t=rel_t,
                    rel_r=rel_r,
                )
                if args.save_debug_images:
                    imageio.imwrite(os.path.join(args.debug_dir, "event_track_vis", f"{frame_name}.png"), event_track_vis)
                timing["event_track_vis_ms"] += elapsed_ms(t_stage)

                t_stage = time.perf_counter()
                center_pose = pose_filtered @ inv_to_origin
                vis = color.copy()
                vis = draw_posed_3d_box(
                    reader.K, img=vis, ob_in_cam=center_pose, bbox=bbox,
                    line_color=(0, 255, 0), linewidth=3,
                )
                vis = draw_xyz_axis(
                    vis, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                    thickness=3, transparency=0, is_input_rgb=True,
                )
                timing["track_vis_draw_ms"] = elapsed_ms(t_stage)

                if args.save_debug_images:
                    t_stage = time.perf_counter()
                    imageio.imwrite(os.path.join(args.debug_dir, "track_vis", f"{frame_name}.png"), vis)
                    timing["track_vis_save_ms"] = elapsed_ms(t_stage)
                if not args.no_show_window:
                    t_stage = time.perf_counter()
                    panel = stack_visual_panels(vis, event_track_vis) if args.event_track_in_window else vis
                    panel = resize_for_window(
                        panel,
                        args.window_max_width,
                        args.window_max_height,
                        args.window_scale,
                    )
                    panel_bgr = panel[..., ::-1].copy()
                    try:
                        cv2.imshow("V38 Event-contour FoundationPose tracking", panel_bgr)
                        key = cv2.waitKey(max(args.window_delay, 1)) & 0xFF
                        stop_requested = key in (27, ord("q"))
                    except cv2.error as exc:
                        logging.warning(f"OpenCV window unavailable, continuing without display: {exc}")
                        args.no_show_window = True
                    timing["show_window_ms"] = elapsed_ms(t_stage)

            t_stage = time.perf_counter()
            writer.writerow({
                "frame_index": frame_i,
                "frame_name": frame_name,
                "timestamp": timestamp,
                "dt": dt,
                "event_count": event_count,
                "du": du,
                "dv": dv,
                "event_confidence": event_confidence,
                "dominant_vote_count": dominant_vote_count,
                "local_vector_count": local_vector_count,
                "dominant_ratio": dominant_ratio,
                "event_large_motion_gate": int(event_large_motion_gate),
                "event_count_trend": event_count_trend,
                "event_count_trend_consistency": event_count_trend_consistency,
                "event_count_median": event_count_median,
                "bidirectional_rotation_active": int(bidirectional_rotation_active),
                "prior_base_source": prior_base_source,
                "raw_pose_selected": int(kf_update_accepted),
                "state_update_source": kf_update_source,
                "state_select_reason": kf_gate_reason,
                "prior_residual_trans_m": prior_residual_trans_m,
                "prior_residual_rot_deg": prior_residual_rot_deg,
                "event_used": int(event_used),
                "event_contour_update_count": event_kf_update_count,
                "event_contour_skip_count": event_kf_predict_only_count,
                "event_contour_mean_score": event_kf_mean_confidence,
                "delta_rgb_m": float(np.linalg.norm(delta_rgb)),
                "z_used": z_used,
                "pose_select_source": pose_select_source,
                "pose_raw_score": pose_raw_score,
                "pose_baseline_score": pose_baseline_score,
                "pose_event_score": pose_event_score,
                "pose_event_prior_score": pose_prior_score,
                "pose_previous_score": pose_previous_score,
                "pose_raw_abrupt": pose_raw_abrupt,
                "pose_event_supports_raw": pose_event_supports_raw,
                "pose_event_prior_plausible": pose_event_prior_plausible,
                "pose_select_reason": pose_select_reason,
                "rotation_recovery_triggered": int(rotation_recovery_triggered),
                "rotation_hypothesis_count": rotation_hypothesis_count,
                "rotation_cue_deg": rotation_cue_deg,
                "selected_event_hypothesis": selected_event_hypothesis,
                "stable_rotation_active": int(stable_rotation_active),
                "stable_rotation_step_deg": stable_rotation_step_deg,
                "stable_rotation_axis_cosine": stable_rotation_axis_cosine,
                "stable_rotation_override": stable_rotation_override,
                "stable_contracted_rejected": stable_contracted_rejected,
                "stable_score_drop": float(pose_scores.get("stable_score_drop", 0.0)),
                "reversal_rotation_active": int(reversal_rotation_active),
                "rotation_compensation_active": int(rotation_compensation_active),
                "rotation_compensation_step_deg": rotation_compensation_step_deg,
                "rotation_compensation_axis_cosine": rotation_compensation_axis_cosine,
                "relative_trans_m": rel_t,
                "relative_rot_deg": rel_r,
            })
            f.flush()
            timing["metrics_write_ms"] = elapsed_ms(t_stage)

            if prev_prev_pose is not None:
                prev_prev_prev_pose = prev_prev_pose.copy()
            if prev_pose is not None:
                prev_prev_pose = prev_pose.copy()
            prev_pose = pose_for_next.copy()
            prev_frame_num = frame_num
            timing["total_ms"] = elapsed_ms(frame_t0)
            timing_row = {
                "frame_index": frame_i,
                "frame_name": frame_name,
                "step_i": step_i,
                "is_init": int(step_i == 0),
                "raw_event_count": raw_event_count,
                "mask_event_count": event_count,
                "rgb_vis_event_count": rgb_vis_event_count,
                "sub_window_count": sub_window_count,
                "debug": args.debug,
                "save_debug_images": int(args.save_debug_images),
                "dominant_vote_count": dominant_vote_count,
                "local_vector_count": local_vector_count,
                "dominant_ratio": dominant_ratio,
                "event_large_motion_gate": int(event_large_motion_gate),
                "event_count_trend": event_count_trend,
                "event_count_trend_consistency": event_count_trend_consistency,
                "event_count_median": event_count_median,
                "bidirectional_rotation_active": int(bidirectional_rotation_active),
                "event_contour_update_count": event_kf_update_count,
                "event_contour_skip_count": event_kf_predict_only_count,
                "event_contour_mean_score": event_kf_mean_confidence,
                "prior_base_source": prior_base_source,
                "pose_select_source": pose_select_source,
                "pose_raw_score": pose_raw_score,
                "pose_baseline_score": pose_baseline_score,
                "pose_event_score": pose_event_score,
                "pose_event_prior_score": pose_prior_score,
                "pose_previous_score": pose_previous_score,
                "pose_raw_abrupt": pose_raw_abrupt,
                "pose_event_supports_raw": pose_event_supports_raw,
                "pose_event_prior_plausible": pose_event_prior_plausible,
                "rotation_recovery_triggered": int(rotation_recovery_triggered),
                "rotation_hypothesis_count": rotation_hypothesis_count,
                "rotation_cue_deg": rotation_cue_deg,
                "selected_event_hypothesis": selected_event_hypothesis,
                "stable_rotation_active": int(stable_rotation_active),
                "stable_rotation_step_deg": stable_rotation_step_deg,
                "stable_rotation_axis_cosine": stable_rotation_axis_cosine,
                "stable_rotation_override": stable_rotation_override,
                "stable_contracted_rejected": stable_contracted_rejected,
                "stable_score_drop": float(pose_scores.get("stable_score_drop", 0.0)),
                "reversal_rotation_active": int(reversal_rotation_active),
                "rotation_compensation_active": int(rotation_compensation_active),
                "rotation_compensation_step_deg": rotation_compensation_step_deg,
                "rotation_compensation_axis_cosine": rotation_compensation_axis_cosine,
                "raw_pose_selected": int(kf_update_accepted),
                "state_update_source": kf_update_source,
                "prior_residual_trans_m": prior_residual_trans_m,
                "prior_residual_rot_deg": prior_residual_rot_deg,
                "track_refine_iter": args.track_refine_iter,
            }
            timing_row.update(timing)
            timing_writer.writerow(timing_row)
            timing_f.flush()
            timing_rows.append(timing_row)

            elapsed = timing["total_ms"] / 1000.0
            fps_times.append(elapsed)
            logging.info(
                f"event du/dv=({du:.2f},{dv:.2f}), used={event_used}, "
                f"contour_updates={event_kf_update_count}/{sub_window_count}, "
                f"dom={dominant_ratio:.2f} ({dominant_vote_count}/{local_vector_count}), "
                f"large_event_gate={int(event_large_motion_gate)}, "
                f"count_trend={int(bidirectional_rotation_active)}/{event_count_trend:.3f}/"
                f"{event_count_trend_consistency:.2f}, "
                f"base={prior_base_source}, "
                f"rot_recovery={int(rotation_recovery_triggered)} cue={rotation_cue_deg:.1f}deg "
                f"hyp={rotation_hypothesis_count} selected_hyp={selected_event_hypothesis or '-'}, "
                f"stable={int(stable_rotation_active)}/{stable_rotation_step_deg:.1f}deg/"
                f"cos={stable_rotation_axis_cosine:.2f}/override={stable_rotation_override}, "
                f"contracted_reject={stable_contracted_rejected}, "
                f"score_drop={float(pose_scores.get('stable_score_drop', 0.0)):.3f}, "
                f"reversal={int(reversal_rotation_active)}, "
                f"rot_comp={int(rotation_compensation_active)}/"
                f"{rotation_compensation_step_deg:.1f}deg/"
                f"cos={rotation_compensation_axis_cosine:.2f}, "
                f"select={pose_select_source} baseline/event/prior/prev="
                f"{pose_baseline_score:.3f}/{pose_event_score:.3f}/"
                f"{pose_prior_score:.3f}/{pose_previous_score:.3f}, "
                f"prior_res={prior_residual_trans_m:.4f}m/{prior_residual_rot_deg:.1f}deg, "
                f"|delta_rgb|={np.linalg.norm(delta_rgb):.4f} m, "
                f"relative={rel_t:.4f} m/{rel_r:.2f} deg, "
                f"total={timing['total_ms']:.1f} ms, "
                f"track={timing['foundationpose_track_ms']:.1f} ms, "
                f"event={timing['event_motion_total_ms']:.1f} ms, "
                f"vis={timing['event_vis_ms'] + timing['event_track_vis_ms'] + timing['track_vis_draw_ms'] + timing['track_vis_save_ms']:.1f} ms"
            )
            if stop_requested:
                logging.info("Visualization stopped by user.")
                break

    summarize_timing_rows(timing_rows, timing_summary_path)

    if not args.no_show_window:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    avg_fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0
    logging.info("=" * 60)
    logging.info(f"Done. Avg FPS: {avg_fps:.2f}")
    logging.info(f"Metrics: {metrics_path}")
    logging.info(f"Timing: {timing_path}")
    logging.info(f"Timing summary: {timing_summary_path}")
    logging.info(f"Event-contour sub-window stats: {corr_stats_path}")
    logging.info(f"Visuals: {args.debug_dir}/track_vis, {args.debug_dir}/event_vis, "
                 f"{args.debug_dir}/event_track_vis, and {args.debug_dir}/event_contour_vis")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
