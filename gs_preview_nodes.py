import os
import sys
import json
import struct
import glob
import re
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
import folder_paths

def _load_preview_camera_tensors_from_json(camera_json):
    camera_json = Path(camera_json)
    if not camera_json.exists():
        return torch.empty((0, 4, 4)), torch.empty((0, 3, 3))
    with open(camera_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    poses = []
    intrs = []
    if isinstance(data, dict) and "cameras" in data:
        cameras = data.get("cameras") or {}
        order = data.get("camera_order") or sorted(cameras.keys(), key=lambda value: str(value))
        for camera_id in order:
            entry = cameras.get(str(camera_id), cameras.get(camera_id))
            if not isinstance(entry, dict):
                continue
            if "camera_pose" in entry:
                pose = np.asarray(entry["camera_pose"], dtype=np.float32)
            elif "extrinsic" in entry:
                pose = np.linalg.inv(np.asarray(entry["extrinsic"], dtype=np.float32))
            else:
                continue
            if "intrinsic" not in entry:
                continue
            poses.append(pose)
            intrs.append(np.asarray(entry["intrinsic"], dtype=np.float32))
    elif isinstance(data, dict):
        for camera_id in sorted(data.keys(), key=lambda value: str(value)):
            entry = data[camera_id]
            if not isinstance(entry, dict) or "extrinsic" not in entry or "intrinsic" not in entry:
                continue
            poses.append(np.linalg.inv(np.asarray(entry["extrinsic"], dtype=np.float32)))
            intrs.append(np.asarray(entry["intrinsic"], dtype=np.float32))

    if not poses:
        return torch.empty((0, 4, 4)), torch.empty((0, 3, 3))
    return torch.from_numpy(np.stack(poses)).float(), torch.from_numpy(np.stack(intrs)).float()


def _find_preview_camera_json_for_ply(ply_path):
    ply_path = Path(ply_path)
    candidates = []
    if re.fullmatch(r"point_cloud_(\d+)\.ply", ply_path.name):
        step = re.fullmatch(r"point_cloud_(\d+)\.ply", ply_path.name).group(1)
        candidates.append(ply_path.with_name(f"trainer_cameras_{step}.json"))
    candidates.extend([
        ply_path.with_name("trainer_cameras.json"),
        ply_path.parent / "cameras.json",
        ply_path.parent.parent / "gs_data" / "cameras.json",
        ply_path.parent.parent / "cameras.json",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    json_candidates = sorted(ply_path.parent.glob("trainer_cameras_*.json"))
    return json_candidates[-1] if json_candidates else None


def _read_ply_header_bytes(path, max_bytes=262144):
    with open(path, "rb") as handle:
        data = handle.read(max_bytes)
    marker = b"end_header\n"
    index = data.find(marker)
    if index < 0:
        marker = b"end_header\r\n"
        index = data.find(marker)
    if index < 0:
        return ""
    return data[:index + len(marker)].decode("ascii", errors="replace")


def _ply_header_has_gaussian_fields(path):
    header = _read_ply_header_bytes(path)
    if not header:
        return False
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    names = set()
    for line in header.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "property":
            names.add(parts[2])
    return required.issubset(names)


def _quat_wxyz_multiply(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _preview_worldmirror_basis_cache_path(ply_path):
    path = Path(ply_path)
    stat = path.stat()
    key = f"basis_v2:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(key).hexdigest()[:16]
    cache_dir = Path(folder_paths.get_temp_directory()) / "hyworld2_preview_basis"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{path.stem}.worldmirror_{digest}.ply"


def _convert_gaussian_ply_to_worldmirror_preview_basis(ply_path):
    path = Path(ply_path)
    if not path.exists() or not _ply_header_has_gaussian_fields(path):
        return str(path)

    out_path = _preview_worldmirror_basis_cache_path(path)
    if out_path.exists() and out_path.stat().st_mtime >= path.stat().st_mtime:
        return str(out_path)

    with open(path, "rb") as handle:
        header = b""
        vertex_count = None
        props = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Invalid PLY header in {path}")
            header += line
            text = line.decode("ascii", "replace").strip()
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property"):
                parts = text.split()
                if len(parts) >= 3 and parts[1] != "list":
                    props.append((parts[1], parts[2]))
            elif text == "end_header":
                data_offset = handle.tell()
                break

    if vertex_count is None:
        raise ValueError(f"PLY has no vertex count: {path}")

    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "short": "<i2",
        "ushort": "<u2",
    }
    dtype = np.dtype([(name, type_map.get(kind, "<f4")) for kind, name in props])
    vertices = np.fromfile(path, dtype=dtype, count=vertex_count, offset=data_offset).copy()
    prop_names = set(vertices.dtype.names or [])
    if not {"x", "y", "z"}.issubset(prop_names):
        return str(path)

    old_x = vertices["x"].copy()
    old_y = vertices["y"].copy()
    old_z = vertices["z"].copy()
    vertices["x"] = old_x
    vertices["y"] = old_z
    vertices["z"] = -old_y

    if {"nx", "ny", "nz"}.issubset(prop_names):
        old_nx = vertices["nx"].copy()
        old_ny = vertices["ny"].copy()
        old_nz = vertices["nz"].copy()
        vertices["nx"] = old_nx
        vertices["ny"] = old_nz
        vertices["nz"] = -old_ny

    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if set(rot_names).issubset(prop_names):
        quats = np.stack([vertices[name] for name in rot_names], axis=-1).astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-8
        quats[valid] = quats[valid] / norms[valid]
        basis_quat = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)
        quats[valid] = _quat_wxyz_multiply(basis_quat, quats[valid])
        for idx, name in enumerate(rot_names):
            vertices[name] = quats[:, idx]

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    os.replace(tmp_path, out_path)
    print(f"[DJBFilmz_HyWorldPreview] Converted HYWorld2 PLY to preview basis: {out_path}")
    return str(out_path)


def _preview_stat_line(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "empty"
    return (
        f"min={float(arr.min()):.6g}, "
        f"max={float(arr.max()):.6g}, "
        f"mean={float(arr.mean()):.6g}"
    )


def _read_preview_ply_sample(path, sample_count=200000):
    path = Path(path)
    with open(path, "rb") as handle:
        header_bytes = bytearray()
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            header_bytes.extend(line)
            if line.strip() == b"end_header":
                break
        data_offset = len(header_bytes)
        header = header_bytes.decode("ascii", errors="replace")

        fmt = None
        vertex_count = 0
        in_vertex = False
        props = []
        list_props = []
        for raw_line in header.splitlines():
            parts = raw_line.strip().split()
            if not parts:
                continue
            if parts[:2] == ["format", "binary_little_endian"]:
                fmt = "binary_little_endian"
            elif parts[:2] == ["format", "ascii"]:
                fmt = "ascii"
            elif len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and parts[0] == "property":
                if len(parts) >= 5 and parts[1] == "list":
                    list_props.append(" ".join(parts[1:]))
                elif len(parts) >= 3:
                    props.append((parts[1], parts[2]))

        type_info = {
            "char": ("b", 1), "int8": ("b", 1),
            "uchar": ("B", 1), "uint8": ("B", 1),
            "short": ("h", 2), "int16": ("h", 2),
            "ushort": ("H", 2), "uint16": ("H", 2),
            "int": ("i", 4), "int32": ("i", 4),
            "uint": ("I", 4), "uint32": ("I", 4),
            "float": ("f", 4), "float32": ("f", 4),
            "double": ("d", 8), "float64": ("d", 8),
        }
        offsets = {}
        stride = 0
        for prop_type, name in props:
            if prop_type not in type_info:
                continue
            offsets[name] = (prop_type, stride)
            stride += type_info[prop_type][1]
        if fmt != "binary_little_endian" or stride <= 0 or vertex_count <= 0:
            return {
                "format": fmt or "unknown",
                "vertex_count": vertex_count,
                "properties": [name for _, name in props],
                "list_properties": list_props,
                "unsupported": True,
            }

        read_count = min(int(sample_count), int(vertex_count))
        payload = handle.read(read_count * stride)

    values = {name: [] for _, name in props}
    for index in range(read_count):
        base = index * stride
        for name, (prop_type, offset) in offsets.items():
            code, size = type_info[prop_type]
            if base + offset + size <= len(payload):
                values[name].append(struct.unpack_from("<" + code, payload, base + offset)[0])

    summary = {
        "format": fmt,
        "vertex_count": vertex_count,
        "sample_count": read_count,
        "stride": stride,
        "properties": [name for _, name in props],
        "list_properties": list_props,
        "values": values,
        "data_offset": data_offset,
    }
    return summary


def _log_preview_ply_diagnostics(path, label):
    try:
        stats = _read_preview_ply_sample(path)
        print(f"[DJBFilmz_HyWorldPreview][diag] {label} PLY: {path}")
        print(
            f"[DJBFilmz_HyWorldPreview][diag] {label} format={stats.get('format')}, "
            f"vertices={stats.get('vertex_count')}, sample={stats.get('sample_count')}, "
            f"stride={stats.get('stride')}, properties={stats.get('properties')}"
        )
        if stats.get("list_properties"):
            print(f"[DJBFilmz_HyWorldPreview][diag] {label} list_properties={stats['list_properties']}")
        values = stats.get("values") or {}
        for group_name, names in (
            ("xyz", ("x", "y", "z")),
            ("f_dc", ("f_dc_0", "f_dc_1", "f_dc_2")),
            ("opacity_raw", ("opacity",)),
            ("scale_raw", ("scale_0", "scale_1", "scale_2")),
            ("rot", ("rot_0", "rot_1", "rot_2", "rot_3")),
        ):
            present = [name for name in names if name in values and values[name]]
            if not present:
                continue
            if len(present) == 1:
                print(f"[DJBFilmz_HyWorldPreview][diag] {label} {group_name}: {_preview_stat_line(values[present[0]])}")
            else:
                joined = "; ".join(f"{name}: {_preview_stat_line(values[name])}" for name in present)
                print(f"[DJBFilmz_HyWorldPreview][diag] {label} {group_name}: {joined}")
        if all(name in values for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
            sh_c0 = 0.28209479177387814
            rgb = np.stack([0.5 + sh_c0 * np.asarray(values[f"f_dc_{idx}"], dtype=np.float64) for idx in range(3)], axis=1)
            print(
                f"[DJBFilmz_HyWorldPreview][diag] {label} rgb_from_sh: "
                f"r({_preview_stat_line(rgb[:, 0])}); "
                f"g({_preview_stat_line(rgb[:, 1])}); "
                f"b({_preview_stat_line(rgb[:, 2])})"
            )
        if "opacity" in values and values["opacity"]:
            alpha = 1.0 / (1.0 + np.exp(-np.asarray(values["opacity"], dtype=np.float64)))
            print(f"[DJBFilmz_HyWorldPreview][diag] {label} alpha_sigmoid: {_preview_stat_line(alpha)}")
    except Exception as exc:
        print(f"[DJBFilmz_HyWorldPreview][diag] {label} PLY diagnostics failed: {type(exc).__name__}: {exc}")


def _log_preview_splat_diagnostics(path, label, sample_count=200000):
    try:
        path = Path(path)
        if not path.exists() or path.suffix.lower() != ".splat":
            return
        size = path.stat().st_size
        count = size // 32
        read_count = min(int(sample_count), int(count))
        xyz = [[], [], []]
        scales = [[], [], []]
        alpha = []
        with open(path, "rb") as handle:
            for _ in range(read_count):
                row = handle.read(32)
                if len(row) < 32:
                    break
                vals = struct.unpack_from("<6f", row, 0)
                for idx in range(3):
                    xyz[idx].append(vals[idx])
                    scales[idx].append(vals[idx + 3])
                alpha.append(row[27] / 255.0)
        print(f"[DJBFilmz_HyWorldPreview][diag] {label} SPLAT: {path}")
        print(f"[DJBFilmz_HyWorldPreview][diag] {label} splats={count}, sample={read_count}, bytes={size}")
        print(
            f"[DJBFilmz_HyWorldPreview][diag] {label} splat xyz: "
            f"x({_preview_stat_line(xyz[0])}); y({_preview_stat_line(xyz[1])}); z({_preview_stat_line(xyz[2])})"
        )
        print(
            f"[DJBFilmz_HyWorldPreview][diag] {label} splat scale: "
            f"x({_preview_stat_line(scales[0])}); y({_preview_stat_line(scales[1])}); z({_preview_stat_line(scales[2])})"
        )
        print(f"[DJBFilmz_HyWorldPreview][diag] {label} splat alpha: {_preview_stat_line(alpha)}")
    except Exception as exc:
        print(f"[DJBFilmz_HyWorldPreview][diag] {label} SPLAT diagnostics failed: {type(exc).__name__}: {exc}")


def _camera_debug_angles_from_pose(pose):
    pose = np.asarray(pose, dtype=np.float32)
    forward = pose[:3, 2]
    norm = max(float(np.linalg.norm(forward)), 1e-8)
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(forward[1]) / norm))))
    up = pose[:3, 1]
    roll = math.degrees(math.atan2(float(up[0]), max(float(abs(up[1])), 1e-8)))
    return yaw, pitch, roll


def _fmt_vec3_np(vec):
    return f"({float(vec[0]): .4f},{float(vec[1]): .4f},{float(vec[2]): .4f})"


# --- Core Classes ---

class DJBFilmz_HyWorldPreview:
    """
    Preview Gaussian Splatting PLY files with interactive gsplat.js viewer.
    
    Displays 3D Gaussian Splats in an interactive WebGL viewer with orbit controls.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "ply_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to a Gaussian Splatting PLY file"
                }),
                "camera_poses": ("TENSOR", {
                    "tooltip": "Optional: camera poses tensor from WorldMirror V2. Used to initialize the viewer camera."
                }),
                "camera_intrinsics": ("TENSOR", {
                    "tooltip": "Optional: camera intrinsics tensor from WorldMirror V2. Used for viewer FOV."
                }),
                "coordinate_basis": (["auto", "worldmirror", "hyworld2_worldgen"], {
                    "default": "auto",
                    "tooltip": "Coordinate basis of the PLY/camera tensors. Use hyworld2_worldgen for official HYWorld2 trainer outputs."
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("video_path", "ply_path",)
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "DJBFilmz/3D"
    OUTPUT_IS_LIST = (False, False)
    
    @classmethod
    def IS_CHANGED(cls, ply_path=None, coordinate_basis="auto", **kwargs):
        coordinate_basis = cls._normalize_coordinate_basis(coordinate_basis)
        state = [str(ply_path or ""), coordinate_basis]
        if ply_path:
            try:
                stat = os.stat(ply_path)
                state.append(f"{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                state.append("missing")
        return "|".join(state)

    @staticmethod
    def _normalize_coordinate_basis(coordinate_basis):
        value = str(coordinate_basis or "auto").strip()
        if not value:
            return "auto"
        if value not in {"auto", "worldmirror", "hyworld2_worldgen"}:
            print(f"[DJBFilmz_HyWorldPreview] Unknown coordinate_basis={coordinate_basis!r}; falling back to auto")
            return "auto"
        return value

    def _tensor_to_list(self, value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().float()
            return value.tolist()
        if isinstance(value, np.ndarray):
            return value.astype(np.float32).tolist()
        if isinstance(value, (list, tuple)):
            return value
        return None

    def _camera_intrinsics_to_preview(self, camera_intrinsics):
        if camera_intrinsics is None or not hasattr(camera_intrinsics, "detach"):
            return None
        intrs = camera_intrinsics.detach().cpu().float()
        if intrs.dim() == 4:
            intrs = intrs[0]
        if intrs.dim() == 3:
            intrs = intrs[0]
        if intrs.shape[-2:] != (3, 3):
            return None
        return intrs.tolist()

    def _infer_coordinate_basis(self, ply_path, coordinate_basis="auto"):
        requested = self._normalize_coordinate_basis(coordinate_basis)
        if requested != "auto":
            return requested
        try:
            path = Path(ply_path)
            candidates = [
                path.parent.parent / "train_command.json",
                path.parent.parent / "train_config.json",
                path.parent / "train_command.json",
            ]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                with open(candidate, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                basis = data.get("ply_basis") or data.get("camera_pose_basis")
                if isinstance(basis, str):
                    basis = basis.lower()
                    if "hyworld2" in basis:
                        return "hyworld2_worldgen"
                    if "worldmirror" in basis:
                        return "worldmirror"
            if (path.parent / "trainer_cameras.json").exists() or path.name.startswith("point_cloud_"):
                return "hyworld2_worldgen"
        except Exception as exc:
            print(f"[DJBFilmz_HyWorldPreview] coordinate_basis auto-detect skipped: {type(exc).__name__}: {exc}")
        return "worldmirror"

    def _camera_poses_to_preview_basis(self, poses, coordinate_basis):
        if coordinate_basis != "hyworld2_worldgen":
            return poses
        basis = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=poses.dtype,
            device=poses.device,
        )
        return basis.unsqueeze(0) @ poses

    def _camera_poses_to_preview_extrinsics(self, camera_poses, coordinate_basis="worldmirror"):
        if camera_poses is None or not hasattr(camera_poses, "detach"):
            return None
        poses = camera_poses.detach().cpu().float()
        if poses.dim() == 4:
            poses = poses[0]
        if poses.dim() == 2:
            poses = poses.unsqueeze(0)
        if poses.dim() != 3 or poses.shape[-2] not in (3, 4) or poses.shape[-1] != 4:
            return None
        if poses.shape[0] == 0:
            print("[DJBFilmz_HyWorldPreview][diag] camera_poses is empty; skipping extrinsics.")
            return None

        if poses.shape[-2] == 3:
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=poses.dtype).view(1, 1, 4)
            poses = torch.cat([poses, bottom.repeat(poses.shape[0], 1, 1)], dim=1)
        poses = self._camera_poses_to_preview_basis(poses, coordinate_basis)

        c2w = poses[0].clone()
        centers = poses[:, :3, 3]
        c2w[:3, 3] = centers.mean(dim=0)

        # WorldMirror/Panorama nodes output c2w. The browser viewer receives
        # w2c-style extrinsics and reconstructs camera center as -R^T t.
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        w2c = torch.eye(4, dtype=c2w.dtype)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -(R.T @ t)
        return w2c.tolist()

    def _view_info_for_path(self, path):
        output_dir = folder_paths.get_output_directory()
        temp_dir = folder_paths.get_temp_directory()

        path_norm = path.replace("\\", "/")
        output_dir_norm = output_dir.replace("\\", "/")
        temp_dir_norm = temp_dir.replace("\\", "/")

        if path_norm.startswith(output_dir_norm):
            rel_path = os.path.relpath(path, output_dir).replace("\\", "/")
            file_type = "output"
        elif path_norm.startswith(temp_dir_norm):
            rel_path = os.path.relpath(path, temp_dir).replace("\\", "/")
            file_type = "temp"
        else:
            rel_path = os.path.basename(path)
            file_type = "output"

        return {
            "rel_path": rel_path,
            "subfolder": os.path.dirname(rel_path).replace("\\", "/"),
            "filename": os.path.basename(rel_path),
            "type": file_type,
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
        }

    def preview(
        self,
        ply_path=None,
        camera_poses=None,
        camera_intrinsics=None,
        coordinate_basis="auto",
        **kwargs,
    ):
        """Prepare PLY file for gsplat.js preview."""
        coordinate_basis = self._normalize_coordinate_basis(coordinate_basis)

        # If no path provided, we can't preview
        if not ply_path:
            return {"ui": {}, "result": ("", "")}
        
        # Validate ply_path
        if not os.path.exists(ply_path):
            print(f"[DJBFilmz_HyWorldPreview] PLY file not found: {ply_path}")
            return {"ui": {"error": [f"File not found: {ply_path}"]}, "result": ("", "")}

        output_dir = folder_paths.get_output_directory()
        ply_info = self._view_info_for_path(ply_path)
        resolved_coordinate_basis = self._infer_coordinate_basis(ply_path, coordinate_basis)
        camera_coordinate_basis = resolved_coordinate_basis

        try:
            is_gaussian_ply = _ply_header_has_gaussian_fields(ply_path)
        except Exception:
            is_gaussian_ply = False

        if is_gaussian_ply:
            if resolved_coordinate_basis == "hyworld2_worldgen":
                preview_path = _convert_gaussian_ply_to_worldmirror_preview_basis(ply_path)
                frontend_coordinate_basis = "worldmirror"
            else:
                preview_path = ply_path
                frontend_coordinate_basis = resolved_coordinate_basis
            preview_info = ply_info
            preview_format = "ply"
            if preview_path != ply_path:
                preview_info = self._view_info_for_path(preview_path)
                print("[DJBFilmz_HyWorldPreview] Gaussian PLY detected; loading preview-basis PLY.")
            else:
                print("[DJBFilmz_HyWorldPreview] Gaussian PLY detected; loading original PLY directly.")
        else:
            frontend_coordinate_basis = resolved_coordinate_basis
            preview_path = os.path.splitext(ply_path)[0] + ".splat"
            if os.path.exists(preview_path):
                preview_info = self._view_info_for_path(preview_path)
                preview_format = "splat"
            else:
                preview_path = ply_path
                preview_info = ply_info
                preview_format = "ply"
        
        print(f"🔍 [DJBFilmz_HyWorldPreview] Preparing UI Data:")
        print(f"   - Full Path: {ply_path}")
        print(f"   - Filename: {ply_info['filename']}")
        print(f"   - Subfolder: {ply_info['subfolder']}")
        print(f"   - Type: {ply_info['type']}")
        print(f"   - Size: {ply_info['size_mb']:.2f} MB")
        if preview_path != ply_path:
            print(f"   - Preview cache: {preview_info['filename']} ({preview_info['size_mb']:.2f} MB)")
        print(f"   - Resolved coordinate basis: {resolved_coordinate_basis}")
        if frontend_coordinate_basis != resolved_coordinate_basis:
            print(f"   - Frontend coordinate basis: {frontend_coordinate_basis}")
        if hasattr(camera_poses, "shape"):
            print(f"   - camera_poses shape: {tuple(camera_poses.shape)}")
        else:
            print("   - camera_poses shape: None")
        if hasattr(camera_intrinsics, "shape"):
            print(f"   - camera_intrinsics shape: {tuple(camera_intrinsics.shape)}")
        else:
            print("   - camera_intrinsics shape: None")
        _log_preview_ply_diagnostics(ply_path, "source")
        if preview_path != ply_path and preview_format == "splat":
            _log_preview_splat_diagnostics(preview_path, "preview")
        elif preview_path != ply_path and preview_format == "ply":
            _log_preview_ply_diagnostics(preview_path, "preview")

        # Find latest recorded video from the viewer, if one was exported.
        video_path = ""
        try:
            pattern = os.path.join(output_dir, "gaussian-recording-*.mp4")
            video_files = glob.glob(pattern)
            if video_files:
                video_files.sort(key=os.path.getmtime, reverse=True)
                video_path = os.path.abspath(video_files[0])
                print(f"   - Found video: {os.path.basename(video_path)}")
        except Exception as e:
            print(f"   ⚠️ Error finding video: {e}")
            
        ui_data = {
            "filename": [ply_info["filename"]],
            "subfolder": [ply_info["subfolder"]],
            "type": [ply_info["type"]],
            "ply_path": [ply_info["rel_path"]],
            "file_size_mb": [ply_info["size_mb"]],
            "preview_filename": [preview_info["filename"]],
            "preview_subfolder": [preview_info["subfolder"]],
            "preview_type": [preview_info["type"]],
            "preview_path": [preview_info["rel_path"]],
            "preview_file_size_mb": [preview_info["size_mb"]],
            "preview_format": [preview_format],
            "coordinate_basis": [frontend_coordinate_basis],
        }
        
        # Add camera parameters if provided
        extrinsics = self._camera_poses_to_preview_extrinsics(camera_poses, camera_coordinate_basis)
        # Do not forward WorldMirror intrinsics to the browser viewer by default:
        # viewer_gaussian.html switches into native-resolution rendering whenever
        # intrinsics are present, which is much heavier for multi-million splats.
        if extrinsics is not None:
            ui_data["extrinsics"] = [extrinsics]
        
        print(f"✅ [DJBFilmz_HyWorldPreview] UI data ready. Returning to frontend.")
        return {"ui": ui_data, "result": (video_path, ply_path)}


class DJBFilmz_LoadPLYFile:
    """Load a PLY path and optional camera JSON for preview/testing nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path to a .ply file.",
                }),
            },
            "optional": {
                "camera_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional camera JSON. If empty, tries trainer_cameras.json near the PLY.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("ply_path", "camera_poses", "camera_intrinsics", "info")
    FUNCTION = "load"
    CATEGORY = "DJBFilmz/3D"

    @classmethod
    def IS_CHANGED(cls, ply_path, camera_json="", **kwargs):
        values = []
        for raw_path in (ply_path, camera_json):
            value = str(raw_path or "").strip().strip('"')
            path = Path(value) if value else None
            if path and path.exists():
                stat = path.stat()
                values.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
            else:
                values.append(value)
        return "|".join(values)

    def load(self, ply_path, camera_json=""):
        ply_path = str(ply_path or "").strip().strip('"')
        if not ply_path:
            raise ValueError("Load PLY File requires ply_path.")
        path = Path(ply_path)
        if not path.exists():
            raise ValueError(f"PLY file not found: {path}")
        if path.suffix.lower() != ".ply":
            raise ValueError(f"Load PLY File expects a .ply file: {path}")

        camera_path = Path(str(camera_json).strip().strip('"')) if camera_json else None
        if not camera_path or not camera_path.exists():
            camera_path = _find_preview_camera_json_for_ply(path)

        if camera_path and camera_path.exists():
            poses, intrs = _load_preview_camera_tensors_from_json(camera_path)
        else:
            poses, intrs = torch.empty((0, 4, 4)), torch.empty((0, 3, 3))

        info = {
            "ply_path": str(path),
            "camera_json": str(camera_path) if camera_path else "",
            "camera_count": int(poses.shape[0]) if poses.ndim >= 1 else 0,
        }
        print(f"[DJBFilmz_LoadPLYFile] PLY: {path}")
        print(f"[DJBFilmz_LoadPLYFile] Camera JSON: {info['camera_json'] or 'not found'}")
        print(f"[DJBFilmz_LoadPLYFile] Cameras: {info['camera_count']}")
        return (str(path), poses, intrs, json.dumps(info, indent=2))
