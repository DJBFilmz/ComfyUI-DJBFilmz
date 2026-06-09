import os
import json
import torch
import numpy as np
from plyfile import PlyData, PlyElement

class DJBFilmz_LoadHYWorldCamera:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_path": ("STRING", {"default": ""}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("EXTRINSICS", "INTRINSICS")
    RETURN_NAMES = ("extrinsics", "intrinsics")
    FUNCTION = "load_camera"
    CATEGORY = "DJBFilmz/3D"

    def load_camera(self, json_path, frame_index):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON camera file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            cam_data = json.load(f)
            
        # Standard format check
        # HY-World usually outputs a list of camera frames, or a dictionary mapping frames
        if isinstance(cam_data, list):
            if frame_index >= len(cam_data):
                frame_index = len(cam_data) - 1
            frame = cam_data[frame_index]
        elif isinstance(cam_data, dict):
            # If it's mapped by file names, grab the keys
            keys = list(cam_data.keys())
            if frame_index >= len(keys):
                frame_index = len(keys) - 1
            frame = cam_data[keys[frame_index]]
        else:
            frame = cam_data
            
        # 1. Extract Intrinsics (3x3 Matrix)
        # Search for K, intrinsics, or construct from fx, fy, cx, cy
        if "K" in frame:
            intrinsics = np.array(frame["K"])
        elif "intrinsics" in frame:
            intrinsics = np.array(frame["intrinsics"])
        else:
            # Fallback construct from common keys
            fx = frame.get("fx", 1000.0)
            fy = frame.get("fy", 1000.0)
            cx = frame.get("cx", 960.0)
            cy = frame.get("cy", 540.0)
            intrinsics = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ])
            
        # 2. Extract Extrinsics (4x4 Matrix)
        # Search for world2cam (W2C), extrinsics, or R and T
        if "W2C" in frame:
            extrinsics = np.array(frame["W2C"])
        elif "extrinsics" in frame:
            extrinsics = np.array(frame["extrinsics"])
        elif "R" in frame and "T" in frame:
            R = np.array(frame["R"])
            T = np.array(frame["T"]).flatten()
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = R
            extrinsics[:3, 3] = T
        else:
            extrinsics = np.eye(4)

        # Ensure correct shape
        if intrinsics.shape != (3, 3):
            # Reshape if flat
            intrinsics = intrinsics.reshape(3, 3)
        if extrinsics.shape != (4, 4):
            extrinsics = extrinsics.reshape(4, 4)

        # Convert to PyTorch Tensors (standard format for ComfyUI matrices)
        extrinsics_tensor = torch.from_numpy(extrinsics).float()
        intrinsics_tensor = torch.from_numpy(intrinsics).float()

        return (extrinsics_tensor, intrinsics_tensor)

class DJBFilmz_FlipPLYCoordinates:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ply_path": ("STRING", {"default": ""}),
                "rotate_180_x": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "flip_ply"
    CATEGORY = "DJBFilmz/3D"

    def flip_ply(self, ply_path, rotate_180_x):
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
            
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        
        # Make a mutable copy of the numpy structured array
        data = np.array(vertex.data)
        
        if rotate_180_x:
            # 180-degree rotation around X-axis translates to:
            # Y -> -Y and Z -> -Z
            data['y'] = -data['y']
            data['z'] = -data['z']
            
            # Correct the local orientations (quaternions) to match the flip
            # Quaternions are typically stored as rot_0 (w), rot_1 (x), rot_2 (y), rot_3 (z)
            if 'rot_0' in data.dtype.names:
                w = data['rot_0'].copy()
                x = data['rot_1'].copy()
                y = data['rot_2'].copy()
                z = data['rot_3'].copy()
                
                # Apply 180-deg rotation around X-axis (q_rot = [0, 1, 0, 0])
                # q_new = q_rot * q_orig
                data['rot_0'] = -x
                data['rot_1'] = w
                data['rot_2'] = -z
                data['rot_3'] = y
                
            # If normals are present, they also need inversion
            if 'nx' in data.dtype.names:
                data['ny'] = -data['ny']
                data['nz'] = -data['nz']
                
        # Rebuild and write PLY
        new_vertex = PlyElement.describe(data, 'vertex')
        new_plydata = PlyData([new_vertex], text=plydata.text)
        
        base, ext = os.path.splitext(ply_path)
        output_path = f"{base}_flipped{ext}"
        new_plydata.write(output_path)
        
        return (output_path,)

NODE_CLASS_MAPPINGS = {
    "DJBFilmz_LoadHYWorldCamera": DJBFilmz_LoadHYWorldCamera,
    "DJBFilmz_FlipPLYCoordinates": DJBFilmz_FlipPLYCoordinates
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DJBFilmz_LoadHYWorldCamera": "Load HYWorld Camera",
    "DJBFilmz_FlipPLYCoordinates": "Flip PLY Coordinates (OpenCV to WebGL)"
}