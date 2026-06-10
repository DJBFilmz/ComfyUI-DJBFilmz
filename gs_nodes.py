import os
import json
import torch
import numpy as np
from plyfile import PlyData, PlyElement

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
        data = np.array(vertex.data)
        
        if rotate_180_x:
            data['y'] = -data['y']
            data['z'] = -data['z']
            
            if 'rot_0' in data.dtype.names:
                w = data['rot_0'].copy()
                x = data['rot_1'].copy()
                y = data['rot_2'].copy()
                z = data['rot_3'].copy()
                
                data['rot_0'] = -x
                data['rot_1'] = w
                data['rot_2'] = -z
                data['rot_3'] = y
                
            if 'nx' in data.dtype.names:
                data['ny'] = -data['ny']
                data['nz'] = -data['nz']
                
        new_vertex = PlyElement.describe(data, 'vertex')
        new_plydata = PlyData([new_vertex], text=plydata.text)
        
        base, ext = os.path.splitext(ply_path)
        output_path = f"{base}_flipped{ext}"
        new_plydata.write(output_path)
        
        return (output_path,)

NODE_CLASS_MAPPINGS = {
    "DJBFilmz_FlipPLYCoordinates": DJBFilmz_FlipPLYCoordinates
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DJBFilmz_FlipPLYCoordinates": "Flip PLY Coordinates (OpenCV to WebGL)"
}
