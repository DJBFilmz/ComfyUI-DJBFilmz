import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
subnode_path = os.path.join(curr_dir, "gs_viewer")

if subnode_path not in sys.path:
    sys.path.append(subnode_path)

# Import your original camera nodes
from .gs_nodes import DJBFilmz_FlipPLYCoordinates

# Import the standalone viewer nodes
from .gs_preview_nodes import DJBFilmz_HyWorldPreview, DJBFilmz_LoadPLYFile

WEB_DIRECTORY = "web"

NODE_CLASS_MAPPINGS = {
    "DJBFilmz_FlipPLYCoordinates": DJBFilmz_FlipPLYCoordinates,
    "DJBFilmz_HyWorldPreview": DJBFilmz_HyWorldPreview,
    "DJBFilmz_LoadPLYFile": DJBFilmz_LoadPLYFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DJBFilmz_FlipPLYCoordinates": "Flip PLY Coordinates (OpenCV to WebGL)",
    "DJBFilmz_HyWorldPreview": "HYWorld PLY Preview",
    "DJBFilmz_LoadPLYFile": "Load PLY File"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
