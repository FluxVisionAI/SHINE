# written by Dexiang Zong
# CardioVascular Systems Imaging and Artificial Intelligence (CVS.AI)
# National Heart Research Institute Singapore (NHRIS)
# national heart center singarpore

import numpy as np
import trimesh


class MeshProcessor:
    """Handles loading and parsing of the 3D anatomical heart meshes (ventricles)."""
    def __init__(self, mesh_path: str):
        self.mesh_path = mesh_path
        self.mesh = self._load_mesh()

    def _load_mesh(self) -> trimesh.Trimesh:
        """Loads mesh from standard file format (e.g. .obj, .ply, .gii)."""
        # trimesh auto-detects formats
        return trimesh.load(self.mesh_path)

    def get_vertices(self) -> np.ndarray:
        """Returns vertices array (N, 3)."""
        return np.array(self.mesh.vertices, dtype=np.float32)

    def get_faces(self) -> np.ndarray:
        """Returns faces array (F, 3)."""
        return np.array(self.mesh.faces, dtype=np.int32)

    def sample_surface_points(self, num_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Randomly samples points on the mesh surface.
        
        Returns:
            points: Array of coordinates (num_points, 3)
            face_indices: Indices of the faces containing the sampled points
        """
        points, face_indices = trimesh.sample.sample_surface_even(self.mesh, num_points)
        return points.astype(np.float32), face_indices
