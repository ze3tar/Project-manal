import argparse
import os

import numpy as np
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from STL/PLY/etc.
    Handles scenes and meshes, and checks faces/vertices.
    """
    mesh = trimesh.load(path, force='mesh', skip_materials=True, process=False)

    # If it's a Scene, merge all geometries into a single mesh
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Scene in {path} has no geometries")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object from {path} is not a Trimesh")

    if mesh.vertices.shape[0] == 0:
        raise ValueError(f"Mesh {path} has no vertices")
    return mesh


def mesh_to_points(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    """
    Convert mesh to point cloud.
    If the mesh has faces, sample points on the surface.
    If it has no faces (already a point cloud), just use vertices.
    """
    # If no faces → treat as point cloud vertices
    if mesh.faces is None or mesh.faces.shape[0] == 0:
        print("[Info] Mesh has no faces; using vertices directly as points.")
        pts = mesh.vertices
        if pts.shape[0] > n_points:
            idx = np.random.choice(pts.shape[0], size=n_points, replace=False)
            pts = pts[idx]
        return pts.astype(np.float32)

    # Proper mesh: sample on surface
    print(f"[Info] Mesh has {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces.")
    print(f"[Info] Sampling {n_points} points on surface...")
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def save_points_as_ascii_ply(points: np.ndarray, path: str) -> None:
    """
    Save (N,3) points to ASCII PLY (x y z).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n = points.shape[0]
    print(f"[Save] Writing {n} points to {path}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DTU GT mesh (stl029_total.ply) to ASCII point-cloud PLY"
    )
    parser.add_argument(
        "--input",
        default="scan29/stl029_total.ply",
        help="Path to DTU GT mesh (binary PLY/STL)",
    )
    parser.add_argument(
        "--output",
        default="scan29/stl029_total_points.ply",
        help="Output ASCII point-cloud PLY",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=6000000,
        help="Number of points to sample (if mesh has faces)",
    )
    args = parser.parse_args()

    print(f"[Load] {args.input}")
    mesh = load_mesh(args.input)

    pts = mesh_to_points(mesh, args.num_points)
    print(f"[Info] Generated point cloud with {pts.shape[0]} points")

    save_points_as_ascii_ply(pts, args.output)
    print("[Done] Conversion finished.")


if __name__ == "__main__":
    main()
