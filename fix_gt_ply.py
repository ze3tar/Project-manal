import struct
import numpy as np

def read_binary_ply(path):
    print("[Load] Reading binary PLY:", path)

    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break

        # Parse header
        for line in header:
            if line.startswith("element vertex"):
                num_verts = int(line.split()[2])
                print("Vertex count:", num_verts)

        # Each vertex = 3 floats (xyz) + 3 floats (normals) + 3 uchar (rgb)
        stride = 4*6 + 3
        data = f.read(num_verts * stride)

    print("[Parse] Extracting XYZ only...")

    points = []
    offset = 0

    for _ in range(num_verts):
        # Read 6 floats = xyz, nx, ny, nz
        xyz_n = struct.unpack("<ffffff", data[offset : offset + 24])
        x, y, z = xyz_n[:3]
        points.append((x, y, z))
        offset += stride

    pts = np.array(points, dtype=np.float32)
    
    # Clean invalid data
    mask = np.isfinite(pts).all(axis=1)
    removed = len(pts) - mask.sum()
    pts = pts[mask]

    print(f"[Clean] Removed {removed} invalid points.")
    print(f"[OK] Loaded {len(pts)} clean XYZ points.")
    return pts


def save_ascii_ply(points, outpath):
    print("[Save] Writing ASCII PLY:", outpath)
    with open(outpath, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(points)))
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")

    print("[DONE] Saved clean GT:", outpath)


# MAIN
pts = read_binary_ply("scan29/stl029_total.ply")
save_ascii_ply(pts, "scan29/stl029_gt_ascii_fixed.ply")
