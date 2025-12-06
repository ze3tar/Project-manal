import open3d as o3d

pcd = o3d.io.read_point_cloud("outputs/scan29_clean.ply")
o3d.visualization.draw_geometries([pcd])   # works only if GUI is available
