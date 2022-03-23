import open3d

def draw_pointcloud(np_pointcloud):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pointcloud)
    open3d.visualization.draw_geometries([pcd],
                                    zoom=0.5,
                                    front=[0.5, -0.2, -1],
                                    lookat=[0, 0, 0],
                                    up=[0, -1, 0.2])