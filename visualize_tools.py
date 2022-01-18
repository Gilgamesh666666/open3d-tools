'''
Date: 2021-04-07 11:18:37
LastEditors: ze bai
LastEditTime: 2021-06-25 12:30:47
FilePath: /mm/o3d_tools/visualize_tools.py
'''
import open3d as o3d
import numpy as np
import yaml
import os

# with open('o3d_tools/color.yaml', 'r') as f:
#     color = yaml.safe_load(f)

# COLOR_MAP = color['color_map']
# COLOR_MAP_NROM = color['color_map_norm']

with open('o3d_tools/color137.yaml', 'r') as f:
    color = yaml.safe_load(f)

COLOR_MAP = list(color['color_map'].values())
COLOR_MAP_NROM = list(color['color_map_norm'].values())

def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def get_green():
    """
    Get color green for rendering
    """
    return [0.651, 0.929, 0]
def get_orange():
    """
    Get color orange for rendering
    """
    return [1, 0.3, 0.05]
SRC_COLOR = [0, 0.651, 0.929] # blue
TGT_COLOR = [0.651, 0.929, 0] # green
GT_COLOR = [1,1,0]
SPHERE_COLOR = [0,1,0.1]
SPHERE_COLOR_2 = [0.5,1,0.1]

INLIER_COLOR = [0, 0.9, 0.1]
OUTLIER_COLOR = [1, 0.1, 0.1]
def make_o3d_PointCloud(input_nparr:np.array, color:np.array=None):
    # [n, 3]
    pcd = o3d.geometry.PointCloud()
    assert len(input_nparr.shape) == 2
    assert input_nparr.shape[1] == 3
    pcd.points = o3d.utility.Vector3dVector(input_nparr)
    if color is not None:
        #assert color.shape == (3, 1)
        pcd.paint_uniform_color(color)
    return pcd
def visualize_correspondences_official(source_pcd, target_pcd, inliers, color):
    inliers_temp = []
    if isinstance(inliers, np.ndarray):
        for item in inliers:
            inliers_temp.append((item[0], item[1]))
    else:
        inliers_temp = inliers
    if isinstance(source_pcd, np.ndarray):
        source_pcd = make_o3d_PointCloud(source_pcd)
    if isinstance(target_pcd, np.ndarray):
        target_pcd = make_o3d_PointCloud(target_pcd)
    if isinstance(color, np.ndarray):
        color = list(color)
    corr = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_pcd, target_pcd, inliers_temp)
    corr.paint_uniform_color(color)
    return corr

def visualize_pcd(pc, color):
    # [n, 3] or str
    if isinstance(pc, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.paint_uniform_color(color)
    else:
        print('Input must be numpy!')
        raise ValueError
    return pcd

# Visualize the detected keypts on src_pcd and tgt_pcd
def visualize_keypoint(keypts, color=[0, 0, 1], size=0.03):
    # input: numpy[n, 3]
    # output: List of open3d object (which can directly add to open3d.visualization.draw_geometries())
    box_list0 = []
    for i in range(keypts.shape[0]):
        # Which request open3d 0.9
        # For open3d 0.7: open3d.geometry.create_mesh_sphere(radius=size)
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_box.translate(keypts[i].reshape([3, 1]))
        mesh_box.paint_uniform_color(color)
        box_list0.append(mesh_box)
    return box_list0

def visualize_correspondences(
    source_corrs_points, target_corrs_points, gt_inliers, translate=[-1.3,-1.5,0]
):
    """
    Helper function for visualizing the correspondences

    Just plot segments and two vertex of segments

    [n,3], [n,3], [m,]

    gt_inliers is the indices in "source_corrs_points" which mark the inliers in "source_corrs_points"
    """

    if isinstance(gt_inliers, (list, set, tuple)):
        gt_inliers = np.asarray(list(gt_inliers))

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_corrs_points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_corrs_points)
    
    target.translate(translate)
    
    # get inliers
    source_inlier_points = source_corrs_points[gt_inliers, :]
    target_inlier_points = target_corrs_points[gt_inliers, :]
    
   
    source_inlier_spheres = visualize_keypoint(source_inlier_points, color=INLIER_COLOR, size=0.01)
    target_inlier_spheres = visualize_keypoint(target_inlier_points, color=INLIER_COLOR, size=0.01)
    
    source_all_spheres = visualize_keypoint(source_corrs_points, color=OUTLIER_COLOR, size=0.01)
    target_all_spheres = visualize_keypoint(target_corrs_points, color=OUTLIER_COLOR, size=0.01)

    inlier_line_mesh = LineMesh(source_inlier_points, target_inlier_points, None, INLIER_COLOR, radius=0.012)
    inlier_line_mesh_geoms = inlier_line_mesh.cylinder_segments

    all_line_mesh = LineMesh(source_corrs_points, target_corrs_points, None, OUTLIER_COLOR, radius=0.001)
    all_line_mesh_geoms = all_line_mesh.cylinder_segments
    
    # estimate normals
    vis_list = [*source_all_spheres, *target_all_spheres, *source_inlier_spheres, *target_inlier_spheres]
    vis_list.extend([*all_line_mesh_geoms, *inlier_line_mesh_geoms])
    return vis_list


# Credit to JeremyBYU in this Open3D issue: https://github.com/intel-isl/Open3D/pull/738
# Modified to fit the latest version of Open3D

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, first_points, second_points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.first_points = first_points
        self.second_points = second_points
        if lines is None:
            self.lines = np.tile(np.arange(first_points.shape[0])[:, None], (1, 2))
        else:
            self.lines = np.array(lines)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.first_points[self.lines[:, 0], :]
        second_points = self.second_points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                #print(cylinder_segment)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a)
                cylinder_segment = cylinder_segment.rotate(
                    R)#, center=True)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)
