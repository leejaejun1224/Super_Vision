#!/usr/bin/env python3

# Import modules
import pcl

# Returns Downsampled version of a point cloud
# The bigger the leaf size the less information retained
def do_voxel_grid_filter(point_cloud, LEAF_SIZE = 0.01):
  voxel_filter = point_cloud.make_voxel_grid_filter()
  voxel_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE) 
  return voxel_filter.filter()

# Returns only the point cloud information at a specific range of a specific axis
def do_passthrough_filter(point_cloud, name_axis, min_axis, max_axis):
  pass_filter = point_cloud.make_passthrough_filter()
  pass_filter.set_filter_field_name(name_axis)
  pass_filter.set_filter_limits(min_axis, max_axis)
  return pass_filter.filter()

# Use RANSAC planse segmentation to separate plane and not plane points
# Returns inliers (plane) and outliers (not plane)
def do_ransac_plane_segmentation(point_cloud, max_distance):

  segmenter = point_cloud.make_segmenter_normals(ksearch=50)
  segmenter.set_optimize_coefficients(True)
  segmenter.set_model_type(pcl.SACMODEL_NORMAL_PLANE)  #pcl_sac_model_plane
  segmenter.set_normal_distance_weight(0.1)
  segmenter.set_method_type(pcl.SAC_RANSAC) #pcl_sac_ransac
  segmenter.set_max_iterations(1000)
  segmenter.set_distance_threshold(max_distance) #0.03)  #max_distance
  indices, coefficients = segmenter.segment()

  inliers = point_cloud.extract(indices, negative=False)
  outliers = point_cloud.extract(indices, negative=True)

  return indices, inliers, outliers