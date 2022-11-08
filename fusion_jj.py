#!/usr/bin/env python
#-*- coding: utf-8 -*-

#sdfsd


"""
pcl 가공을 여기서 다 해버려야 하는건가
점군데이터는 필요가 없을까
나도 다른 곳에서 pcd를 가공하는 가공코드를 따로 짜보자.
"""


from __future__ import print_function
from distutils.spawn import find_executable
from symbol import try_stmt
from unittest.util import three_way_cmp
import numpy as np
import os
import sys
import threading
import time
import math
import rospy
import multiprocessing
from rospy.numpy_msg import numpy_msg


import message_filters

from pcl_helper import *
from filtering_helper import *

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point32
from sensor_msgs.msg import ChannelFloat32
from vision_msgs.msg import Detection2DArray, Detection2D
from sensor_msgs.msg import PointField
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
import cv2
import ros_numpy
#from cv_bridge import CvBridge, CvBridgeError


"""
# using namespace에 대한 설명
sensor_msgs::PointCloud
using namespace sensor_msgs

PointCloud



# c++
cin >> a; # input()
cout << "c++" <<endl; print("C++\n")
std::cout << "C++" << endl;
using namespace std;
endl = '\n' (속도 : endl < '\n')
#define endl '\n'
"""




"""
여러가지 파라미터들 전역변수로 선언 여기서도 함수로 바꿀 수 있는건 바꾸고 그냥 쓸건 쓰자
"""
tr_x = 0.0
tr_y = 0.0
tr_z = 0.0

CameraIntrintrinsicParam = [490.0542667650563, 0.0, 317.3826837893605, 0.0, 491.0918311743009, 256.63115327011445, 0.0, 0.0, 1.0]
Camera_Intrinsic = np.array(CameraIntrintrinsicParam).reshape(3,3)
CameraLidarRotation = [0,0,0,0,0,0,0,0,0]
CamearLidarTranslation = [0.7, 0.3,0.01]

pan = 0.0
tilt = 0.0

x_i = 0.0
x_e = 0.0

y_i = 0.0
y_e = 0.0

bbox_xmin = 0
bbox_xmax = 2000
bbox_ymin = 0
bbox_ymax = 2000

extrinsic_example = np.array([[   0.999715  ,-0.00610437 ,0.023068, -0.0149389236], 
  [0.0113064 , 0.972496   ,-0.232646, -0.059792],  
 [-0.0210134 , 0.23284    ,0.972288 ,-0.11993134]])

params_cam = {
    "WIDTH": 640, # image width
    "HEIGHT": 480, # image height
    "FOV": 60, # Field of view
    "X": 0.78, # meter
    "Y": -0.55,
    "Z": -0.4,
    "YAW": 0.0, # deg
    "PITCH": 0.0,
    "ROLL": 0.0
}




"""
extrinsic matrix인데 이거보단 아래 ExtrinsicMat 함수를 쓰는게 나을 듯 하다.
"""
def CamLiD_Callibration():
    CamLidRot = np.array(CameraLidarRotation).reshape(3,3)
    CamLidTr = np.array(CamearLidarTranslation).reshape(3,1)
    return CamLidRot, CamLidTr




"""
geometry 이론 간단하게 사용해서 matrix 그냥 만들었음. 어떻게 만들어진 matrix인지는 설명 필요하면 말하삼
돌고돌아 가장 쉬울만한 matrix 선정. extrinsic matrix는 3x3 rotation(RotMat) 행렬과 translation(CamLidTr) 3x1행렬의 조합임. rotation
행렬은 만든 식이고 translation 행렬은 거리 행렬이니 직접 자로 재서 구하면 됨. 그렇게 해서 결과 행렬 3x4행렬을
만들어 놓고 그 밑에 [0,0,0,1] 행렬 만들어서 붙이면 된다. 어차피 0만 나오는 행렬이기도 하고 행렬 차원 맞추기용으로 붙인 
행렬이니 별 의미는 없음! 결과로 Extrinsic_param이 나옴
여기에 pan은 카메라를 좌우로 얼마나 돌렸냐이고 tilt는 위아래로 얼마나 기울였냐 각도임.
"""    
def ExtrinsicMat(pan,tilt,CamearLidarTranslation):   

    cosP = math.cos(pan*math.pi/180)
    sinP = math.sin(pan*math.pi/180)
    cosT = math.cos(tilt*math.pi/180)
    sinT = math.sin(tilt*math.pi/180)
    
    RotMat = np.array([[sinP, -cosP,0],[sinT*cosP, sinT*sinP,-cosT],[cosT*cosP,cosT*sinP,sinT]]) #u,v,a축
    CamLidTr = np.array(CamearLidarTranslation).reshape(3,1)    
    Rot_Tr = np.hstack((RotMat, CamLidTr))
    np_zero = np.array([0,0,0,1])
    Extrinsic_param = np.vstack((Rot_Tr, np_zero))    
    
    return Extrinsic_param




"""
카메라 내부 행렬을 구하는 첫 번째 방법. 카메라 제원표에 나와있는 값을 이용하여 식에다 대입해서 쓰는 방법.
calibration해서 구하는 두 가지 방법이 있다. 바로 아래 방법은 calibration을 해서 나온 결과값을 사용하는 방법
"""
def Cam_Intrinsic(Camera_Intrinsic):
    np_zero = np.array([[0,0,0]]).T
    Intrinsic_param = np.hstack((Camera_Intrinsic, np_zero))
    return Intrinsic_param




"""
카메라 내부 행렬을 구하는 두 번째 방법. 내부행렬식에다가 카메라 제원을 집어넣어서 내부 행렬을 만든 것.
두 방법 중에서 어느것이 더 정확한지는 아직 모르겠음.... 이 코드에서는 이 방법으로 내부행렬 정의해서 사용했음.
"""
def  getCameraMat(params_cam):
    focalLength = params_cam["WIDTH"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
    principalX = params_cam["WIDTH"]/2
    principalY = params_cam["HEIGHT"]/2
    CameraMat = np.array([focalLength,0.,principalX,0,focalLength,principalY,0,0,1]).reshape(3,3)
    return CameraMat




"""
Extrinsic_points는 extrinsic matrix (R|t 행렬)을 거쳐서 카메라 좌표계로 나온 라이다 점들임. 이걸 Cam_Intrinsic 함수에 넣으면 
"""
def fixel_coordinate(Int_param, Extrinsic_points):
    fixel_coord = Cam_Intrinsic(Int_param).dot(Extrinsic_points)
    return fixel_coord




"""
extrinsic 행렬이랑 라이다 좌표계 4x4랑 4x1 곱해서 4x1행렬로 만든 다음 맨 아래 값 어차피 1이라 지워서 3x1행렬로 바꾸어줌
"""
def get_lidar_to_camera(TransformMat, pointcloud):
    Lid2cam = TransformMat.dot(pointcloud)
    Lid2cam = np.delete(Lid2cam, 3, axis=0)
    return Lid2cam




"""
clustering 하는 함수 많이 봤겠고
"""
def get_clusters(cloud, tolerance, min_size, max_size):
    tree = cloud.make_kdtree()
    extraction_object = cloud.make_EuclideanClusterExtraction()
    extraction_object.set_ClusterTolerance(tolerance)
    extraction_object.set_MinClusterSize(min_size)
    extraction_object.set_MaxClusterSize(max_size)
    extraction_object.set_SearchMethod(tree)
    clusters = extraction_object.Extract()
    
    return clusters

"""
이거 안씀
"""    
def pcd_inner_fov(fixel_coord,width,height,new_world_xyz):
    inner_fov_pts = {}
    for idx in range(0,fixel_coord.shape[1]):
        if 0 < int(fixel_coord[0,idx]) < width and 0 < int(fixel_coord[1,idx]) < height:
            if int(new_world_xyz[0,idx]) > 0:
                new_new_world_xyz = new_world_xyz[:,idx]
                inner_fov_pts[idx] = [int(fixel_coord[0,idx]), int(fixel_coord[1,idx])]
                return new_new_world_xyz, idx
            else:
                continue    
        else:
            continue  




"""
파라미터부터 하나씩 설명하자면 img는 종이 하나를 가져왔다고 보면 되고, new_world_xyz는 x,y,z,1이 있는 4x1행렬이 되겠다. 사실 정확히 말하면 for문 돌면서 x,y,z,1값을 append 했으니 
[4x검출된 점의 갯수] 인 행렬인데 일단 각 점 하나에 대한 4x1행렬이라고 보자. 

"""
def get_lidar_in_image_fov(img, new_world_xyz, fixel_coord, xmin, ymin, width, height, min_distance=1.0):
    inner_fov_pts = {}
    for idx in range(0,fixel_coord.shape[1]):
        if xmin < int(fixel_coord[0,idx]) < width and ymin < int(fixel_coord[1,idx]) < height:
            if int(new_world_xyz[0,idx]) > 0:
                #new_world_xyz = new_world_xyz
                inner_fov_pts[idx] = [int(fixel_coord[0,idx]), int(fixel_coord[1,idx])]
                cv2.circle(img, (int(fixel_coord[0,idx]), int(fixel_coord[1,idx])), 1, (255,255,255), cv2.FILLED, cv2.LINE_4)
    return img, inner_fov_pts




"""
yolo로 부터 bounding box의 위치를 반환 받으면 그 안에 있는 점들만 딕셔너리 형태로 저장해줌
"""
def inner_roi(img, bbox_left,bbox_top, w,h, new_fixel_coord):
    roi_pts = {}
    for idx in range(0,new_fixel_coord.shape[1]):
        if bbox_left < int(new_fixel_coord[0,idx]) < bbox_left + w -1 and bbox_top < int(new_fixel_coord[1,idx]) < bbox_top + h -1:
            roi_pts[idx] = [int(new_fixel_coord[0,idx]), int(new_fixel_coord[1,idx])]
            #cv2.circle(img, (int(new_fixel_coord[0,idx]),int(new_fixel_coord[1,idx])),1, (255,0,0), cv2.FILLED, cv2.LINE_4)
            #inner_roi_lidar_pts = new_new_world_xyz[:,idx]
    return img, roi_pts 




def inner_cluster(img, bbox_left,bbox_top, w,h, colored_points):
    # roi_pts = {}
    for idx in range(0,len(colored_points)):
        print(colored_points[idx][0],colored_points[idx][1])
        
        #cv2.circle(img, (int(colored_points[idx][0]),int(colored_points[idx][1])),1, (255,255,255), cv2.FILLED, cv2.LINE_4)
            #inner_roi_lidar_pts = new_new_world_xyz[:,idx]
    return img



"""
pointcloud형식으로 들어온 데이터를 clustering해주는 내장함수 선언 주의할 것은 이에 들어가는 pointcloud형식엔
RGB값이 빠져있어야 하므로 cloud에 들어갈 인자는 colorless_cloud가 들어와야한다.
"""
def get_clusters(cloud, tolerance, min_size, max_size):

    tree = cloud.make_kdtree()
    extraction_object = cloud.make_EuclideanClusterExtraction()

    extraction_object.set_ClusterTolerance(tolerance)
    extraction_object.set_MinClusterSize(min_size)
    extraction_object.set_MaxClusterSize(max_size)
    extraction_object.set_SearchMethod(tree)
    
  # Get clusters of indices for each cluster of points, each clusterbelongs to the same object
  # 'clusters' is effectively a list of lists, with each list containing indices of the cloud
    clusters = extraction_object.Extract()

    return clusters




"""
이건 roi를 자르기 위한 함수 많이 봤지?
"""
def do_passthrough(cloud, filter_axis, axis_min, axis_max):     #point들의 roi를 설정하기 위해 만든 함수
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()




"""
행렬식을 pointcloud2형식의 메시지로 바꿔주는 함수. pointcloud2는 통신하는 메시지 형식이라서 이 메시지 안의
점들 데이터를 roi를 자른다던가 clustering을 한다던가 하려면 pcl 즉 pointcloud형식으로 바꾸어 주여햐 함
그래서 마지막에 ros_to_pcl 라이브러리 함수를 써서 pointcloud(pcl) 형식으로 바꾸어 줌 사실 바로 pointcloud
형식을 바꾸어도 되긴 한데 혹시 pointcloud2 형식도 나중에 쓸 일이 있을까봐 만들어 놓았음 
"""
def array_to_pointcloud(inner_world_xyz):
    
    #헤더 정의
    header = Header()
    header.frame_id = 'os_sensor'
    header.stamp = rospy.Time.now()


    #안에 들어가야할 정보 정의
    pc_array = np.zeros(len(inner_world_xyz), dtype=[
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('intensity', np.float32),
    ])
    pc_array['x'] = inner_world_xyz[:, 0]
    pc_array['y'] = inner_world_xyz[:, 1]
    pc_array['z'] = inner_world_xyz[:, 2]
    pc_array['intensity'] = inner_world_xyz[:, 3]

    pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=header.stamp, frame_id=header.frame_id)




    return pc_msg

def cluster_mask(clusters, clouds):
    cluster_color = [255,255,255]
    cluster_point_list = []
    for j, idx in enumerate(clusters):
        for i, indice in enumerate(idx):
            cluster_point_list.append([clouds[indice][0],
                                        clouds[indice][1],
                                        clouds[indice][2],
                                        rgb_to_float(cluster_color)])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(cluster_point_list)
    return cluster_cloud

def get_colored_clusters(clusters, cloud):
  
#   Get a random unique colors for each object
    number_of_clusters = len(clusters)
#   print(number_of_clusters)
    colors = get_color_list(number_of_clusters)

    colored_points = []

  
    # Assign a color for each point
    # Points with the same color belong to the same cluster
    for cluster_id, cluster in enumerate(clusters):
        tmp_colored_points = []
        for c, i in enumerate(cluster):
            x, y, z = cloud[i][0], cloud[i][1], cloud[i][2]
            color = rgb_to_float(colors[cluster_id])
            colored_points.append([x, y, z, color])
            tmp_colored_points.append([x, y, z, color])
      


    #print(colored_points)
    clusters_cloud = pcl.PointCloud_PointXYZRGB()
    clusters_cloud.from_list(colored_points)
    clusters_msg = pcl_to_ros(clusters_cloud)
    clusters_msg.header.frame_id = "os_sensor"
  
  # print(clusters_msg[0])



    return clusters_msg, colored_points
"""
대망의 콜백함수
"""
def pcl_callback(lidar_subscriber,image_subscriber):
    
    #clustering에 쓸 parameter들을 정의해 두었음
    tolerance = 0.8
    min_size = 25.0
    max_size = 400.0

    # try:
    #     img = bridge.imgmsg_to_cv2(image_subscriber, 'bgr8')
    # except CvBridgeError as e:
    #     rospy.logerr("CvBridge Error: {}".format(e))

    # img = ros_numpy.numpify(image_subscriber)

    # img = np.frombuffer(image_subscriber.data, dtype=np.uint8).reshape(image_subscriber.height, image_subscriber.width, -1)
    #위에서 말한 calibration말고 camera 제원값을 식에 집어 넣어서 행렬값을 구해서 3x3 내부행렬을 정의했음
    arr = ros_numpy.numpify(image_subscriber)
    img = np.array(arr, dtype=np.uint8)

    CameraMat = getCameraMat(params_cam)
    # CameraMat =  Cam_Intrinsic(Camera_Intrinsic) 
    #회전 + 이동 행렬을 더해서 extrinsic matrix를 만들어 놓았음
    Extrinsic_param = ExtrinsicMat(0,0,CamearLidarTranslation)
    
    #밑에서 쓸 값인데 먼저 선언해서 이따 width만 쓰게 정의함 중요하지x
    width = 640
    height = 480


    #yolo로 detecion된 객체의 좌표와 그 박스의 가로세로 크기를 받아서 쓸 건데 그 전에 미리 시험에 보려고 내 임의대로 bounding box의 위치 정의했음.
    bbox_top = 190
    bbox_left = 350
    h = 210
    w = 80
    #callback 함수에서 받아온 pcl_msg를(pointcloud2 즉 ros 메시지 형식을) 사용할 것임
    world = []
    for n in point_cloud2.read_points(lidar_subscriber, skip_nans=True):  #여기서 x,y,z 값과 1을 world라는 리스트에 추가해줌 
        world.append((n[0], n[1], n[2],1))

    world_xyz = np.array(world, np.float32) #world라는 리스트를 행렬로 바꿔줌

    new_world_xyz = world_xyz[:,:].T #이 행렬을 세로로 바꾸어줌 그러니 4x1 행렬이 되겠고

    Ext = get_lidar_to_camera(Extrinsic_param, new_world_xyz) #new_world_xyz 행렬을 extrinsic 행렬에 곱하면 4x4 행렬이 나올텐데 만 아래줄 어차피 0이니까 지워서 3x4 행렬로 만들어줌
    
    fixel_coord = CameraMat.dot(Ext) #callback 함수 위쪽에서 정의한 CameraMat과 회전행렬 결과값을 곱해서 카메라 픽셀 단위에 맞추어진 새로운 좌표를 만들었음. 이 결과는 3x1 행렬이겠지 카메라 3x3행렬과 회전행렬 결과값 3x1행렬을 곱했으니
    fixel_coord /= fixel_coord[2,:] #3x1 행렬 중에서 맨 아래 값은 픽셀에서 얼마나 멀리 떨어져 있는지를 의미하는데 카메라에선 그런 건 필요없으니 그 값으로 전체를 나눠줌. 그럼 맨 아래값은 1이 되겠지.
    
    # img = np.zeros((480,640,3), np.uint8) #가로 640, 세로480 크기의 검정색 이미지를 선언하고
    img, inner_fov_pts = get_lidar_in_image_fov(img, new_world_xyz,fixel_coord, 0, 0, width, height, min_distance=1.0) #픽셀좌표로 넘어온 점들중에 위에 만든 검정색 이미지 안에 들어가는 점들만 추출함 이 점들의 인덱스와 좌표를 딕셔너리 형태로 반환  ex){2456(번째 점): [x,y]}
    idxs = list(inner_fov_pts.keys()) #위에서 딕셔너리의 key들 즉 인덱스들을 list로 저장을 한 다음
    new_new_world_xyz = new_world_xyz[:, idxs] #그 인덱스에 해당하는 라이다 월드 좌표들을 저장해주고
    new_fixel_coord = fixel_coord[:,idxs] #그 인덱스 안에 해당하는 픽셀 좌표들도 저장해줌

    roi = img[bbox_top:bbox_top+h, bbox_left:bbox_left+w]  #픽셀 좌표내에서 roi 일단 내 맘대로 설정해줌 이거 함수로 만들어야 함 두 개 이상 객체 검출되는 경우가 많을거라
    
    #inner_roi(img, bbox_left,bbox_top, w,h, new_fixel_coord) 
    img, inner_roi_pts = inner_roi(img, bbox_left,bbox_top, w,h, new_fixel_coord) #roi안에 해당하는 점들에 대해서만 위에랑 똑같이 딕셔너리 형태로 반환하고 그 좌표들을 점으로 찍어놓음 반환해주고
    inner_idx = list(inner_roi_pts) #roi안에 있는 점들의 인덱스를 리스트로 반환하고

    inner_world_xyz = new_new_world_xyz[:,inner_idx].T #그 roi안에 있는 점들의 월드좌표들을 반환해서 새로 저장하고
    #print(inner_world_xyz.shape)
    #print(inner_world_xyz.shape[1])
    xyz_tmp = inner_world_xyz[0:3,:].T

    pc2 = array_to_pointcloud(inner_world_xyz) #그 월드 좌표들은 배열의 형태일텐데 이걸 pointcloud2형식 즉 rosmsg형식으로 변환을 해줌. 이를 다시 pointcloud 형식으로 바꿔서 우리가 가공할 수 있는 pcl형식으로 바꾸어 줄것임 
    pc2_publisher.publish(pc2)
    #print(inner_world_xyz)

    clouds = ros_to_pcl(pc2) #우리가 가공해야하므로 pointcloud형식 즉 pcl형식으로 바꾸어줌
    #print(clouds)
    
    filter_axis = 'x'
    axis_min = 3.5
    axis_max = 4.5
    clouds = do_passthrough(clouds, filter_axis, axis_min, axis_max)  # 일단은 지금 사람이 4m에 있으니 4m지점에 대해서만 월드좌표들을 남겨놓고
    
    colorless_cloud = XYZRGB_to_XYZ(clouds) #clustering 함수에는 rgb 값이 필요가 없으니 rgb값 지워주고 
    clusters = get_clusters(colorless_cloud, 1.0, 20, 500) #clustering 해서 
    print("number of objects : {0}".format(len(clusters)))
    clusters_msg, colored_points = get_colored_clusters(clusters, colorless_cloud)
    
    clusters_publisher.publish(clusters_msg)
    distance = []

    for i in range(0,len(colored_points)):
        distance.append(math.sqrt((colored_points[i][0])**2 + (colored_points[i][1])**2))
        
        

    # min_distance = min(distance)
    
    # print("distance to objects : {0}".format(min_distance))    

   

    cv2.rectangle(roi, (0,0), (w-1,h-1), (0,255,0)) #이건 그냥 그리는 함수
    cv2.imshow("projection", img)
    cv2.waitKey(1)




    #앞으로 할 것들 : clustering한 객체들 정확하게 객수 파악하는 거랑(사실 안중요할 수도 있음 거리 정보가 끊긴다고 하면 잠깐 그 전의 거리값을 그대로 반환하도록 하면 되니까 그런데 그래도 최대한 정확하게 검출이 되도록 하는게 best)


if __name__ == '__main__':
    

    rospy.init_node('sensor_fusion', anonymous = True)
    
    lidar_subscriber = message_filters.Subscriber('/os_cloud_node/points', PointCloud2)
    image_subscriber = message_filters.Subscriber('/usb_cam/image_raw', Image)
    # yolo_subscriber = message_filters.Subscriber('/yolo_bbox', )


    clusters_publisher = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size = 1)
    # inner_cloud_msg_pub = rospy.Publisher("/inner_cloud", PointCloud2, queue_size = 1)
    # Lid2Cam_publisher = rospy.Publisher("/lid2cam", PointCloud2, queue_size=1)
    pc2_publisher = rospy.Publisher("roi", PointCloud2, queue_size=1)
    
    ts = message_filters.ApproximateTimeSynchronizer([lidar_subscriber,image_subscriber],queue_size=10, slop=0.1)
    ts.registerCallback(pcl_callback)
    
    get_color_list.color_list = []
    
    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     rate.sleep()

    try:
        rospy.spin()
        print("ddddddddddddddddddddddddddd")

    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')    

