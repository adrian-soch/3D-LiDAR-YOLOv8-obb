import numpy as np

MIN_POINT_COUNT = 4

# Emergency_vehicle and van merged
CLASS_NAME_TO_ID = {'BICYCLE':0, 'BUS':1, 'CAR':2, 'EMERGENCY_VEHICLE':3,
                   'MOTORCYCLE':4, 'PEDESTRIAN':5, 'TRAILER':6, 'TRUCK':7, 'VAN':8, 'OTHER':9}

colours = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
          [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255], [255, 255, 180]]

'''
Parameters for point cloud manipulation
'''
boundary = {
    "minX": -12,
    "maxX": 52,
    "minY": -32,
    "maxY": 32,
    "minZ": -0.2,
    "maxZ": 3.8,
}

bound_size_x = boundary['maxX'] - boundary['minX']
bound_size_y = boundary['maxY'] - boundary['minY']
bound_size_z = boundary['maxZ'] - boundary['minZ']

BEV_WIDTH = 1024
BEV_HEIGHT = 1024

DISCRETIZATION = max(bound_size_x, bound_size_y) / BEV_HEIGHT


# A9 North Lidar to Ground Transform
lidar2ground = np.array([[0.9998000,  0.0000000,  0.0199990, 0],
                         [0.0000000,  1.0000000,  0.0000000, 0],
                         [-0.0199990,  0.0000000,  0.9998000, 8.8],
                         [0, 0, 0, 1]])
