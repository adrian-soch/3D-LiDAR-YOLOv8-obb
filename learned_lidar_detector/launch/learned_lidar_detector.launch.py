import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription

share_dir = get_package_share_directory('learned_lidar_detector')
model_path = os.path.join(share_dir, 'yolov8n-obb_range.pt')

rviz_config_file = os.path.join(share_dir, 'rviz.rviz')

CONFIG_NAME = 'r02_s03_cam1South_lidarSouth_config.yaml'
data_dependant_params = os.path.join(
    share_dir, CONFIG_NAME)


def generate_launch_description():

    lidar_perception_node = Node(
        package='learned_lidar_detector',
        executable='learned_lidar_detector',
        output='screen',
        parameters=[
            data_dependant_params,
            {'lidar_topic': 'points'},
            {'model_path': model_path}
        ]
    )

    lidar_viz = Node(
        package='obj_tracker',
        executable='tracker_bbox_viz',
        name='lidar_det_viz',
        output='screen',
        parameters=[
            {'topic_name': 'ld_proc/dets'},
            {'bbox_marker_topic': 'ld_proc/bboxs'},
            {'tracklet_topic': 'ld_proc/tracklets'}
        ]
    )

    lidar_tracker = Node(
        package='obj_tracker',
        executable='object_tracker',
        name='learned_lidar_tracker',
        output='screen',
        parameters=[
            {'detection_topic': 'ld_proc/dets'},
            {'det_pub_topic': 'ld_proc/tracks'},
            {'marker_pub_topic': 'ld_proc/id_markers'}
        ]
    )

    lidar_tracker_viz = Node(
        package='obj_tracker',
        executable='tracker_bbox_viz',
        name='lidar_track_viz',
        output='screen',
        parameters=[
            {'topic_name': 'ld_proc/tracks'},
            {'bbox_marker_topic': 'ld_proc/track_bboxs'},
            {'tracklet_topic': 'ld_proc/tracklets'}
        ]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file]
    )

    launch_list = [
        lidar_perception_node,
        lidar_viz,
        lidar_tracker,
        lidar_tracker_viz,
        rviz_node
    ]

    return LaunchDescription(launch_list)
