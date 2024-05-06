"""
This node subscribes to a Detection3DArray which stores tracker results
It will publish the tracking results as marker array to be visualized in rviz

To see the result:
    1. open Rviz.
    2. add the marker topic that is published here.
    3. set the frame in rviz to the one that the markers are set to.
"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

import random


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        # Get the topic name from the ROS parameter server
        topic_name = self.declare_parameter(
            'topic_name', '/lidar_proc/tracks').get_parameter_value().string_value

        bbox_marker_topic = self.declare_parameter(
            'bbox_marker_topic', 'lidar_proc/tracker_bboxs').get_parameter_value().string_value

        tracklet_topic = self.declare_parameter(
            'tracklet_topic', '/lidar_proc/tracklets').get_parameter_value().string_value

        # Subscribe to the detection messages
        self.subscription = self.create_subscription(
            Detection3DArray, topic_name, self.detection_callback, 5)

        # Publish the marker messages
        self.bbox_publisher = self.create_publisher(
            MarkerArray, bbox_marker_topic, 5)
        self.tracklet_publisher = self.create_publisher(
            MarkerArray, tracklet_topic, 5)

        self.get_logger().info('Starting Tracker BBox Visualization')

        # Index to prevent us from overwriting a tracklet that we
        # still want to see in Rviz
        self.tracklet_idx = 0

    def detection_callback(self, msg):
        # Create a MarkerArray message
        track_bbox_array = MarkerArray()
        tracklet_array = MarkerArray()

        # Loop through each detection
        idx = 0
        for detection in msg.detections:

            # Get track ID
            trk_id = detection.results[0].hypothesis.score

            # Get a colour based on the track ID
            # set the seed value so the same colour is applied
            # to the same track each time
            random.seed(trk_id)
            r = random.random()
            g = random.random()
            b = random.random()

            # Get the position and size of the detection
            pos = detection.bbox.center.position
            size = detection.bbox.size
            quat = detection.bbox.center.orientation

            # Create a cube marker for the detection
            marker = Marker()
            marker.id = idx
            marker.header = msg.header
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = pos
            marker.scale = size

            marker.pose.orientation = quat

            marker.color.a = 0.6
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.lifetime = Duration(seconds=0.1).to_msg()

            # Create a sphere marker for the track
            tracklet = Marker()
            tracklet.id = self.tracklet_idx
            tracklet.header = msg.header
            tracklet.type = Marker.SPHERE
            tracklet.action = Marker.ADD
            tracklet.pose.position = pos
            tracklet.scale.x = 0.3
            tracklet.scale.y = 0.3
            tracklet.scale.z = 0.3
            tracklet.color.a = 0.6
            tracklet.color.r = r
            tracklet.color.g = g
            tracklet.color.b = b
            tracklet.lifetime = Duration(seconds=1.5).to_msg()

            # Add the marker to the array
            idx += 1
            self.tracklet_idx = (self.tracklet_idx + 1) % 400000
            track_bbox_array.markers.append(marker)
            tracklet_array.markers.append(tracklet)

        # Publish the marker array
        self.bbox_publisher.publish(track_bbox_array)
        self.tracklet_publisher.publish(tracklet_array)


def main(args=None):
    rclpy.init(args=args)
    detector_node = DetectorNode()
    rclpy.spin(detector_node)
    detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
