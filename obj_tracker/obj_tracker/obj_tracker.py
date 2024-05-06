"""
 * @file obj_tracker.py
 * @brief Perform 2D Object tracking
 * @author Adrian Sochaniwsky (sochania@mcmaster.ca)
 * @version 0.1
 * @date 2023-03-29

    NOTE: Default is using the oriented bounding box detections,
        This can be changed to axis aligned detections

    NOTE: Relative imports only work when running via ROS
        to run via python the relative import must be removed.
        Relative import is `.` in front of the imported module
 *
 * @copyright Copyright (c) 2023
"""
# fmt: off
# This limits CPU usage
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from .obj_tracker_utils import *

import time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
# fmt: on


class ObjectTracker(Node):

    def __init__(self):
        super().__init__('object_tracker')

        # Get the topic name from the ROS parameter server
        self.world_frame = self.declare_parameter(
            'world_frame', 'map').get_parameter_value().string_value

        det_pub_topic = self.declare_parameter(
            'det_pub_topic', 'lidar_proc/tracks').get_parameter_value().string_value

        marker_pub_topic = self.declare_parameter(
            'marker_pub_topic', 'lidar_proc/track_markers').get_parameter_value().string_value

        # Current default is aa_detections for axis algined and o_detections for oriented detections
        detection_topic = self.declare_parameter(
            'detection_topic', '/lidar_proc/detections_with_class').get_parameter_value().string_value
        self.isOBB = self.declare_parameter(
            'isOBB', True).get_parameter_value().bool_value

        if self.isOBB:
            from .sort import sort_rotated_bbox as s
        else:
            from .sort import sort as s

        # create instance of SORT
        self.tracker = s.Sort(
            max_age=3, min_hits=3, iou_threshold=0.01, dt=0.1, output_unmatched=False)

        self.subscription = self.create_subscription(
            Detection3DArray,
            detection_topic,
            self.callback,
            1)
        self.subscription  # prevent unused variable warning

        self.track_publisher_ = self.create_publisher(
            Detection3DArray, det_pub_topic, 2)
        self.marker_publisher_ = self.create_publisher(
            MarkerArray, marker_pub_topic, 2)

        self.get_logger().info('Tracker Module initialized.')

    def callback(self, msg):
        """Takes new detections and updates the tracker.
        Execution time must be < detection publish rate

        Args:
            msg (vision_msgs/Detection3DArray)
        """
        t1 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

        # Convert detections to np.ndarray for the tracker
        detections = detection3DArray2Numpy(msg.detections, self.isOBB)

        # update SORT with detections
        track_ids = self.tracker.update(detections)

        # Create and Publish 3D Detections with Track IDs
        track_msg_arr = createDetection3DArr(track_ids, msg.header, self.isOBB)
        self.track_publisher_.publish(track_msg_arr)

        # Create and publish Text Marker Array
        m_arr = self.track2MarkerArray(track_ids, msg.header.stamp, self.isOBB)
        self.marker_publisher_.publish(m_arr)

        # Print execution time to ros log
        t2 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        self.get_logger().info('Tracked {:4d} objects in {:.1f} msec.'.format(
            len(m_arr.markers), (t2-t1)*1000))

    def track2MarkerArray(self, track_ids, stamp, isOBB) -> MarkerArray:
        """Create ROS 2 markers for track results

        Args:
            track_ids (np.ndarray): Tracker results
            stamp (ros timestamp): time stamp of the message
            isOBB (bool): Is oriented or axis-aligned bbox

        Returns:
            MarkerArray: array of markers for rviz
        """
        m_arr = MarkerArray()
        idx = 0
        for trk in track_ids:
            marker = Marker()
            marker.id = idx
            marker.header.stamp = stamp
            marker.header.frame_id = self.world_frame
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.scale.z = 0.8  # height of `A` in meters

            if isOBB:
                id_str = str(int(trk[6]))
            else:
                id_str = str(int(trk[4]))

            marker.text = id_str
            marker.pose.position.x = trk[0]
            marker.pose.position.y = trk[1]

            marker.lifetime = Duration(seconds=0.1).to_msg()

            marker.color.a = 1.0
            marker.color.g = 0.8
            marker.color.b = 0.6

            idx += 1
            m_arr.markers.append(marker)

        return m_arr


def main(args=None):
    rclpy.init(args=args)
    object_tracker = ObjectTracker()
    rclpy.spin(object_tracker)

    object_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
