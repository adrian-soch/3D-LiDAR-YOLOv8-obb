"""
This node subscribes to a Detection3DArray which stores tracker results.
It will convert the Detection3D object into a csv entry and save the csv file

Saves in the MOTChallenge 2D Format, using the 2D ground plane

This was created to save the results for offline evaluation.
"""
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

FOLDER_PATH = '/home/adrian/dev/metrics/SORT_Results'


class MotEntry:
    """Contains the values and helper functions for the detections in MOT Challenge format
        See comment at the top of the file for more details.
    """

    def __init__(self, frame, id=None, bb_left=None, bb_top=None, bb_width=None, bb_height=None, conf=-1):
        self.frame = frame
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.x = -1
        self.y = -1
        self.z = -1

    def toStr(self):
        return "{},{},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{}".format(
            self.frame, self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height,
            self.conf, self.x, self.y, self.z)


def create_folder_path(file_path):
    """
    Creates the folders in the given file path if they don't already exist.
    """
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        # Get the topic name from the ROS parameter server
        topic_name = self.declare_parameter(
            'topic_name', '/lidar_proc/tracks').get_parameter_value().string_value

        # Subscribe to the detection messages
        self.subscription = self.create_subscription(
            Detection3DArray, topic_name, self.detection_callback, 5)

        # Get the current date and time
        now = datetime.now()
        time = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.out_path = os.path.join(FOLDER_PATH, time + '_tracker_output.txt')
        create_folder_path(self.out_path)

        self.frame_count = 0

    def detection_callback(self, msg):
        self.frame_count += 1

        # Hold all the tracks in a frame
        tracks = []

        # Loop through each detection
        for detection in msg.detections:
            # Get the position and size of the detection
            pos = detection.bbox.center.position
            size = detection.bbox.size

            result = detection.results[0]
            id = int(result.hypothesis.score)

            entry = MotEntry(self.frame_count, id,
                             bb_left=pos.y - size.y/2.0,
                             bb_top=pos.x - size.x/2.0,
                             bb_width=size.y,
                             bb_height=size.x)
            tracks.append(entry)

        # Print to txt file
        with open(self.out_path, "a") as f:
            for trk in tracks:
                f.write(trk.toStr() + "\n")


def main(args=None):
    rclpy.init(args=args)
    detector_node = DetectorNode()
    rclpy.spin(detector_node)
    detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
