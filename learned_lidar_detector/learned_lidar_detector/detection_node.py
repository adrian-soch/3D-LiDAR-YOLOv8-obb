# fmt: off
# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter, ParameterType
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D
from vision_msgs.msg import ObjectHypothesisWithPose

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from scipy.spatial.transform import Rotation as R
import time
from ultralytics import YOLO

import ros2_numpy
from learned_lidar_detector.lidar_2_bev import transform_pc, array_to_image, fast_bev
import learned_lidar_detector.configs.a9_config as cfg
# fmt: on


class LidarProcessingNode(Node):
    def __init__(self):
        super().__init__('lidar_processing_node')

        self.declare_parameter('lidar_topic', 'points')
        self.declare_parameter('detection_topic', 'ld_proc/dets')
        self.declare_parameter('pc_topic', 'ld_proc/cloud')
        self.declare_parameter('confidence', 0.6)
        self.declare_parameter(
            'model_path', Parameter.Type.STRING)

        lidar_topic = self.get_parameter(
            'lidar_topic').get_parameter_value().string_value
        detection_topic = self.get_parameter(
            'detection_topic').get_parameter_value().string_value
        pc_topic = self.get_parameter(
            'pc_topic').get_parameter_value().string_value
        confidence = self.get_parameter(
            'confidence').get_parameter_value().double_value
        model_path = self.get_parameter(
            'model_path').get_parameter_value().string_value

        self.subscription = self.create_subscription(
            PointCloud2,
            lidar_topic,
            self.lidar_callback,
            1)
        self.det3d_pub = self.create_publisher(
            Detection3DArray,
            detection_topic,
            1)
        self.pc_pub = self.create_publisher(
            PointCloud2,
            pc_topic,
            1)

        self.model = YOLO(model_path, task='obb')
        '''
        USE pre-transformed cloud.
        '''
        self.transform = self.get_transform_param()
        self.confidence = confidence
        self.device = 'cuda:0'  # or 'cpu'

        self.debug = False
        self.world_frame = "map"
        self.get_logger().info('3D BEV LiDAR detection initialized.')

    def get_transform_param(self) -> np.ndarray:

        self.declare_parameter('lidar2world_transform.quaternion', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter('lidar2world_transform.translation', descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE_ARRAY))

        l2g_quat = self.get_parameter(
            'lidar2world_transform.quaternion').get_parameter_value().double_array_value

        l2g_translation = self.get_parameter(
            'lidar2world_transform.translation').get_parameter_value().double_array_value

        # x,y,z,w
        l2g_rotation = R.from_quat(
            (l2g_quat[1], l2g_quat[2], l2g_quat[3], l2g_quat[0])).as_matrix()

        l2g_transformation = np.eye(4)
        l2g_transformation[:3, :3] = l2g_rotation
        l2g_transformation[:3, 3] = l2g_translation

        return l2g_transformation

    def lidar_callback(self, msg):
        t1 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        pc = ros2_numpy.numpify(msg)

        # Only retain x, y, z, intensity
        pc = pc.reshape(pc.size)
        # pc = pc[['x', 'y', 'z', 'intensity']]
        pc = pc[['x', 'y', 'z']]

        # Convert to unstructured so we can operate easier
        pc = structured_to_unstructured(pc)
        pc = transform_pc(pc, self.transform)

        # Crop point cloud based on paramters
        pc = pc[np.logical_not((pc[:, 0] <= cfg.boundary['minX']) | (
            pc[:, 0] > cfg.boundary['maxX']))]
        pc = pc[np.logical_not((pc[:, 1] <= cfg.boundary['minY']) | (
            pc[:, 1] > cfg.boundary['maxY']))]
        pc = pc[np.logical_not((pc[:, 2] <= cfg.boundary['minZ']) | (
            pc[:, 2] > cfg.boundary['maxZ']))]

        t2 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

        bev_img = fast_bev(pc=pc, im_width=cfg.BEV_WIDTH, im_height=cfg.BEV_HEIGHT, discretization=cfg.DISCRETIZATION,
                           min_x=cfg.boundary['minX'], max_x=cfg.boundary['maxX'], min_y=cfg.boundary['minY'], max_y=cfg.boundary['maxY'],
                           min_z=cfg.boundary['minZ'], max_z=cfg.boundary['maxZ'])

        t3 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

        results = self.model(array_to_image(bev_img), device='cuda:0',  imgsz=(
            1024), conf=self.confidence, verbose=False)
        obb_results = results[0].obb.cpu().numpy()
        class_dict = results[0].names

        t4 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

        if self.debug:
            from PIL import Image
            for r in results:
                im_bgr = r.plot()  # BGR-order numpy array
                im_rgb = Image.fromarray(
                    im_bgr[..., ::-1])  # RGB-order PIL image
                r.show()

        detection_array = Detection3DArray()
        detection_array.header.frame_id = self.world_frame
        for det in obb_results:
            x, y, w, h, r = self.px_to_meters(det.xywhr[0])
            q = self.quaternion_from_euler(r)
            cls = class_dict[det.cls[0]]

            detection = Detection3D()
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = cls
            result.hypothesis.score = float(det.conf[0])
            detection.results.append(result)
            detection.id = str(int(det.cls[0]))

            detection.bbox.size.x = w
            detection.bbox.size.y = h
            detection.bbox.size.z = self.height_from_class(cls)

            detection.bbox.center.position.x = x
            detection.bbox.center.position.y = y
            detection.bbox.center.position.z = detection.bbox.size.z / 2.0

            detection.bbox.center.orientation.w = q[0]
            detection.bbox.center.orientation.x = q[1]
            detection.bbox.center.orientation.y = q[2]
            detection.bbox.center.orientation.z = q[3]

            detection_array.detections.append(detection)

        self.det3d_pub.publish(detection_array)
        self.publish_pc(self.pc_pub, pc, self.world_frame)

        t5 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        self.get_logger().info(
            f'Time (msec): pre {(t2-t1)*1000:.2f} bev {(t3-t2)*1000:.2f} ref {(t4-t3)*1000:.2f} pub {(t5-t4)*1000:.2f}')

    @staticmethod
    def height_from_class(cls: str) -> float:
        CLASS_NAME_TO_HEIGHT = {'BICYCLE': 1.6, 'BUS': 2.75, 'CAR': 2.0, 'EMERGENCY_VEHICLE': 2.3,
                                'MOTORCYCLE': 1.6, 'PEDESTRIAN': 1.9, 'TRAILER': 3.2, 'TRUCK': 3.2, 'VAN': 2.0, 'OTHER': 2.0}
        return CLASS_NAME_TO_HEIGHT[cls]

    def px_to_meters(self, xywhr: np.ndarray) -> np.ndarray:
        """Convert BEV psuedo image coordinates into meters

        Args:
            xywhr (np.ndarray): OBB from YOLO model

        Returns:
            np.ndarray: OBB in meters/radians
        """
        x, y, w, h, r = xywhr

        # Weird switcharoo for correct mapping, don't ask
        r *= -1.0
        y = (y * cfg.DISCRETIZATION) + cfg.boundary['minX']
        x = (x * cfg.DISCRETIZATION) + cfg.boundary['minY']
        x, y = y, x

        w *= cfg.DISCRETIZATION
        h *= cfg.DISCRETIZATION

        return (float(x), float(y), float(h), float(w), float(r))

    @staticmethod
    def quaternion_from_euler(yaw_rad: float):
        """
        Converts yaw to quaternion
        quat = [w, x, y, z]
        """
        cy = np.cos(yaw_rad * 0.5)
        sy = np.sin(yaw_rad * 0.5)
        cp = 1.0
        sp = 0.0
        cr = 1.0
        sr = 0.0

        q = [0] * 4
        q[0] = cy * cp * cr + sy * sp * sr
        q[1] = cy * cp * sr - sy * sp * cr
        q[2] = sy * cp * sr + cy * sp * cr
        q[3] = sy * cp * cr - cy * sp * sr

        return q

    @staticmethod
    def publish_pc(publisher, cloud, frame_id: str) -> None:
        pc = unstructured_to_structured(cloud, dtype=np.dtype(
            [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]))
        msg = ros2_numpy.msgify(PointCloud2, pc)
        msg.header.frame_id = frame_id
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
