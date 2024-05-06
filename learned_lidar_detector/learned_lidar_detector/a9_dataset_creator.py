'''
This script converts the A9 data into BEV psuedo images
based on the config file values
'''

# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import cv2
from glob import glob
import json
import os
import open3d as o3d
import numpy as np
from pypcd4 import PointCloud
import time
from typing import List
import random

from configs import a9_config as cfg
from lidar_2_bev import radius_outlier_removal, transform_pc, array_to_image, euler_from_quaternion


class A9LidarBevCreator():
    def __init__(self, input_list: List[str], useIntensity=False):
        """Create bird's-eye-view (BEV) psuedo images from LiDAR point clouds

        This is custom for A9 (https://innovation-mobility.com/en/project-providentia/a9-dataset/) format data.

        Args:
            input_list (List[str]): Folders with lidar point clouds (from A9 dataset)
            useIntensity (Bool): Use intensity as the 3rd channel of the image, if false, range from sensor is used.
        """
        print('Getting files from path.')
        self.lidar_list = []
        for seq_path in input_list:
            self.lidar_list += (get_files(seq_path, 'pcd'))

        self.useIntensity = useIntensity
        random.seed(69)
        print(f'Found {len(self.lidar_list)} data samples.')

    def create_yolo_obb_dataset(self, output_path: str, test_fraction=0.2, val_fraction=0.2, percent_background=0.0):
        """Create a data set with train, val, test folders. Format is in the YOLOv8-OBB format.
            The Train split proportion is automatically calculated from val and test.

        Args:
            output_path (str): Location to save the dataset.
            test_fraction (float, optional): Test set proportion. Defaults to 0.2.
            val_fraction (float, optional): Validation set proportion. Defaults to 0.2.
            percent_background (float, optional): Controls the proportion of empty images in the final dataset. Defaults to 0.0.
        """
        assert output_path is not None, "Output folder must be not be None"
        start_time = time.time()

        background_set = ['background.pcd'] * \
            int(percent_background*len(self.lidar_list))
        self.lidar_list += background_set

        val_size, test_size = int(
            len(self.lidar_list)*val_fraction), int(len(self.lidar_list)*test_fraction)
        train_size = len(self.lidar_list) - val_size - test_size
        assert train_size > 0, "Invalid train/val/test split."

        shuffle_list(self.lidar_list)

        folders = [('train', range(0, train_size)), ('val', range(
            train_size, val_size+train_size)), ('test', range(train_size+val_size, test_size+val_size+train_size))]
        for folder, split_range in folders:
            img_path = os.path.join(output_path, 'images', folder)
            gt_path = os.path.join(output_path, 'labels', folder)
            os.makedirs(img_path)
            os.makedirs(gt_path)

            for idx in split_range:
                bev_image, det_list = self.get_bev_and_label(idx=idx)
                det_list = self.__normalize_labels(det_list)

                if det_list is None:
                    print('   No lables for this frame.')

                file_name = str(idx).zfill(7)
                self.__write_label_file(
                    det_list, name=os.path.join(gt_path, f'{file_name}.txt'))
                save_img(os.path.join(
                    img_path, f'{file_name}.jpg'), array_to_image(bev_image))

                print(f'Saving image/label {idx} to {folder}')

        end_time = time.time()
        print(f'Processing time: {end_time - start_time:.2f} seconds')

    def demo_pc_to_image(self, debug=False):
        for idx in range(len(self.lidar_list)):
            self.get_bev_and_label(idx=idx, visualize=True, debug=debug)

    def get_bev_and_label(self, idx: int, lidar_frame_path=None, visualize=False, debug=False):
        pc_path = lidar_frame_path
        if lidar_frame_path is None:
            pc_path = self.lidar_list[idx]

        if pc_path == 'background.pcd':
            return np.zeros((3, cfg.BEV_HEIGHT, cfg.BEV_WIDTH)), None

        pc = self.get_pc(pc_path)
        pc = transform_pc(pc, cfg.lidar2ground)

        # Get detection bboxes in the ground plane
        gt_json = get_gt(pc_path)
        det_list = self.__convert_a9_json(
            gt_json, min_points=cfg.MIN_POINT_COUNT)
        det_list = self.__crop_labels(
            det_list, height=cfg.BEV_HEIGHT, width=cfg.BEV_WIDTH)

        # Convert to BEV
        bev_image = self.create_bev(pc, visualize=visualize, labels=det_list)

        if debug:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
            triad = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=10, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, triad])

        return bev_image, det_list

    @staticmethod
    def __normalize_labels(labels):
        if labels is None:
            return None
        labels = np.array(labels).astype(float)
        labels[:, [1, 3, 5, 7]] = labels[:, [1, 3, 5, 7]] / cfg.BEV_WIDTH
        labels[:, [2, 4, 6, 8]] = labels[:, [2, 4, 6, 8]] / cfg.BEV_HEIGHT

        # Clip labels
        labels[:, [1, 3, 5, 7]] = np.clip(labels[:, [1, 3, 5, 7]], 0, 1)
        labels[:, [2, 4, 6, 8]] = np.clip(labels[:, [2, 4, 6, 8]], 0, 1)
        return labels.tolist()

    @staticmethod
    def __write_label_file(data, name):
        with open(name, 'w') as file:
            if data is None:
                file.write('')
                return
            for d in data:
                entry = f'{int(d[0])} {d[1]} {d[2]} {d[3]} {d[4]} {d[5]} {d[6]} {d[7]} {d[8]} \n'
                file.write(entry)

    @staticmethod
    def __crop_labels(labels, height, width):
        labels = np.array(labels).astype(float)
        # All 4 points must be outside image to be deleted
        mask = ~((labels[:, 1] > width) | (labels[:, 2] > height) &
                 (labels[:, 3] > width) | (labels[:, 4] > height) &
                 (labels[:, 5] > width) | (labels[:, 6] > height) &
                 (labels[:, 7] > width) | (labels[:, 8] > height))

        out = labels[mask].tolist()
        return None if len(out) == 0 else out

    def __convert_a9_json(self, gt_json, min_points=0, debug=False):
        '''
        Covnert the A9 .json gt format into lists of yolo-obb format
        list = [[class, x1, y1, x2, y2, x3, y3, x4, y4]]
        '''
        frames = gt_json['openlabel']['frames']

        det_list = []

        # Only one frame per file
        frame_num = list(frames.keys())[0]
        objects = frames[frame_num]['objects']
        for item in objects.items():
            data = item[1]['object_data']

            # Skip if not enough points in bbox
            num_points = data['cuboid']['attributes']['num'][0]['val'] if data['cuboid'][
                'attributes']['num'][0]['name'] == 'num_points' else min_points
            if num_points < min_points:
                if debug:
                    print(f'    Skipping object with {num_points} points.')
                continue
            bbox = data['cuboid']['val']
            bbox_bev = self.__convert_labels(bbox)
            corners = self.__bbox3d_to_corners(bbox_bev)
            detection = [cfg.CLASS_NAME_TO_ID[data['type']]] + corners[:]
            det_list.append(detection)
        det_list = None if len(det_list) == 0 else det_list
        return det_list

    @staticmethod
    def __bbox3d_to_corners(bbox_bev):
        '''
        Convert a9 label cuboid format to 4 BEV ground plane corners
        '''
        x, y, z = bbox_bev[0], bbox_bev[1], bbox_bev[2]
        w, l, h = bbox_bev[3], bbox_bev[4], bbox_bev[5]
        yaw = bbox_bev[6]

        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)

        # Rotate the point and then add absolute position
        x1 = (l/2 * cos_yaw - w/2 * sin_yaw) + x
        y1 = (l/2 * sin_yaw + w/2 * cos_yaw) + y

        x2 = (-l/2 * cos_yaw - w/2 * sin_yaw) + x
        y2 = (-l/2 * sin_yaw + w/2 * cos_yaw) + y

        x3 = (-l/2 * cos_yaw + w/2 * sin_yaw) + x
        y3 = (-l/2 * sin_yaw - w/2 * cos_yaw) + y

        x4 = (l/2 * cos_yaw + w/2 * sin_yaw) + x
        y4 = (l/2 * sin_yaw - w/2 * cos_yaw) + y

        return [x1, y1, x2, y2, x3, y3, x4, y4]

    def __convert_labels(self, bbox):
        '''
        Convert A9 label data into the psuedo image pixel space
        '''
        x, y, z = bbox[0], bbox[1], bbox[2]
        qx, qy, qz, qw = bbox[3], bbox[4], bbox[5], bbox[6]
        w, l, h = bbox[7], bbox[8], bbox[9]

        yaw = euler_from_quaternion(qw, qx, qy, qz)

        yaw = -yaw
        y1 = int((x - cfg.boundary['minX']) / cfg.DISCRETIZATION)
        x1 = int((y - cfg.boundary['minY']) / cfg.DISCRETIZATION)
        z1 = z
        w1 = int(w / cfg.DISCRETIZATION)
        l1 = int(l / cfg.DISCRETIZATION)
        h1 = h

        return x1, y1, z1, w1, l1, h1, yaw

    def get_pc(self, lidar_file):
        pc = PointCloud.from_path(lidar_file)
        return pc.numpy()


    def create_bev(self, pointCloud: np.ndarray, visualize=False, labels=None) -> np.ndarray:
        '''
        Based on: https://github.com/maudzung/SFA3D

        create 3 channel image
        1) density
        2) height map
        3) Options: intensity, range image (dist2sensor)
        '''
        # Normalize pointcloud orientation and height, align road plane with x-y plane
        '''
        TODO add transform that rotates yaw angle for better cropping
        OR use a transformed cropbox that is the size of the RoI
        '''
        t1 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)

        # Crop point cloud based on paramters
        pointCloud = pointCloud[np.logical_not((pointCloud[:, 0] <= cfg.boundary['minX']) | (
            pointCloud[:, 0] > cfg.boundary['maxX']))]
        pointCloud = pointCloud[np.logical_not((pointCloud[:, 1] <= cfg.boundary['minY']) | (
            pointCloud[:, 1] > cfg.boundary['maxY']))]
        pointCloud = pointCloud[np.logical_not((pointCloud[:, 2] <= cfg.boundary['minZ']) | (
            pointCloud[:, 2] > cfg.boundary['maxZ']))]

        # Apply radius removal
        pointCloud = radius_outlier_removal(pointCloud, num_points=12, r=0.8)

        Height = cfg.BEV_HEIGHT + 1
        Width = cfg.BEV_WIDTH + 1

        if not self.useIntensity:
            # Use range instead
            pointCloud = pointCloud[:, :3]

            range = np.sqrt(pow(pointCloud[:, 0], 2.0) +
                            pow(pointCloud[:, 1], 2.0)).reshape(-1, 1)
            pointCloud = np.hstack([pointCloud, range])

        pointCloud[:, 0] = np.int_(
            np.floor(pointCloud[:, 0] / cfg.DISCRETIZATION) - Height*cfg.boundary['minX']/cfg.bound_size_x)
        pointCloud[:, 1] = np.int_(
            np.floor(pointCloud[:, 1] / cfg.DISCRETIZATION) - Width*cfg.boundary['minY']/cfg.bound_size_y)

        # sort-3times
        sorted_indices = np.lexsort(
            (-pointCloud[:, 2], pointCloud[:, 1], pointCloud[:, 0]))

        pointCloud = pointCloud[sorted_indices]
        _, unique_indices, unique_counts = np.unique(
            pointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
        PointCloud_top = pointCloud[unique_indices]

        # Height Map, Intensity Map & Density Map
        heightMap = np.zeros((Height, Width))
        rangeMap = np.zeros((Height, Width))
        densityMap = np.zeros((Height, Width))

        max_height = float(np.abs(cfg.boundary['maxZ'] - cfg.boundary['minZ']))
        heightMap[np.int_(PointCloud_top[:, 0]), np.int_(
            PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

        max_range = pointCloud[:, 3].max()
        rangeMap[np.int_(PointCloud_top[:, 0]), np.int_(
            PointCloud_top[:, 1])] = PointCloud_top[:, 3] / max_range

        normalizedCounts = np.minimum(
            1.0, np.log(unique_counts + 1) / np.log(64))
        densityMap[np.int_(PointCloud_top[:, 0]), np.int_(
            PointCloud_top[:, 1])] = normalizedCounts

        RGB_Map = np.zeros((3, Height - 1, Width - 1))
        RGB_Map[2, :, :] = densityMap[:cfg.BEV_HEIGHT, :cfg.BEV_WIDTH]
        RGB_Map[1, :, :] = heightMap[:cfg.BEV_HEIGHT, :cfg.BEV_WIDTH]
        RGB_Map[0, :, :] = rangeMap[:cfg.BEV_HEIGHT, :cfg.BEV_WIDTH]

        t2 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        print(f'Time (msec): {(t2-t1)*1000:.2f}')

        if visualize:
            image = array_to_image(RGB_Map)
            if labels is not None:
                image = annotate_bev(labels, image)

            cv2.imshow('Numpy Array as Image', image)
            if user_input_handler() < 0:
                exit(0)

        return RGB_Map


def user_input_handler() -> int:
    out = 0
    key = cv2.waitKey(0)

    # Escape key
    if (key == 27):
        out = -1
    return out


def annotate_bev(labels, image):
    for obj in labels:
        colour = cfg.colours[int(obj[0])]
        image = draw_r_bbox(obj[1:], image, colour)
    return image


def draw_r_bbox(corners, img, colour):
    '''
    Draw rotated bbox on the psuedo image
    '''
    corners_int = np.array(corners).astype(int)

    img = cv2.line(img, (corners_int[0], corners_int[1]),
                   (corners_int[2], corners_int[3]), colour, 1)
    img = cv2.line(img, (corners_int[2], corners_int[3]),
                   (corners_int[4], corners_int[5]), colour, 1)
    img = cv2.line(img, (corners_int[4], corners_int[5]),
                   (corners_int[6], corners_int[7]), colour, 1)
    img = cv2.line(img, (corners_int[6], corners_int[7]),
                   (corners_int[0], corners_int[1]), colour, 1)

    return img


def shuffle_list(list: list) -> None:
    random.shuffle(list)


def get_files(path, ext: str) -> list:
    assert os.path.isdir(path)
    files = glob(os.path.join(path, f'*.{ext}'))
    return sorted(files)


def save_img(filename: str, cv_image):
    cv2.imwrite(filename, cv_image)


def get_gt(lidar_file) -> dict:
    head, tail = os.path.split(lidar_file)
    name, _ = os.path.splitext(tail)
    gt_path = os.path.join(head.replace("/point_clouds/", "/labels_point_clouds/"),
                           name + '.json')
    with open(gt_path) as f:
        return json.load(f)


def main(args):
    seq_list = [
        '/home/adrian/dev/A9_images_and_points/a9_dataset_r02_s01/point_clouds/s110_lidar_ouster_north',
        '/home/adrian/dev/A9_images_and_points/a9_dataset_r02_s02/point_clouds/s110_lidar_ouster_north',
        '/home/adrian/dev/A9_images_and_points/a9_dataset_r02_s03/point_clouds/s110_lidar_ouster_north',
        '/home/adrian/dev/A9_images_and_points/a9_dataset_r02_s04/point_clouds/s110_lidar_ouster_north',
        '/home/adrian/dev/A9_images_and_points/a9_dataset_r02_s01/point_clouds/s110_lidar_ouster_south',
        '/home/adrian/dev/A9_images_and_points/a9_dataset_r02_s02/point_clouds/s110_lidar_ouster_south',
        '/home/adrian/dev/A9_images_and_points/a9_dataset_r02_s03/point_clouds/s110_lidar_ouster_south',
        '/home/adrian/dev/A9_images_and_points/a9_dataset_r02_s04/point_clouds/s110_lidar_ouster_south',
    ]

    lbc = A9LidarBevCreator(input_list=seq_list)

    # UNCOMMENT For Demo
    # lbc.demo_pc_to_image(debug=False)

    lbc.create_yolo_obb_dataset(
        output_path=args.output, val_fraction=0.2, test_fraction=0.25, percent_background=0.0)


# check if the script is run directly and call the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes the LiDAR images into BEV psuedo images in the YOLO-obb format.")
    parser.add_argument(
        "-o", "--output", help="The path where the results are saved.", default='/home/adrian/dev/A9_images_and_points/bev_lidar_range')
    args = parser.parse_args()
    main(args)
