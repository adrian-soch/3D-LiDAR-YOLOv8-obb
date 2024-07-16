'''
This script converts the KITTI data into BEV psuedo images
and YOLO-OBB training labels based on the config file
'''
# fmt: off
# limit the number of cpus used by high performance libraries
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from lidar_2_bev import transform_pc, array_to_image
from configs import kitti_config as cfg
import random
import time
import numpy as np
import open3d as o3d
from glob import glob
import cv2
# fmt: off


class LidarBevCreator():
    def __init__(self, lidar_path: str, image_path: str, label_path: str, useIntensity=True):
        """Create bird's-eye-view (BEV) psuedo images from LiDAR point clouds

        Args:
            input_list (List[str]): Folder(s) with lidar point clouds
            useIntensity (Bool): Use intensity as the 3rd channel of the image, if false, range from sensor is used.
        """
        print('Getting files from path.')

        self.lidar_list = (get_files(lidar_path, 'bin'))
        self.image_path = image_path
        self.label_path = label_path

        self.useIntensity = useIntensity
        random.seed(69)
        assert len(self.lidar_list) > 0, "No point cloud files found."
        print(f'Found {len(self.lidar_list)} data samples.')

    def create_yolo_obb_dataset(self, output_path: str, test_fraction=0.2, val_fraction=0.2):
        """Create a data set with train, val, test folders. Format is in the YOLOv8-OBB format.
            The Train split proportion is automatically calculated from val and test.

        Args:
            output_path (str): Location to save the dataset.
            test_fraction (float, optional): Test set proportion. Defaults to 0.2.
            val_fraction (float, optional): Validation set proportion. Defaults to 0.2.
        """
        assert output_path is not None, "Output folder must be not be None"
        start_time = time.time()

        val_size, test_size = int(
            len(self.lidar_list)*val_fraction), int(len(self.lidar_list)*test_fraction)
        train_size = len(self.lidar_list) - val_size - test_size
        assert train_size > 0, "Invalid train/val/test split."

        folders = [('train', range(0, train_size)), ('val', range(
            train_size, val_size+train_size)), ('test', range(train_size+val_size, test_size+val_size+train_size))]
        for folder, split_range in folders:
            img_path = os.path.join(output_path, 'images', folder)
            gt_path = os.path.join(output_path, 'labels', folder)
            os.makedirs(img_path)
            os.makedirs(gt_path)

            for idx in split_range:
                bev_image = self.get_bev(idx=idx)
                det_list = self.get_label(idx=idx)

                norm_det_list = self.__normalize_labels(det_list)

                if norm_det_list is None:
                    print('   No lables for this frame.')

                file_name = str(idx).zfill(7)
                self.__write_label_file(
                    norm_det_list, name=os.path.join(gt_path, f'{file_name}.txt'))
                save_img(os.path.join(
                    img_path, f'{file_name}.jpg'), array_to_image(bev_image))

                print(f'Saving image/label {idx} to {folder}')

        end_time = time.time()
        print(f'Processing time: {end_time - start_time:.2f} seconds')

    def create_test_bevs(self, test_data:str, output_path:str, visualize=False):
        test_list = get_files(test_data, 'bin')
        assert not os.path.exists(output_path), "Output folder exists already"
        os.makedirs(output_path)

        for path in test_list:
            _, tail = os.path.split(path)
            idx, _ = os.path.splitext(tail)

            bev_image = self.get_bev(lidar_frame_path=path, idx=idx, visualize=visualize)
            file_name = str(idx).zfill(6)

            save_img(os.path.join(output_path, f'{file_name}.jpg'), array_to_image(bev_image))

            print(f'Saving bev image {idx} to {output_path}')

    def demo_pc_to_image(self, debug=False):
        for idx in range(len(self.lidar_list)):
            det_list = self.get_label(idx=idx)
            self.get_bev(idx=idx, visualize=True, labels=det_list, debug=debug)

    def label_3d_on_image(self, line_width=1):
        """For each image, project 3D bounding boxes to the image

        Args:
            line_width (int, optional): Width of bounding box line. Defaults to 2.
        """
        for idx in range(len(self.lidar_list)):
            pc_path = self.lidar_list[idx]

            # Get detection points in the image plane
            _, tail = os.path.split(pc_path)
            name, _ = os.path.splitext(tail)
            gt_path = os.path.join(self.label_path, name + '.txt')
            det_list = get_gt(gt_path, convertToLidar=True)
            detections = self.get_image_points(det_list)

            img_path = os.path.join(self.image_path, name + '.png')
            img = cv2.imread(img_path)

            for det in detections:
                colour = cfg.colours[int(det['class'])]
                corners = det['points'].astype(int)

                # corners_int = np.array(corners).astype(int)
                img = cv2.line(img, (corners[0]), (corners[1]), colour, line_width)
                img = cv2.line(img, (corners[1]), (corners[2]), colour, line_width)
                img = cv2.line(img, (corners[2]), (corners[3]), colour, line_width)
                img = cv2.line(img, (corners[3]), (corners[0]), colour, line_width)

                img = cv2.line(img, (corners[4]), (corners[5]), colour, line_width)
                img = cv2.line(img, (corners[5]), (corners[6]), colour, line_width)
                img = cv2.line(img, (corners[6]), (corners[7]), colour, line_width)
                img = cv2.line(img, (corners[7]), (corners[4]), colour, line_width)

                img = cv2.line(img, (corners[0]), (corners[4]), colour, line_width)
                img = cv2.line(img, (corners[1]), (corners[5]), colour, line_width)
                img = cv2.line(img, (corners[2]), (corners[6]), colour, line_width)
                img = cv2.line(img, (corners[3]), (corners[7]), colour, line_width)

                img = cv2.line(img, (corners[0]), (corners[7]), colour, line_width)
                img = cv2.line(img, (corners[4]), (corners[3]), colour, line_width)

            image = cv2.putText(image, f'{name}.png', org=(20, 20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(255,255,255), thickness=2)
            cv2.imshow('Demo 3D detection on image.', img)
            if user_input_handler() < 0:
                exit(0)


    def get_image_points(self, bboxes):
        dets = []
        for bbox in bboxes:
            lidar_points = self.__bbox3d_to_corners(bbox[1:], is3D=True)
            image_points = self.project_to_image(np.array([lidar_points]), cfg.P_velo_to_image)
            if image_points.shape[0] != 8:
                continue
            dets.append({'class': bbox[0], 'points':image_points})
        return dets


    def get_label(self, idx:int, label_path=None):

        path = label_path
        if label_path is None:
            path = self.lidar_list[idx]

        # Create label name from LiDAR file name
        _, tail = os.path.split(path)
        name, _ = os.path.splitext(tail)
        gt_path = os.path.join(self.label_path, name + '.txt')

        # Get Ground truth results from label file
        det_list = get_gt(gt_path)
        det_list = self.__convert_labels(det_list)
        det_list = self.__crop_labels(
            det_list, height=cfg.BEV_HEIGHT, width=cfg.BEV_WIDTH)

        return det_list

    def get_bev(self, idx: int, lidar_frame_path=None, visualize=False, labels=None, debug=False):
        pc_path = lidar_frame_path
        if lidar_frame_path is None:
            pc_path = self.lidar_list[idx]

        pc = self.get_pc(pc_path)

        # Convert to BEV
        bev_image = self.create_bev(pc, visualize=visualize, labels=labels, idx=idx)

        if debug:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
            triad = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=10, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, triad])

        return bev_image

    @staticmethod
    def project_to_image(points : np.ndarray, proj_mat: np.ndarray) -> np.ndarray:
        image_points = transform_pc(np.array(points), proj_mat)
        image_points = image_points[:, :2]/image_points[:, 2:]

        # Filter out-of-bounds points
        image_width, image_height = 1242*1.25, 375*1.25  # KITTI image dimensions
        valid_indices = np.logical_and.reduce((
            image_points[:, 0] >= 0,
            image_points[:, 0] < image_width,
            image_points[:, 1] >= 0,
            image_points[:, 1] < image_height
        ))
        image_points = image_points[valid_indices]

        return image_points

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

    @staticmethod
    def __bbox3d_to_corners(bbox_bev, is3D=False):
        '''
        Convert a9 label cuboid format to 4 BEV ground plane corners
        '''
        x, y, z = bbox_bev[0], bbox_bev[1], bbox_bev[2]
        h, w, l = bbox_bev[3], bbox_bev[4], bbox_bev[5]
        if is3D:
            yaw = bbox_bev[6]-np.pi/2
        else:
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

        if not is3D:
            return [x1, y1, x2, y2, x3, y3, x4, y4]
        else:
            return [[x1, y1, z/2 - h/2],
                    [x2, y2, z/2 - h/2],
                    [x3, y3, z/2 - h/2],
                    [x4, y4, z/2 - h/2],
                    [x1, y1, z/2 + h/2],
                    [x2, y2, z/2 + h/2],
                    [x3, y3, z/2 + h/2],
                    [x4, y4, z/2 + h/2]]

    def __convert_labels(self, bboxes):
        '''
        Convert Label data into the psuedo image pixel space

        object_label = [cat_id, x, y, z, h, w, l, ry]
        output = [class, x1, y1, x2, y2, x3, y3, x4, y4]
        '''
        labels = []
        for bbox in bboxes:
            x, y, z = bbox[1], bbox[2], bbox[3]
            h, w, l, = bbox[4], bbox[5], bbox[6]
            yaw = -bbox[7]
            y1 = int((x - cfg.boundary['minX']) / cfg.DISCRETIZATION)
            x1 = int((y - cfg.boundary['minY']) / cfg.DISCRETIZATION)
            z1 = z
            w1 = int(w / cfg.DISCRETIZATION)
            l1 = int(l / cfg.DISCRETIZATION)
            h1 = h

            bbox = bbox[0], x1, y1, z1, h1, w1, l1, yaw
            bbox = [bbox[0]] + self.__bbox3d_to_corners(bbox[1:])
            labels.append(bbox)
        return labels

    def get_pc(self, lidar_file):
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def create_bev(self, pointCloud: np.ndarray, visualize=False, labels=None, idx=None) -> np.ndarray:
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

        # Shift Pointcloud to align with minZ=0 metres
        pointCloud[:, 2] -= cfg.boundary['minZ']

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
        intensityMap = np.zeros((Height, Width))
        densityMap = np.zeros((Height, Width))

        max_height = float(np.abs(cfg.boundary['maxZ'] - cfg.boundary['minZ']))
        heightMap[np.int_(PointCloud_top[:, 0]), np.int_(
            PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

        max_range = pointCloud[:, 3].max()
        intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(
            PointCloud_top[:, 1])] = PointCloud_top[:, 3] / max_range

        normalizedCounts = np.minimum(
            1.0, np.log(unique_counts + 1) / np.log(64))
        densityMap[np.int_(PointCloud_top[:, 0]), np.int_(
            PointCloud_top[:, 1])] = normalizedCounts

        RGB_Map = np.zeros((3, Height - 1, Width - 1))
        RGB_Map[2, :, :] = densityMap[:cfg.BEV_HEIGHT, :cfg.BEV_WIDTH]  # r_map
        RGB_Map[1, :, :] = heightMap[:cfg.BEV_HEIGHT, :cfg.BEV_WIDTH]  # g_map
        RGB_Map[0, :, :] = intensityMap[:cfg.BEV_HEIGHT, :cfg.BEV_WIDTH]  # b_map

        t2 = time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)
        print(f'Time (msec): {(t2-t1)*1000:.2f}')

        if visualize:
            image = array_to_image(RGB_Map)
            if labels is not None:
                image = annotate_bev(labels, image)

            idx = '' if idx is None else idx
            image = cv2.putText(image, f'{idx}.png', org=(20, 20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(255,255,255), thickness=2)
            cv2.imshow(f'BEV example', image)
            if user_input_handler() < 0:
                cv2.destroyAllWindows()
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
                   (corners_int[4], corners_int[5]), [173, 5, 247], 1)
    img = cv2.line(img, (corners_int[4], corners_int[5]),
                   (corners_int[6], corners_int[7]), colour, 1)
    img = cv2.line(img, (corners_int[6], corners_int[7]),
                   (corners_int[0], corners_int[1]), colour, 1)

    return img


def shuffle_list(list: list) -> None:
    random.shuffle(list)


def get_files(path:str, ext: str) -> list:
    """Get a sorted list of files matching the extension in the provided folder.

    Args:
        path (str): Path of data
        ext (str): Extension of data, Ex. 'txt'

    Returns:
        list: Sorted list of full filepaths to each item matching the extension
    """
    assert os.path.isdir(path), "Path is not a valid folder."
    files = glob(os.path.join(path, f'*.{ext}'))
    return sorted(files)


def save_img(filename: str, cv_image):
    cv2.imwrite(filename, cv_image)


def get_gt(label_path, convertToLidar=True):
    labels = []
    for line in open(label_path, 'r'):
        line = line.rstrip()
        line_parts = line.split(' ')
        obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
        cat_id = int(cfg.CLASS_NAME_TO_ID[obj_name])
        if cat_id <= -99:  # ignore Tram and Misc
            continue
        truncated = int(float(line_parts[1]))  # truncated pixel ratio [0..1]
        # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        occluded = int(line_parts[2])
        alpha = float(line_parts[3])  # object observation angle [-pi..pi]
        # xmin, ymin, xmax, ymax
        bbox = np.array([float(line_parts[4]), float(
            line_parts[5]), float(line_parts[6]), float(line_parts[7])])
        # height, width, length (h, w, l)
        h, w, l = float(line_parts[8]), float(
            line_parts[9]), float(line_parts[10])
        # location (x,y,z) in camera coord.
        x, y, z = float(line_parts[11]), float(
            line_parts[12]), float(line_parts[13])
        # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        ry = float(line_parts[14])

        object_label = [cat_id, x, y, z, h, w, l, ry]
        labels.append(object_label)

    if convertToLidar:
        labels = camera_to_lidar_box(labels, C2V_Mat=cfg.Tr_velo_to_cam_inv, hasClass=True)
    return labels

def camera_to_lidar_box(boxes, C2V_Mat, hasClass=False):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        if hasClass:
            cls, x, y, z, h, w, l, ry = box
        else:
            x, y, z, h, w, l, ry = box

        (x, y, z), h, w, l, rz = transform_pc(np.array([[x, y, z]]), C2V_Mat).tolist()[0], h, w, l, -ry #- np.pi / 2
        if hasClass:
            ret.append([cls, x, y, z, h, w, l, rz])
        else:
            ret.append([x, y, z, h, w, l, rz])
    return ret

def main(args):
    lidar_path = '/home/adrian/dev/kitti/data_object_velodyne/training/velodyne'
    image_path = '/home/adrian/dev/kitti/data_object_image_2/training/image_2'
    label_path = '/home/adrian/dev/kitti/data_object_label_2/training/label_2'

    lbc = LidarBevCreator(lidar_path=lidar_path,
                          image_path=image_path, label_path=label_path)

    # UNCOMMENT For Demo
    # lbc.demo_pc_to_image(debug=False)
    # lbc.label_3d_on_image()

    # lbc.create_yolo_obb_dataset(
        # output_path=args.output, val_fraction=0.78, test_fraction=0.2)

    lbc.create_test_bevs(test_data='/home/adrian/dev/kitti/data_object_velodyne/testing/velodyne/', output_path='/home/adrian/dev/kitti/test_608')


# check if the script is run directly and call the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processes the LiDAR images into BEV psuedo images in the YOLO-obb format.")
    parser.add_argument(
        "-o", "--output", help="The path where the results are saved.", default='/home/adrian/dev/kitti/kitti_bev/')
    args = parser.parse_args()
    main(args)
