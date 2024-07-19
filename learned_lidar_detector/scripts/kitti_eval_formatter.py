"""
Formats test results into the KITTI format for evalulation.
"""
# fmt: off
# limit the number of cpus used by high performance libraries
import os

import argparse
import glob
from os import path
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path

HERE = Path(__file__).parent

sys.path.append(str(HERE / '../learned_lidar_detector'))
import configs.kitti_config as cfg
from lidar_2_bev import transform_pc
# fmt: on


class KittiEval():
    def __init__(self, bev_path: str, method: str, out_path: str, imgsz: int):
        self.method = method
        self.out_path = out_path
        self.bev_list = (get_files(bev_path, 'jpg'))
        self.imgsz = imgsz
        assert len(self.bev_list) > 0, "No files found."
        print(f'Found {len(self.bev_list)} data samples.')

        assert not os.path.exists(
            self.out_path), "Output folder exists already"
        os.makedirs(self.out_path)

    def get_model_output(self, model_path: str, conf=0.2, device='cuda:0'):
        model = YOLO(model_path, task='obb')

        batch_size = 1
        path_list = self.bev_list[:]
        while path_list:

            # Get batch of paths
            batch = []
            while path_list and len(batch) < batch_size:
                batch.append(path_list.pop(0))

            results = model(batch, device=device, imgsz=(
                self.imgsz), conf=conf, verbose=True, half=False)

            import time
            time.sleep(0.02)
            self.create_prediction_files(results)

    def create_prediction_files(self, results):
        for res in results:
            obb_results = res.obb.cpu().numpy()
            class_dict = res.names

            kitti_entries = []
            for det in obb_results:
                cls = class_dict[det.cls[0]]
                conf = det.conf[0]
                lidar_det = self.px_to_meters(det.xywhr[0])

                if lidar_det is not None:
                    kitti_entries.append(self.get_kitti_entry(
                        cls=cls, lidar_det=lidar_det, conf=conf))
                else:
                    kitti_entries.append(None)

            tail = os.path.split(res.path)[-1]
            file_name = os.path.splitext(tail)[0]

            file_path = path.join(self.out_path, f'{file_name}.txt')
            self.write_predicted_labels(file_path, kitti_entries)

    def get_kitti_entry(self, cls: str, lidar_det, conf: float) -> str:
        """Get kitti formatted detection

        Args:
            cls (str): Predicted Class
            lidar_det (tuple): (x, y, w, h, r)
        """
        truncated = -1
        occluded = -1
        alpha = -10
        h = self.height_from_class(cls)
        w, l = lidar_det[2], lidar_det[3]

        lidar_frame_coord = [lidar_det[0],
                             lidar_det[1], h/2 - 1.7]
        # 3D in camera coordinates
        [x, y, z] = transform_pc(
            np.array([lidar_frame_coord]), cfg.Tr_velo_to_cam).reshape(-1)

        # Get 3D bbox in image coordinates
        lidar_3d_corners = self.bbox3d_to_corners(
            lidar_frame_coord + [h, w, l, lidar_det[4]])
        image_3d_corners = self.project_to_image(
            np.array(lidar_3d_corners), cfg.P_velo_to_image).astype(int)

        if image_3d_corners.shape[0] != 8:
            return None
        min_x, max_x = min(image_3d_corners[:, 0]), max(image_3d_corners[:, 0])
        min_y, max_y = min(image_3d_corners[:, 1]), max(image_3d_corners[:, 1])

        bbox = [min_x, min_y, max_x, max_y]

        ry = lidar_det[4]

        if self.method == '2D':
            h, w, l = 0, 0, 0
            x, y, z = -1000, -1000, -1000
        elif self.method == 'BEV':
            h = 0

        return (f'{cls} {truncated} {occluded} {alpha} {bbox[0]} '
                f'{bbox[1]} {bbox[2]} {bbox[3]} {h:.3f} {w:.3f} {l:.3f} '
                f'{x:.3f} {y:.3f} {z:.3f} {ry:.3f} {conf:.3f}')

    def px_to_meters(self, xywhr: np.ndarray) -> np.ndarray:
        """Convert BEV psuedo image coordinates into meters

        Args:
            xywhr (np.ndarray): OBB from YOLO model

        Returns:
            np.ndarray: OBB in meters/radians
        """
        x, y, w, h, r = xywhr

        # Weird switcharoo for correct mapping, don't ask
        y = (y * cfg.DISCRETIZATION) + cfg.boundary['minX']
        x = (x * cfg.DISCRETIZATION_y) + cfg.boundary['minY']
        x, y = y, x

        w *= cfg.DISCRETIZATION_y
        h *= cfg.DISCRETIZATION_y

        return (float(x), float(y), float(h), float(w), float(r))

    @staticmethod
    def bbox3d_to_corners(bbox_bev):
        '''
        Convert center point to corner representation
        '''
        x, y, z = bbox_bev[0], bbox_bev[1], bbox_bev[2]
        h, w, l = bbox_bev[3], bbox_bev[4], bbox_bev[5]

        yaw = bbox_bev[6]-np.pi/2

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

        return [[x1, y1, z - h/2.0],
                [x2, y2, z - h/2.0],
                [x3, y3, z - h/2.0],
                [x4, y4, z - h/2.0],
                [x1, y1, z + h/2.0],
                [x2, y2, z + h/2.0],
                [x3, y3, z + h/2.0],
                [x4, y4, z + h/2.0]]

    @staticmethod
    def project_to_image(points: np.ndarray, proj_mat: np.ndarray) -> np.ndarray:
        image_points = transform_pc(np.array(points), proj_mat)
        image_points = image_points[:, :2]/image_points[:, 2:]

        # Filter out-of-bounds points
        # image_width, image_height = 1242*1.2, 375*1.2  # KITTI image dimensions
        # valid_indices = np.logical_and.reduce((
        #     image_points[:, 0] >= 0,
        #     image_points[:, 0] < image_width,
        #     image_points[:, 1] >= 0,
        #     image_points[:, 1] < image_height
        # ))
        # image_points = image_points[valid_indices]

        return image_points

    @staticmethod
    def height_from_class(cls: str) -> float:
        CLASS_NAME_TO_HEIGHT = {'Cyclist': 1.6,
                                'Truck': 3.2, 'Car': 1.7, 'Pedestrian': 1.9}
        return CLASS_NAME_TO_HEIGHT[cls]

    def write_predicted_labels(self, file_path, data):
        with open(file_path, 'w') as file:
            for entry in data:
                if entry is None:
                    file.write('')
                    return
                else:
                    file.write(entry + '\n')


def get_files(path, ext: str) -> list:
    assert os.path.isdir(path)
    files = glob.glob(os.path.join(path, f'*.{ext}'))
    return sorted(files)


def main(args):
    kiti_eval_creator = KittiEval(
        bev_path=args.bev_path, method=args.method, out_path=args.output, imgsz=args.imgsz)

    kiti_eval_creator.get_model_output(
        model_path='/home/adrian/dev/kitti/yolov8n-obb-p2.pt')


# check if the script is run directly and call the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Formats test results into the KITTI format for evalulation.")
    parser.add_argument(
        "-m", "--method", help="The evaluation method ['2D', 'BEV', '3D'].", default='BEV', type=str)
    parser.add_argument(
        "-l", "--bev_path", help="The lidar BEV psuedo image path.", default='/home/adrian/dev/kitti/kitti_bev_last/images/val/', type=str)
    parser.add_argument(
        "-o", "--output", help="The path where the results are saved.", default='/home/adrian/dev/kitti/bev_p21024_val_predictions/data', type=str)
    parser.add_argument(
        "--imgsz", help="The image size.", default=1024, type=int)
    
    args = parser.parse_args()

    if args.method.upper() not in ['2D', 'BEV', '3D']:
        exit("Not a valid evaluation method, must be in ['2D', 'BEV', '3D']")
    main(args)
