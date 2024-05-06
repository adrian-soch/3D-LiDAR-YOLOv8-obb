"""
Functions for converting Point clouds into BEV psuedo images
"""
import numpy as np
import open3d as o3d


def fast_bev(pc: np.ndarray, im_width: int, im_height: int, discretization: float, min_x: float,
             max_x: float, min_y: float, max_y: float, min_z: float, max_z: float) -> np.ndarray:
    """Convert point cloud into a Birds-Eye-View psuedo image.

    Note: only square images supported.

    Args:
        pc (np.ndarray): Input Point Cloud
        im_width (int): Image width
        im_height (int): Image height
        discretization (float): Maters per pixel value
        min_x (float): Min pointcloud x distance from sensor (meters)
        max_x (float): Max pointcloud x distance from sensor (meters)
        min_y (float): Min pointcloud y distance from sensor (meters)
        max_y (float): Max pointcloud y distance from sensor (meters)
        min_z (float): Min pointcloud z distance from sensor (meters)
        max_z (float): Max pointcloud z distance from sensor (meters)

    Returns:
        np.ndarray: Output image
    """
    X_RANGE = max_x - min_x
    Y_RANGE = max_y - min_y

    range = np.sqrt(pow(pc[:, 0], 2.0) +
                    pow(pc[:, 1], 2.0)).reshape(-1, 1)
    pc = np.hstack([pc, range])

    # Apply radius removal
    pc = radius_outlier_removal(pc, num_points=12, r=0.8)

    HEIGHT = im_height + 1
    WIDTH = im_width + 1
    pc[:, :2] = np.int_(
        np.floor(pc[:, :2] / discretization) - np.array([WIDTH, HEIGHT]) * np.array(
            [min_x, min_y]) / np.array([X_RANGE, Y_RANGE])
    )

    sorted_indices = np.lexsort(
        (-pc[:, 2], pc[:, 1], pc[:, 0]))
    pc = pc[sorted_indices]

    # Getting unique points with counts
    _, unique_indices, unique_counts = np.unique(
        pc[:, :2], axis=0, return_index=True, return_counts=True
    )
    PointCloud_top = pc[unique_indices]

    # Pre-computed constants
    max_height = float(np.abs(max_z - min_z))
    max_range = pc[:, 3].max()

    # Maps initialization
    heightMap = np.zeros((HEIGHT, WIDTH))
    rangeMap = np.zeros((HEIGHT, WIDTH))
    densityMap = np.zeros((HEIGHT, WIDTH))

    # Filling the maps
    heightMap[tuple(PointCloud_top[:, :2].T.astype(int))
              ] = PointCloud_top[:, 2] / max_height
    rangeMap[tuple(PointCloud_top[:, :2].T.astype(int))
             ] = PointCloud_top[:, 3] / max_range
    densityMap[tuple(PointCloud_top[:, :2].T.astype(int))] = np.minimum(
        1.0, np.log(unique_counts + 1) / np.log(64))

    RGB_Map = np.zeros((3, HEIGHT - 1, WIDTH - 1))
    RGB_Map[2, :, :] = densityMap[:im_height, :im_width]
    RGB_Map[1, :, :] = heightMap[:im_height, :im_width]
    RGB_Map[0, :, :] = rangeMap[:im_height, :im_width]

    return RGB_Map


def array_to_image(array: np.ndarray) -> np.ndarray:
    """Convert [3 x N x M] array into an image

    Args:
        array (np.ndarray):

    Returns:
        np.ndarray: Image format
    """
    image = (array*255).astype(np.uint8)
    image = image.transpose((1, 2, 0))  # HWC to CHW
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return image


def transform_pc(pc: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform the [x, y, z] values of an array given a Homogenous transformation matrix [4x4].
    Assumes the first 3 coloumns are the x, y, z values.

    Args:
        pc (np.ndarray): [3+n X M] pointcloud info
        transform (np.ndarray): 4x4 transformation matrix

    Returns:
        np.ndarray: Transformed pc
    """
    xyz1 = np.hstack(
        [pc[:, :3], np.ones((pc.shape[0], 1), dtype=np.float32)])
    xyz1 = np.matmul(transform, xyz1.T).T

    pc[:, :3] = xyz1[:, :3]
    return pc


def radius_outlier_removal(pc: np.ndarray, num_points=12, r=0.8) -> np.ndarray:
    """Remove outlier points. See o3d docuemntation for `remove_radius_outlier`.

    Args:
        pc (np.ndarray): Pointcloud
        num_points (int, optional): Number of points to consider. Defaults to 12.
        r (float, optional): Radius. Defaults to 0.8.

    Returns:
        np.ndarray: Array without outlier points.
    """
    pc = pc.T if pc.shape[1] > 9 else pc
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    _, ind = pcd.remove_radius_outlier(nb_points=num_points, radius=r)

    mask = np.zeros(pc.shape[0], dtype=bool)
    mask[ind] = True
    return pc[mask]


def euler_from_quaternion(qw, qx, qy, qz) -> float:
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)

    Note: only returns yaw about z axis, in radians
    """
    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(t3, t4)
