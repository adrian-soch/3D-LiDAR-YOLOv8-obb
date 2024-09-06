# 3D-LiDAR-YOLOv8-obb
3D LiDAR Object Detection using YOLOv8-obb (oriented bounding box). 2 scenarios were tested, the A9-Intersection dataset [1] and the ubiquitous KITTI dataset. The LiDAR pointclouds are converted into in a Bird'e-Eye-View image [2]. The YOLOv8-obb [3] model is used to predict bounding boxes and classes in the BEV image. A separate ROS node for tracking the detections is provided, it is based on SORT [4], and uses rotated bounding boxes.

**Example output from A9 Intersection dataset**
![Screenshot from 2024-09-06 16-07-45](https://github.com/user-attachments/assets/506fc6a1-67ca-419a-93e9-c630f48232ce)

**Example BEV Psuedo Image from KITTI dataset**
![000021](https://github.com/user-attachments/assets/070227a7-050e-42a5-b5c3-1303b675c0b4)


## TODO
- [ ] Seperate the ros package from scripts for creating training data
- [x] Add gif/image
- [x] Add support for KITTI data
- [x] Add example ROS bag
- [x] Add documentation
- [x] Create branch for Ubuntu 22/ROS2 Humble


# Install
Tested on:

| Ubunbtu 22 |
|:-------------:|
|  i7-11800H @ 2.30GHz × 16|
|   32 GB RAM   |
|  NVIDIA Quadro T1200 |
| CUDA Version: 11.8? - just let apt decide what it installs |

> If you prefer to use ROS 2 Galctic, a few changes in the code are required (vision_msgs/msg/Detection2D has a slightly different definition)
1. **ROS 2 Humble**: Follow https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html.
2. Clone repo: 
    ```
    cd <ROS2_WS>/src
    git clone --recurse-submodules https://github.com/adrian-soch/cam_lidar_tools.git
    ```
3. Run rosdep tool to automatically download ROS dependencies `rosdep install --from-paths . -y --ignore-src`.
4. `sudo apt install nvidia-cudnn`
5. Python deps:
    ```
    pip3 install ros2-numpy==0.0.4 \
    transforms3d==0.4.1 \
    ultralytics==8.2.2 \
    open3d==0.18.0 \
    filterpy==1.4.5 \
    lap==0.4.0 \
    shapely>=2.0.4 \
    numpy==1.23.5
    ```

    > Always source the ROS2 install and the local ros2 worksapce via `source /opt/ros/humble/setup.bash` and `. install/setup.bash` respectively.

## Usage

1. Download a sample rosbag here: https://drive.google.com/drive/folders/1rVnvbeyFYvG2HnFULadnT8nWH69EnOCm?usp=sharing. (Or create your own)
2. **OPTIONAL** Export the PyTorch Model to TensorRT for faster inference, see [here](#exporting-model-to-tensorrt-spped-up-inference-for-nvidia).
3. Build and run the nodes, see the example launch file for details.

### Exporting model to TensorRT (Spped up inference for Nvidia)

Assuming the `ultralytics` package is installed, run this command with the PyTorch weights to compile an `.engine` file for faster inference speeds on Nvidia GPUs.

```
yolo export model=yolo8n-obb_range.pt format=engine imgsz=1024 half=True simplify=True
```

## References
- [1]
    ```
    @inproceedings{zimmer2023tumtrafintersection,
            title={TUMTraf Intersection Dataset: All You Need for Urban 3D Camera-LiDAR Roadside Perception},
            author={Zimmer, Walter and Cre{\ss}, Christian and Nguyen, Huu Tung and Knoll, Alois C},
            booktitle={2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)},
            pages={1030--1037},
            year={2023},
            organization={IEEE}
        }
    ```
- [2]
    ```
    @misc{Super-Fast-Accurate-3D-Object-Detection-PyTorch,
    author =       {Nguyen Mau Dung},
    title =        {{Super-Fast-Accurate-3D-Object-Detection-PyTorch}},
    howpublished = {\url{https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection}},
    year =         {2020}
    }
    ```
- [3]
    ```
    @software{Jocher_Ultralytics_YOLO_2023,
    author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
    license = {AGPL-3.0},
    month = jan,
    title = {{Ultralytics YOLO}},
    url = {https://github.com/ultralytics/ultralytics},
    version = {8.0.0},
    year = {2023}
    }
    ```
- [4]
    ```
    @inproceedings{Bewley2016_sort,
    author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
    booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
    title={Simple online and realtime tracking},
    year={2016},
    pages={3464-3468},
    keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
    doi={10.1109/ICIP.2016.7533003}
    }
    ```