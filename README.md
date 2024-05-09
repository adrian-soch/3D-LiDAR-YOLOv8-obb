# 3D-LiDAR-YOLOv8-obb
3D LiDAR Object Detection using YOLOv8-obb (oriented bounding box). The pre-trained weights were trained on the A9-Intersection dataset [1]. The YOLOv8-obb [2] model is used to predict bounding boxes and class in a Bird'e-Eye-View image created from a LiDAR point cloud [3]. A separate ROS node for tracking the detections is provided, it is based on SORT [4], and uses rotated bounding boxes.

## TODO
- [ ] Add example ROS bag
- [ ] Add documentation
- [ ] Add gif/image
- [ ] Create branch for Ubuntu 22/ROS2 Humble

## Usage

1. Install ROS 2 Galactic (for Ubuntu 20.04)
2. Clone this repo to your ROS 2 workspace
3. Clone https://github.com/adrian-soch/ros2_numpy to your ROS 2 workspace
4. Build and run the nodes, see the example launch file for details.

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
- [3]
    ```
    @misc{Super-Fast-Accurate-3D-Object-Detection-PyTorch,
    author =       {Nguyen Mau Dung},
    title =        {{Super-Fast-Accurate-3D-Object-Detection-PyTorch}},
    howpublished = {\url{https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection}},
    year =         {2020}
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