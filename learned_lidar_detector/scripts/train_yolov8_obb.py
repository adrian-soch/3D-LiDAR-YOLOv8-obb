from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')

# Train the model
results = model.train(data='kitti_bev.yaml', epochs=80, imgsz=808, close_mosaic=0, batch=0.7,
                      patience=10, plots=True, cache=True, box=7.5, cls=0.5, dfl=0.7,
                      single_cls=False, scale=0.0, hsv_s=0.01, hsv_v=0.01,
                      degrees=0.05, crop_fraction=0.05, mosaic=0.0)
