from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n-obb.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='./configs/a9_bev.yaml', epochs=30, imgsz=1024, close_mosaic=0, batch=4)
