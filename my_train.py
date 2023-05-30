from ultralytics import YOLO

model = YOLO('datasets/yolov8.yaml')

model.train(data='yolo-dc.yaml', workers=2, epochs=50, batch=16)
