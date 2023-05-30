from ultralytics import YOLO

model = YOLO('my_model.pt')

model.train(data='yolo-dc.yaml', workers=0, epochs=50, batch=16)
