from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.predict('input_videos/image.png', save=True)