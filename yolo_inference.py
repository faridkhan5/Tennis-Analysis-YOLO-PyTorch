from ultralytics import YOLO

model = YOLO('models/params/last.pt')

result = model.predict('input_videos/input_video.mp4', save=True)