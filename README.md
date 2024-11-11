# Tennis-Analysis-YOLO-PyTorch

## About
This project implements a real-time analysis of tennis match videos, tracking players and ball movements using YOLOv5. The system detects player positions and the ball, maps them to a mini-court overlay, and computes key statistics such as shot speed and player speed.

## Features
* **Detection and Tracking:** YOLOv5 detects players and ball; Roboflow YOLO model enhances ball detection
* **Keypoint and Court Detection:** Trained YOLOv5 on player keypoints, ResNet on court keypoints
* **Mini-Court Overlay:** Maps player and ball movements onto a mini-court displayed in the video corner
* **Player and Ball Stats:** Calculates ball speed and player speed once ball direction changes

## Methodology
* **Object Detection:** Detect players and the ball in each frame using YOLOv5
* **Improved Ball Detection:** Integrated a pre-trained YOLO model from Roboflow to enhance ball detection in difficult frames
* **Court Keypoints Prediction:** Trained a ResNet model on tennis court images and used it to predict court keypoints
* **Mini-Court Visualization and Stats Calculation:**
  - Converted bounding box coordinates to mini-court coordinates
  - Calculated shot speed and player speed based on direction changes of the ball

## Tech Stack
* YOLOv5: Real-time object detection
* Roboflow: Enhanced YOLO model for robust ball tracking
* ResNet: Custom-trained for tennis court keypoints detection
* OpenCV: Visualizations and mini-court overlay drawing
