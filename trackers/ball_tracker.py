from ultralytics import YOLO
import cv2
import pickle


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        # load saved ball detections
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
            
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections


    def detect_frame(self, frame):
        """ detects a ball class object and finds out its bbox coords
        Returns:
            dict: {id: [bbox coords]}
        """
        result = self.model.predict(frame, conf=0.15)[0]
        
        ball_dict = {}
        for box in result.boxes:
            bbox_coords = box.xyxy.tolist()[0]
            ball_dict[1] = bbox_coords
        
        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        '''draws a bbox around the ball'''
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # draw bboxes
            for tracking_id, bbox_coords in ball_dict.items():
                x1, y1, x2, y2 = bbox_coords
                # text
                cv2.putText(frame, f"Ball ID: {tracking_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
                # bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames 

