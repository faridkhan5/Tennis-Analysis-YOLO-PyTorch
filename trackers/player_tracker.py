from ultralytics import YOLO
import cv2
import pickle


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        # load saved player detections
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
            
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections


    def detect_frame(self, frame):
        """ tracks each person class object using a unique tracking id and finds out their respective bbox coords
        Returns:
            dict: {tracking_id of people: [bbox coords]}
        """
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names  # {id1: obj1, ..., id2: obj2}
        
        player_dict = {}
        """
        Results for Frame 1:
        {
            boxes: [
                {
                    'id': tensor([3]),               # Tracking ID for this object
                    'cls': tensor([0]),               # Class ID (e.g., 0 for "person")
                    'xyxy': tensor([[100, 200, 300, 400]]),  # Bounding box coordinates
                    'confidence': tensor([0.95])      # Confidence score
                },
                {
                    'id': tensor([4]),
                    'cls': tensor([0]),
                    'xyxy': tensor([[150, 250, 350, 450]]),
                    'confidence': tensor([0.92])
                }
            ],
            masks: None,  # Only present if segmentation model is used
            names: {0: "person", 1: "car"},  # Dictionary mapping class IDs to names
        }
        """
        for box in results.boxes:
            # convert tracking_id to int
            tracking_id = int(box.id.tolist()[0])
            # xyxy: [x1, y1, x2, y2]
                # (x1, y1) - top-left
                # (x2, y2) - bottom-right
            bbox_coords = box.xyxy.tolist()[0]
            obj_cls_id = box.cls.tolist()[0]
            obj_cls_name = id_name_dict[obj_cls_id]
            # only consider boxes with people
            if obj_cls_name == "person":
                player_dict[tracking_id] = bbox_coords
        
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        '''draws bbox around all person class objects in the video'''
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # draw bboxes
            for tracking_id, bbox_coords in player_dict.items():
                x1, y1, x2, y2 = bbox_coords
                # text
                cv2.putText(frame, f"Player ID: {tracking_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)
                # bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            output_video_frames.append(frame)
        
        return output_video_frames 

