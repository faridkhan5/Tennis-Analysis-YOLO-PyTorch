from ultralytics import YOLO
import cv2
import pickle
from utils import get_center_of_bbox, euclidean_distance

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """filters the 2 tennis players
        Returns:
            list: [{tracking_id_1: bbox_coords},
                    {tracking_id_2: bbox_coords}]
        """
        player_detections_first_frame = player_detections[0]
        chosen_two_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            # filter only the 2 tennis players in each frame
            filtered_player_dict = {tracking_id: bbox_coords for tracking_id, bbox_coords in player_dict.items() if tracking_id in chosen_two_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections
    
    def choose_players(self, court_keypoints, player_dict):
        """finds the 2 tennis players among all people in the frame
        Returns:
            list: tracking_ids of the 2 nearest people to the court
        """
        distances = []
        for tracking_id, bbox_coords in player_dict.items():
            player_center = get_center_of_bbox(bbox_coords)
            min_dist = float('inf')
            cumulative_dist = 0
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                curr_dist = euclidean_distance(player_center, court_keypoint)
                cumulative_dist += curr_dist
                #if curr_dist <= min_dist:
                    # shortest distance for a player to a court_keypoint among all other court_keypoints
                #   min_dist = curr_dist
            distances.append((tracking_id, cumulative_dist))
        distances.sort(key=lambda x: x[1])
        # choose top 2 shortest distances
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        # load saved player detections
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        print('frame detections len: ', len(frames))
        for frame in frames:

            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        print('player detections len: ', len(player_detections))
        return player_detections

    def detect_frame(self, frame):
        """ tracks each person class object using a unique tracking id and finds out their respective bbox coords
        Returns:
            dict: {tracking_id of person 1: [bbox coords], ..., 
                    tracking_id of person n: [bbox_coords]}
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