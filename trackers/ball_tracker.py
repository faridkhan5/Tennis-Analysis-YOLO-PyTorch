from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_detections):
        """detects ball in every frame by interpolating the missing bbox coordinates
        Returns:
            dict: imputed ball bbox coords
        """
        ball_positions = [x.get(1, []) for x in ball_detections]

        # convert list to pandas df to interpolate the missing vals
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # interpolate missing vals
        df_ball_positions = df_ball_positions.interpolate(method='polynomial', order=3)
        
        # since default direction is forward, we need to fill the 1st row
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """detects the ball bbox coords of each frame
        Returns:
            list of dicts: [{id1: [bbox_coords]}, ...,
                            {id1: [bbox_coords]}]
        """
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

    def get_ball_hit_frames(self, ball_detections):
        '''outputs the indices of the frames where the ball has been hit'''
        ball_positions = [x.get(1, []) for x in ball_detections]

        # convert list to pandas df to interpolate the missing vals
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # get smooth detections in all frames
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        # reduce the effect of outliers
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()  # subtracts 2 consecutive rows from each other

        # init a new col to track frames where ball is hit
        df_ball_positions['ball_hit'] = 0
        
        # atleast for 25 frames ball moves in one direction (increasing or decreasing)
        min_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions) - int(min_change_frames_for_hit*1.2)):
            negative_pos_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            positive_pos_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

            # count pos changes in one direction
            if negative_pos_change or positive_pos_change:
                pos_change_count = 0
                for nxt_frame_idx in range(i+1, i + int(min_change_frames_for_hit * 1.2) + 1):
                    negative_pos_change_nxt_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[nxt_frame_idx] < 0
                    positive_pos_change_nxt_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[nxt_frame_idx] > 0

                    if negative_pos_change and negative_pos_change_nxt_frame:
                        pos_change_count += 1
                    elif positive_pos_change and positive_pos_change_nxt_frame:
                        pos_change_count += 1

                if pos_change_count > min_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        ball_hit_frames_idx = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return ball_hit_frames_idx

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
                cv2.putText(frame, f"Ball ID: {tracking_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 2)
                # bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
    