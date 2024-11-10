import cv2
import numpy as np

import constants
from utils import (convert_pixel_distance_to_meters,
                   convert_meters_to_pixel_distance,
                   get_foot_position,
                   get_closest_keypoint_index,
                   get_bbox_height,
                   measure_xy_distance,
                   get_center_of_bbox,
                   euclidean_distance,
                   keypoints_to_idx,
                   midpoint)


class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        # padding in the frame
        self.buffer = 50
        # padding in the canvas inside the frame
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_minicourt_position()
        self.set_court_drawing_keypoints()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters, constants.DOUBLES_BASELINE_WIDTH, self.court_drawing_width)

    def set_court_drawing_keypoints(self):
        drawing_keypoints = [0] * 38
        # Doubles Court included -> (topleft, topright, bottomleft, bottomright)
        ## point i -> drawing_keypoints([2*i], [2*i+1])
        ## point 0
        drawing_keypoints[0], drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y)
        ## point 1
        drawing_keypoints[2], drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y)
        ## point 2
        drawing_keypoints[4] = int(self.court_start_x)
        drawing_keypoints[5] = self.court_start_y + self.convert_meters_to_pixels(constants.COURT_LENGTH)
        ## point 3
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_drawing_width
        drawing_keypoints[7] = self.court_start_y + self.convert_meters_to_pixels(constants.COURT_LENGTH)
        # Singles Court only -> (topleft, bottomleft, topright, bottomright)
        ## point 4
        drawing_keypoints[8] = drawing_keypoints[0] + self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_WIDTH)
        drawing_keypoints[9] = drawing_keypoints[1]
        ## point 5
        drawing_keypoints[10] = drawing_keypoints[8]
        drawing_keypoints[11] = drawing_keypoints[5]
        ## point 6
        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_WIDTH)
        drawing_keypoints[13] = drawing_keypoints[3]
        ## point 7
        drawing_keypoints[14] = drawing_keypoints[6] - self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_WIDTH)
        drawing_keypoints[15] = drawing_keypoints[7]
        # Service Box -> (topleft, topright, bottomleft, bottomright)
        ## point 8
        drawing_keypoints[16] = drawing_keypoints[8]
        drawing_keypoints[17] = drawing_keypoints[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_LENGTH)
        ## point 9
        drawing_keypoints[18] = drawing_keypoints[16] + self.convert_meters_to_pixels(constants.SINGLES_BASELINE_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17]
        ## point 10
        drawing_keypoints[20] = drawing_keypoints[10]
        drawing_keypoints[21] = drawing_keypoints[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_LENGTH)
        ## point 11
        drawing_keypoints[22] = drawing_keypoints[20] + self.convert_meters_to_pixels(constants.SINGLES_BASELINE_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21]
        ## point 12 -> top half T point
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18]) / 2)
        drawing_keypoints[25] = drawing_keypoints[17]
        ## point 13 -> bottonm half T point
        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22]) / 2)
        drawing_keypoints[27] = drawing_keypoints[21]
        # Baseline Center
        ## point 14 -> top baseline center
        drawing_keypoints[28] = int((drawing_keypoints[8] + drawing_keypoints[12]) / 2)
        drawing_keypoints[29] = drawing_keypoints[9]
        ## point 15 -> bottom baseline center
        drawing_keypoints[30] = int((drawing_keypoints[10] + drawing_keypoints[14]) / 2)
        drawing_keypoints[31] = drawing_keypoints[11]
        # Net
        ## point 16 -> net center
        drawing_keypoints[32] = drawing_keypoints[24]
        drawing_keypoints[33] = int((drawing_keypoints[25] + drawing_keypoints[27]) / 2)
        ## point 17 -> net left
        drawing_keypoints[34] = drawing_keypoints[16]
        drawing_keypoints[35] = drawing_keypoints[33]
        ## point 18 -> net right
        drawing_keypoints[36] = drawing_keypoints[18]
        drawing_keypoints[37] = drawing_keypoints[33]

        self.drawing_keypoints = drawing_keypoints

    def set_court_lines(self):
        self.lines = [
            # each tuple represents a pair of (ith, jth) keypoints
            (0,2), 
            (4,5),
            (6,7),
            (1,3),

            (0,1),
            (8,9), 
            (10,11),
            (2,3)
        ]

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_minicourt_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_length = self.court_end_y - self.court_start_y

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_keypoints), 2):
            if i < 28:
                x = int(self.drawing_keypoints[i])
                y = int(self.drawing_keypoints[i+1])
                cv2.circle(frame, (x,y), 5, (0,0,255), -1)
        # draw lines
        for line in self.lines:
            start_point = (int(self.drawing_keypoints[line[0]*2]), int(self.drawing_keypoints[line[0]*2+1]))  # (0,1), (8,9)
            end_point = (int(self.drawing_keypoints[line[1]*2]), int(self.drawing_keypoints[line[1]*2+1]))  # (4,5), (10,11)
            cv2.line(frame, start_point, end_point, (0,0,0), 2)
        # draw net
        net_start_point = (self.drawing_keypoints[0], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        net_end_point = (self.drawing_keypoints[2], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255,0,0), 2)
        return frame

    def draw_background_rectangle(self, frame):
        '''draws a transparent white rectangle on a given frame'''
        # fully black image
        shapes = np.zeros_like(frame, np.uint8)
        # draw a filled white rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255,255,255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5  # frame transparency ratio
        beta = 1 - alpha  # shapes_img transparency ratio
        mask = shapes.astype(bool)
        # blends the frame and shapes_img
            # - only modifies pixels in out_img, where mask is True
        out[mask] = cv2.addWeighted(frame, alpha, shapes, beta, 0)[mask]
        return out
    
    def draw_minicourt(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def get_start_point_of_minicourt(self):
        return (self.court_start_x, self.court_end_y)
    
    def get_width_of_minicourt(self):
        return self.court_drawing_width
    
    def get_length_of_minicourt(self):
        return self.court_drawing_length
    
    def get_court_drawing_keypoints(self):
        return self.drawing_keypoints
    
    def get_minicourt_coordinates(self,
                                  object_position,
                                  closest_keypoint,
                                  closest_keypoint_index,
                                  player_height_in_pixels,
                                  player_height_in_meters):
        '''converts given pos from court dim to minicourt dim'''
        # init minicourt obj pos
        closest_minicourt_kp = (self.drawing_keypoints[closest_keypoint_index*2],
                                self.drawing_keypoints[closest_keypoint_index*2+1])
        minicourt_obj_pos = [closest_minicourt_kp[0], closest_minicourt_kp[1]]

        # player-to-kp dist
        dist_obj_to_kp_x_pixels, dist_obj_to_kp_y_pixels = measure_xy_distance(object_position, closest_keypoint)
        ## convert to meters
        dist_obj_to_kp_x_meters = convert_pixel_distance_to_meters(dist_obj_to_kp_x_pixels,
                                                                 player_height_in_pixels,
                                                                 player_height_in_meters)
        dist_obj_to_kp_y_meters = convert_pixel_distance_to_meters(dist_obj_to_kp_y_pixels,
                                                                 player_height_in_pixels,
                                                                 player_height_in_meters)
        ## convert back to mini court coords in pixels
        minicourt_dist_obj_to_kp_x_pixels = self.convert_meters_to_pixels(dist_obj_to_kp_x_meters)
        minicourt_dist_obj_to_kp_y_pixels = self.convert_meters_to_pixels(dist_obj_to_kp_y_meters)

        # update minicourt obj pos using calculated distances
        ## obj is right(+) or left(-) of kp
        if object_position[0] >= closest_keypoint[0]:
            minicourt_obj_pos[0] = minicourt_obj_pos[0] + minicourt_dist_obj_to_kp_x_pixels
        else:
            minicourt_obj_pos[0] = minicourt_obj_pos[0] - minicourt_dist_obj_to_kp_x_pixels
        ## obj is above(-) or below(+) kp
        if object_position[1] <= closest_keypoint[1]:
            minicourt_obj_pos[1] = minicourt_obj_pos[1] - minicourt_dist_obj_to_kp_y_pixels
        else:
            minicourt_obj_pos[1] = minicourt_obj_pos[1] + minicourt_dist_obj_to_kp_y_pixels

        return tuple(minicourt_obj_pos)

    def convert_bboxes_to_minicourt_coordinates(self, player_bboxes, ball_bboxes, original_court_keypoints):
        """iterates over player and ball bboxes to find each player's and ball's loc in minicourt coords
        Returns:
            tuple of 2 lists of dicts: (
                [
                    {  # frame 1
                        player_id_1: (minicourt_pos_x, minicourt_pos_y),
                        player_id_2: (minicourt_pos_x, minicourt_pos_y)
                    }, ...,
                    {  # frame n
                        player_id_1: (minicourt_pos_x, minicourt_pos_y),
                        player_id_2: (minicourt_pos_x, minicourt_pos_y)
                    }
                ],
                [
                    {  # frame 1
                        1: (minicourt_ball_pos_x, minicourt_ball_pos_y)
                    }, ...,
                    {  # frame n
                        1: minicourt_ball_pos_x, minicourt_ball_pos_y)
                    }
                ]
            )
        """
        player_ids = list(player_bboxes[0].keys())
        player_heights = {
            player_ids[0]: constants.PLAYER_1_HEIGHT,
            player_ids[1]: constants.PLAYER_2_HEIGHT
        }

        output_player_bboxes = []
        output_ball_bboxes = []
        for frame_num, player_bbox in enumerate(player_bboxes):
            ball_bbox = ball_bboxes[frame_num][1]
            ball_pos = get_center_of_bbox(ball_bbox)
            # dist(center of ball bbox, center of bottom of player bbox)
            closest_player_id_to_ball = min(player_bbox.keys(),
                                            key=lambda x: euclidean_distance(ball_pos, get_center_of_bbox(player_bbox[x])))
            
            # kps to consider for closest kps
            kp_indices = [i for i in range(4, 19)]
            
            # each frame is rep as a dict in the op list
            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_pos = get_foot_position(bbox)

                # closest kp to player in pixels
                closest_kp_to_player_idx = get_closest_keypoint_index(foot_pos,
                                                                      original_court_keypoints,
                                                                      kp_indices)
                closest_kp_to_player = (original_court_keypoints[closest_kp_to_player_idx*2],
                                        original_court_keypoints[closest_kp_to_player_idx*2+1])

                # player height in pixels
                    # - max height over 50 frames
                frame_idx_min = max(0, frame_num-20)
                frame_idx_max = min(len(player_bboxes), frame_num+50)
                player_bboxes_height_in_pixels = [get_bbox_height(player_bboxes[i][player_id]) for i in range(frame_idx_min, frame_idx_max)]
                max_player_height_in_pixels = max(player_bboxes_height_in_pixels)

                # convert closest kp to meters
                minicourt_player_pos = self.get_minicourt_coordinates(foot_pos,
                                                                      closest_kp_to_player,
                                                                      closest_kp_to_player_idx,
                                                                      max_player_height_in_pixels,
                                                                      player_heights[player_id])
                
                output_player_bboxes_dict[player_id] = minicourt_player_pos

                if closest_player_id_to_ball == player_id:
                    # closest kp to ball in pixels
                    closest_kp_to_ball_idx = get_closest_keypoint_index(ball_pos,
                                                                original_court_keypoints,
                                                                kp_indices)
                    closest_kp_to_ball = (original_court_keypoints[closest_kp_to_ball_idx*2],
                                original_court_keypoints[closest_kp_to_ball_idx*2+1])
                
                    minicourt_ball_pos = self.get_minicourt_coordinates(ball_pos,
                                                                    closest_kp_to_ball,
                                                                    closest_kp_to_ball_idx,
                                                                    max_player_height_in_pixels,
                                                                    player_heights[player_id])
                    output_ball_bboxes.append({1: minicourt_ball_pos})
            output_player_bboxes.append(output_player_bboxes_dict)
        return output_player_bboxes, output_ball_bboxes
    
    def draw_points_on_minicourt(self, frames, positions, color=(255,0,0)):
        for frame_num, frame in enumerate(frames):
            for i, pos in positions[frame_num].items():
                x, y = pos
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames