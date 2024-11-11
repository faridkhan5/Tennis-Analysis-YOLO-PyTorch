import cv2

from utils import (read_video,
                   save_video,
                   euclidean_distance,
                   convert_pixel_distance_to_meters,
                   draw_player_stats)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from copy import deepcopy
import pandas as pd


def main():
    # read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # detect players and ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolov5_150_best.pt")

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    print('-'*30)
    print('Player detections done')

    ball_detections = ball_tracker.detect_frames(video_frames,
                                                read_from_stub=True,
                                                stub_path="tracker_stubs/ball_detections_yolov5_150.pkl")
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    print('-'*30)
    print('Ball detections done')
    
    # court line detector model
    court_model_path = "models/keypoints_model_30.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    print('-'*30)
    print('Keypoints prediction done')

    # choose tennis players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Mini court
    minicourt = MiniCourt(video_frames[0])
    ## convert court positions to minicourt positions
    minicourt_player_detections, minicourt_ball_detections = minicourt.convert_bboxes_to_minicourt_coordinates(player_detections, ball_detections, court_keypoints)
    print('-'*30)
    print('Conversion to minicourt coordinates done')
    
    player_stats = [{
            'frame_num': 0,
            'player_1_number_of_shots': 0,
            'player_1_cumulative_shot_speed': 0,
            'player_1_curr_shot_speed': 0,
            'player_1_cumulative_speed': 0,
            'player_1_curr_speed': 0,

            'player_2_number_of_shots': 0,
            'player_2_cumulative_shot_speed': 0,
            'player_2_curr_shot_speed': 0,
            'player_2_cumulative_speed': 0,
            'player_2_curr_speed': 0
    }]

    # detect ball hits
    ball_hit_frames = ball_tracker.get_ball_hit_frames(ball_detections)
    for ball_hit_idx in range(len(ball_hit_frames) - 1):
        start_frame = ball_hit_frames[ball_hit_idx]
        end_frame = ball_hit_frames[ball_hit_idx + 1]

        # time taken for next player to hit the ball once curr player has hit
        ball_hit_time_taken_s = (end_frame - start_frame) / 24  # 24 fps
        # dist covered by ball
        ball_hit_dist_covered_pixels = euclidean_distance(minicourt_ball_detections[start_frame][1],
                                                         minicourt_ball_detections[end_frame][1])
        ball_hit_dist_covered_meters = convert_pixel_distance_to_meters(ball_hit_dist_covered_pixels,
                                                                        minicourt.get_length_of_minicourt(),
                                                                        constants.COURT_LENGTH) * 2.2  # dist * court_angle_adjustment * trajectory_adjustment
        # speed of ball
        ball_hit_speed = ball_hit_dist_covered_meters / ball_hit_time_taken_s * 3.6  # 1 m/s = 3.6 km/h
        
        # home player - hits the ball
        player_positions = minicourt_player_detections[start_frame]
        ball_hit_player_id = min(player_positions.keys(),
                                 key=lambda player_id: euclidean_distance(player_positions[player_id], minicourt_ball_detections[start_frame][1]))
        # opponent player speed
        opp_player_id = 1 if ball_hit_player_id == 2 else 2
        opp_player_dist_covered_pixels = euclidean_distance(minicourt_player_detections[start_frame][opp_player_id],
                                                            minicourt_player_detections[end_frame][opp_player_id])
        opp_player_dist_covered_meters = convert_pixel_distance_to_meters(opp_player_dist_covered_pixels,
                                                                          minicourt.get_width_of_minicourt(),
                                                                          constants.DOUBLES_BASELINE_WIDTH) * 2.2
        opp_player_speed = opp_player_dist_covered_meters / ball_hit_time_taken_s * 3.6

        curr_player_stats = deepcopy(player_stats[-1])
        curr_player_stats['frame_num'] = start_frame
        curr_player_stats[f"player_{ball_hit_player_id}_number_of_shots"] += 1
        curr_player_stats[f"player_{ball_hit_player_id}_cumulative_shot_speed"] += ball_hit_speed
        curr_player_stats[f"player_{ball_hit_player_id}_curr_shot_speed"] = ball_hit_speed

        curr_player_stats[f"player_{opp_player_id}_cumulative_speed"] += opp_player_speed
        curr_player_stats[f"player_{opp_player_id}_curr_speed"] = opp_player_speed

        player_stats.append(curr_player_stats)
    
    # player stats df
    df_player_stats = pd.DataFrame(player_stats)
    df_frames = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    df_player_stats = pd.merge(df_frames, df_player_stats, on='frame_num', how='left')
    df_player_stats = df_player_stats.ffill()
    print('-'*30)
    print('Player stats done')

    # average hit speed using cumulative hit speed
    df_player_stats['player_1_avg_shot_speed'] = df_player_stats['player_1_cumulative_shot_speed'] / df_player_stats['player_1_number_of_shots']
    df_player_stats['player_2_avg_shot_speed'] = df_player_stats['player_2_cumulative_shot_speed'] / df_player_stats['player_2_number_of_shots']
    df_player_stats['player_1_avg_speed'] = df_player_stats['player_1_cumulative_speed'] / df_player_stats['player_1_number_of_shots']
    df_player_stats['player_2_avg_speed'] = df_player_stats['player_2_cumulative_speed'] / df_player_stats['player_2_number_of_shots']

    # Draw output
    ## player bbox
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## draw ball bbox
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

    ## court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    ## minicourt
    output_video_frames = minicourt.draw_minicourt(output_video_frames)
    output_video_frames = minicourt.draw_points_on_minicourt(output_video_frames,
                                                             minicourt_player_detections)
    output_video_frames = minicourt.draw_points_on_minicourt(output_video_frames,
                                                             minicourt_ball_detections,
                                                             color=(0,255,255))
    print('-'*30)
    print('Minicourt done')

    ## player stats
    output_video_frames = draw_player_stats(output_video_frames, df_player_stats)
    ## frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i+1}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # save video
    save_video(output_video_frames, "output_videos/output_video.avi")
    print('-'*30)
    print('Video saved successfully')
    print('-'*30)


if __name__=='__main__':
    main()