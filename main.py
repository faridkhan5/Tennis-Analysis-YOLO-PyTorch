import cv2

from utils import (read_video,
                   save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt


def main():
    # read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # detect players and ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolov5_last.pt")

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                read_from_stub=True,
                                                stub_path="tracker_stubs/ball_detections.pkl")
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # court line detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose tennis players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Mini court
    minicourt = MiniCourt(video_frames[0])
    ## convert court positions to minicourt positions
    minicourt_player_detections, minicourt_ball_detections = minicourt.convert_bboxes_to_minicourt_coordinates(player_detections, ball_detections, court_keypoints)
    
    # detect ball hits
    ball_hit_frames = ball_tracker.get_ball_hit_frames(ball_detections)

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
    ## frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i+1}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # save video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__=='__main__':
    main()