from utils import (read_video,
                   save_video)
from trackers import PlayerTracker, BallTracker


def main():
    # read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # detect players & ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolov5_last.pt")

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                read_from_stub=True,
                                                stub_path="tracker_stubs/ball_detections.pkl")

    # draw output
    ## draw player bbox
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    ## draw ball bbox
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

    # save video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__=='__main__':
    main()