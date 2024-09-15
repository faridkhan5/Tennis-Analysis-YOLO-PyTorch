from utils import (read_video,
                   save_video)
from trackers import PlayerTracker


def main():
    # read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # detect players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")

    # draw output
    ## draw player bbox
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # save video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__=='__main__':
    main()