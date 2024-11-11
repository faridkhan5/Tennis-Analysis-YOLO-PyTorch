import numpy as np
import cv2


def draw_player_stats(output_video_frames, player_stats):
    for idx, row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_curr_shot_speed']
        player_2_shot_speed = row['player_2_curr_shot_speed']
        
        player_1_speed = row['player_1_curr_speed']
        player_2_speed = row['player_2_curr_speed']

        player_1_avg_shot_speed = row['player_1_avg_shot_speed']
        player_2_avg_shot_speed = row['player_2_avg_shot_speed']

        player_1_avg_speed = row['player_1_avg_speed']
        player_2_avg_speed = row['player_2_avg_speed']

        frame = output_video_frames[idx]  # extracts curr frame

        # transparent stats box
        shapes = np.zeros_like(frame, np.uint8)  # creates a blank img same as frame dim
        # draw overlay rectangle on blank frame
        width = 350
        height = 230
        start_x = frame.shape[1] - 400  # frame.shape -> (b * l) -> (1080 * 1920)
        start_y = frame.shape[0] - 500
        end_x = start_x + width
        end_y = start_y + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0,0,0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        output_video_frames[idx] = frame

        # add text to overlay rectangle
        text = "    Player 1    Player 2"
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               text,
                                               (start_x+90, start_y+30),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.6,
                                               (255,255,255),
                                               2)

        text = "Shot Speed:"
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               text,
                                               (start_x+10, start_y+80), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 
                                               0.45, 
                                               (0,255,255),
                                               2)
        player_1_text = f"{player_1_shot_speed:.1f} km/h"
        player_2_text = f"{player_2_shot_speed:.1f} km/h"
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               player_1_text, 
                                               (start_x + 130, start_y + 80),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.45,
                                               (0, 255, 255),
                                               2)
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               player_2_text,
                                               (start_x + 130 + 120, start_y + 80),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.45,
                                               (0, 255, 255),
                                               2)
        
        

        text = "Player Speed:"
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               text,
                                               (start_x+10, start_y+120),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.45,
                                               (255,255,255),
                                               2)
        player_1_text = f"{player_1_speed:.1f} km/h"
        player_2_text = f"{player_2_speed:.1f} km/h"
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               player_1_text, 
                                               (start_x + 130, start_y + 120),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.45,
                                               (255, 255, 255),
                                               2)
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               player_2_text,
                                               (start_x + 130 + 120, start_y + 120),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.45,
                                               (255, 255, 255),
                                               2)


        text = "Avg Sh Speed:"
        output_video_frames[idx] = cv2.putText(output_video_frames[idx], text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)
        player_1_text = f"{player_1_avg_shot_speed:.1f} km/h"
        player_2_text = f"{player_2_avg_shot_speed:.1f} km/h"
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               player_1_text, 
                                               (start_x + 130, start_y + 160),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.45,
                                               (0, 255, 255),
                                               2)
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               player_2_text,
                                               (start_x + 130 + 120, start_y + 160),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.45,
                                               (0, 255, 255),
                                               2)
       
        text = "Avg Pl Speed:"
        output_video_frames[idx] = cv2.putText(output_video_frames[idx], text, (start_x+10, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        player_1_text = f"{player_1_avg_speed:.1f} km/h"
        player_2_text = f"{player_2_avg_speed:.1f} km/h"
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               player_1_text, 
                                               (start_x + 130, start_y + 200),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.45,
                                               (255, 255, 255),
                                               2)
        output_video_frames[idx] = cv2.putText(output_video_frames[idx],
                                               player_2_text,
                                               (start_x + 130 + 120, start_y + 200),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.45,
                                               (255, 255, 255),
                                               2)
    
    return output_video_frames