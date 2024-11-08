import torch
from torchvision import models, transforms
import cv2
import numpy as np
from utils import keypoints_to_idx, midpoint


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=None)
        # modify fc layer
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        # load saved wts
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # (C, H, W) -> (3,224,224)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """ predicts the keypoints using the loaded model
        Returns:
            1-D np arr: x,y coordinates of all 14 keypoints
        """
        # only predict on the 1st frame, because camera remains stationary
            # - tennis court remains static across all frames
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # model needs inp in a list format for pred
        img_tensor = self.transform(img_rgb).unsqueeze(0) # img -[unsqueeze]-> [img]
        with torch.no_grad():
            outputs = self.model(img_tensor)  # (1,28)
        keypoints = outputs.squeeze().cpu().numpy()  # (28)

        # add additional points as center of baseline
        kp_4 = keypoints_to_idx(keypoints, 4)
        kp_5 = keypoints_to_idx(keypoints, 5)
        kp_6 = keypoints_to_idx(keypoints, 6)
        kp_7 = keypoints_to_idx(keypoints, 7)
        kp_8 = keypoints_to_idx(keypoints, 8)
        kp_9 = keypoints_to_idx(keypoints, 9)
        kp_10 = keypoints_to_idx(keypoints, 10)
        kp_11 = keypoints_to_idx(keypoints, 11)
        kp_12 = keypoints_to_idx(keypoints, 12)
        kp_13 = keypoints_to_idx(keypoints, 13)
        
        kp_14 = midpoint(kp_4, kp_6)
        kp_15 = midpoint(kp_5, kp_7)
        kp_16 = midpoint(kp_12, kp_13)
        kp_17 = midpoint(kp_8, kp_10)
        kp_18 = midpoint(kp_9, kp_11)

        keypoints = np.append(keypoints, kp_14[0])
        keypoints = np.append(keypoints, kp_14[1])
        keypoints = np.append(keypoints, kp_15[0])
        keypoints = np.append(keypoints, kp_15[1])
        keypoints = np.append(keypoints, kp_16[0])
        keypoints = np.append(keypoints, kp_16[1])
        keypoints = np.append(keypoints, kp_17[0])
        keypoints = np.append(keypoints, kp_17[1])
        keypoints = np.append(keypoints, kp_18[0])
        keypoints = np.append(keypoints, kp_18[1])

        # map back transformed img to original img dims
        og_h, og_w = img_rgb.shape[:2]
        keypoints[::2] *= og_w / 224.0
        keypoints[1::2] *= og_h / 224.0

        return keypoints  # [x1, y1, x2, y2, x1, y1, x2, y2, ...]
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            if i < 28:
                x = int(keypoints[i])
                y = int(keypoints[i+1])

                # draw dot on the coords
                cv2.putText(image, str(i//2), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.circle(image, (x,y), 5, (0,0,255), -1)  # -1 -> filled circle
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames