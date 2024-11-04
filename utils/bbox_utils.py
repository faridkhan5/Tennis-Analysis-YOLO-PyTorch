def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def euclidean_distance(p1, p2):
    return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_kp_dist = float('inf')
    closest_kp_idx = keypoint_indices[0]
    for kp_idx in keypoint_indices:
        kp = keypoints[kp_idx*2], keypoints[kp_idx*2+1]
        kp_dist = abs(point[1] - kp[1])
    
        if kp_dist < closest_kp_dist:
            closest_kp_dist = kp_dist
            closest_kp_idx = kp_idx
    return closest_kp_idx

def get_bbox_height(bbox):
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))