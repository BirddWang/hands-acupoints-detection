import cv2
import os
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt # for testing

def _load_jsonl(file_path):
    data = []
    with open('file_path', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
        return data
    
def count_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def _get_keypoint(image_path, label_path):
    img = cv2.imread(image_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 50, 100])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ## we need to change the lower_blue and upper_blue to maintain the size of keypoints being 3
    keypoints = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        keypoints.append((cx, cy))

    merged_contours = []
    if len(keypoints) != 3:
        for i in range(len(contours)):
            merged = False
            for j in range(i + 1, len(contours)):
                M1 = cv2.moments(contours[i])
                M2 = cv2.moments(contours[j])
                if M1['m00'] == 0 or M2['m00'] == 0:
                    continue
                cX1 = int(M1['m10'] / M1['m00'])
                cY1 = int(M1['m01'] / M1['m00'])
                cX2 = int(M2['m10'] / M2['m00'])
                cY2 = int(M2['m01'] / M2['m00'])
                if count_distance(cX1, cY1, cX2, cY2) < 30:
                    merged = True
                    break
            if not merged:
                merged_contours.append(contours[i])

        keypoints = []
        for cnt in merged_contours:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            keypoints.append((cx, cy))
        
    if args.target:
        _tag_keypoint(image_path, keypoints)

    return keypoints

def _tag_keypoint(image_path, keypoints):
    return 0

def main(args):
    img_dir = f"{args.img_dir}/{args.idx}"
    image_len = len(os.listdir(img_dir))
    for i in range(image_len):
        img_path = f"{img_dir}/{i}.jpg"
        label_path = f"{args.label_dir}/{args.idx}.jsonl"
        keypoints = _get_keypoint(img_path, label_path)
        
        # save keypoints to jsonl file
        j_file = {
            'image': img_path,
            'keypoints': keypoints
        }
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        
        with open(f"{args.output}/{args.idx}.jsonl", 'a') as f:
            f.write(json.dumps(j_file) + '\n')

    return 0


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--idx", type=int, required=True, help="video idx")
    argparse.add_argument("--img_dir", type=str, default="datasets/hand/imgs", help="image directory")
    argparse.add_argument("--label_dir", type=str, default="datasets/hand/labels", help="label directory")
    argparse.add_argument("--target", type=bool, default=False, help="if True, get the targeted keypoints")
    argparse.add_argument("--output", type=str, default="datasets/hand/keypoints", help="output directory")
    args = argparse.parse_args()
    main(args)