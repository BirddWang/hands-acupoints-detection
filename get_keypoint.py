import cv2
import os
import numpy as np
import argparse
import json
# import matplotlib.pyplot as plt # for testing

def _load_jsonl(file_path):
    data = []
    with open('file_path', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
        return data


def _get_keypoint(image_path, label_path):
    img = cv2.imread(image_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 50, 100])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        keypoints.append((cx, cy))
    
    data = _load_jsonl(label_path)
    

    
    return 0

def _tag_keypoint(image_path, keypoint):
    return 0

def main(args):
    img_dir = f"{args.img_dir}/{args.idx}"
    len = len(os.listdir(img_dir))
    for i in range(len):
        img_path = f"{img_dir}/{i}.jpg"
        label_path = f"{args.label_dir}/{args.idx}.jsonl"
        _get_keypoint(img_path, label_path)
    return 0


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--idx", type=int, required=True, help="video idx")
    argparse.add_argument("--img_dir", type=str, default="datasets/hand/imgs", help="image directory")
    argparse.add_argument("--label_dir", type=str, default="datasets/hand/labels", help="label directory")
    args = argparse.parse_args()
    main(args)