# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:29:20 2024

@author: user
"""

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import tempfile
import warnings

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction
from mmengine.structures import InstanceData

from mmaction.apis import (detection_inference, inference_recognizer,
                           inference_skeleton, init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract

try:
    from mmdet.apis import init_detector
except ImportError:
    warnings.warn("Failed to load `init_detector` from `mmdet.apis`.")

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError("Install moviepy for video output functionality.")

# UI Parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 255, 255)
THICKNESS = 1


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='MMAction2 Video Demo')
    parser.add_argument('--video', default='demo/test_video.mp4', help='Input video path.')
    parser.add_argument('--output', default='output_video.mp4', help='Output video path.')
    parser.add_argument('--det-config', default='demo/faster_rcnn_config.py', help='Detection config path.')
    parser.add_argument('--det-checkpoint', default='faster_rcnn_checkpoint.pth', help='Detection checkpoint.')
    parser.add_argument('--pose-config', default='demo/pose_hrnet_config.py', help='Pose config path.')
    parser.add_argument('--pose-checkpoint', default='pose_hrnet_checkpoint.pth', help='Pose checkpoint.')
    parser.add_argument('--action-config', default='configs/posec3d_config.py', help='Action config path.')
    parser.add_argument('--action-checkpoint', default='posec3d_checkpoint.pth', help='Action recognition checkpoint.')
    parser.add_argument('--device', default='cuda:0', help='Device for inference.')
    parser.add_argument('--fps', type=int, default=24, help='FPS of output video.')
    return parser.parse_args()


def load_label_map(file):
    """Loads action label map from a file."""
    with open(file, 'r') as f:
        lines = f.readlines()
    return {int(line.split(': ')[0]): line.split(': ')[1].strip() for line in lines}


def infer_action(model, keypoints, shape):
    """Runs action recognition inference."""
    results = inference_skeleton(model, keypoints, shape)
    return results.pred_score.argmax().item()


def annotate_frames(frames, results, action_label, color=(0, 255, 0)):
    """Annotates frames with bounding boxes and action labels."""
    for idx, frame in enumerate(frames):
        detections = results[idx]
        if detections is not None:
            for box, label in detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{action_label}: {label:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), FONT, 0.5, TEXT_COLOR, THICKNESS)
    return frames


def main():
    args = parse_arguments()

    # Extract frames and initialize temporary directories
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames = frame_extract(args.video, out_dir=tmp_dir.name)

    # Initialize models
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    pose_model = init_recognizer(args.pose_config, args.pose_checkpoint, args.device)
    action_model = init_recognizer(args.action_config, args.action_checkpoint, args.device)

    # Run detection
    detections, _ = detection_inference(args.det_config, args.det_checkpoint, frame_paths, 0.9, args.device)

    # Run pose estimation
    pose_results, _ = pose_inference(args.pose_config, args.pose_checkpoint, frame_paths, detections, args.device)

    # Run action recognition
    action_label_map = load_label_map("label_map.txt")
    action_results = []
    for keypoints in pose_results:
        label_index = infer_action(action_model, keypoints, original_frames[0].shape[:2])
        action_results.append(action_label_map[label_index])

    # Annotate frames
    annotated_frames = annotate_frames(original_frames, detections, action_results)

    # Save output video
    output_video = mpy.ImageSequenceClip(annotated_frames, fps=args.fps)
    output_video.write_videofile(args.output)
    tmp_dir.cleanup()


if __name__ == '__main__':
    main()
