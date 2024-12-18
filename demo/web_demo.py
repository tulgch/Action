# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:32:30 2024

@author: user
"""

# Copyright (c) OpenMMLab. All rights reserved.
"""Webcam Real-Time Spatio-Temporal Action Detection Demo.

This script integrates human detection, pose estimation, and action recognition 
in a spatio-temporal pipeline, utilizing pre-trained models from MMAction2 and MMDetection.
"""

import argparse
import logging
import time
import threading
import cv2
import torch
import mmcv
import numpy as np
from mmengine import Config, DictAction

from mmdet.apis import inference_detector, init_detector
from mmaction.structures import ActionDataSample
from mmengine.structures import InstanceData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------
def get_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Webcam Action Detection Demo")
    parser.add_argument('--config', type=str, required=True, help='Action detection model config.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Action model checkpoint file.')
    parser.add_argument('--det-config', type=str, help='Human detection config.')
    parser.add_argument('--det-checkpoint', type=str, help='Human detection checkpoint.')
    parser.add_argument('--label-map', type=str, help='Path to the label map file.')
    parser.add_argument('--device', default='cuda:0', help='Device to run the model on.')
    parser.add_argument('--video-source', default=0, help='Video source (0 for webcam or video path).')
    parser.add_argument('--output-file', default=None, help='Path to save the output video.')
    parser.add_argument('--show', action='store_true', help='Display video output in real-time.')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold.')
    return parser.parse_args()

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def load_label_map(file_path):
    """Load label map from a file."""
    with open(file_path, 'r') as f:
        return {int(line.split(': ')[0]): line.split(': ')[1].strip() for line in f}

def draw_predictions(frame, bboxes, predictions, label_map, color=(0, 255, 0)):
    """Draw bounding boxes and action predictions on the frame."""
    for bbox, preds in zip(bboxes, predictions):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for idx, (label, score) in enumerate(preds):
            if idx > 3:  # Show up to 3 labels
                break
            text = f"{label_map[label]}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10 - idx * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# ------------------------------------------------------------
# Core Components
# ------------------------------------------------------------
class HumanDetector:
    """Human detection using MMDetection."""
    def __init__(self, config_path, checkpoint_path, device, threshold=0.5):
        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.threshold = threshold

    def detect(self, frame):
        """Run human detection and return bounding boxes."""
        results = inference_detector(self.model, frame)
        bboxes = results.pred_instances.bboxes
        scores = results.pred_instances.scores
        valid_idx = scores > self.threshold
        return bboxes[valid_idx]

class ActionDetector:
    """Spatio-Temporal Action Recognition."""
    def __init__(self, config, checkpoint, device, label_map):
        self.model = init_detector(config, checkpoint, device=device)
        self.label_map = label_map

    def recognize(self, frames, bboxes):
        """Run action recognition on a batch of frames."""
        inputs = torch.stack([torch.from_numpy(frame) for frame in frames])
        results = self.model(inputs)
        predictions = results.pred_instances.scores
        preds = [self.get_top_actions(scores) for scores in predictions]
        return preds

    def get_top_actions(self, scores):
        """Return the top predicted actions with confidence scores."""
        return [(i, score) for i, score in enumerate(scores) if score > 0.5]

# ------------------------------------------------------------
# Main Processing Pipeline
# ------------------------------------------------------------
def main():
    args = get_arguments()

    # Initialize models
    logger.info("Initializing models...")
    human_detector = HumanDetector(args.det_config, args.det_checkpoint, args.device, args.confidence_threshold)
    action_detector = ActionDetector(Config.fromfile(args.config), args.checkpoint, args.device, load_label_map(args.label_map))

    # Open video source
    video_source = cv2.VideoCapture(args.video_source)
    if not video_source.isOpened():
        logger.error("Failed to open video source.")
        return

    # Prepare video writer
    out_writer = None
    if args.output_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output_file, fourcc, 20.0, (int(video_source.get(3)), int(video_source.get(4))))

    logger.info("Starting video stream...")
    while True:
        ret, frame = video_source.read()
        if not ret:
            break

        # Human detection
        bboxes = human_detector.detect(frame)

        # Action recognition (process current frame only)
        if bboxes.shape[0] > 0:
            predictions = action_detector.recognize([frame], bboxes)
            frame = draw_predictions(frame, bboxes, predictions, action_detector.label_map)

        # Display or write output
        if args.show:
            cv2.imshow('Action Recognition', frame)
        if out_writer:
            out_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logger.info("Exiting...")
    video_source.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
