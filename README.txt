Webcam Spatio-Temporal Action Detection Demo

This project implements a real-time spatio-temporal action detection system using MMAction2 and MMDetection. 
The system detects humans in video streams, estimates actions, and visualizes results.

------------------------------------------------------------
Overview:
The pipeline consists of:
1. Human Detection: Using Faster R-CNN to detect persons in the video.
2. Action Detection: Using SlowOnly (pre-trained on AVA dataset) for spatio-temporal action recognition.
3. Visualization: Bounding boxes and action labels are overlaid on video frames.

------------------------------------------------------------
Dependencies:
Install the required libraries with the following command:
pip install -r requirements.txt

requirements.txt:
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
mmcv>=2.0.0
mmengine>=0.7.0
mmdet>=3.0.0
mmaction2>=1.0.0
numpy>=1.21.0
moviepy>=1.0.3

------------------------------------------------------------
Demo Command:
Run the following command to test the system:

python demo/webcam_demo.py ^
    --config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py ^
    --checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth ^
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py ^
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth ^
    --label-map tools/data/ava/label_map.txt ^
    --device cuda ^
    --output-fps 15 ^
    --show

------------------------------------------------------------
Arguments:
--config: Path to the spatio-temporal action detection model configuration.
--checkpoint: Pre-trained checkpoint for action detection.
--det-config: Configuration for the human detection model.
--det-checkpoint: Pre-trained checkpoint for the Faster R-CNN human detector.
--label-map: Path to the label map file for action classes.
--device: Device to run the inference (cuda for GPU or cpu).
--output-fps: FPS for output video.
--show: Display the video stream with annotations in real time.

------------------------------------------------------------
How It Works:
1. Human Detection:
   - Faster R-CNN detects human bounding boxes in each frame.
2. Action Recognition:
   - SlowOnly predicts actions for detected persons across multiple frames.
3. Visualization:
   - Bounding boxes and predicted action labels are displayed on the video.

------------------------------------------------------------
Output:
The system:
- Displays a video stream with real-time action detection results.
- Optionally saves the annotated video to a file if --out-filename is specified.

------------------------------------------------------------
Example Actions Detected:
Using the AVA Dataset label map:
- Walking
- Sitting
- Running
- Punching
- Falling
- Picking up objects

------------------------------------------------------------
Device Requirements:
- CUDA GPU for fast inference.
- Tested on an NVIDIA GPU with PyTorch and CUDA installed.

------------------------------------------------------------
Acknowledgments:
This project is based on:
- MMAction2: https://github.com/open-mmlab/mmaction2
- MMDetection: https://github.com/open-mmlab/mmdetection

------------------------------------------------------------
License:
This project uses OpenMMLab’s tools and adheres to their respective licenses.
