# NeuroTouch: Edge HMI
**Bharat AI-SoC Student Challenge 2026 Submission**

NeuroTouch is a low-latency, contactless Human-Machine Interface (HMI) designed for resource-constrained Edge AI devices (NVIDIA Jetson Nano). It enables users to control media applications through semantic hand gestures and spatial analog volume control on a strict 5-Watt power budget.

## Features
* **Hybrid Compute Engine:** Utilizes CPU for MediaPipe Neural Network inference and GPU (OpenCV UMat) for vision pre-processing.
* **Hardware-Level Optimization:** Overrides USB bus protocols to force MJPEG compression, preventing bandwidth saturation.
* **Threaded I/O:** Decouples camera polling from the inference loop to achieve stable 30 FPS and sub-50ms latency.
* **Memory Stewardship:** Implements active garbage collection to prevent Linux OOM-killer termination during extended use.

## Files Included
* `neuro_final_lite.py`: The main execution script containing the threaded camera class and inference loop.
* `gesture_brain.pkl`: The trained K-Nearest Neighbors (KNN) classifier weights.
* `Project_Report.pdf`: Comprehensive documentation covering methodology, hardware utilization, and telemetry results.

## Setup & Execution
1. Ensure the Jetson Nano is set to 5W mode: `sudo nvpmodel -m 1`
2. Install requirements: `pip3 install mediapipe opencv-python pynput psutil`
3. Run the interface: `python3 neuro_final_lite.py`
