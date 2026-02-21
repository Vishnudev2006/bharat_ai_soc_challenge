import cv2
import mediapipe as mp
import pickle
import numpy as np
from pynput.keyboard import Key, Controller
from collections import deque
from statistics import mode
import time
import psutil
import os
import threading

# --- OPTIMIZATION 1: CPU HYBRID COMPUTE ---
# Force AI inference to CPU to prevent Jetson GPU/OS driver Segmentation Faults
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- CONFIGURATION ---
CAM_WIDTH = 320
CAM_HEIGHT = 240
FRAME_SKIP = 2
STABILITY_BUFFER = 5

# --- OPTIMIZATION 2: ASYNCHRONOUS I/O THREAD & MJPEG ---
class CamStream:
    """Daemon thread for asynchronous MJPEG camera polling to prevent I/O blocking."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # Force MJPEG compression to save USB 2.0 bandwidth
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.stream.set(3, CAM_WIDTH)
        self.stream.set(4, CAM_HEIGHT)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()

# --- SYSTEM MONITOR ---
class JetsonMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.gpu_load_path = "/sys/devices/gpu.0/load"
        self.power_path = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input"
        
    def get_stats(self):
        cpu_per_core = psutil.cpu_percent(percpu=True)
        try: proj_ram = self.process.memory_info().rss / (1024 * 1024) 
        except: proj_ram = 0
        gpu_usage = 0
        try:
            with open(self.gpu_load_path, 'r') as f:
                content = f.read().strip()
                if content: gpu_usage = float(content) / 10.0
        except: pass
        power_w = 0
        try:
            with open(self.power_path, 'r') as f:
                content = f.read().strip()
                if content: power_w = float(content) / 1000.0
        except: pass
        return cpu_per_core, proj_ram, gpu_usage, power_w

# --- SETUP ---
print("--- NEURO-TOUCH: CHALLENGE EDITION (OPTIMIZED) ---")
monitor = JetsonMonitor()

try:
    with open('gesture_brain.pkl', 'rb') as f: brain = pickle.load(f)
    print("SUCCESS: Brain loaded.")
except: 
    print("ERROR: gesture_brain.pkl not found!")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
keyboard = Controller()

# Start the threaded camera
cam = CamStream(0).start()
time.sleep(1.0) # Allow camera warmup

history = deque(maxlen=STABILITY_BUFFER)
current_action = "WAITING"
last_gesture = -1
frame_counter = 0

# Stats placeholders
stat_fps, stat_pwr, stat_gpu, stat_ram = "FPS: 0", "PWR: 0W", "GPU: 0%", "RAM: 0MB"
stat_latency, stat_accuracy, stat_stability = "Lat: 0ms", "Acc: 0%", "Stab: 0%"
cpus = [0, 0, 0, 0]
prev_frame_time = time.time()

while True:
    # Read from the background thread instantly
    img_raw = cam.read()
    if img_raw is None: continue
    
    # --- TIMING START ---
    start_time = time.time()
    
    # --- OPTIMIZATION 3: GPU OFFLOAD VIA UMAT ---
    # Transfer frame to GPU memory for rapid preprocessing
    umat_img = cv2.UMat(img_raw)
    umat_img = cv2.flip(umat_img, 1)
    img = umat_img.get() # Download back to CPU for drawing
    
    # --- STATS UPDATE (Every 10 frames) ---
    if frame_counter % 10 == 0:
        cpus, proj_ram, gpu, power = monitor.get_stats()
        # FPS Calculation
        fps = 1.0 / (time.time() - prev_frame_time) if (time.time() - prev_frame_time) > 0 else 0
        stat_fps = f"FPS: {int(fps)}"
        stat_pwr = f"PWR: {power:.1f}W"
        stat_gpu = f"GPU: {int(gpu)}%"
        stat_ram = f"RAM: {int(proj_ram)}MB"
    prev_frame_time = time.time()

    # --- AI PROCESSING ---
    if frame_counter % FRAME_SKIP == 0:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                features = []
                for lm in handLms.landmark: features.extend([lm.x, lm.y])
                
                try:
                    # 1. Get Probability (Accuracy/Confidence)
                    probs = brain.predict_proba([features])[0]
                    confidence = np.max(probs) * 100
                    prediction = np.argmax(probs)
                    
                    history.append(prediction)
                    
                    # 2. Calculate Stability
                    if len(history) > 0:
                        most_common = mode(history)
                        stability_score = (history.count(most_common) / len(history)) * 100
                    else:
                        stability_score = 0

                    # 3. Update Stats Strings
                    stat_accuracy = f"Acc: {int(confidence)}%"
                    stat_stability = f"Stab: {int(stability_score)}%"
                
                except: pass
                
                if len(history) == STABILITY_BUFFER:
                    try:
                        stable_gesture = mode(history)
                        
                        # --- GESTURE MAPPING ---
                        if stable_gesture == 0: # OPEN -> PAUSE
                            current_action = "PAUSE (OPEN)"
                            if last_gesture != 0:
                                keyboard.press(Key.space)
                                keyboard.release(Key.space)
                                last_gesture = 0
                                
                        elif stable_gesture == 1: # TWO FINGERS -> PLAY
                            current_action = "PLAY (2 FINGERS)"
                            if last_gesture != 1:
                                keyboard.press(Key.space)
                                keyboard.release(Key.space)
                                last_gesture = 1
                                
                        elif stable_gesture == 2: # THUMB UP -> VOL UP
                            current_action = "VOL + (UP)"
                            keyboard.press('0')
                            keyboard.release('0')
                            last_gesture = 2

                        elif stable_gesture == 3: # THUMB DOWN -> VOL DOWN
                            current_action = "VOL - (DOWN)"
                            keyboard.press('9')
                            keyboard.release('9')
                            last_gesture = 3
                            
                        elif stable_gesture == 4: # THUMB LEFT -> REWIND
                            current_action = "SEEK << (LEFT)"
                            keyboard.press(Key.left)
                            keyboard.release(Key.left)
                            last_gesture = 4
                            
                        elif stable_gesture == 5: # THUMB RIGHT -> FORWARD
                            current_action = "SEEK >> (RIGHT)"
                            keyboard.press(Key.right)
                            keyboard.release(Key.right)
                            last_gesture = 5
                            
                    except: pass
        else:
            history.clear()
            current_action = "NO HAND"
            stat_accuracy = "Acc: --"
            stat_stability = "Stab: --"

    # --- LATENCY CALCULATION ---
    latency_ms = (time.time() - start_time) * 1000
    stat_latency = f"Lat: {int(latency_ms)}ms"

    # --- DASHBOARD DRAWING ---
    overlay = img.copy()
    cv2.rectangle(overlay, (220, 0), (320, 240), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, 180), (120, 240), (0, 0, 0), -1) 
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    font = cv2.FONT_HERSHEY_PLAIN
    
    # 1. System Stats
    cv2.putText(img, stat_fps, (230, 20), font, 1, (0, 255, 0), 1)
    cv2.putText(img, stat_pwr, (230, 35), font, 1, (0, 255, 255), 1)
    cv2.putText(img, stat_gpu, (230, 50), font, 1, (255, 0, 255), 1)
    cv2.putText(img, stat_ram, (230, 65), font, 1, (200, 200, 200), 1)
    
    # 2. CPU Bars
    cv2.putText(img, "CORES:", (230, 90), font, 1, (255, 255, 255), 1)
    for i, usage in enumerate(cpus):
        y = 105 + (i * 12)
        cv2.rectangle(img, (230, y), (310, y+8), (50, 50, 50), -1)
        bar_w = int((usage / 100) * 80)
        cv2.rectangle(img, (230, y), (230 + bar_w, y+8), (0, 150, 255), -1)

    # 3. AI Performance Metrics
    cv2.putText(img, "AI METRICS:", (5, 195), font, 1, (255, 255, 255), 1)
    cv2.putText(img, stat_latency, (5, 210), font, 1, (0, 165, 255), 1) 
    cv2.putText(img, stat_accuracy, (5, 225), font, 1, (0, 255, 0), 1) 
    cv2.putText(img, stat_stability, (65, 225), font, 1, (255, 255, 0), 1) 

    # 4. Action Text
    color = (255, 255, 255)
    if "PAUSE" in current_action: color = (0, 0, 255)
    elif "PLAY" in current_action: color = (0, 255, 0)
    elif "VOL" in current_action: color = (255, 255, 0)
    
    text_size = cv2.getTextSize(current_action, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = (CAM_WIDTH - text_size[0]) // 2
    cv2.putText(img, current_action, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("NeuroTouch Challenge", img)
    
    frame_counter += 1
    if cv2.waitKey(1) == ord('q'): break

cam.stop()
cv2.destroyAllWindows()
