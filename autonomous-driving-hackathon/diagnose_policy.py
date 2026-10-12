#!/usr/bin/env python3
from pathlib import Path
import time
import numpy as np
from stable_baselines3 import PPO
from config import CFG
from ros_io import RosIO

candidates = [
    Path('models/ppo_fastlap_best.zip'),
    Path('models/ppo_fastlap_progress_best.zip'),
    Path('models/ppo_fastlap_latest.zip'),
]
model_path = next((p for p in candidates if p.exists()), None)
if model_path is None:
    raise SystemExit('No model found')

print('model:', model_path)
model = PPO.load(str(model_path), device='cpu')
io = RosIO()
if not io.wait_ready():
    raise SystemExit('Camera/LiDAR not ready')
io.set_camera(); io.stop()
try:
    for i in range(30):
        obs, _ = io.observation()
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        speed = CFG.min_train_speed + 0.5 * (float(action[1]) + 1.0) * (CFG.final_max_speed - CFG.min_train_speed)
        steer = float(action[0]) * CFG.max_steer_rad
        print(f'[{i:02d}] action=[{float(action[0]):+.3f}, {float(action[1]):+.3f}] speed={speed:.3f} m/s steer={steer:+.3f}')
        time.sleep(0.15)
finally:
    io.close()
