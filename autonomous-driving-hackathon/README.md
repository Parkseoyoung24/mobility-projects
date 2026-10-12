# 2026 Autonomous Driving Hackathon

Code from the 2026 Autonomous Driving Hackathon (August 2026, team of five), hosted by the Korea Association of Autonomous Mobility Industry, where a small vehicle had to drive a rail track while recognizing traffic lights and obstacles.

## Traffic light detection
- `split_dataset.py`: converts LabelMe JSON labels to YOLO format and splits the images into train, validation and test sets (80/10/10) for a YOLOv8 detector with four classes (red, yellow, green, left arrow).
- `dataset.yaml`: YOLO dataset config. The image dataset is not included.

## Reinforcement learning driving (PPO)
Lap driving with PPO (Stable-Baselines3) in the PhysiCar simulator. The agent observes a camera image, 9 LiDAR sectors and its own speed and steering, and outputs steering and throttle. The reward encourages track progress and penalizes low speed, leaving the track, collisions and getting stuck.
- `config.py`, `env_fastlap.py`, `train_ppo.py`, `callbacks.py`, `diagnose_policy.py`

These files modify the simulator's base environment, which is not included here.

## What I learned
The line-tracing model trained with reinforcement learning in simulation made errors on the real track because the lighting and track colors were different. This showed me the sim-to-real gap firsthand.
