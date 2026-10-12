# 2026 Autonomous Driving Hackathon

Code from the 2026 Autonomous Driving Hackathon (August 2026, team of five), hosted by the Korea Association of Autonomous Mobility Industry, where a small vehicle had to drive a rail track while recognizing traffic lights and obstacles.

## traffic-light-detection
Data preparation for a YOLOv8 traffic light detector with four classes (red, yellow, green, left arrow).
`split_dataset.py` converts JSON labels to YOLO format and splits the images into train, validation and test sets (80/10/10).
The image dataset is not included.

## rl-driving-ppo
Reinforcement learning for lap driving with PPO (Stable-Baselines3) in the PhysiCar simulator.
The agent observes a camera image, 9 LiDAR sectors and its own speed and steering, and outputs steering and throttle.
The reward encourages track progress and penalizes low speed, leaving the track, collisions and getting stuck.

These files modify the simulator's base environment, which is not included here.

## What I learned
The model trained in simulation made errors on the real track because of differences in lighting and track color. This showed me the sim-to-real gap firsthand.
