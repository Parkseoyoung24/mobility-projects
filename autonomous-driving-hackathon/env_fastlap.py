import math
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import CFG
from ros_io import RosIO
from sim_api import SimAPI
from track import TrackGeometry


class PhysicarFastLapEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_speed=3.0, random_starts=False):
        super().__init__()
        self.io = RosIO()
        self.sim = SimAPI()
        self.track = TrackGeometry(self.sim.route())
        self.max_speed = float(max_speed)
        self.random_starts = False

        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(3, CFG.image_h, CFG.image_w), dtype=np.uint8),
            "lidar": spaces.Box(0.0, 1.0, shape=(CFG.lidar_sectors,), dtype=np.float32),
            "vehicle": spaces.Box(
                low=np.asarray([0.0, -1.0], dtype=np.float32),
                high=np.asarray([1.5, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.step_dt = 1.0 / CFG.control_hz

        self.prev_s = 0.0
        self.total_forward_progress = 0.0
        self.prev_steer_norm = 0.0
        self.stuck_reference_progress = 0.0
        self.stuck_reference_time = time.monotonic()
        self.episode_start_t = time.monotonic()

        if not self.io.wait_ready():
            raise RuntimeError("Camera/LiDAR data is not ready")
        self.io.set_camera()

    def set_max_speed(self, value):
        self.max_speed = float(np.clip(value, 0.0, CFG.final_max_speed))

    def _map_action(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        steer_norm = float(a[0])
        throttle_norm = float(a[1])
        steering = steer_norm * CFG.max_steer_rad
        u = 0.5 * (throttle_norm + 1.0)
        speed = CFG.min_train_speed + u * (self.max_speed - CFG.min_train_speed)
        return float(speed), float(steering), steer_norm

    def _vehicle_pose(self):
        try:
            return self.sim.extract_vehicle_pose(self.sim.state())
        except Exception:
            return self.sim.extract_vehicle_pose(self.sim.pose())

    def _front_lidar_m(self, obs):
        lidar_m = np.asarray(obs["lidar"], dtype=np.float32) * CFG.lidar_clip_m
        c = len(lidar_m) // 2
        return float(np.min(lidar_m[max(0, c - 1):min(len(lidar_m), c + 2)]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.io.stop()
        self.sim.reset()

        x, y, yaw = self.track.start_pose(0, 0.0, 0.0)
        self.sim.teleport(x, y, yaw)
        time.sleep(0.25)
        self.io.set_camera()
        obs, _ = self.io.observation()

        x, y, _ = self._vehicle_pose()
        idx, _ = self.track.nearest(x, y)
        self.prev_s = self.track.progress_s(idx)
        self.total_forward_progress = 0.0
        self.prev_steer_norm = 0.0
        self.stuck_reference_progress = 0.0
        self.stuck_reference_time = time.monotonic()
        self.episode_start_t = time.monotonic()
        return obs, {}

    def step(self, action):
        t0 = time.monotonic()
        speed, steering, steer_norm = self._map_action(action)
        self.io.drive(speed, steering)

        remaining = self.step_dt - (time.monotonic() - t0)
        if remaining > 0:
            time.sleep(remaining)

        obs, _ = self.io.observation()
        x, y, yaw = self._vehicle_pose()
        idx, dist_center = self.track.nearest(x, y)
        cur_s = self.track.progress_s(idx)
        ds = self.track.cyclic_delta_s(self.prev_s, cur_s)
        self.prev_s = cur_s

        # Net progress: moving backward removes previous progress.
        self.total_forward_progress = max(0.0, self.total_forward_progress + ds)

        safe_half = self.track.safe_half_width(idx)
        boundary_ratio = dist_center / max(safe_half, 1e-6)
        offtrack = boundary_ratio >= 1.0

        front_m = self._front_lidar_m(obs)
        collision = front_m < CFG.collision_lidar_m

        tangent_yaw = self.track.tangent_yaw(idx)
        heading_error = math.atan2(
            math.sin(yaw - tangent_yaw),
            math.cos(yaw - tangent_yaw),
        )

        reward = CFG.reward_progress * ds
        reward -= CFG.reward_time_penalty
        reward += CFG.reward_heading * math.cos(heading_error)

        if speed < CFG.low_speed_threshold_mps:
            reward -= CFG.reward_low_speed_penalty

        if boundary_ratio > 0.70:
            risk = float(np.clip((boundary_ratio - 0.70) / 0.30, 0.0, 1.0))
            reward -= CFG.reward_boundary_risk * risk * risk

        if front_m < CFG.obstacle_risk_m:
            risk = float(np.clip((CFG.obstacle_risk_m - front_m) / CFG.obstacle_risk_m, 0.0, 1.0))
            reward -= CFG.reward_obstacle_risk * risk * risk

        reward -= CFG.reward_steer_jerk * abs(steer_norm - self.prev_steer_norm)
        self.prev_steer_norm = steer_norm

        now = time.monotonic()
        if self.total_forward_progress - self.stuck_reference_progress >= CFG.stuck_progress_m:
            self.stuck_reference_progress = self.total_forward_progress
            self.stuck_reference_time = now
        stuck = (now - self.stuck_reference_time) >= CFG.stuck_time_s

        start_pt = self.track.center[0]
        start_distance = math.hypot(x - float(start_pt[0]), y - float(start_pt[1]))
        start_yaw = self.track.tangent_yaw(0)
        start_heading_error = math.atan2(
            math.sin(yaw - start_yaw),
            math.cos(yaw - start_yaw),
        )

        enough_progress = (
            self.total_forward_progress >= CFG.lap_completion_ratio * self.track.length
        )
        near_start = start_distance <= CFG.finish_radius_m
        correct_heading = (
            abs(math.degrees(start_heading_error)) <= CFG.finish_heading_tolerance_deg
        )
        finished = enough_progress and near_start and correct_heading

        terminated = False
        reason = ""
        if collision:
            reward += CFG.penalty_collision
            terminated = True
            reason = "collision"
        elif offtrack:
            reward += CFG.penalty_offtrack
            terminated = True
            reason = "offtrack"
        elif finished:
            reward += CFG.reward_finish
            terminated = True
            reason = "lap_complete"
        elif stuck:
            reward += CFG.penalty_stuck
            terminated = True
            reason = "stuck"

        truncated = False
        ep_time = time.monotonic() - self.episode_start_t

        if terminated:
            self.io.stop()
            print(
                f"episode end: {reason:12s} | "
                f"progress={self.total_forward_progress:.2f}/{self.track.length:.2f}m | "
                f"time={ep_time:.1f}s | speed={speed:.2f} | "
                f"center={dist_center:.3f} | front={front_m:.2f}m"
            )

        info = {
            "reason": reason,
            "lap_time_s": ep_time if finished else None,
            "progress_m": self.total_forward_progress,
            "progress_ratio": self.total_forward_progress / max(self.track.length, 1e-6),
            "track_length_m": self.track.length,
            "start_distance_m": start_distance,
            "boundary_ratio": boundary_ratio,
            "front_lidar_m": front_m,
            "speed_cmd_mps": speed,
            "steer_cmd_rad": steering,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.io.close()
        finally:
            super().close()
