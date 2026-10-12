from dataclasses import dataclass

@dataclass
class FastLapConfig:
    camera_topic: str = "/camera/image_raw/compressed"
    scan_topic: str = "/scan"
    speed_topic: str = "/speed"
    steering_topic: str = "/steering"
    camera_pan_topic: str = "/camera/pan"
    camera_tilt_topic: str = "/camera/tilt"
    sim_base: str = "http://localhost/sim/api"

    image_w: int = 96
    image_h: int = 96
    lidar_sectors: int = 9
    lidar_clip_m: float = 4.0

    min_train_speed: float = 0.0
    final_max_speed: float = 3.0
    deploy_max_speed: float = 3.0
    max_steer_rad: float = 0.349

    camera_pan_rad: float = 0.0
    camera_tilt_rad: float = -0.262
    control_hz: float = 15.0
    sensor_timeout_s: float = 8.0

    fallback_track_half_width_m: float = 0.55
    car_half_width_m: float = 0.11
    boundary_buffer_m: float = 0.04
    collision_lidar_m: float = 0.11
    obstacle_risk_m: float = 0.45

    lap_completion_ratio: float = 0.90
    finish_radius_m: float = 0.35
    finish_heading_tolerance_deg: float = 60.0

    stuck_time_s: float = 3.0
    stuck_progress_m: float = 0.12
    low_speed_threshold_mps: float = 0.12

    reward_progress: float = 8.0
    reward_time_penalty: float = 0.03
    reward_low_speed_penalty: float = 0.18
    reward_heading: float = 0.05
    reward_boundary_risk: float = 0.25
    reward_obstacle_risk: float = 0.15
    reward_steer_jerk: float = 0.03

    reward_finish: float = 250.0
    penalty_offtrack: float = -80.0
    penalty_collision: float = -100.0
    penalty_stuck: float = -120.0

CFG = FastLapConfig()
