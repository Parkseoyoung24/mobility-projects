#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from callbacks import FastLapCallback
from config import CFG
from env_fastlap import PhysicarFastLapEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    base_env = PhysicarFastLapEnv(max_speed=CFG.final_max_speed, random_starts=False)
    env = Monitor(base_env, filename="logs/monitor.csv")

    latest = Path("models/ppo_fastlap_latest.zip")
    if args.resume and latest.exists():
        print("resuming from", latest)
        model = PPO.load(str(latest), env=env, device="auto")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device="auto",
            tensorboard_log="logs/tb",
            learning_rate=2e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[128, 64], vf=[128, 64])),
        )

    cb = FastLapCallback("models", hours=args.hours, verbose=1)

    try:
        model.learn(
            total_timesteps=100_000_000,
            reset_num_timesteps=False,
            callback=cb,
        )
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        model.save("models/ppo_fastlap_latest")
        Path("models/metadata.json").write_text(json.dumps({
            "min_speed_mps": CFG.min_train_speed,
            "max_speed_mps": CFG.final_max_speed,
            "max_steer_rad": CFG.max_steer_rad,
            "objective": "start -> full loop -> same start; no offtrack/collision; minimize lap time",
        }, indent=2), encoding="utf-8")
        env.close()
        print("saved models/ppo_fastlap_latest.zip")
        if Path("models/ppo_fastlap_best.zip").exists():
            print("best lap model: models/ppo_fastlap_best.zip")
        else:
            print("no full lap yet, best progress model: models/ppo_fastlap_progress_best.zip")


if __name__ == "__main__":
    main()
