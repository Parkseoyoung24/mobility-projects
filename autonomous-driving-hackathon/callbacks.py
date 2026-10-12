import time
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

class FastLapCallback(BaseCallback):
    def __init__(self, model_dir="models", hours=1.0, verbose=1):
        super().__init__(verbose)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + float(hours) * 3600.0
        self.best_lap = None
        self.best_progress = -1.0
        self.last_periodic = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            progress = float(info.get("progress_m", 0.0))
            reason = info.get("reason", "")

            if self.best_lap is None and progress > self.best_progress:
                self.best_progress = progress
                self.model.save(str(self.model_dir / "ppo_fastlap_progress_best"))

            if reason == "lap_complete":
                lap = float(info["lap_time_s"])
                if self.best_lap is None or lap < self.best_lap:
                    self.best_lap = lap
                    self.model.save(str(self.model_dir / "ppo_fastlap_best"))
                    print(f"new best lap: {lap:.2f}s, saved ppo_fastlap_best.zip")

        if self.num_timesteps - self.last_periodic >= 5000:
            self.model.save(str(self.model_dir / "ppo_fastlap_latest"))
            self.last_periodic = self.num_timesteps

        if time.monotonic() >= self.deadline:
            print("training time limit reached")
            return False
        return True
