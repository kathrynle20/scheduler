from schedulers.base import Scheduler
from schedulers.baseline import BaselineScheduler
from schedulers.hybrid import HybridScheduler

__all__ = ["Scheduler", "BaselineScheduler", "HybridScheduler"]


def build(name: str, config: dict) -> Scheduler:
    if name == "baseline":
        return BaselineScheduler()
    if name == "hybrid":
        return HybridScheduler(**(config.get("hybrid") or {}))
    raise ValueError(f"unknown scheduler: {name}")
