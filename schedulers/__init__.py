from schedulers.base import Scheduler
from schedulers.baseline import BaselineScheduler
from schedulers.hybrid import HybridScheduler
from schedulers.work_stealing import WorkStealingScheduler

__all__ = [
    "Scheduler",
    "BaselineScheduler",
    "HybridScheduler",
    "WorkStealingScheduler",
]


def build(name: str, config: dict) -> Scheduler:
    if name == "baseline":
        return BaselineScheduler()
    if name == "hybrid":
        return HybridScheduler(**(config.get("hybrid") or {}))
    if name == "work_stealing":
        return WorkStealingScheduler(**(config.get("work_stealing") or {}))
    raise ValueError(f"unknown scheduler: {name}")
