"""Simulation configuration: sequence, hardware, and the field of view.

Deliberately shaped like SNAKE's ``SimConfig`` (``GreConfig`` + ``HardwareConfig`` +
FOV), because the two simulators are meant to be compared and a reader of one should
recognise the other. The differences are only where ours is continuous: the FOV here
describes the *evaluation* grid and the reference raster, never the object, which has no
grid at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..conventions import FOV


@dataclass
class GreConfig:
    """Spoiled gradient-echo timing. Times in ms, flip angle in degrees."""

    tr_ms: float = 50.0
    te_ms: float = 30.0
    flip_deg: float = 15.0

    def __post_init__(self) -> None:
        if self.te_ms >= self.tr_ms:
            raise ValueError(f"TE ({self.te_ms}) must be shorter than TR ({self.tr_ms})")
        if not 0.0 <= self.flip_deg <= 180.0:
            raise ValueError(f"flip angle {self.flip_deg} out of range")
        if self.tr_ms <= 0 or self.te_ms <= 0:
            raise ValueError("TR and TE must be positive")


@dataclass
class HardwareConfig:
    """Scanner side. ``field`` keys the relaxation table and is never defaulted silently."""

    n_coils: int = 8
    dwell_time_ms: float = 1e-3
    field: str = "1.5T"

    def __post_init__(self) -> None:
        from ..physics.tissue_props import table

        table(self.field)          # raises with the valid field list if unknown


@dataclass
class SimConfig:
    """Everything an acquisition needs that is not the object or the trajectory."""

    fov: FOV
    seq: GreConfig = field(default_factory=GreConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    seed: int = 0

    @property
    def field(self) -> str:
        return self.hardware.field
