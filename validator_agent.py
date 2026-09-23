"""Physical safety checks shared by the simulator and an eventual serving API."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhysicsLimits:
    cut_in_ms: float = 3.0
    cut_out_ms: float = 25.0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.cut_in_ms) and np.isfinite(self.cut_out_ms)):
            raise ValueError("Wind speed limits must be finite.")
        if not 0 <= self.cut_in_ms < self.cut_out_ms:
            raise ValueError("Require 0 <= cut-in < cut-out.")


def validate_power(
    predicted_power: np.ndarray,
    wind_speed_ms: np.ndarray,
    limits: PhysicsLimits | None = None,
    *,
    apply_wind_limits: bool = True,
) -> np.ndarray:
    """Bound power to [0, 1]; apply shutdowns only for comparable hub-height wind.

    Arrays must have the same nonempty [hours, turbines] shape. Invalid numeric
    inputs fail explicitly: NaN forecasts must never become plausible output.
    The exact cut-in/cut-out boundaries remain eligible for generation.
    """
    limits = limits or PhysicsLimits()
    power = np.asarray(predicted_power, dtype=np.float64)
    wind = np.asarray(wind_speed_ms, dtype=np.float64)
    if power.ndim != 2 or min(power.shape) == 0 or power.shape != wind.shape:
        raise ValueError("Power and wind require identical nonempty [hours, turbines] shapes.")
    if not np.isfinite(power).all() or not np.isfinite(wind).all():
        raise ValueError("Power and wind must contain only finite numbers.")
    if (wind < 0).any():
        raise ValueError("Wind speed cannot be negative.")
    result = np.clip(power, 0.0, 1.0)
    if apply_wind_limits:
        result[(wind < limits.cut_in_ms) | (wind > limits.cut_out_ms)] = 0.0
    return result.astype(np.float32)


def modulus_flow_features(topography: Any, boundary_conditions: Any) -> Any:
    """Extension point for a calibrated NVIDIA Modulus flow model, not a mock.

    Requires terrain, turbine geometry and validated boundary conditions before
    physics-derived features can be claimed or used in training.
    """
    raise NotImplementedError("A calibrated Modulus flow model and terrain inputs are required.")
