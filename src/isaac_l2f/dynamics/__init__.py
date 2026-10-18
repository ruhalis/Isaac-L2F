"""Pure-PyTorch L2F quadrotor dynamics (batched, device-agnostic)."""

from .init import DISTILLATION_INIT, sample_initial_state
from .model import (
    Derivative,
    action_to_command,
    dxdt,
    motor_step_closed_form,
    rk4_step,
    rotor_thrust,
    rotor_thrust_mean,
    rotor_wrench,
    step,
    wrench_from_thrust,
)
from .observation import privileged_observation, student_observation
from .params import Params, load_registry
from .randomize import sample_parameters
from .reward import reward, terminated
from .state import State

__all__ = [
    "DISTILLATION_INIT", "Derivative", "Params", "State", "action_to_command", "dxdt", "load_registry",
    "motor_step_closed_form", "privileged_observation", "reward", "rk4_step", "rotor_thrust",
    "rotor_thrust_mean", "rotor_wrench", "sample_initial_state", "sample_parameters", "step", "student_observation",
    "terminated", "wrench_from_thrust",
]  # fmt: skip
