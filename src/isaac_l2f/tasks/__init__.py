"""Isaac Lab task registrations for Isaac-L2F.

Isaac Lab 3.0 discovers this module through the ``isaaclab.tasks`` entry point and imports it
before every ``isaaclab`` CLI command so that the ``gym.register`` calls below run. Entry points
are strings, so this module needs only ``gymnasium`` and stays importable without Isaac Lab;
the env modules themselves import Isaac Lab and are Linux-only.

Each env registers one id; physics is Newton only, through the ``newton_mjwarp`` preset of its
``PresetCfg`` (also the default), selected on the command line::

    isaaclab train --task Isaac-L2F-Hover-Direct --rl_library rsl_rl --num_envs 1024 physics=newton_mjwarp --viz none
"""

from __future__ import annotations

import gymnasium as gym

HOVER_TASK_ID = "Isaac-L2F-Hover-Direct"

_HOVER = "isaac_l2f.tasks.hover"

gym.register(
    id=HOVER_TASK_ID,
    entry_point=f"{_HOVER}.hover_env:HoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_HOVER}.hover_env_cfg:HoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_HOVER}.agents.rsl_rl_ppo_cfg:HoverPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

__all__ = ["HOVER_TASK_ID"]
