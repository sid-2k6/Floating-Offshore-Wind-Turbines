"""Physics-informed RL dataset construction for floating offshore wind turbines.

This package converts the FLOATBench tower-fatigue benchmark (a static
condition -> damage regression dataset) into a reinforcement-learning dataset
with explicit blade-pitch / yaw / IPC control actions, an IoT sensor
observation layer, and a load-relief reward.

Modules
-------
turbine     IEA-22-280-RWT reference properties + baseline operating schedule
aero        Calibrated BEM-lite rotor model (pitch / yaw action sensitivity)
load_model  Aero vs wave tower-load decomposition, calibrated on FLOATBench
damage      Damage rescaling under control action (ratio-anchored)
actions     Action space, actuator rate limits and duty cost
iot         IoT sensor layer (noise, quantisation, dropout, latency)
mdp         Sea-state episode construction, transitions and reward
build_dataset  End-to-end pipeline entry point
env         Offline replay environment over the generated transitions
"""

__version__ = "0.1.0"

__all__ = [
    "actions",
    "aero",
    "build_dataset",
    "config",
    "damage",
    "env",
    "floatbench",
    "iot",
    "load_model",
    "mdp",
    "turbine",
]
