"""Load the agent-wide YAML config once at import time and expose it as `cfg`."""

import os

from omegaconf import OmegaConf

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_YAML = os.path.join(_CONFIG_DIR, "agent.yaml")

cfg = OmegaConf.load(_AGENT_YAML)
