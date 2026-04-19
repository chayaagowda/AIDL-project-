"""
Config loader — reads config/config.yaml and returns a nested namespace.
"""

import yaml
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(path: str = "config/config.yaml") -> SimpleNamespace:
    """
    Load YAML config file and return as a nested SimpleNamespace.

    Usage:
        cfg = load_config()
        print(cfg.training.batch_size)
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw)
