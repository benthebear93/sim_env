#!/usr/bin/env python3
"""Assemble Exp 4 with only Exp 3's right UR10e and HX5-D20 hand."""
from __future__ import annotations

import argparse
from pathlib import Path

from build_exp3 import build as build_ur10e_scene


def build(config_path=None):
    build_ur10e_scene(config_path, setup=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    build(parser.parse_args().config)
