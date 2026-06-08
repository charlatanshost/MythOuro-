"""
MythOuro training scripts.

The training entry points live in script files whose names start with a
parameter-budget tag (`1b_fine_web_edu.py`, `3b_fine_web_edu.py`).
Python module names can't start with a digit, so the console-script
entry points in `pyproject.toml` go through `training.cli`, which uses
`runpy` to execute the script files directly.
"""
