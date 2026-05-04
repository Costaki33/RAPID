"""Execution strategies that wrap the backends.

- :mod:`rapid.runners.single_gpu` — one backend instance on one device.
- :mod:`rapid.runners.dual_gpu` — two backend instances on two GPUs via Ray,
  station shards split evenly.
- :mod:`rapid.runners.cpu_worker_sweep` — a single GPU backend paired with a
  variable-size Ray pool of CPU preprocessing workers.
"""
