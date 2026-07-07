"""
Reproducibility utilities.

Seeds every relevant RNG (Python, NumPy, PyTorch CPU/CUDA/MPS) and enables
deterministic algorithms so that an experiment run is bit-for-bit (or as close
as the backend allows) reproducible given a fixed `seed`.

Usage:
    from src.utils.reproducibility import set_seed, make_generator, seed_worker
    set_seed(config['seed'])            # call once at the start of a run
    DataLoader(..., shuffle=True,
               generator=make_generator(config['seed']),
               worker_init_fn=seed_worker)

Note on MPS (Apple Silicon): PyTorch does not guarantee bitwise determinism for
every operator on the MPS backend. Seeding makes runs reproducible up to any
residual backend nondeterminism; for fully reproducible numbers prefer the CPU
backend or pin the exact PyTorch / OS versions recorded by `log_environment()`.
"""

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed all RNGs and (optionally) request deterministic algorithms.

    Returns the seed so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the default CPU generator

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        # Prefer deterministic kernels; warn instead of crashing on ops that
        # lack a deterministic implementation (common on MPS).
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            pass

    return seed


def make_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """A CPU torch.Generator seeded for deterministic DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id: int) -> None:
    """worker_init_fn that seeds NumPy/Python RNGs inside DataLoader workers."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def log_environment() -> dict:
    """Return a dict of version/seed-relevant environment info for the run log."""
    info = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    return info
