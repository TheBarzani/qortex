"""
Classical generator–discriminator (GAN-style) pipeline for swaption surfaces.

This package provides:
- PyTorch models for Generator and Discriminator
- Dataset wrapper for context + target surfaces
- Training loop for a conditional GAN
- Utilities for parameter counting, seeding, etc.
"""

from .models import Generator, Discriminator
from .utils import count_parameters