"""
Configuration for Uniform Rectangular Array (URA).
"""

import numpy as np


class URAConfig:
    """Configuration for Uniform Rectangular Array."""
    def __init__(self, rows=16, cols=16, dx=0.5, dy=0.7, freq_ghz=10.0):
        self.rows = rows
        self.cols = cols
        self.N = rows * cols  # Total elements: 256
        self.dx = dx  # Horizontal spacing in wavelengths
        self.dy = dy  # Vertical spacing in wavelengths
        self.freq = freq_ghz * 1e9
        self.c = 3e8
        self.wavelength = self.c / self.freq
        self.dx_meters = dx * self.wavelength
        self.dy_meters = dy * self.wavelength
