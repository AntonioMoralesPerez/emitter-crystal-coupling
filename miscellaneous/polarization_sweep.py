"""
    Docstring for parameter_sweep.py

This is a script for internal check on how polarization affects the emitter
dynamics. An emitter position is fixed and a sweep for different polarizations is
perform.
"""

# %% Import necessary packages
import numpy as np
import meep as mp  # needed for the numerical simulation of the photonic crystal
from meep import mpb  # same case as the previous package
import pythtb as tb  # for easy simulation of the tight-binding model

# some imports for plotting utilities
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import src as mf  # our package functions to work with
