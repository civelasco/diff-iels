"""
Optimizer for the temporal compression of the free-electron pulses. We tune the
trainable physical parameters of the ElectronModulator, the laser field
amplitude, the frequency difference, the electron kinetic energy, and the
free-propagation distance, to maximize the compression of the pulse after the
interaction with the lasers and the subsequent free propagation.

Compression is quantified by the degree of coherence DOC_m (see metrics.py). We
optimize against the first order, m = 1. Since the whole forward model is
differentiable, we minimize the infidelity

    L = 1 - DOC_1

by gradient descent (the gradients flow from the loss, through the alpha
coefficients, back to the physical parameters). L lies in [0,1], with L = 0
corresponding to perfect point-like bunching.
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.constants import pi
from physics import Config,ElectronModulator

