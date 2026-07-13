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

import torch
import scipy
import warnings
from metric import degree_of_coherence

eV_au = (scipy.constants.physical_constants['electron volt-joule relationship'][0]
       / scipy.constants.physical_constants['atomic unit of energy'][0] )

mm_au = 1/(scipy.constants.physical_constants['atomic unit of length'][0]*1e3)

nm_au = 1/(scipy.constants.physical_constants['atomic unit of length'][0]*1e9)

def optimize(model, steps, learning_rate, order=1):
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    alpha0 = torch.zeros((model.L,model.L),dtype=torch.complex128)
    alpha0[model.lmax,model.lmax] = 1.0
    history = []
    for step in range(steps):
        opt.zero_grad()
        alpha = model(alpha0)
        doc  = degree_of_coherence(model,alpha,order=order)
        loss = 1 - doc
        # we check that the loss does not blow-up
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        opt.step()
        with torch.no_grad():
            # we set limits to some parameters so they do not escape the 
            # physics-based bounds (kinetic > 0) and (w2 > w1 so fr_diff < 0).
            model.kinetic.clamp_(min = 1.0*eV_au)
            model.fr_diff.clamp_(max = -1e-5*eV_au)
            leak = model.edge_leakage(alpha)
        if leak > 1e-2:
            warnings.warn(f"edge leakage {leak:.1e} at step {step}")
        history.append((loss.item(), doc.item(), leak
                        ,model.electric.item()*nm_au/eV_au
                        ,model.kinetic.item()/eV_au
                        ,model.fr_diff.item()/eV_au
                        ,model.distance.item()/mm_au))
    return history

