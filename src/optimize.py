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
c_au = scipy.constants.physical_constants['inverse fine-structure constant'][0]

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
            leak = model.edge_leakage(alpha)
        if leak > 1e-2:
            warnings.warn(f"edge leakage {leak:.1e} at step {step}")
        history.append((loss.item(), doc.item(), leak
                        ,model.electric.item()*nm_au/eV_au
                        ,model.kinetic.item()/eV_au
                        ,model.fr_diff.item()/eV_au
                        ,model.distance.item()/mm_au))
    return history

"""
---------------------- Single value opimization test --------------------------
We try out the optimizer by taking the free-propagation distance as the one 
parameter to optimize. We contrast the obtained result of the DOC_1 with the 
analytical PINEM maximum of DOC_1 ~ 0.34.
"""
if __name__ == '__main__':
    from physics import Config,ElectronModulator
    import matplotlib.pyplot as plt
    import numpy as np
    freq1    = 1.0
    numap    = 0.2
    beta_tgt = 1e-3
    pts_p_rl = 64
    kinetic  = 30e3
    field    = 0.2
    freq_dif = -0.5

    cfg = Config(freq1,numap,beta_tgt,pts_p_rl)
    m   = ElectronModulator(cfg,kinetic,field,freq_dif,0.0)

    def freeze_all_but_distance(model):
        for name, p in model.named_parameters():
            p.requires_grad_(name == 'log_distance')
    print(f'zT/zR = {(m.talbot_distance/m.rayleigh_range):.2f} \n',
        f'lmax  = {m.lmax} \n',
        f'v/c   = {(m.velocity/c_au):.3f}\n'
        f'initial value distance: {(m.distance/mm_au):.3f} mm')
    freeze_all_but_distance(m)
    steps = 150
    lr    = 5e-3
    alpha0 = torch.zeros((m.L,m.L),dtype=torch.complex128)
    alpha0[m.lmax,m.lmax] = 1.0
    print(f'initial DOC_1 before optimization: {degree_of_coherence(m,m(alpha0),1):.3f}')

    hist = optimize(m,steps,lr,1)
    print(f'final DOC_1 after optimization: {hist[-1][1]:.3f}')
    print(f'optimum analytical PINEM limit   : {0.34}')

    doc_hist = np.array([h[1] for h in hist])
    plt.plot(doc_hist,label = 'optimization')
    plt.plot([0,steps],[0.34,0.34],color='k',linestyle='--'
            ,label='analytical limit (non-recoil)')
    plt.legend()
    plt.ylabel('DOC1')
    plt.xlabel('training steps')
    plt.title('single-parameter training run (non-recoil regime)\n' 
                + f'--parameter: free-propagation distance, lr = {lr}--')
    plt.show()
