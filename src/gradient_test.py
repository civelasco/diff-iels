"""
Finite-difference validation of autograd gradients for ElectronModulator

Here we compare the gradient obtained by loss.backward() vs a finite-difference 
scheme obtained by slightly varying the trainable parameters. We want to check 
that everything works for the dimensionless los-scale trainable parameters, which 
carry both a frozen scale (fixing the magnitude) and a frozen width (fixing the 
curvature), see physics.py.

NOTE: consider when choosing eps that the error is not monotonic in this 
      quantity, and picking eps badly makes a correct gradient look broken. 
      Here two error sources compete:
      1) truncation: linked to the fact that the finite difference approximates
         the derivative over a finite interval (error scales as eps**2)
      2) round-off: error associated with machine precision of floating point
         numbers (grows as 1/eps)
      The sum of these two errors should give a U-shaped curve in log-log, with
      straight line slopes of +2 and -1 on either side of the minimum. 
      The position of the minima depend on how strongly the loss cuves with respect
      to each parameter. The width normalization in physics.py removes most of the 
      spread that is present in a naive log parametrization, so a single eps should 
      be good enough for all parameters. The dictionary option is kept as a fallback
      and for diagnostic purposes.
"""
import torch
from metric import degree_of_coherence
def gradient_test(model,alpha0,eps = 1e-4,doc_order = 1,rtol = 1e-3
                  ,print_test=True,raise_on_fail=True):

    params = {nm: p for nm,p in model.named_parameters() if p.requires_grad}
    if not params:
        raise ValueError("no parameters require grad...")
    
    def loss():
        alpha = model(alpha0)
        return 1.0 - degree_of_coherence(model,alpha,doc_order)
    
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        loss().backward()
        analytic = {nm: p.grad.item() for nm, p in params.items()}

    results = {}
    with torch.no_grad():
        for nm,p in params.items():
            e  = eps[nm] if isinstance(eps, dict) else eps
            x0 = p.item()
            p.fill_(x0+e); Lp = loss().item()
            p.fill_(x0-e); Lm = loss().item()
            p.fill_(x0)
            num = (Lp - Lm) / (2*e)
            rel = abs(analytic[nm] - num) / (abs(num)+1e-12) 
            results[nm] = (analytic[nm],num,rel)

    if print_test:
        print(f'{'parameter':<14}{'analytic':>14}{'numeric':>14}{'rel err':>12}')
        for nm, (a,n,rel) in results.items():
            flag = '' if rel < rtol else " <--- oh no... CHECK!"
            print(f'{nm:<14}{a:>14.3e}{n:>14.3e}{rel:>12.2e}{flag}')
    
    if raise_on_fail:
        bad = {nm:r[2] for nm, r in results.items() if r[2]>=rtol}
        assert not bad, f"gradient check failed for {bad} (rtol={rtol})"
    
    return results

if __name__ == "__main__":
    from physics import Config, ElectronModulator
    import numpy as np
    import matplotlib.pyplot as plt
    frequency1  = 1.0  
    num_apt     = 0.01
    beta_target = 1e-4
    pts_per_z0  = 32
    cfg = Config(frequency1, num_apt, beta_target, pts_per_z0)
    kinetic     =  31e3    
    electric1   =  8e-3   
    fr_diff     =  0.5*frequency1
    d0          =  0.0
    model = ElectronModulator(cfg, kinetic, electric1, fr_diff, d0
                              ,use_gouy=True)
    alpha0 = torch.zeros((model.L, model.L), dtype=torch.complex128)
    alpha0[model.lmax, model.lmax] = 1.0
    eps_range = np.logspace(-1,-12,12)
    curves = {}
    for eps in eps_range:
        res = gradient_test(model, alpha0, eps=eps
                            ,print_test=False,raise_on_fail=False)
        for nm, (a,n,rel) in res.items():
            curves.setdefault(nm,[]).append(rel)
        print(f'---> done for {eps:g}')

    fig, ax = plt.subplots(figsize=(6,4))
    for nm, rels in curves.items():
        ax.loglog(eps_range,rels,'o-',label=nm)
    ax.set_xlabel(r'$\epsilon$, finite-difference step')
    ax.set_ylabel('relative error')
    ax.axhline(1e-3,ls=':',color='k')
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig('../plots/gradient test')
    plt.show()
