"""
Test for physics.py based on the analytical PINEM limit. Complements the 
self-convergence sweep of convergence_test.py, which only tests when the 
integrator results stop shifting. Here, we check that the integrator actually
lands in the right result for a given case.

In the phase-matched, non-recoil limit (and neglecting the Gouy phase) 
the interaction reduces in the anti-diagonal l2 = -l1 to a single PINEM ladder 
driven by the difference-frequency cross term b12 (the single-beam terms 
a11/a22 are non-phase-matched and drop out with one beam alone giving no 
modulation). The exact reduction is

        P_n  ==  J_n( 4 * b12(0) * pi * zR )**2 ,      l1 = n, l2 = -n

    b12(0)   = e1 e2 / (2 v gamma w1 w2),   e2 = e1 (w2/w1)**0.5
    pi * zR  = integral of scale_fact along the grid (with extent -> inf)
    
This checks the matrix-exponential propagation, Hermiticity, phase matching, and
the coupling normalization all at once, against a closed form.
"""

import numpy as np
import torch
from scipy.special import jv
from physics import Config, ElectronModulator, interaction, c_au, eV_au

torch.set_grad_enabled(False)


def pinem_limit_test(field=0.2, freq1=1.0, freq2=0.5, na=0.2, verbose=True):
    # -- phase-matched electron energy: v/c = (w1 - w2)/(w1 + w2) -------------
    beta_pm  = (freq1 - freq2) / (freq1 + freq2)
    gamma_pm = 1.0 / np.sqrt(1.0 - beta_pm**2)
    ke_eV    = (gamma_pm - 1.0) * c_au**2 / eV_au # (gamma-1) m c^2, in eV
    diff     = freq2 - freq1                      # fr_diff so w2 is as asked

    cfg = Config(frequency1=freq1, numerical_aperture=na,
                 beta_target=1e-4, points_per_rayleigh=64) 
    m = ElectronModulator(cfg, ke_eV, field, diff, 0.0)
    
    a0 = torch.zeros((m.L, m.L), dtype=torch.complex128)
    a0[m.lmax, m.lmax] = 1.0
    out = interaction(m, a0, use_gouy=False)   # Gouy off = pure PINEM
    p = (torch.abs(out)**2).numpy()

    # -- anti-diagonal l2 = -l1  ->  column index L-1-i -----------------------
    L, lmax = m.L, m.lmax
    idx    = np.arange(L)
    P_anti = p[idx, L - 1 - idx]
    n      = idx - lmax

    # -- effective coupling from the model's OWN generator (finite-range) -----
    v, gm  = m.velocity.item(), m.gamma.item()
    f1, f2 = m.frequency1, m.frequency2.item()
    e1     = m.electric.item()
    e2     = e1 * (f2 / f1)**0.5
    b12_0  = e1 * e2 / (2 * v * gm * f1 * f2)
    I_env  = np.pi * m.rayleigh_range
    arg    = 4 * b12_0 * I_env                # Bessel argument

    # -- diagnostics ----------------------------------------------------------
    # analytic mismatch
    err       = float(np.abs(P_anti - jv(n, arg)**2).max())  
    # off-anti-diagonal pop
    leak_off  = float(1 - P_anti.sum())                      
    # non-recoil check (n=1)
    recoil_ph = (f1 + f2)**2 / (2 * v * gm**3 * c_au**2) * I_env
    # compare to paper's beta
    beta_paper = 2 * b12_0 * I_env                           

    if verbose:
        print(f"phase-matched: v/c={ (v/c_au):.4f}  KE={ke_eV/1e3:.2f} keV   "
              f"L={L} lmax={lmax} nseg={len(m.z_mid)}")
        print(f"b12(0)={b12_0:.3e}  I_env={I_env:.1f}  (pi*zR={np.pi*m.rayleigh_range:.1f})  "
              f"arg=4*b12*I_env={arg:.4f}")
        print(f"\n  n :   P_anti        J_n(arg)^2")
        for i in range(max(0, lmax-4), min(L, lmax+5)):
            print(f" {i-lmax:+2d} : {P_anti[i]:.6f}      {jv(i-lmax, arg)**2:.6f}")
        print(f"\nmax|P_anti - J_n^2|      = {err:.2e}   (analytic match)")
        print(f"off-anti-diagonal pop    = {leak_off:.2e}   (reduction valid if << 1)")
        print(f"recoil phase (n=1)       = {recoil_ph:.2e}   (non-recoil if << 1)")
        print(f"beta (=2*b12*I_env)      = {beta_paper:.4f}   <- compare to the paper's coupling")
    return {"err": err, "leak_off": leak_off, "recoil_phase": recoil_ph,
            "arg": arg, "beta_paper": beta_paper, "P_anti": P_anti}


if __name__ == "__main__":
    pinem_limit_test()
