"""
Convergence test for the tan-sampled interaction module in physics.py
  Two parameters rule the interaction convergence 
   1) beta_target -> interaction extent (how far from the lasers focus we have 
      to go so that the interaction does not modify the electron state any more) 
   2) points_per_rayleigh -> the number of points in each rayleigh_range, it 
      gives us the point density used
  They converge to different limits, so we sweep each with the other fixed

  We compare successive refinements. "Converged" means the answer stops 
  changing as we refine. Thus, we look for a *first plateau*, not the smallest 
  value of difference.
"""
# %%
import time
import numpy as np
import torch
from physics import Config,ElectronModulator,interaction
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Physical parameters, fixed for the convergence test
# -----------------------------------------------------------------------------
freq1_init   = 1.0
num_apt_init = 0.2
kinetic_init = 30e3
field_init   = 0.75
diff_init    = freq1_init*(0.5-1.0) # phase matched so interaction is maximized
dist_init    = 0.0

cfg0 = Config(freq1_init,num_apt_init,1e-3,10)
m0   = ElectronModulator(cfg0,kinetic_init,field_init,diff_init,0.0)
print(m0.lmax) # check that the number of sidebands is a reasonable number

# -----------------------------------------------------------------------------
# fixed tolerance for convergence
# -----------------------------------------------------------------------------
tolerance = 1e-3 # converged when max|dP| < tolerance 

# we set to no gradients, as they are not needed for a convergence test
torch.set_grad_enabled(False)  

def single_test(beta_target, pts, track=False):
    cfg = Config(frequency1=freq1_init,numerical_aperture=num_apt_init,
                 beta_target=beta_target,points_per_rayleigh=pts)
    m   = ElectronModulator(cfg,kinetic_init,field_init,diff_init,dist_init)
    a0  = torch.zeros((m.L,m.L), dtype=torch.complex128)
    a0[m.lmax,m.lmax] = 1.0
    t   = time.time()
    # we also neglect the Gouy phase to characterize the PINEM-like interaction
    out = interaction(m,a0,tracking=track,use_gouy=False) 
    dt  = time.time() - t
    if track:
        peak = max(m.edge_leakage(out[i]) for i in range(out.shape[0]))
        return {"peak_leak": peak, "lmax": m.lmax, "nseg": len(m.z_mid)
                , "dt": dt}
    p = (torch.abs(out) ** 2).numpy()
    return {"p": p, "nseg": len(m.z_mid), "lmax": m.lmax, 
            "ZLP weight": p[m.lmax,m.lmax]
            , "edge": m.edge_leakage(out), "dt": dt}

def spread(p,lmax):
    lv = np.abs(np.arange(p.shape[0]) - lmax)
    return float((p * (lv[:,None] + lv[None,:])).sum())

def dmax(pa,pb):
    return float(np.abs(pa-pb).max())

def sweep_test(name, knob_name, knob_values, make_cfg):
    print(f"\n==== {name} ====")
    print(f"{'knob':^10} {'nseg':^6} {'ZLP weight':^14} "
          f"{'edge(1r)':^14} {'vs_prev':^12} {'time':^9}")
    results,deltas = [], []
    prev = None
    for kv in knob_values:
        bt,pts = make_cfg(kv)
        r = single_test(bt,pts)
        results.append(r)
        d = dmax(r["p"], prev["p"]) if prev is not None else None
        deltas.append(d)
        dstr = " --" if d is None else f"{d:.2e}"
        print(f" {kv:^10} {r['nseg']:^7d} {r['ZLP weight']:^14.4f}"
              f"{r['edge']:^14.2e} {dstr:^12} {r['dt']:^9.1f}s")
        prev = r
    # first plateau: first refinement whose change (and the next) is below 
    # tolerance
    plateau = None
    for i in range(1,len(deltas)):
        if deltas[i] is not None and deltas[i] < tolerance:
            plateau = knob_values[i]
            break
    if plateau is not None:
        print(f"-> first plateau (dP < {tolerance:.0e}): {knob_name} = {plateau}")
    else:
        print(f"-> no plateau below {tolerance:.0e} in this range")
    return results,deltas

# -----------------------------------------------------------------------------
# Point density sweep, we first do this one to know where to fix this for the
# extent sweep
# -----------------------------------------------------------------------------
betamax_fixed  = 1e-3
density_values = [4, 8, 16, 32, 64, 128, 256, 512]
dens_res, dens_delt = sweep_test(
    f"Point density (beta target = {betamax_fixed:.0e}) fixed",
    "points_per_rayleigh", density_values, 
    make_cfg = lambda k: (betamax_fixed, k))

# -----------------------------------------------------------------------------
# Interaction extent sweep we fix the density at 16
# -----------------------------------------------------------------------------
betamax_values = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4] 
density_fixed  = 256
beta_res, beta_delt = sweep_test(
    f"Interaction extent (points_per_rayleigh = {density_fixed:.0e}) fixed",
    "beta_target", betamax_values, 
    make_cfg = lambda bt: (bt, density_fixed))

# %%
