"""
Convergence test for the tan-sampled interaction module in physics.py.
  Two parameters rule the interaction convergence
   1) beta_target -> interaction extent (how far from the laser focus we have
      to go so that the interaction no longer modifies the electron state)
   2) points_per_rayleigh -> the number of points per Rayleigh range, i.e. the
      segment density used
  They converge to different limits, so we sweep each with the other fixed.

  We converge on the degree of coherence (DOC_m), the quantity we actually
  optimize. "Converged" means the objective stops changing as we refine 
  -> we look for a *first plateau* in DOC, not the smallest change.
"""
# %%
import time
import numpy as np
import torch
from physics import (Config, ElectronModulator, interaction,
                     free_propagation, mm_au)
from metric import degree_of_coherence

# -----------------------------------------------------------------------------
# Physical parameters, fixed for the convergence test
# -----------------------------------------------------------------------------
freq1_init   = 1.0
num_apt_init = 0.2
kinetic_init = 30e3
field_init   = 0.75
# phase matched so the interaction is maximized
diff_init    = freq1_init*(0.5-1.0) 
order        = 1  # DOC order to converge (the one we optimize)

# reference model: gives lmax for a sanity check and the Talbot distance which
#                  fixes the periodicity of compression
cfg0 = Config(freq1_init, num_apt_init, 1e-3, 10)
m0   = ElectronModulator(cfg0, kinetic_init, field_init, diff_init, 0.0)
print(f"lmax = {m0.lmax}")  # check the number of sidebands is reasonable
# representative compression distance held fixed for the whole test
prop_dist_mm = (m0.talbot_distance/6).item() / mm_au

# -----------------------------------------------------------------------------
# fixed tolerance for convergence (absolute change in DOC_m)
# -----------------------------------------------------------------------------
tolerance = 1e-2   # converged when |dDOC| < tolerance

# no gradients needed for a convergence test
torch.set_grad_enabled(False)


def single_test(beta_target, pts):
    cfg = Config(frequency1=freq1_init, numerical_aperture=num_apt_init,
                 beta_target=beta_target, points_per_rayleigh=pts)
    m   = ElectronModulator(cfg, kinetic_init, field_init, diff_init, prop_dist_mm)
    a0  = torch.zeros((m.L, m.L), dtype=torch.complex128)
    a0[m.lmax, m.lmax] = 1.0
    t   = time.time()
    # neglect the Gouy phase to characterize the PINEM-like interaction, then
    # free-propagate (DOC needs the post-propagation, bunched state)
    out = interaction(m, a0, use_gouy=False)
    out = free_propagation(m, out)
    doc = degree_of_coherence(m, out, order=order).item()  # DOC_order
    dt  = time.time() - t
    return {"doc": doc, "nseg": len(m.z_mid), "lmax": m.lmax,
            "edge": m.edge_leakage(out), "dt": dt}

def sweep_test(name, knob_name, knob_values, make_cfg):
    print(f"\n==== {name} ====")
    print(f"{'knob':^10} {'nseg':^7} {'DOC_1':^12} "
          f"{'dDOC':^12} {'edge':^12} {'time':^9}")
    results, deltas = [], []
    prev = None
    for kv in knob_values:
        bt, pts = make_cfg(kv)
        r = single_test(bt, pts)
        results.append(r)
        d = abs(r["doc"] - prev["doc"]) if prev is not None else None
        deltas.append(d)
        dstr = " --" if d is None else f"{d:.2e}"
        print(f" {kv:^10} {r['nseg']:^7d} {r['doc']:^12.5f} "
              f"{dstr:^12} {r['edge']:^12.2e} {r['dt']:^8.1f}s")
        prev = r
    # first plateau: first refinement whose change is below tolerance
    plateau = None
    for i in range(1, len(deltas)):
        if deltas[i] is not None and deltas[i] < tolerance:
            plateau = knob_values[i]
            break
    if plateau is not None:
        print(f"-> first plateau (|dDOC| < {tolerance:.0e}): {knob_name} = {plateau}")
    else:
        print(f"-> no plateau below {tolerance:.0e} in this range")
    return results, deltas

# -----------------------------------------------------------------------------
# Point density sweep -- do this first to know where to fix pts for the extent
# sweep
# -----------------------------------------------------------------------------
betamax_fixed  = 1e-4
density_values = [4, 8, 16, 32, 64, 128, 256, 512]
dens_res, dens_delt = sweep_test(
    f"Point density (beta_target = {betamax_fixed:.0e}) fixed",
    "points_per_rayleigh", density_values,
    make_cfg = lambda k: (betamax_fixed, k))

# -----------------------------------------------------------------------------
# Interaction extent sweep -- fix the density on the plateau found above
# -----------------------------------------------------------------------------
betamax_values = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
density_fixed  = 256
beta_res, beta_delt = sweep_test(
    f"Interaction extent (points_per_rayleigh = {density_fixed}) fixed",
    "beta_target", betamax_values,
    make_cfg = lambda bt: (bt, density_fixed))

# %%
