"""
Convergence test for the tan-sampled interaction module in physics.py.
  Two parameters rule the interaction convergence
   1) points_per_rayleigh -> the number of points per Rayleigh range, i.e. the
      segment density used
   2) beta_target -> interaction extent (how far from the laser focus we have
      to go so that the interaction no longer modifies the electron state)
  
  They converge to different limits, so we sweep each with the other fixed.

  We converge on the degree of coherence (DOC_m), the quantity we actually
  optimize. "Converged" means the objective stops changing as we refine 
  -> we look for a *first plateau* in DOC, not the smallest change.

  NOTE: Because theta_max = arctan(intdist/2z0) is bounded by pi/2 
  num_segments = 2*theta_max*points_per_rayleigh saturates no matter how small 
  beta_target gets, while the physical extent keeps growing. The outermost 
  segments then span a larger dz and become under-resolved, so pushing 
  beta_target too low makes the results worse rather than better. The extent 
  sweep should then be read where nseg still grows with beta_target, and treat 
  the PINEM analytic test as the more robust benchmark for the extent.
"""

import time
import torch
from physics import (Config, ElectronModulator, interaction,
                     free_propagation, mm_au)
from metric import degree_of_coherence

# -----------------------------------------------------------------------------
# Physical parameters, fixed for the convergence test
# -----------------------------------------------------------------------------
freq1_init   = 1.0
num_apt_init = 0.2
kinetic_init = 31e3
field_init   = 0.25
# phase matched so the interaction is maximized
diff_init    = freq1_init*0.5 
order        = 1  # DOC order to converge (the one we optimize)

# reference model: gives lmax for a sanity check and the Talbot distance which
#                  fixes the periodicity of compression
cfg0 = Config(freq1_init, num_apt_init, 1e-3, 10)
m0   = ElectronModulator(cfg0, kinetic_init, field_init, diff_init, 0.0)
print(f"lmax = {m0.lmax}")  # check the number of sidebands is reasonable
# representative compression distance near where the DOC_1 is maximized in the 
# PINEM nonrecoil limit
prop_dist_mm = 0.073*m0.talbot_distance/(m0._beta0_init*mm_au)

# -----------------------------------------------------------------------------
# fixed tolerance for convergence (absolute change in DOC_m)
# -----------------------------------------------------------------------------
tolerance = 1e-3 # converged when |dDOC| < tolerance

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


if __name__ == "__main__":
    # point density sweep
    betamax_fixed  = 1e-4
    density_values = [4, 8, 16, 32, 64, 128, 256]
    dens_res, dens_delt = sweep_test(
        f"Point density (beta_target = {betamax_fixed:.0e}) fixed",
        "points_per_rayleigh", density_values,
        make_cfg = lambda k: (betamax_fixed, k))

    # extent sweep
    betamax_values = [1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4]
    density_fixed  = 128
    beta_res, beta_delt = sweep_test(
        f"Interaction extent (points_per_rayleigh = {density_fixed}) fixed",
        "beta_target", betamax_values,
        make_cfg = lambda bt: (bt, density_fixed))

    import matplotlib.pyplot as plt

    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (10,4))

    d_vals = density_values[1:]
    ax1.semilogy(d_vals, dens_delt[1:])
    ax1.axhline(tolerance,ls = ':',c = 'k')
    ax1.set_xlabel(r'point density per $z_0$')
    ax1.set_ylabel(r'|$\Delta$ DOC| vs previous')
    ax1.set_title('Density: successive change (extent fixed)')
    
    b_vals = betamax_values
    ax2.semilogx(b_vals,[r['doc'] for r in beta_res],color = 'C1')
    ax2.invert_xaxis()
    ax2.set_xlabel(r'$\beta_{\rm target}$' + '\n' 
                   + r'(smaller $\to$ longer interaction)')
    ax2.set_ylabel(r'DOC$_1$')
    ax2.set_title('Extent: DOC vs window')

    axn = ax2.twinx()
    axn.semilogy(b_vals,[r['nseg'] for r in beta_res],color = 'C2')
    axn.set_ylabel('number of segments',color = 'C2')
    
    plt.tight_layout()
    plt.savefig('../plots/convergence test')
    plt.show()
    
