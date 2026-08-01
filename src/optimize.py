"""
Optimization of the temporal compression of the free-electron pulses. 
Compression is quantified by the degree of coherence DOC_m (see 
metrics.py). By default we work with the first order, m = 1, minimizing
the infidelity

    L = 1 - DOC_1

which lies in the [0,1], with L = 0 corresponding to perfect point-like 
bunching.

Two complementary routines are provided:
    - optimize: performs gradient descent on the trainable parameters of the
    differentiable forward model, Electron modulator, so gradients flow from 
    the loss through the alpha coefficients back to the physical parameters. 
    - scal_latent: maps the loss along one latent on a grid. This is needed 
    because the kinetic energy and frequency difference enter through a narrow 
    phase-matching resonance condition, which makes the landscape along those 
    directions multi-modal, with maxima spaced by roughly one resonance width.
    Gradient descent converges to whichever basin it starts in, so a scan is 
    the only way to see the others.
    - profile_distance: sweeps the free-propagation distance, which does not 
    enter the interaction, it only multiplies the comb by the quadratic Talbot 
    phase. This means the sweep is nearly free, since the interaction only 
    needs to be coputed once. Additionally, in normal conditions where only the
    antidiagonal of the energy matrix is populated, this phase is periodic in
    the distance, resulting in several local maxima within one period. Descent
    will only find the neares one, while the sweep finds the best. Finally, it
    removes a further problem from the scans: because the Talbot distance 
    depends on velocity (kinetic energy) and on the frequency difference, a 
    scan over these parameters at fixed distance would drift away from its 
    optimal phase and report a decrease that is unrelated to the interaction. 
    Profiling the distance at every scan point means the reported quantity is 
    max_z DOC_1, a function only of the interaction parameters alone.
"""

from xml.parsers.expat import model

import torch
import scipy
import warnings
from collections import namedtuple
from metric import degree_of_coherence,sinc_kernels
from physics import interaction, free_propagation

pi    = scipy.constants.pi
eV_au = (scipy.constants.physical_constants['electron volt-joule relationship'][0]
       / scipy.constants.physical_constants['atomic unit of energy'][0] )
mm_au = 1/(scipy.constants.physical_constants['atomic unit of length'][0]*1e3)
nm_au = 1/(scipy.constants.physical_constants['atomic unit of length'][0]*1e9)
c_au = scipy.constants.physical_constants['inverse fine-structure constant'][0]
_UNITS = {'log_electric': ('eV/nm', eV_au/nm_au),
          'log_kinetic': ('eV', eV_au),
          'log_fr_diff': ('eV', eV_au),
          'log_distance': ('mm', mm_au)}
Profile = namedtuple('Profile', 'doc distance talbot_fraction leak')

def _drift_phase(model):
    """
    Phase acquired by each sideband per unit Talbot fraction: the drift phase
    at z = f*zT is _drift_phase(model)*f.

    zT cancels in this parameterization, so the phase depends only on the
    frequency ratio -- at fixed Talbot fraction the drift is independent of the
    kinetic energy.
    """
    l = torch.arange(model.L, dtype=torch.float64) - model.lmax
    w = l[:, None]*model.frequency1 + l[None, :]*model.frequency2
    return 2*pi*w**2/model.fr_diff**2

# ------------------- gradient-based optimization -----------------------------
def optimize(model, steps, learning_rate, order=1, ref_freq=None,
             alpha_int=None, profile=False, n_z=400,span=0.5):
    """
    Gradient descent on the trainable parameters, minimizing L = 1 - DOC_order.

    Which parameters are trained is set by the caller through requires_grad,
    not by this function.

    Parameters
    ----------
    model: ElectronModulator -> optimized in place.
    steps: int -> number of Adam steps.
    learning_rate: float -> Adam learning rate, applied to every trainable
                   latent. A single value is appropriate because the latents
                   are normalized by their conditioning widths (see the width
                   buffers in physics.py).
    order: int -> order m of the degree of coherence (see metric.py).
    ref_freq: float or None -> probe frequency in eV, passed through to
              degree_of_coherence. None uses the model's own frequency
              difference.
    alpha_int: (L,L) complex128 tensor or None -> precomputed comb, i.e. the
               output of interaction(model,alpha0). Supply it when no
               interaction parameter is being trained, so that the interaction
               is not recomputed identically at every step. Raises if an
               interaction parameter does require grad, since the comb would
               then be stale.
    profile: bool -> if True the distance is maximized by scanning one 
             period of the Talbot range, otherwise it is optimized with 
             the Adam optimizer, the same as the other parameters. The sweep
             optimization is done under no_grad
    n_z: int -> number of points used in the scanning optimization
    span: float -> fraction of zT that the sweep utilizes, note that DOC_m is 
          usually periodic in zT/2 (as long as the energy matrix is dominantly
          populated in the antidiagonal), so by default it is set as 0.5 

    Returns
    -------
    list of tuple -> one row per step,
        (loss, doc, leak, electric, kinetic, fr_diff, distance)
    with the last four in input units: eV/nm, eV, eV and mm.
    """
    if alpha_int is not None:
        live = {nm for nm, p in model.named_parameters() if p.requires_grad}
        stale = live & {'log_electric', 'log_kinetic', 'log_fr_diff'}
        if stale:
            raise ValueError(
                f"alpha_int was precomputed but {sorted(stale)} require grad; "
                 "the interaction would be stale. Pass alpha_int=None.")
    if profile and model.log_distance.requires_grad:
        raise ValueError("profile=True maximizes over the distance directly; "
                         "freeze log_distance before calling.")
    
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    alpha0 = torch.zeros((model.L,model.L),dtype=torch.complex128)
    alpha0[model.lmax,model.lmax] = 1.0
    history = []
    for step in range(steps):
        opt.zero_grad()
        comb = (alpha_int if alpha_int is not None
                else interaction(model, alpha0, use_gouy=model.use_gouy))
        if profile:
            psi  = _drift_phase(model)
            frac = span*torch.arange(n_z, dtype=torch.float64)/n_z
            with torch.no_grad():                    # locate maximizer, no graph
                kern = sinc_kernels(model, order=order, ref_freq=ref_freq)
                k = int(torch.argmax(torch.stack([
                    degree_of_coherence(model, comb*torch.exp(1j*psi*f),
                                        order=order, ref_freq=ref_freq,
                                        kernels=kern)
                    for f in frac])))
            
            # NOTE: the call below deliberately omits kernels=kern. kern is
            # detached (built under no_grad), and the kernels depend on
            # log_fr_diff through freq2 -- reusing them here would drop that
            # gradient. Rebuilding once is cheap next to the n_z sweep.
            alpha   = comb*torch.exp(1j*psi*frac[k])  # single eval, with graph
            dist_mm = (frac[k]*model.talbot_distance).item()/mm_au
        else:
            alpha   = free_propagation(model, comb)
            dist_mm = model.distance.item()/mm_au
        doc  = degree_of_coherence(model, alpha, order=order, ref_freq=ref_freq)
        loss = 1 - doc
        # we check that the loss does not blow-up
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        with torch.no_grad():
            leak = float(model.edge_leakage(alpha))
            row = (loss.item(), doc.item(), leak,
                   model.electric.item()*nm_au/eV_au,
                   model.kinetic.item()/eV_au,
                   model.fr_diff.item()/eV_au,
                   dist_mm)
        opt.step()
        if leak > 1e-2:
            warnings.warn(f"edge leakage {leak:.1e} at step {step}")
        history.append(row)

    return history

# ------------------- grid-based profiling and scans --------------------------
def profile_distance(model, alpha0, n_z=400, order=1, ref_freq=None, 
                     alpha_int=None, return_curve=False, span=0.5):
    """
    Profile out the free-propagation distance: for the current interaction
    parameters, find the drift distance maximizing DOC_order.

    The distance does not enter the interaction. It only multiplies the comb
    by the quadratic Talbot phase, so the comb is computed once and only the 
    phase factor is swept: the cost is one interaction plus n_z metric 
    evaluations, rather than one interaction per trial distance. Sweeping 
    is worth it rather than descending because along the anti-diagonal l2 = -l1
    the phase reduces to 2 pi l1^2 z/zT, so the compression revives at 
    fractional multiples of zT and DOC_m has several local maxima within one 
    period.
  

    Parameters
    ----------
    model: ElectronModulator -> supplies the interaction, the frequencies, the
           Talbot distance and the frozen grid (lmax, L).
    alpha0: (L,L) complex128 tensor -> initial sideband amplitudes, normally
            the ZLP alpha0[lmax,lmax] = 1. Ignored if alpha_int is given.
    n_z: int -> number of trial distances, spanning [0,zT)*span uniformly. The
         grid spacing zT/n_z sets the resolution of the returned distance.
    order: int -> order m of the degree of coherence (see metric.py).
    ref_freq: float or None -> probe frequency in eV, passed through to
              degree_of_coherence. None uses the model's own frequency
              difference.
    alpha_int: (L,L) complex128 tensor or None -> precomputed comb, i.e. the
               output of interaction(model,alpha0). Supply it to avoid
               repeating the interaction when profiling several times at fixed
               interaction parameters. None computes it here.
    return_curve: bool -> if True, also return the full sweep, so the profile
                  can be plotted rather than only its maximum. This changes the
                  return arity, see below.
    span: float -> fraction of zT that the sweep utilizes, note that DOC_m is 
          usually periodic in zT/2 (as long as the energy matrix is dominantly
          populated in the antidiagonal), so by default it is set as 0.5 

    Returns
    -------
    Profile(doc,distance,talbot_fraction,leak) -> namedtuple with
        doc: () float64 tensor -> DOC_order at the best distance. Kept as a
             tensor (no .item()) so that the profiled objective stays
             differentiable. Wrap the call in torch.no_grad() when the gradient
             is not needed: the graph otherwise retains n_z copies of the
             (L,L,L,L) intermediates built inside degree_of_coherence.
        distance: () float64 tensor -> best distance, in mm. Carries the
                  gradient through zT, and hence through the kinetic energy and
                  the frequency difference.
        talbot_fraction: () float64 tensor -> the same distance as a fraction
                         of the Talbot distance. This is exactly k/n_z for the
                         winning grid index k, so zT cancels and it carries no
                         gradient.
        leak: float -> edge leakage of the comb, model.edge_leakage(alpha_int).
              Evaluated once rather than per distance: free propagation is a
              pure phase, so |alpha| does not depend on the distance.

    If return_curve is True, returns the 3-tuple (profile,z,docs) instead:
        profile: Profile -> as above.
        z: (n_z,) float64 tensor -> the swept distances in mm, in grid order,
           with z[argmax(docs)] equal to profile.distance.
        docs: (n_z,) float64 tensor -> DOC_order along that grid.
    """

    if alpha_int is None:
        alpha_int = interaction(model, alpha0,use_gouy=model.use_gouy)

    psi  = _drift_phase(model)
    zt   = model.talbot_distance
    frac = span*torch.arange(n_z, dtype=torch.float64)/n_z
    kern = sinc_kernels(model, order=order, ref_freq=ref_freq)
    docs = torch.stack([
        degree_of_coherence(model, alpha_int*torch.exp(1j*psi*f),
                            order=order, ref_freq=ref_freq, kernels=kern)
        for f in frac])
    k = int(torch.argmax(docs))
    z = frac * zt

    # free propagation is a pure phase, so |alpha| -- and the edge population
    # -- is the same at every trial distance: evaluate the leak once
    with torch.no_grad():
        leak = float(model.edge_leakage(alpha_int))

    # ---------------- unit conversion, return boundary only ----------------
    out = Profile(doc=docs[k], distance=z[k]/mm_au,
                  talbot_fraction=z[k]/zt, leak=leak)
    if return_curve:
        return out, z/mm_au, docs
    return out

def scan_latent(model, alpha0, name, values, order=1, ref_freq=None,n_z=400
                ,span=0.5):
    """
    Map DOC_order along one latent coordinate, with the free-propagation
    distance profiled out at every point.

    The distance is profiled rather than held fixed because zT depends on the
    velocity and on the frequency difference: a scan at fixed distance would
    drift away from its optimal drift phase and report a decrease that has
    nothing to do with the interaction. The reported quantity is therefore
    max_z DOC_order, a function of the interaction parameters alone.

    The model is mutated in place and restored on exit, rather than rebuilt at
    each point. Rebuilding would recompute the frozen grid (lmax, L, segments)
    and the frozen conditioning widths from the perturbed parameters, so lmax
    could change mid-scan and 'values' would mean a different physical detuning
    at every point.

    Parameters
    ----------
    model: ElectronModulator -> scanned in place, restored before returning.
    alpha0: (L,L) complex128 tensor -> initial sideband amplitudes.
    name: str -> name of the latent to scan, as it appears in
          model.named_parameters(), e.g. 'log_kinetic'. 'log_distance' is
          rejected, since it is the parameter being profiled out.
    values: sequence of float -> latent values to visit. These are absolute,
            not offsets: pass model.log_kinetic.item() + offsets to scan around
            the current point.
    order: int -> order m of the degree of coherence, passed to
           profile_distance.
    ref_freq: float or None -> probe frequency in eV, passed to
              profile_distance.
    n_z: int -> number of trial distances per point, passed to
         profile_distance.
    span: float -> fraction of zT utilized by the distance sweep passed to 
          profile_distance 

    Returns
    -------
    list of tuple -> one row per scanned value,
        (latent, physical, doc, distance, talbot_fraction, leak)
    with
        latent: float -> the value taken from 'values'. For log_kinetic and
                log_fr_diff one unit is one resonance width; for log_electric
                one unit is one e-fold.
        physical: float -> the corresponding physical value in input units,
                  i.e. _UNITS[name][0]: eV/nm for the field amplitude, eV for
                  the kinetic energy and the frequency difference.
        doc: float -> max_z DOC_order at that point.
        distance: float -> the maximizing distance, in mm.
        talbot_fraction: float -> the same distance as a fraction of zT.
        leak: float -> edge leakage of the comb at that point. Worth watching:
              a wide scan moves the phase matching enough to redistribute
              sideband weight, and the grid is frozen.

    All columns are plain floats, so the result can be passed straight to
    numpy.array. Unlike profile_distance, nothing here is differentiable: the
    sweep runs under torch.no_grad().
    """
    if name == 'log_distance':
        raise ValueError("cannot scan log_distance while profiling it out")
    
    p  = dict(model.named_parameters())[name]
    phys = name.removeprefix('log_')
    conv = _UNITS[name][1]
    rows = []
    x0 = p.item()
    with torch.no_grad():
        try:
            for v in values:
                p.fill_(float(v))
                pr = profile_distance(model, alpha0, n_z=n_z, order=order,
                                        ref_freq=ref_freq,span=span)
                rows.append((float(v),
                                getattr(model, phys).item()/conv,
                                pr.doc.item(),
                                pr.distance.item(),
                                pr.talbot_fraction.item(),
                                pr.leak))
        finally:
            p.fill_(x0)
    return rows


"""
---------------------- Single value opimization test --------------------------
We try out the optimizer by taking the free-propagation distance as the one 
parameter to optimize. We contrast the obtained result of the DOC_1 with the 
analytical PINEM maximum of DOC_1 ~ 0.34.

NOTE: unlike the rest of the project, the model here is built with
      use_gouy = False. This is deliberate rather than an oversight: the 0.34
      reference is the non-recoil PINEM limit, and it is the Gouy phase acting
      together with recoil that allows that limit to be exceeded. Switching it
      off is what makes the comparison below the right one to draw.
NOTE: The two distances printed below need not agree. In this configuration the
      drift landscape has four near-degenerate maxima within one Talbot period
      (DOC_1 spread ~4e-4, separated by ~50 mm), so descent and the sweep can
      settle in different basins at essentially the same compression. It is the
      DOC_1 values that are compared, and only in the direction the sweep can
      guarantee.
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
    m   = ElectronModulator(cfg,kinetic,field,freq_dif,0.0,use_gouy=False)

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

    # the interaction does not depend on the propagation distance, so with only
    # log_distance trainable the comb is identical at every step and at every
    # trial distance: compute it once and hand it to both routines
    with torch.no_grad():
        alpha_int = interaction(m, alpha0, use_gouy=m.use_gouy)
        doc0 = degree_of_coherence(m, free_propagation(m, alpha_int), 1)
    print(f'initial DOC_1 before optimization: {doc0:.3f}')

    hist = optimize(m,steps,lr,1,alpha_int=alpha_int)
    print(f'final DOC_1 after optimization: {hist[-1][1]:.3f}')
    print(f'optimum analytical PINEM limit   : {0.34}')

    with torch.no_grad():
        prof = profile_distance(m,alpha0,n_z=200,alpha_int=alpha_int)
    gap = prof.doc.item() - hist[-1][1]
    print(f'descent: DOC_1 = {hist[-1][1]:.4f} at d = {hist[-1][6]:.3f} mm')
    print(f'sweep  : DOC_1 = {prof.doc.item():.4f} at d = {prof.distance.item():.3f} mm')
    print(f'sweep - descent = {gap:+.4f}  (>= 0 up to the grid resolution)')
    assert gap > -5e-3, (
        'the sweep is a global maximum over the same objective and cannot be '
        'worse than descent: the two paths are not evaluating the same model '
        '(check use_gouy, ref_freq, and alpha_int staleness)')

    fig, ax = plt.subplots(1,1,figsize = (4,4))
    doc_hist = np.array([h[1] for h in hist])
    ax.plot(doc_hist,label = 'optimization')
    ax.axhline(0.34,color='k',linestyle='--'
            ,label='analytical limit (non-recoil)')
    ax.legend()
    ax.set_ylabel('DOC1')
    ax.set_xlabel('training steps')
    ax.set_title('single-parameter training run (non-recoil regime)\n' 
                + f'--parameter: free-propagation distance, lr = {lr}--')
    plt.show()
