"""
Compression metric for the free-electron pulses. We use the degree of coherence
(DOC_m) as the measure of temporal compression, computed from the alpha
coefficients obtained after the interaction and the subsequent free 
propagation. This quantity is the absolute square value of the Fourier 
transform of the electron probability density at a frequency m * omega (with m 
an integer) taken over a single optical period 2pi/omega. By default, we fix 
omega at the frequency difference omega1 - omega2. For our purposes, this 
function measures how much the mode of frequency m * omega "sees" the electron 
as a point particle within an optical period. Simply put, DOC_m = 0 for a broad 
electron and DOC_m = 1 for all m's in the case of a point-like electron (i.e. a 
Dirac delta distribution in time).

Mathematically:
DOC_m = |I_m/I_0|^2
where I_m = ∑_{l1 l1',l2 l2'} sinc[(m W - (l1-l1') w1 - (l2-l2') w2) pi/W] D_12
with sinc(x) = sin(x)/x, W = w1 - w2 and
D_12 = alpha*_{l1',l2'} alpha_{l1,l2}
"""

# %%
import torch
import scipy.constants
from collections import namedtuple

eV_au = (scipy.constants.physical_constants['electron volt-joule relationship'][0]
       / scipy.constants.physical_constants['atomic unit of energy'][0] )

Kernels = namedtuple('Kernels', 'sincm sinc0 order ref_freq')

def sinc_kernels(params, order=1, ref_freq=None):
    """
    The alpha-independent part of DOC_m: the sinc weights entering I_m and I_0.

    These depend only on the frequencies, lmax, the order and the probe, so for
    a sweep in which only alpha changes (e.g. over the free-propagation
    distance) they can be built once and passed to degree_of_coherence instead
    of being rebuilt on every call.

    Parameters
    ----------
    params: ElectronModulator -> gives the frequencies and the frozen grid.
    order: int -> order m, baked into sincm as sinc(m - argsinc).
    ref_freq: float or None -> probe frequency in eV, baked into argsinc. None
              uses the model's own frequency difference.

    Returns
    -------
    Kernels(sincm,sinc0,order,ref_freq) -> the two (L,L,L,L) float64 weight
        tensors, together with the order and the probe they were built for, so
        that degree_of_coherence can reject a mismatched pair. ref_freq is
        stored as passed, i.e. in eV, before the conversion to atomic units.

    NOTE: these tensors carry gradient. argsinc depends on freq2 = freq1 +
          fr_diff, hence on log_fr_diff. Kernels built inside a torch.no_grad()
          block are detached, and reusing them for a gradient-carrying
          evaluation silently drops that contribution.
    """
    freq1 = params.frequency1
    freq2 = params.frequency2
    lmax  = params.lmax
    if ref_freq is None:
       ref_au = params.fr_diff
    else:
       ref_au = ref_freq*eV_au

    l = torch.arange(-lmax,lmax+1,1)
    argsinc = ((l[:,None,None,None]-l[None,None,:,None])*freq1
              +(l[None,:,None,None]-l[None,None,None,:])*freq2)/ref_au
    sincm   = torch.sinc(order-argsinc)
    sinc0   = torch.sinc(argsinc)

    return Kernels(sincm, sinc0, order, ref_freq)

def degree_of_coherence(params, alpha, order=1, ref_freq=None, kernels=None):
    """ 
       Our measure of compression, computed through the alpha coefficients 
       obtained after interaction and free propagation.
    
       Parameters 
       ----------
       params: ElectronModulator -> gives the physical quantities (frequencies,
               and the frozen grid (lmax, L). 
       alpha: (L,L) complex128 tensor -> sideband amplitudes. |alpha[i,j]|^2 is 
               the probability that the electron exchanged l1 = i-lmax photons with
               beam 1 and l2 = j - lmax with beam 2. alpha[lmax,lmax] corresponds 
               to the ZLP. The amplitude is normalized to sum |alpha|**2 = 1.
       order: int -> order in which the degree of coherence is calculated, for 
               order=m measures the compression of the pulse against a mode of 
               frequency W*m, with W the reference frequency (see below). 
       ref_freq: frequency (in eV) of the mode against which compression is 
                 measured, by default it is set to the difference between w1 and 
                 w2. When it differs from the frequency difference, the window of 
                 optimization and the periodicity of the pulse train might be 
                 mismatched. Passed to sinc_kernels.
       kernels: Kernels or None -> precomputed output of sinc_kernels. Supply
                 it to avoid rebuilding the alpha-independent part on every
                 call. Must have been built with the same order and ref_freq;
                 a mismatch raises rather than silently returning the wrong
                 order. None computes it here.
    
       Returns
       -------
       float64 -> DOC_m, degree of coherence of order m, which is 0 for a spread
                  out electron pulse, and 1 for a compressed pulse.
       """
    if kernels is None: 
       kernels = sinc_kernels(params, order=order, ref_freq=ref_freq)
    elif kernels.order != order or kernels.ref_freq != ref_freq:
       raise ValueError(
           f"kernels built for order={kernels.order}, ref_freq={kernels.ref_freq}; "
           f"called with order={order}, ref_freq={ref_freq}")
    sincm, sinc0 = kernels.sincm, kernels.sinc0
    d12   = alpha[:,:,None,None]*alpha[None,None,:,:].conj()
    im_i0 = (sincm*d12).sum()/(sinc0*d12).sum()
    return im_i0.real**2+im_i0.imag**2