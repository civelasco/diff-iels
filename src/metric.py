"""
Compression metric for the free-electron pulses. We use the degree of coherence
(DOC_m) as the measure of temporal compression, computed from the alpha
coefficients obtained after the interaction and the subsequent free 
propagation. This quantity is the absolute square value of the Fourier 
transform of the electron probability density at a frequency m * omega (with m 
an integer) taken over a single optical period 2pi/omega. In particular, we fix 
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.constants import pi
from physics import Config,ElectronModulator,interaction

def degree_of_coherence(params,alpha,order=1):
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
    order:  order in which the degree of coherence is calculated, for order=m
            measures the compression of the pulse against a mode of frequency
            W*m, with W the frequency difference between w1 and w2. 

    Returns
    -------
    float64 -> DOC_m, degree of coherence of order m, which is 0 for a spread
               out electron pulse, and 1 for a compressed pulse.
    """

    freq1 = params.frequency1
    freq2 = params.frequency2
    dfreq = params.fr_diff
    lmax  = params.lmax

    l = torch.arange(-lmax,lmax+1,1)
    argsinc = ((l[:,None,None,None]-l[None,None,:,None])*freq1
               +(l[None,:,None,None]-l[None,None,None,:])*freq2)/dfreq
    d12     = alpha[:,:,None,None]*alpha[None,None,:,:].conj()
    sincm   = torch.sinc(order-argsinc)
    sinc0   = torch.sinc(argsinc)
    
    im_i0 = ((sincm*d12).sum()/(sinc0*d12).sum())
    out   = im_i0.real**2+im_i0.imag**2
    return out

