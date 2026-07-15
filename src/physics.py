"""
===============================================================================
This code is based on Free-Space Optical Modulation of Free Electrons in the 
Continuous-Wave Regime by C.I. Velasco and F.J. García de Abajo, PRL 134 
123804 (2025).
===============================================================================
 
Here, we compute the evolution of a free electron that interacts with two
counter-propagating laser beams and its subsequent free-space propagation. 
 
The lasers are modeled as gaussian beams and we consider the electron is 
traveling along their optical axis. Each beam is characterized by their 
frequencies w1 and w2, the electric field amplitudes E1 and E2, and the 
numerical aperture of the optical system NA, which fixes the rayleigh range 
(zR), i.e. the distance in which the lasers are focused and their amplitudes 
are more intense. Besides the amplitude, each gaussian beam has an associated 
Gouy phase exp(i arctan(z/zR)), which adds a "momentum kick" that also 
affects the interaction probability between the electron and the lasers. 
Note that for the scheme to work, we must have w2<w1, which is automatically
enforced by construction in the module.
 
The electron is characterized by a kinetic energy distribution tightly 
centered around an average value Ke0, the spread is much smaller than the 
photon energies hbar(w1,w2). (so we consider a point-like distribution in 
energy space). 
 
The evolution comes in two steps, first we modulate the energy during the 
interaction of the electron with the laser beams, and then the electron is
spatially modulated by the interaction.
 
Numerically, the electron state is a 2D array of complex amplitudes 
alpha[l1,l2], where l1 and l2 the emitted (l1,l2 < 0) or absorbed (l1,l2 > 0)
photons with lasers 1 and 2 (energy shift is l1*hbar*w1 + l2*hbar*w2), 
truncated at |l1|,|l2| <= lmax (set by a coupling strenght that depends on 
interaction parameters). The interaction is integrated by slicing the optical 
axis into segments and applying a matrix_exp(-i M(z) dz) to alpha on each 
segment. Free propagation is instead a element-by-element multiplication by 
the Talbot phase.  

The module is a differentiable PyTorch nn.Module: the physical inputs (field
amplitudes, frequency difference, electron kinetic energy and final 
propagation distance) are trainable parameters, so that temporal compression 
of the electron pulse in a single optical period 2*pi*c/|w1-w2| can be 
optimized by gradient descent through the full evolution.  
===============================================================================
"""
import torch
import torch.nn as nn
import scipy
import math
from dataclasses import dataclass
# We use atomic units (electron charge = electron mass = hbar = 1.0) but we 
# expect the input to be using different system of units
# we define basic physical and mathematical constants we use along the code
pi = scipy.constants.pi # pi
# speed of light in atomic units = 1/\alpha (the fine-structure constant)
c_au = scipy.constants.physical_constants['inverse fine-structure constant'][0]
# electron volt to atomic units (multiply a quantity in eV to get atomic units 
# of energy)
eV_au = (scipy.constants.physical_constants['electron volt-joule relationship'][0]
       / scipy.constants.physical_constants['atomic unit of energy'][0] )
# nanometer (10**-9 meters) and millimeter (10**-3 meters) in atomic units
#  (multiply a quantity in nm or mm to get atomic units of length)
nm_au = 1/(scipy.constants.physical_constants['atomic unit of length'][0]*1e9)
mm_au = 1/(scipy.constants.physical_constants['atomic unit of length'][0]*1e3)

# -----------------------------------------------------------------------------
@dataclass
class Config:
    # these parameters are fixed
    frequency1: torch.Tensor  # frequency of laser 1
    numerical_aperture: float # numerical aperture of the laser system
    beta_target: float        # maximum sideband order considered
    points_per_rayleigh: int  # number of points used per Rayleigh range
    
# -----------------------------------------------------------------------------
def interaction(params,alpha,tracking = False,use_gouy = False):
    """Propagate the electron energy-state amplitudes through the two laser
       beam interaction.
    
    Parameters 
    ----------
    params: ElectronModulator -> gives the physical quantities (velocity, 
    gamma, frequencies, field, rayleigh_range) and the frozen grid (z_mid, dz, 
    lmax, L). 
    alpha: (L,L) complex128 tensor -> sideband amplitudes. |alpha[i,j]|^2 is 
           the probability that the electron exchanged l1 = i-lmax photons with
           beam 1 and l2 = j - lmax with beam 2. alpha[lmax,lmax] corresponds 
           to the ZLP. The amplitude is normalized to sum |alpha|**2 = 1.
    tracking: bool -> if True, returns the state after every segment
    use_gouy: bool -> includes the Gouy phase that dramatically modifies the 
              interaction

    Returns
    -------
    (L, L) complex128 -> if tracking is False (final state)
    (n_segments, L, L) complex128 -> if tracking True (per-segment trajectory)
    """

    v  = params.velocity
    gm = params.gamma
    freq1 = params.frequency1
    freq2 = params.frequency2
    electric1 = params.electric
    rayleigh = params.rayleigh_range
    lmax = params.lmax
    L    = params.L
    n_segments = len(params.z_mid)
    # -------------------------------------------
    def coupling_amplitudes():
        # here we define the position independent coupling amplitudes A_ij
        # and B_ij used for all segments. These amplitudes hold information 
        # about how strong is the coupling between the electron and the laser
        # light.
        e1 = electric1
        # the amplitude of the laser 2 is defined such that both lasers have 
        # the same **power** 
        e2 = electric1 * (freq2/freq1)**0.5
        e1c = e1.conj()
        e2c = e2.conj()

        a11 = e1*e1/(2*v*gm*freq1**2)
        a22 = e2*e2/(2*v*gm*freq2**2)
        a12 = e1*e2/(2*v*gm*freq1*freq2)
        a21 = a12
        a_row1  = torch.stack((a11,a12))
        a_row2  = torch.stack((a21,a22))
        a_coefs = torch.stack((a_row1,a_row2))

        b11 = e1*e1c/(2*v*gm*freq1**2)
        b22 = e2*e2c/(2*v*gm*freq2**2)
        b12 = e1*e2c/(2*v*gm*freq1*freq2)
        b21 = e2*e1c/(2*v*gm*freq1*freq2)
        b_row1  = torch.stack((b11,b12))
        b_row2  = torch.stack((b21,b22))
        b_coefs = torch.stack((b_row1,b_row2))

        return a_coefs,b_coefs
    # -------------------------------------------
    def segment_matrix(zn,amplitudes,use_gouy):
        """
        we add the segment-dependent factors to the coefficients
        since we take z0 equal for both lasers, the Gouy phase only 
        influences terms which have (Ei * Ei) or (Ei * Ej^*) and cancels
        otherwise
        """
        
        acf,bcf = amplitudes
        scale_fact = (1+(zn/rayleigh)**2)**-1
        if use_gouy:
            gouy_phase = torch.exp(2j*torch.arctan(zn/rayleigh))
        else:
            gouy_phase = 1.0
        a11 = acf[0,0] * scale_fact * gouy_phase
        a22 = acf[1,1] * scale_fact / gouy_phase
        a12 = acf[0,1] * scale_fact
        b11 = bcf[0,0] * scale_fact
        b22 = bcf[1,1] * scale_fact
        b12 = bcf[0,1] * scale_fact * gouy_phase
        kron    = torch.kron # kronecker deltas to vectorize the filling of M
        def shift(k):
            s   = torch.zeros(L,L,dtype=torch.complex128)
            idx = torch.arange(L-abs(k))
            if k >= 0: s[idx+k,idx] = 1.0
            else:      s[idx,idx-k] = 1.0
            return s
        S = {k: shift(k) for k in (-2,-1,0,1,2)}
        I = S[0]

        l = torch.arange(L,dtype=torch.float64)-lmax
        l1,l2 = l[:,None],l[None,:]
        # first, we fill the diagonal
        bsum  = b11+b22+b11.conj()+b22.conj()
        vfac_m = (1/v-1/c_au)
        vfac_p = (1/v+1/c_au)
        Dvals = (bsum+vfac_m*l1*freq1+vfac_p*l2*freq2
            +(freq1*l1-freq2*l2)**2/(2*v*gm**3*c_au**2)).to(torch.complex128)
        M = torch.diag(Dvals.reshape(L*L))
        M = M + (-a12)          *kron(S[-1], S[-1])
        M = M + (2*b12.conj())  *kron(S[-1], S[ 1])
        M = M + (-a12.conj())   *kron(S[ 1], S[ 1])
        M = M + (2*b12)         *kron(S[ 1], S[-1])
        M = M + (-a11.conj())   *kron(S[-2], I    )
        M = M + (-a11)          *kron(S[ 2], I    )
        M = M + (-a22.conj())   *kron(I    , S[-2])
        M = M + (-a22)          *kron(I    , S[ 2])
        return M
    # -------------------------------------------
    alpha = alpha.reshape(L*L).to(torch.complex128)
    # tracking will be used when we want to generate the sideband probability
    # maps. Otherwise, no need to save this information
    if tracking:
        alpha_i = torch.zeros((n_segments,L,L)
                              ,dtype = torch.complex128)
    amplitudes = coupling_amplitudes()
    for n,(zc,dzc) in enumerate(zip(params.z_mid,params.dz)):
        M  = segment_matrix(zc,amplitudes,use_gouy=use_gouy)
        U  = torch.matrix_exp(-1j * M * dzc)
        alpha = U @ alpha
        if tracking:
            alpha_i[n] = alpha.reshape(L,L)
    alpha = alpha.reshape(L,L)
    return alpha_i if tracking else alpha

# -----------------------------------------------------------------------------
def free_propagation(params,alpha):
    """Apply the free-space Talbot phase to the sideband amplitudes.

    Pure phase, applied elementwise (no mixing between sidebands): each 
    alpha[i,j] is multiplied by exp(i * phi_{l1,l2} * distance), where 
    l1 = i - lmax, l2 = j - lmax, and 
       phi = (l1 w1 + l2 w2)^2 / (2 v^3 gamma^3)
    We calculate the propagation phase as (2 pi/zT)(...)^2/dw^2, where 
    zT = 4 pi v^3 gamma^3 / dw^2 is the Talbot distance.
    We compute the propagation in the co-moving frame of the electron since it 
    removes the ballistic phase which would introduce strong oscillations that 
    do affect the duration of the compressed pulses.

    Parameters
    ----------
    params: ElectronModulator 
       Supplies velocity, frequencies, talbot_distance, distance, lmax, L
    alpha: (L, L) complex128 tensor
       Sideband amplitudes (same convention as interaction).
    
    Returns
    -------
    (L, L) complex128 tensor
       Amplitudes after propagating params.distance. |alpha|^2 is unchanged 
       (energy populations are conserved, only relative phases evolve).
    """

    v     = params.velocity
    freq1 = params.frequency1
    freq2 = params.frequency2
    dw    = params.fr_diff
    zt    = params.talbot_distance
    lmax  = params.lmax
    dist  = params.distance
    L     = params.L
    # Think about the propagation for the entire spectrum
    l     = torch.arange(0,L,1)-lmax
    l1,l2 = l[:,None], l[None,:]
    argument = (2*pi/zt)*(l1*freq1+l2*freq2)**2*dw**-2
    free_prop_mat = torch.exp(1j*argument*dist)
    return free_prop_mat*alpha

# =============================================================================
class ElectronModulator(nn.Module):
    def __init__(self,cfg,initial_kinetic,initial_field,
                 initial_freq_diff,initial_prop_dist,use_gouy=False):
        # ke0  -> initial guess for the kinetic energy of the electron
        # e0   -> initial guess for the electric field amplitude
        # dw0  -> initial guess for the frequency difference
        # d0   -> initial guess for the free-propagation distance
        # the parameters placed here are the ones to be optimized
        super().__init__()
        self.cfg = cfg
        self.use_gouy = use_gouy
        # -- convert from the input units to atomic untis ---------------------
        e0  = initial_field     * (eV_au/nm_au) 
        dw0 = initial_freq_diff * eV_au
        ke0 = initial_kinetic   * eV_au    
        d0  = initial_prop_dist * mm_au
        self.frequency1 = self.cfg.frequency1*eV_au
        # -- trainable parameters  --------------------------------------------
        # the parameters in atomic units are very different in magnitude, so 
        # we use the inputs as a frozen scale and optimize a quantity in log 
        # scale, that remians O(1) with reasonable learning rates.
        # we are missing our fourth trainable parameter: the free-propagation 
        # distance, which we introduce later after defining a few quantities
        self.register_buffer('electric_scale'
                             ,torch.as_tensor(e0,dtype=torch.float64))
        self.register_buffer('fr_diff_scale'
                             ,torch.as_tensor(abs(dw0),dtype=torch.float64))
        self.register_buffer('kinetic_scale'
                             ,torch.as_tensor(ke0,dtype=torch.float64))
        
        self.log_electric = nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.log_fr_diff  = nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.log_kinetic  = nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.log_distance = nn.Parameter(torch.zeros((),dtype=torch.float64))        
        # -- discretization parameters (FROZEN): computed once from initial 
        # params, not again. These set the shapes of the tensors used so they
        # are not differentiable (and thus should not influence gradients)
        with torch.no_grad():
            rayleigh0 = self._init_rayleigh()
            beta0     = self._init_beta(self.electric,self.kinetic)
            intdist0  = rayleigh0 * (beta0/self.cfg.beta_target)**0.5
        self._beta0_init = float(beta0)
        self.lmax     = max(5,math.ceil(3*beta0))
        self.L        = 2*self.lmax + 1
        self.interaction_distance = intdist0
        # taking a linear sampling of the interaction region takes too many 
        # steps unitl convergence. To fix this, we take a tan sampling, where
        # zn = zR * tan(fn) which is natural given the position dependence of 
        # the interaction matrix (1+(z0/zR)**2) = sec(fn)**2 and trivially so 
        # for the Gouy phase
        theta_max  = torch.arctan(intdist0 / (2*rayleigh0)).item()
        pts_per_zR = self.cfg.points_per_rayleigh
        self.num_segments = max(8,int(2*theta_max*pts_per_zR))
        theta      = torch.linspace(-theta_max,theta_max,self.num_segments+1
                                    ,dtype = torch.float64)
        z_edges    = rayleigh0*torch.tan(theta)
        # we use the mid-point rule, with the interaction amplitude calculated 
        # at the middle of each segment
        self.z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
        self.dz = z_edges[1:] - z_edges[:-1]

        # -- trainable parameter (free-propagation distance) ------------------
        # this distance is being set at 0.0 by default (the specific values it 
        # will usually take depend strongly on the Talbot distance), but the 
        # log-scale values rely on a scale that cannot be 0.0, so we instead 
        # build a safe initial guess in terms of z_T.
        with torch.no_grad():
            dist_scale = d0 if d0 != 0.0 else 0.25*self.talbot_distance.item()
        self.register_buffer('distance_scale'
                             ,torch.as_tensor(dist_scale,dtype = torch.float64))

    def __repr__(self): 
        return (f"ElectronModulator(L={self.L}, segments={len(self.z_mid)},"
        f"beta0={self._beta0_init:.2f}, v/c={(self.velocity/c_au).item():.3f},"
        f"intdist/zR = {(self.interaction_distance/
                         self.rayleigh_range).item():.1f}")

    # - these compute from explicit arguments (init will call them w/no grad) -
    @staticmethod
    def _init_velocity(kinetic):
        return c_au * (1-(kinetic/c_au**2+1)**-2.0)**0.5
    def _init_rayleigh(self):
        return 2*c_au / (self.frequency1 * self.cfg.numerical_aperture**2)
    def _init_beta(self,electric1,kinetic):
        v   = self._init_velocity(kinetic)
        gm  = (1-(v/c_au)**2.0)**-0.5
        electric2 = electric1 * (self.frequency2/self.frequency1)**0.5
        eleprod = electric1 * electric2
        b12 = eleprod / (2 * v * gm * self.frequency1 * self.frequency2) 
        return abs(2 * b12 * pi * self.rayleigh_range)

    @staticmethod
    def edge_leakage(alpha):
        # this function tells what fraction of the population is living at the 
        # sidebands, if this is larger than 0.1, it is very possible we have
        # saturated lmax for the given beta. 
        p = torch.abs(alpha)**2
        edge = p[0,:].sum()+p[-1,:].sum()+p[:,0].sum()+p[:,-1].sum()
        return (edge/p.sum()).item()    
    
    # --- conversion of the trainable physical parameters into linear-scale ---
    @property
    def electric(self): return self.electric_scale*torch.exp(self.log_electric)
    @property
    def fr_diff(self): 
        # NOTE: the sign is forced negative here wo that w2 < w1 always.
        #       this is a physical requirement, if it is not met the phase 
        #       matching condition v/c = (w1-w2)/(w1+w2) cannot be fulfilled.
        return -self.fr_diff_scale*torch.exp(self.log_fr_diff)
    @property
    def kinetic(self): return self.kinetic_scale*torch.exp(self.log_kinetic)
    @property
    def distance(self): return self.distance_scale*torch.exp(self.log_distance)
    # - live derived quantities (smoot functions of params, admit gradient) ---
    # electron velocity, depends on kinetic energy
    @property 
    def velocity(self): return c_au*(1-(self.kinetic/c_au**2.0+1)**-2.0)**0.5 
    # lorentz gamma factor, depends on velocity
    @property
    def gamma(self): return (1-(self.velocity/c_au)**2.0)**-0.5
    # frequency of the laser beam 2, derived from frequency1 and fr_diff
    @property
    def frequency2(self): return self.frequency1 + self.fr_diff
    # Rayleigh Range: distance near the foucs where the laser beam is 
    # considered focused
    @property
    def rayleigh_range(self): return (2*c_au
                        /(self.frequency1*self.cfg.numerical_aperture**2))
    # Talbot distance, sets the periodicity of the free-propagation compression
    @property
    def talbot_distance(self): return (4*pi*self.velocity**3*self.gamma**3
                                                        /(self.fr_diff**2))
    def forward(self,alpha):
        alpha = interaction(self,alpha,tracking=False,use_gouy=self.use_gouy)
        alpha = free_propagation(self,alpha)
        return alpha

# =============================================================================
# --- GRADIENT TEST  ----------------------------------------------------------
# testing that the graph is connected and tracking the gradient for one run
# we also test the spilling test to check everything works

if __name__ == "__main__":
    cfg = Config(frequency1=1.0, numerical_aperture=0.2,
                 beta_target=1e-3, points_per_rayleigh=8)
    m = ElectronModulator(cfg, initial_kinetic=30e3, initial_field=0.25,
                          initial_freq_diff=1.0*(0.5-1), initial_prop_dist=0.0)
    print(f"L = {m.L}   segments = {len(m.z_mid)}   "
        f"intdist/zR = {(m.interaction_distance/m.rayleigh_range).item():.1f}")
 
    alpha0 = torch.zeros((m.L,m.L),dtype=torch.complex128)
    alpha0[m.lmax,m.lmax] = 1.0
 
    out = m(alpha0)
    # not our measure of compression, but good enough to test the gradients
    w    = (torch.arange(m.L,dtype=torch.float64)-m.lmax).abs()
    loss = (torch.abs(out)**2 * (w[:,None]+w[None,:])).sum()
    loss.backward()
 
    print(f"loss = {loss.item():.6e}")
    for name,p in m.named_parameters():
        g = "None" if p.grad is None else f"{p.grad.item():.3e}"
        finite = (p.grad is not None) and torch.isfinite(p.grad).all().item()
        print(f"  grad {name:9s} = {g:>12s}   finite={finite}")
    print(f"edge leakage = {m.edge_leakage(out):.2e}   ")
