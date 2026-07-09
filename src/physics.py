# %%
import torch
import torch.nn as nn
import scipy
from dataclasses import dataclass

# building blocks to compute the evolution of a free electron interacting with
# two counter-propagating laser beams of frequencies w1 and w2, electric field 
# amplitudes e1 and e2. The electron has an initial kinetic energy kinetic with 
# the energy spread being much smaller than w1-w2.
# The evolution comes in two steps, first we modulate the energy during the 
# interaction of the electron with the laser beams, and then the electron is
# spatially modulated by the interaction.


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
    
# -----------------------------------------------------------------------------
def interaction(params,alpha,tracking = False):
    # computes the interaction between the electron and the lasers
    v  = params.velocity
    gm = params.gamma
    freq1 = params.cfg.frequency1
    freq2 = params.frequency2
    electric1 = params.electric
    rayleigh = params.rayleigh_range
    lmax = params.lmax
    L    = params.L
    interaction_distance = params.interaction_distance
    n_segments = params.num_segments
    dz = params.dz
    # -------------------------------------------
    def coupling_amplitudes(zn,use_gouy = True):
        if use_gouy:
            gouy_phase = torch.exp(1j*torch.arctan(zn/rayleigh))
        else: 
            gouy_phase = 1.0
        scale_fact = (1+(zn/rayleigh)**2)**-1
        e1 = electric1 #* gouy_phase
        # the amplitude of the laser 2 is defined such that both lasers have 
        # the same **power** 
        e2 = electric1 * (freq2/freq1)**0.5 #* gouy_phase**-1
        e1c = e1.conj()
        e2c = e2.conj()

        a11 = e1*e1/(2*v*gm*freq1**2)
        a22 = e2*e2/(2*v*gm*freq2**2)
        a12 = e1*e2/(2*v*gm*freq1*freq2)
        a21 = a12
        a_row1  = torch.stack((a11,a12))
        a_row2  = torch.stack((a21,a22))
        a_coefs = torch.stack((a_row1,a_row2))*scale_fact

        b11 = e1*e1c/(2*v*gm*freq1**2)
        b22 = e2*e2c/(2*v*gm*freq2**2)
        b12 = e1*e2c/(2*v*gm*freq1*freq2)
        b21 = e2*e1c/(2*v*gm*freq1*freq2)
        b_row1  = torch.stack((b11,b12))
        b_row2  = torch.stack((b21,b22))
        b_coefs = torch.stack((b_row1,b_row2))*scale_fact

        return a_coefs,b_coefs
    # -------------------------------------------
    def segment_matrix(zn):
        acf,bcf = coupling_amplitudes(zn)
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
        bsum  = bcf[0,0]+bcf[1,1]+bcf[0,0].conj()+bcf[1,1].conj()
        vfac_m = (1/v-1/c_au)
        vfac_p = (1/v+1/c_au)
        Dvals = (bsum+vfac_m*l1*freq1+vfac_p*l2*freq2
              +(freq1*l1-freq2*l2)**2/(2*v*gm**3*c_au**2)).to(torch.complex128)
        M = torch.diag(Dvals.reshape(L*L))
        M = M + (-acf[0,1])          *kron(S[-1], S[-1])
        M = M + (2*bcf[0,1].conj())  *kron(S[-1], S[ 1])
        M = M + (-acf[0,1].conj())   *kron(S[ 1], S[ 1])
        M = M + (2*bcf[0,1])         *kron(S[ 1], S[-1])
        M = M + (-acf[0,0].conj())   *kron(S[-2], I    )
        M = M + (-acf[0,0])          *kron(S[ 2], I    )
        M = M + (-acf[1,1].conj())   *kron(I    , S[-2])
        M = M + (-acf[1,1])          *kron(I    , S[ 2])
        return M
    # -------------------------------------------
    alpha = alpha.reshape(L*L).to(torch.complex128)
    # tracking will be used when we want to generate the sideband probability
    # maps. Otherwise, no need to save this information
    if tracking:
        n_track = min(200,n_segments)
        print(n_segments)
        print(n_track)
        interval_skip = int(n_segments/n_track)
        alpha_i = torch.zeros((n_track,L,L)
                            ,dtype = torch.complex128)
    for n in range(n_segments):
        zn = (-interaction_distance * 0.5 + dz*n)
        M  = segment_matrix(zn)
        U  = torch.matrix_exp(-1j * M * dz)
        alpha = U @ alpha
        if tracking and n%interval_skip == 0:
            print(n)
            alpha_i[n] = alpha.reshape(L,L)
    alpha = alpha.reshape(L,L)
    return alpha_i if tracking else alpha

# -----------------------------------------------------------------------------
def free_propagation(params,alpha):
    v     = params.velocity
    freq1 = params.cfg.frequency1
    freq2 = params.frequency2
    dw    = params.fr_diff
    zt    = params.talbot_distance
    lmax  = params.lmax
    dist  = params.distance
    L     = params.L
    # Think about the propagation for the entire spectrum
    free_prop_mat = torch.zeros_like(alpha)
    l     = torch.arange(0,L,1)-lmax
    l1,l2 = l[:,None], l[None,:]
    argument = l1*freq1/v+l2*freq2/v+(2*pi/zt)*(l1*freq1+l2*freq2)**2*dw**-2
    free_prop_mat = torch.exp(1j*argument*dist)
    return free_prop_mat*alpha

# =============================================================================
class ElectronModulator(nn.Module):
    def __init__(self,cfg,initial_kinetic,initial_field,
                 initial_freq_diff,initial_prop_dist=0):
        # ke0  -> initial guess for the kinetic energy of the electron
        # e0   -> initial guess for the electric field amplitude
        # dw0  -> initial guess for the frequency difference     
        # d0   -> initial guess for the free-propagation distance
        # the parameters placed here are the ones to be optimized
        super().__init__()
        self.cfg = cfg
        # -- convert from the input units to atomic untis ---------------------
        e0 = initial_field      * (eV_au/nm_au) 
        dw0 = initial_freq_diff * eV_au
        ke0 = initial_kinetic   * eV_au    
        d0  = initial_prop_dist * mm_au
        # -- trainable parameters (stored in atomic units) --------------------
        self.electric = nn.Parameter(torch.as_tensor( e0,dtype=torch.float64))
        self.fr_diff  = nn.Parameter(torch.as_tensor(dw0,dtype=torch.float64))
        self.kinetic  = nn.Parameter(torch.as_tensor(ke0,dtype=torch.float64))
        self.distance = nn.Parameter(torch.as_tensor( d0,dtype=torch.float64))
        # -- discretization parameters (FROZEN): computed once from initial 
        # params, not again. These set the shapes of the tensors used so they
        # are not differentiable (and thus should not influence gradients)
        with torch.no_grad():
            rayleigh0 = self._init_rayleigh()
            beta0 = self._init_beta(self.electric,self.kinetic)
            intdist0 = rayleigh0 * (beta0/self.cfg.beta_target)**0.5
        self.lmax = max(5,int(3.0*beta0))
        self.L    = 2*self.lmax + 1
        self.num_segments = max(1,int(10 * intdist0 / rayleigh0))
        self.interaction_distance = intdist0
        self.dz   = self.interaction_distance / self.num_segments

    # - these compute from explicit arguments (init will call them w/no grad) -
    @staticmethod
    def _init_velocity(kinetic):
        return c_au * (1-(kinetic/c_au**2+1)**-2.0)**0.5
    def _init_rayleigh(self):
        return 2*c_au / (self.cfg.frequency1 * self.cfg.numerical_aperture**2)
    def _init_beta(self,electric,kinetic):
        v   = self._init_velocity(kinetic)
        gm  = (1-(v/c_au)**2.0)**-0.5
        ray = 2*c_au / (self.cfg.frequency1 * self.cfg.numerical_aperture**2)
        return abs(electric**2.0 * ray/(2*v*gm*self.cfg.frequency1**2))

    # - live derived quantities (smoot functions of params, admit gradient) ---
    # electron velocity, depends on kinetic energy
    @property 
    def velocity(self): return c_au*(1-(self.kinetic/c_au**2.0+1)**-2.0)**0.5 
    # lorentz gamma factor, depends on velocity
    @property
    def gamma(self): return (1-(self.velocity/c_au)**2.0)**-0.5
    # frequency of the laser beam 2, derived from frequency1 and fr_diff
    @property
    def frequency2(self): return self.cfg.frequency1 + self.fr_diff
    # Rayleigh Range: distance near the foucs where the laser beam is 
    # considered focused
    @property
    def rayleigh_range(self): return (2*c_au
                        /(self.cfg.frequency1*self.cfg.numerical_aperture**2))
    # Talbot distance, sets the periodicity of the free-propagation compression
    @property
    def talbot_distance(self): return (4*pi*self.velocity**3*self.gamma**3
                                                        /(self.fr_diff**2))
    def forward(self,alpha):
        alpha = interaction(self,alpha)
        alpha = free_propagation(self,alpha)
        return alpha
# =============================================================================
