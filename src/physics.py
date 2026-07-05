# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

pi = np.pi
# %%
@dataclass
class Config:
    velocity: torch.Tensor
    frequency1: torch.Tensor
    prop_dist: float
    lmax: int
    n_segments: int

class Modulator(nn.Module):
    def __init__(self,cfg,e0,dw0,ke0): # (self,params)
        # e0   -> initial guess for the electric field amplitude
        # dw0  -> initial guess for the frequency difference     
        # ke0  -> initial guess for the kinetic energy of the electron
        # the parameters placed here are the ones to be optimized
        super().__init__()
        self.cfg = cfg
        self.elec    = nn.Parameter(torch.as_tensor(e0,dtype=torch.float64))
        self.fr_diff = nn.Parameter(torch.as_tensor(dw0,dtype=torch.float64))
        self.kinetic = nn.Parameter(torch.as_tensor(ke0,dtype=torch.float64))
    
    # electron velocity in units of c, depends on kinetic energy
    @property 
    def velocity(self): return (1-(self.kinetic+1)**-2.0)**0.5 
    # lorentz gamma factor, depends on velocity
    @property
    def gamma(self): return (1-self.cfg.velocity**2)**-0.5
    # frequency of the laser beam 2, derived from frequency1 and fr_diff
    @property
    def w2(self): return self.cfg.frequency1 + self.fr_diff

    def coupling_amplitudes(self,zn):
        gouy_phase = np.exp(1j*np.arctan(zn))
        scale_fact = (1+(zn)**2)**-1
        elec1 = self.elec * gouy_phase
        elec2 = elec1   * gouy_phase**-1
        v  = self.velocity
        gm = self.gamma
        freq1 = self.cfg.frequency1
        freq2 = self.w2

        e1c = elec1.conj()
        e2c = elec2.conj()

        a11 = elec1*elec1/(2*v*gm*freq1**2)
        a22 = elec2*elec2/(2*v*gm*freq2**2)
        a12 = elec1*elec2/(2*v*gm*freq1*freq2)
        a21 = a12
        a_row1  = torch.stack((a11,a12))
        a_row2  = torch.stack((a21,a22))
        a_coefs = torch.stack((a_row1,a_row2))*scale_fact

        b11 = elec1*e1c/(2*v*gm*freq1**2)
        b22 = elec2*e2c/(2*v*gm*freq2**2)
        b12 = elec1*e2c/(2*v*gm*freq1*freq2)
        b21 = elec2*e1c/(2*v*gm*freq1*freq2)
        b_row1  = torch.stack((b11,b12))
        b_row2  = torch.stack((b21,b22))
        b_coefs = torch.stack((b_row1,b_row2))*scale_fact

        return a_coefs,b_coefs

    def segment_matrix(self,zn):
        L = 2*self.cfg.lmax + 1
        acf,bcf    = self.coupling_amplitudes(zn)
        freq1 = self.cfg.frequency1
        freq2 = self.w2
        v  = self.velocity
        gm = self.gamma
        bsum = bcf[0,0]+bcf[1,1]+bcf[0,0].conj()+bcf[1,1].conj()

        vfac_m = (1/v-1)
        vfac_p = (1/v+1)
        
        M = torch.zeros(L,L,L,L,dtype=torch.complex128)
        for i1 in range(L):
            l1 = i1-self.cfg.lmax
            for i2 in range(L):
                l2 = i2-self.cfg.lmax
                M[i1,i1,i2,i2] = (bsum+(vfac_m*l1*freq1+vfac_p*l2*freq2
                                        +(freq1*l1-freq2*l2)**2/(2*v*gm**3)))
                if i1 > 0:
                    if i2 > 0:
                        M[i1,i1-1,i2,i2-1] = -acf[0,1]
                    if i2 < L-1:
                        M[i1,i1-1,i2,i2+1] = 2*bcf[0,1].conj()
                if i1 < L-1:
                    if i2 < L-1:
                        M[i1,i1+1,i2,i2+1] = -acf[0,1].conj() 
                    if i2 > 0:
                        M[i1,i1+1,i2,i2-1] = 2*bcf[0,1]
                
                if i1 > 1:
                    M[i1,i1-2,i2,i2] = -acf[0,0].conj()
                if i2 > 1:    
                    M[i1,i1,i2,i2-2] = -acf[1,1].conj()
                if i1 < L-2:
                    M[i1,i1+2,i2,i2] = -acf[0,0]
                if i2 < L-2:
                    M[i1,i1,i2,i2+2] = -acf[1,1]
        return M

    def forward(self,alpha,tracking = False):
        dz = self.cfg.prop_dist / self.cfg.n_segments
        L  = 2*self.cfg.lmax + 1
        alpha = alpha.reshape(L*L).to(torch.complex128)
        if tracking:
            alpha_i = torch.zeros((self.cfg.n_segments,L,L)
                                  ,dtype = torch.complex128)
        for n in range(self.cfg.n_segments):
            zn = -self.cfg.prop_dist * 0.5 + dz*n
            M  = self.segment_matrix(zn).permute(0,2,1,3).reshape(L*L,L*L)
            U  = torch.matrix_exp(-1j * M * dz)
            alpha = U @ alpha
            if tracking:
                alpha_i[n] = alpha.reshape(L,L)
        alpha = alpha.reshape(L,L)
        return alpha_i if tracking else alpha
