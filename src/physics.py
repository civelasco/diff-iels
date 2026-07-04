# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
class Parameters:
    def __init__(self,velocity,electric1,electric2
                 ,frequency1,frequency2):
        self.velocity     = velocity
        self.gamma        = (1-(velocity)**2)**-0.5
        self.electric1    = electric1
        self.electric2    = electric2
        self.frequency1   = frequency1
        self.frequency2   = frequency2

def coupling_amplitudes(params):
    e1 = params.electric1
    e2 = params.electric2
    v  = params.velocity
    gm = params.gamma
    w1 = params.frequency1
    w2 = params.frequency2

    e1c = e1.conjugate()
    e2c = e2.conjugate()

    a11 = e1*e1/(2*v*gm*w1**2)
    a22 = e2*e2/(2*v*gm*w2**2)
    a12 = e1*e2/(2*v*gm*w1*w2)
    a21 = a12

    b11 = e1*e1c/(2*v*gm*w1**2)
    b22 = e2*e2c/(2*v*gm*w2**2)
    b12 = e1*e2c/(2*v*gm*w1*w2)
    b21 = e2*e1c/(2*v*gm*w1*w2)

    a_coefs = torch.tensor([a11,a22,a12,a21])
    b_coefs = torch.tensor([b11,b22,b12,b21])
    return a_coefs,b_coefs

def segment_matrix(params,zn,lmax):
    w1    = params.frequency1
    w2    = params.frequency2
    v     = params.velocity
    gamma = params.gamma
    acf,bcf    = coupling_amplitudes(params)
    scale_prop = (1+zn**2)**-1
    bsum = bcf[0,0]+bcf[1,1]+bcf[0,0].conjugate()+bcf[1,1].conjugate()

    vfac_m = (1/v-1)
    vfac_p = (1/v+1)
    
    matrix = torch.zeros_like((2*lmax,2*lmax,2*lmax,2*lmax))
    for i1 in range(2*lmax):
        l1 = i1-(lmax-1)
        for i2 in range(2*lmax):
            l2 = i2-(lmax-1)
            matrix[i1,i1,i2,i2] = (bsum+(vfac_m*l1*w1+vfac_p*l2*w2
                            +(w1*l1-w2*l2)**2/(2*v*gamma**3)) * scale_prop**-1)
            if i1 > 0:
                if i2 > 0:
                    matrix[i1,i1-1,i2,i2-1] = -acf[0,1]
                if i2 < 2*lmax-1:
                    matrix[i1,i1-1,i2,i2+1] = 2*bcf[0,1].conjugate()
            if i1 < 2*lmax-1:
                if i2 < 2*lmax-1:
                    matrix[i1,i1+1,i2,i2+2] = -acf[0,1].conjugate() 
                if i2 > 0:
                    matrix[i1,i1+1,i2,i2-1] = 2*bcf[0,1]
            
            if i1 > 1:
                matrix[i1,i1-2,i2,i2] = -acf[0,0].conjugate()
            if i2 > 1:    
                matrix[i1,i1,i2-2,i2] = -acf[1,1].conjugate()
            if i1 < 2*lmax-2:
                matrix[i1,i1+2,i2,i2] = -acf[0,0]
            if i2 < 2*lmax-2:
                matrix[i1,i1,i2,i2+2] = -acf[1,1]
    return matrix * scale_prop
