"""
Finie-difference validation of autograd gradients for ElectronModulator

Here we compare the gradient obtained by loss.backward() vs a finite-difference 
scheme obtained by slightly varying the trainable parameters. We want to check 
that everything works for the log-scale trainable parameters, which were 
introduced while building the optimizer.
"""
import torch
from metric import degree_of_coherence
def gradient_test(model,alpha0,eps = 1e-4,doc_order = 1,rtol = 1e-3
                  ,raise_on_fail=True):

    params = {nm: p for nm,p in model.named_parameters() if p.requires_grad}
    if not params:
        raise ValueError("no parameters require grad...")
    
    def loss():
        assert torch.is_grad_enabled(), "grad disabled... can't check anything"
        alpha = model(alpha0)
        return 1.0 - degree_of_coherence(model,alpha,doc_order)
    
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        loss().backward()
        analytic = {nm: p.grad.item() for nm, p in params.items()}

    results = {}
    with torch.no_grad():
        for nm,p in params.items():
            x0 = p.item()
            p.fill_(x0+eps); Lp = loss().item()
            p.fill_(x0-eps); Lm = loss().item()
            p.fill_(x0)
            num = (Lp - Lm) / (2*eps)
            rel = abs(analytic[nm] - num) / (abs(num)+1e-12) 
            results[nm] = (analytic[nm],num,rel)

    print(f'{'parameter':<14}{'analytic':>14}{'numeric':>14}{'rel err':>12}')
    for nm, (a,n,rel) in results.items():
        flag = '' if rel < rtol else " <--- oh no... CHECK!"
        print(f'{nm:<14}{a:>14.3e}{n:>14.3e}{rel:>12.2e}{flag}')
    
    if raise_on_fail:
        bad = {nm:r[2] for nm, r in results.items() if r[2]>=rtol}
        assert not bad, f"gradient check failed for {bad} (rtol={rtol})"
    
    return results

if __name__ == "__main__":
    from physics import Config, ElectronModulator
    frequency1  = 1.0  
    num_apt     = 0.01 
    beta_target = 1e-4
    pts_per_z0  = 128
    cfg = Config(frequency1, num_apt, beta_target, pts_per_z0)
    kinetic     =  31e3    
    electric1   =  0.011   
    fr_diff     =  0.5*frequency1
    d0          =  64.0
    model = ElectronModulator(cfg, kinetic, electric1, fr_diff, d0
                              , use_gouy=True)
    alpha0 = torch.zeros((model.L, model.L), dtype=torch.complex128)
    alpha0[model.lmax, model.lmax] = 1.0
    gradient_test(model, alpha0, eps=1e-4)
