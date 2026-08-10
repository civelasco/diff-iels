"""
learning-rate ceiling for physical parameters in a differentiable 
electron-modulation simulator.

Set RECOMPUTE = True to force a fresh sweep.

Claim under test
----------------
    sigma_dw = 2*pi / T = 2*pi*v / L          half-width to first sinc zero
    dw_res   = (k1 + k2)*v ~= 2*wbar*beta     resonance location
    R        = dw_res / sigma_dw ~= 2L/lambda total phase in units of 2*pi

Under Adam in log coordinates the per-step displacement saturates at
|du| ~= eta, so |dtheta| ~= eta*theta, and resolving the main lobe needs

    eta_max ~= 1/R ~= lambda / (2L)

Log reparametrization does NOT remove this: it fixes scale heterogeneity
across parameters, but R is invariant under the change of variables.

Prediction: eta_max ~ L^-1 for the detuning, and no resolvable ceiling for
the field amplitude (the control). The control is sharp because the
amplitude optimum and its lobe width both scale as 1/L, so their ratio is
L-independent -- whereas the detuning optimum is fixed while its width
shrinks as 1/L.

Reference-backend result: slope -0.973 against a predicted -1, prefactor
1.05-1.24 over a factor 40 in L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIG -- everything tunable lives here
# ===========================================================================

BACKEND = "reference"        # "reference" (closed form) | "ponderomotive"
RECOMPUTE = False            # True forces a fresh sweep, ignoring the cache
SHOW_PLOT = True             # False writes the figure without opening a window

# Physical scales
WAVELENGTH_M = 1.0e-6        # lambda-bar, mean drive wavelength
KINETIC_ENERGY_EV = 100e3    # electron kinetic energy
COUPLING = 1.0               # lumps field units into beta_c
DOC_ORDER = 1                # m in DOC_m; only used by the real backend

# Interaction lengths
L_MIN_UM, L_MAX_UM, N_L = 50.0, 2000.0, 6

# Learning-rate grids, as (min, max, count). Spacing matters more than range:
# 12 points over 4 decades (2.15x spacing) is too coarse to locate a
# boundary and yields a spurious slope. These are ~1.20x and ~1.48x.
ETA_DETUNING = (8e-5, 2e-2, 30)
ETA_AMPLITUDE = (1e-3, 3e-1, 15)

# Optimization
N_SEEDS = 5
N_STEPS = 400
TOL = 1e-3                   # success = within TOL of the known optimum
INIT_LO, INIT_HI = 0.2, 0.8  # init offset, in units of the lobe half-width
ESCAPE_SCALES = 25.0         # abort a run once it wanders this far out
SUCCESS_THRESHOLD = 0.5      # fraction of seeds that must converge
RUN_CONTROL = True
SEED = 0

# Paths are resolved against this file, not the working directory, so
# pressing play from anywhere puts the outputs next to the script.
BEFORE = Path(__file__).resolve().parent.resolve().parent
CACHE_PATH = BEFORE / "data" / "trainability_sweep.npz"
PLOT_PATH = BEFORE / "plots" / "eta_max_vs_L.png"

# ===========================================================================

C_LIGHT = 299_792_458.0
M_E_EV = 510_998.95
DTYPE = torch.float64


@dataclass(frozen=True)
class Physics:
    wavelength_m: float = WAVELENGTH_M
    kinetic_energy_eV: float = KINETIC_ENERGY_EV
    coupling: float = COUPLING

    @property
    def gamma(self) -> float:
        return 1.0 + self.kinetic_energy_eV / M_E_EV

    @property
    def beta(self) -> float:
        return math.sqrt(1.0 - self.gamma ** -2)

    @property
    def v(self) -> float:
        return self.beta * C_LIGHT

    @property
    def omega_bar(self) -> float:
        return 2.0 * math.pi * C_LIGHT / self.wavelength_m

    @property
    def delta_omega_res(self) -> float:
        """(k1 + k2) v, at k1 ~= k2 ~= kbar."""
        return 2.0 * self.omega_bar * self.beta

    def transit_time(self, L: float) -> float:
        return L / self.v

    def sigma_delta_omega(self, L: float) -> float:
        return 2.0 * math.pi / self.transit_time(L)


def sharpness_ratio(L: float, phys: Physics) -> float:
    """R = dw_res / sigma_dw, i.e. total accumulated phase / 2*pi."""
    return phys.delta_omega_res / phys.sigma_delta_omega(L)


def predicted_eta_max(L: float, phys: Physics) -> float:
    return 1.0 / sharpness_ratio(L, phys)


# ---------------------------------------------------------------------------
# Differentiable J_1
# ---------------------------------------------------------------------------

class _BesselJ1(torch.autograd.Function):
    """torch.special.bessel_j1 has no registered derivative -- it returns a
    tensor with grad_fn=None, which surfaces several frames later as
    'element 0 of tensors does not require grad'. Supply the exact one:
    J_1'(x) = J_0(x) - J_1(x)/x, the removable singularity at 0 being 1/2.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.bessel_j1(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        small = x.abs() < 1e-12
        safe = torch.where(small, torch.ones_like(x), x)
        d = torch.where(
            small,
            torch.full_like(x, 0.5),
            torch.special.bessel_j0(x) - torch.special.bessel_j1(x) / safe,
        )
        return grad_out * d


def bessel_j1(x):
    return _BesselJ1.apply(x)


# ---------------------------------------------------------------------------
# Backends: (theta, L) -> scalar objective in [0, 1] with a known argmax
# ---------------------------------------------------------------------------

class ReferenceBackend:
    """Closed-form surrogate: sinc coupling into a J_1 sideband weight.

        beta_c(dw, E) = coupling * E * T * sinc((dw - dw_res) T / 2)
        objective     = |J_1(2 beta_c)|^2, normalised to its global max

    Reproduces both scalings at microseconds per call. Not a substitute for
    the real DOC_1 -- it is the methodology check.
    """

    J1_ARGMAX = 1.8411837813406593
    J1_MAX = 0.5818652069559644

    def __init__(self, phys: Physics):
        self.phys = phys

    def _beta_c(self, dw, E, L):
        T = self.phys.transit_time(L)
        x = (dw - self.phys.delta_omega_res) * T / 2.0
        sinc = torch.sinc(x / math.pi)   # torch.sinc is sin(pi x)/(pi x)
        return self.phys.coupling * E * T * sinc

    def _doc(self, beta_c):
        return (bessel_j1(2.0 * beta_c) / self.J1_MAX) ** 2

    def optimum(self, param: str, L: float) -> float:
        if param == "detuning":
            return self.phys.delta_omega_res
        if param == "amplitude":
            T = self.phys.transit_time(L)
            return self.J1_ARGMAX / (2.0 * self.phys.coupling * T)
        raise ValueError(param)

    def lobe_scale(self, param: str, L: float) -> float:
        """Width of the basin around the optimum, for init and escape."""
        if param == "detuning":
            return self.phys.sigma_delta_omega(L)
        return self.optimum("amplitude", L)

    def __call__(self, theta, param: str, L: float):
        if param == "detuning":
            E = torch.tensor(self.optimum("amplitude", L), dtype=theta.dtype)
            return self._doc(self._beta_c(theta, E, L))
        if param == "amplitude":
            dw = torch.tensor(self.phys.delta_omega_res, dtype=theta.dtype)
            return self._doc(self._beta_c(dw, theta, L))
        raise ValueError(param)


class PonderomotiveBackend:
    """Adapter for the real simulator. Fill in the three ADAPT sites; the
    rest of the file is backend-agnostic.
    """

    def __init__(self, phys: Physics, m: int = DOC_ORDER, **model_kwargs):
        self.phys = phys
        self.m = m
        self.model_kwargs = model_kwargs
        self._models: dict[float, object] = {}
        self._amp_opt: dict[float, float] = {}

    def _model(self, L: float):
        if L not in self._models:
            # ---- ADAPT 1/3: construct the module at this L ----------------
            from physics import ElectronModulator
            self._models[L] = ElectronModulator(
                interaction_length=L,
                wavelength=self.phys.wavelength_m,
                kinetic_energy=self.phys.kinetic_energy_eV,
                **self.model_kwargs,
            )
            # ---------------------------------------------------------------
        return self._models[L]

    def optimum(self, param: str, L: float) -> float:
        if param == "detuning":
            return self.phys.delta_omega_res
        if L not in self._amp_opt:
            # ---- ADAPT 2/3: amplitude optimum. If not closed-form, one
            # forward scan per L, cached here, is enough.
            raise NotImplementedError("amplitude optimum for the real backend")
        return self._amp_opt[L]

    def lobe_scale(self, param: str, L: float) -> float:
        if param == "detuning":
            return self.phys.sigma_delta_omega(L)
        return self.optimum("amplitude", L)

    def __call__(self, theta, param: str, L: float):
        model = self._model(L)
        # ---- ADAPT 3/3: substitute theta for the swept parameter and return
        # the scalar DOC_m. Must stay on the graph: no .item(), no no_grad.
        kwargs = {"detuning": {"delta_omega": theta},
                  "amplitude": {"field_amplitude": theta}}[param]
        return model.doc(self.m, **kwargs)


def make_backend(phys: Physics):
    return ReferenceBackend(phys) if BACKEND == "reference" else PonderomotiveBackend(phys)


# ---------------------------------------------------------------------------
# One Adam run in log-latent coordinates
# ---------------------------------------------------------------------------

def run_single(backend, param, L, eta, init_frac, sign) -> tuple[bool, float]:
    """Initialization sits INSIDE the main lobe at a fixed fraction of its
    half-width. This matters: a fixed *absolute* offset would conflate the
    overshoot mechanism with the shrinking lobe, and starting outside the
    lobe would instead measure side-lobe trapping -- a different, also real,
    failure mode.
    """
    theta_star = backend.optimum(param, L)
    scale = backend.lobe_scale(param, L)

    theta0 = theta_star + init_frac * scale * sign
    if theta0 <= 0:
        raise ValueError("non-positive init; log parametrization needs theta > 0")

    u = torch.tensor(math.log(theta0), dtype=DTYPE, requires_grad=True)
    opt = torch.optim.Adam([u], lr=eta)

    with torch.no_grad():
        f_star = backend(torch.tensor(theta_star, dtype=DTYPE), param, L).item()

    best = -math.inf
    for _ in range(N_STEPS):
        opt.zero_grad()
        f = backend(torch.exp(u), param, L)
        (-f).backward()
        opt.step()

        best = max(best, f.item())
        if best >= (1.0 - TOL) * f_star:
            return True, best / f_star
        # Escaped the basin by many widths: Adam's normalised steps will not
        # walk it back within budget, so stop paying for it.
        if abs(math.exp(u.item()) - theta_star) > ESCAPE_SCALES * scale:
            return False, best / f_star

    return False, best / f_star


def sweep(backend, param, L_values, eta_values, verbose=True) -> dict:
    rng = np.random.default_rng(SEED)
    fracs = rng.uniform(INIT_LO, INIT_HI, size=N_SEEDS)
    signs = rng.choice([-1, 1], size=N_SEEDS)

    success = np.zeros((len(L_values), len(eta_values), N_SEEDS), dtype=bool)
    reached = np.zeros(success.shape, dtype=float)

    for i, L in enumerate(L_values):
        for j, eta in enumerate(eta_values):
            for k in range(N_SEEDS):
                ok, frac = run_single(backend, param, float(L), float(eta),
                                      float(fracs[k]), int(signs[k]))
                success[i, j, k] = ok
                reached[i, j, k] = frac
        if verbose:
            print(f"    L = {L*1e6:7.1f} um   R = {sharpness_ratio(float(L), backend.phys):8.1f}"
                  f"   success {success[i].mean():.0%}")

    return {"L_values": L_values, "eta_values": eta_values,
            "success": success, "reached": reached,
            "success_rate": success.mean(axis=2)}


def extract_boundary(eta_values, success_rate, threshold=SUCCESS_THRESHOLD):
    """Upper edge of the success region, scanning down from the largest eta.

    Downward is the right direction: the amplitude control also fails at
    *small* eta, where the binding constraint is the step budget rather than
    overshoot, so an upward scan would return NaN for it.

    censored[i] is True when the edge sits at the top of the grid -- no
    ceiling was resolved and the value is a lower bound only.
    """
    n = success_rate.shape[0]
    out = np.full(n, np.nan)
    censored = np.zeros(n, dtype=bool)
    for i, row in enumerate(success_rate):
        ok = np.flatnonzero(row >= threshold)
        if ok.size:
            out[i] = eta_values[ok[-1]]
            censored[i] = ok[-1] == len(eta_values) - 1
    return out, censored


# ---------------------------------------------------------------------------

def report(phys, L_values, det, amp):
    meas, cens = extract_boundary(det["eta_values"], det["success_rate"])
    pred = np.array([predicted_eta_max(float(l), phys) for l in L_values])

    print("\n    L (um)         R     predicted     measured    ratio")
    print("    " + "-" * 52)
    for l, pr, m, c in zip(L_values, pred, meas, cens):
        R = sharpness_ratio(float(l), phys)
        ratio = m / pr if np.isfinite(m) else np.nan
        print(f"    {l*1e6:7.1f}  {R:9.1f}    {pr:.3e}    {m:.3e}   {ratio:5.2f}"
              + ("  (censored)" if c else ""))

    good = np.isfinite(meas) & ~cens
    if good.sum() >= 2:
        slope = np.polyfit(np.log(L_values[good]), np.log(meas[good]), 1)[0]
        print(f"\n    fitted slope  d(log eta_max)/d(log L) = {slope:+.3f}"
              f"    (predicted -1)")

    if amp is not None:
        _, cens_a = extract_boundary(amp["eta_values"], amp["success_rate"])
        if cens_a.all():
            print("    control: no ceiling resolved at any L "
                  f"(all censored above eta = {amp['eta_values'][-1]:.1e})")


def make_plot(phys, det, amp, path):
    L = det["L_values"]
    x = L / phys.wavelength_m

    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    meas, _ = extract_boundary(det["eta_values"], det["success_rate"])
    ax.loglog(x, meas, "o", ms=6, zorder=3,
              label=r"measured $\eta_{\max}$ (detuning)")

    dense = np.geomspace(L.min(), L.max(), 200)
    ax.loglog(dense / phys.wavelength_m,
              [predicted_eta_max(float(l), phys) for l in dense],
              "-", lw=1.5, zorder=2,
              label=r"predicted $1/R = \bar\lambda/2L$")

    if amp is not None:
        meas_a, _ = extract_boundary(amp["eta_values"], amp["success_rate"])
        ax.loglog(x, meas_a, "s", ms=6, mfc="none", zorder=3,
                  label=r"measured $\eta_{\max}$ (amplitude, control)")


    ax.set_xlabel(r"$L / \bar\lambda$")
    ax.set_ylabel(r"$\eta_{\max}$")
    ax.set_xlim(x.min() / 1.6, x.max() * 1.6)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    print(f"\n    figure -> {path}")


# ---------------------------------------------------------------------------

def main():
    phys = Physics()
    L_values = np.geomspace(L_MIN_UM, L_MAX_UM, N_L) * 1e-6
    eta_det = np.geomspace(*ETA_DETUNING[:2], ETA_DETUNING[2])
    eta_amp = np.geomspace(*ETA_AMPLITUDE[:2], ETA_AMPLITUDE[2])

    print(f"    backend = {BACKEND}   beta = {phys.beta:.4f}   "
          f"dw_res = {phys.delta_omega_res:.4e} rad/s")

    if CACHE_PATH.exists() and not RECOMPUTE:
        print(f"    loading {CACHE_PATH}   (set RECOMPUTE = True to redo)")
        z = np.load(CACHE_PATH)
        det = {k[4:]: z[k] for k in z.files if k.startswith("det_")}
        amp = {k[4:]: z[k] for k in z.files if k.startswith("amp_")} or None
    else:
        backend = make_backend(phys)
        print("\n  detuning:")
        det = sweep(backend, "detuning", L_values, eta_det)
        amp = None
        if RUN_CONTROL:
            print("\n  amplitude (control):")
            amp = sweep(backend, "amplitude", L_values, eta_amp)

        payload = {f"det_{k}": v for k, v in det.items()}
        if amp is not None:
            payload.update({f"amp_{k}": v for k, v in amp.items()})
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH, **payload)
        print(f"\n    cache  -> {CACHE_PATH}")

    report(phys, det["L_values"], det, amp)
    make_plot(phys, det, amp, PLOT_PATH)

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
