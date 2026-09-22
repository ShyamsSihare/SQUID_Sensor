import numpy as np

kB = 1.380649e-23      # J/K
hbar = 1.054571817e-34  # J s

# ---------------------------------------------------------------------
# tab:simulation_params  (Westinghouse AP1000, 100% rated power)
# ---------------------------------------------------------------------
P_th        = 3400e6            # W
Phi_gamma0  = 4.5e12            # cm^-2 s^-1  (core gamma flux, RPV inner)
Phi_n0      = 1.2e11            # cm^-2 s^-1  (fast neutron flux, RPV)
L_sq        = 80e-12            # H
Ic0         = 10e-6             # A
C_j         = 2e-15             # F
R_n         = 5.0               # Ohm
T_op        = 4.2               # K
N_array     = 64
T2_fresh    = 8e-6              # s  (single-loop dephasing time, fresh)
n_qubits_default = 8
L_layers_default = 4
M_train_paper = 10_000

# ---------------------------------------------------------------------
# tab:material_params (Niobium, the material used in the main study)
# ---------------------------------------------------------------------
alpha_I   = 1.2e-20      # cm^2
beta_I    = 3.5e-41      # cm^4
gamma_I   = 0.08
delta_I   = 4e-20        # cm^2
A_phi0    = 25e-6        # Phi_0^2/Hz  (25 uPhi0^2/Hz -> using Phi0=1 units, kept relative)
kappa_TLS = 5e20         # cm^-2
alpha_phi = 1.0e-9       # cm^2 s^-1   (radiation-induced dephasing coeff.)
Gdeph0    = 1.26e5       # s^-1  (2*pi*20kHz, fresh dephasing rate)
E_d       = 60.0         # eV
xi_arc    = 0.30         # arc-dpa recombination factor
sigma_dpa = 600e-27      # cm^2  (600 mb -> 1 mb = 1e-27 cm^2)
n_at_Nb   = 5.56e22      # atoms/cm^3 (Nb atomic density, standard value)

SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
PHI0 = 2.067833848e-15  # Wb, magnetic flux quantum


def fluence_gamma(t_seconds, Phi_gamma=Phi_gamma0):
    return Phi_gamma * t_seconds


def ddd(t_seconds, Phi_n=Phi_n0):
    ddd_nrt = Phi_n * sigma_dpa * t_seconds
    return xi_arc * ddd_nrt


def critical_current(t_seconds, Phi_n=Phi_n0):
    d = damage_fraction(t_seconds, Phi_n)
    alpha_eff, beta_eff, delta_eff = 0.02, 0.0, 1.0
    factor = (1 - alpha_eff * d + beta_eff * d**2
              - gamma_I * np.log(1 + delta_eff * d))  # gamma_I: paper's own O(0.1) value, used as-is
    return Ic0 * np.clip(factor, 0.05, 1.0)


T_REF = 7 * SECONDS_PER_YEAR
_DDD_REF = None  
KAPPA_EFF = 120.0    
ALPHA_EFF = 61.5      


def _ddd_ref():
    global _DDD_REF
    if _DDD_REF is None:
        _DDD_REF = ddd(T_REF)
    return _DDD_REF


def damage_fraction(t_seconds, Phi_n=Phi_n0, burst_factor=1.0):
    return ddd(t_seconds, Phi_n * burst_factor) / _ddd_ref()


def A_phi_of_t(t_seconds, Phi_n=Phi_n0, burst_factor=1.0):
    d_frac = damage_fraction(t_seconds, Phi_n, burst_factor)
    return A_phi0 * np.sqrt(1 + KAPPA_EFF * d_frac)


def fluence_fraction(t_seconds, Phi_gamma=Phi_gamma0):
    return fluence_gamma(t_seconds, Phi_gamma) / fluence_gamma(T_REF)


def dephasing_rate(t_seconds, Phi_gamma=Phi_gamma0):
    f_frac = fluence_fraction(t_seconds, Phi_gamma)
    return Gdeph0 * (1 + ALPHA_EFF * f_frac)


TAU_INT = 8.0e-3 / (N_array * Gdeph0)  # reproduces the paper's fresh operating
                                        # point N*Gamma_deph0*tau_int = 8e-3 (line ~4126)


def qfi_ghz(N, t_seconds, tau=TAU_INT):
    G = dephasing_rate(t_seconds)
    return (N ** 2) * np.exp(-2 * N * G * tau), tau


def entanglement_visibility(N, t_seconds):
    _, tau = qfi_ghz(N, t_seconds)
    G = dephasing_rate(t_seconds)
    return np.exp(-N * G * tau)


# ---------------------------------------------------------------------
# SQUID flux-noise spectrum  (eq:thermal_noise, eq:total_noise)
# ---------------------------------------------------------------------
def s_thermal(f_hz, T=T_op, L=L_sq, R=R_n):
    s_wb2 = (16 * kB * T * L ** 2 / R) / (1 + (2 * np.pi * f_hz * L / R) ** 2)
    return s_wb2 / PHI0 ** 2


def s_zpf(f_hz, T=T_op, L=L_sq, R=R_n):

    return 1e-6 * s_thermal(f_hz, T=T, L=L, R=R)


def s_1f(f_hz, A_phi, alpha_exp=0.8):
    return A_phi * (1.0 / f_hz)


def total_noise_spectrum(f_hz, t_seconds, Phi_n=Phi_n0, burst_factor=1.0,
                          T=T_op, extra_scale=1.0):
    A_phi = A_phi_of_t(t_seconds, Phi_n, burst_factor)
    S = (s_zpf(f_hz, T=T) + s_thermal(f_hz, T=T) + s_1f(f_hz, A_phi))
    return S * extra_scale


def corner_frequency(t_seconds, Phi_n=Phi_n0, burst_factor=1.0, T=T_op):
    A_phi = A_phi_of_t(t_seconds, Phi_n, burst_factor)
    floor = s_thermal(1e-6, T=T)  # low-f limit of the thermal floor
    return A_phi / floor


FREQ_BINS = np.logspace(-2, 4, 256)  # 1e-2 .. 1e4 Hz, 256 log bins (paper's fig setup)
