"""
Moving Packed Bed (MPB) Reactor Model — Steady-State, Lightly Cooled, Pseudo-Homogeneous
MOLAR FLUX FORM — Effect of inlet water fraction
==========================================================================================

Counter-current flow:
    gas  : z = 0 (inlet, bottom)  ->  z = L (outlet, top)    u_g > 0
    solid: z = L (inlet, top)     ->  z = 0 (outlet, bottom)  u_s > 0 (magnitude)

State variables: F_i [mol/(m²_bed·s)] — molar flux per unit bed cross-section.

Species balance (no u_g, no ε_b):
    dF_i/dz = source_i   [mol/(m²·s) / m] = [mol/(m³_bed·s)]

    dF_CO2/dz = −ρ_bed_cat · r
    dF_H2 /dz = −4 · ρ_bed_cat · r
    dF_CH4/dz = +ρ_bed_cat · r
    dF_H2O/dz = 2·ρ_bed_cat·r − ρ_bed_ads·ads

Partial pressures from mole fractions (ideal gas, constant P):
    p_i = (F_i / F_total) · P_bar

Energy balance (pseudo-homogeneous, counter-current):
    (ΣF_i·Cp_i  −  u_s·ρ_bed·Cp_cat) · dT/dz =
        (−ΔH_r)·ρ_bed_cat·r  +  (−ΔH_ads)·ρ_bed_ads·ads  −  U_a·(T − T_wall)

Regime-switching on u_s*:
    u_s < u_s*  (gas dominates):   denom = ΣF_i·Cp_i − solid_cap > 0
    u_s > u_s*  (solid dominates): T solved in SOLID IVP (ζ = L−z)

Solved by decoupled Gauss-Seidel iteration.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# region 1. PARAMETERS
# =============================================================================

# --- Bed geometry (Bareschino lab setup) ---
d_b   = 0.050
L_b   = 2.000
A_b   = np.pi / 4 * d_b**2
V_bed = A_b * L_b
eps_b = 0.4

# --- Catalyst and sorbent loading ---
M_cat = 0.064
M_ads = 1.22
rho_bed_cat = M_cat / V_bed
rho_bed_ads = M_ads / V_bed
rho_bed_tot = (M_cat + M_ads) / V_bed

# --- Particle properties (13X zeolite pellets) ---
d_p   = 2.5e-3          # particle diameter [m]
eps_p = 0.615
tau_p = 3.0
rho_p = 1400            # [kg/m³] particle density of sorbent (Bareschino)

# --- Dubinin-Astakhov isotherm (H2O on 13X) ---
W0_DA = 190.00e-6
E_DA  = 1190e3
n_DA  = 1.55

# --- LHHW kinetics (Wei et al., K_mix corrected to bar^-0.5) ---
T_ref_K = 555.0
k_ref   = 3.46e-4
Ea_k    = 77.5e3
A_OH    = 0.50;  dH_OH  =  22.4e3
A_H2    = 0.44;  dH_H2  =  -6.2e3
A_mix   = 0.88;  dH_mix = -10.0e3
P_FLOOR = 1e-4

# --- Thermochemistry ---
dH_r   = -165.0e3
dH_ads =  -45.0e3
Cp_cat = 1100.0
Cp_CO2 = 45.4;  Cp_H2 = 29.3;  Cp_CH4 = 46.9;  Cp_H2O = 34.2

# --- Wall heat transfer ---
U_a    = 2000.0

# --- Physical constants ---
R_gas  = 8.314
MW_H2O = 0.018015

# --- Fixed feed fractions (CO2 and H2 stay constant across runs) ---
P_bar = 1.0
P_Pa  = P_bar * 1e5
y_CO2_in = 0.0344
y_H2_in  = 0.1378
# y_H2O_in and y_CH4_in are overwritten per run; base values for printing only
y_H2O_in = 0.001263
y_CH4_in = 1.0 - y_CO2_in - y_H2_in - y_H2O_in

# --- Total molar flux (temperature-independent, set by GHSV) ---
T_STP   = 273.15
GHSV    = 0.5
Q_STP   = GHSV * M_ads / 3600.0
u_g_STP = Q_STP / A_b
F_total_in = u_g_STP * P_Pa / (R_gas * T_STP)

# Per-species inlet fluxes — globals, reassigned for each water-fraction run
F_in_CO2 = y_CO2_in * F_total_in
F_in_H2  = y_H2_in  * F_total_in
F_in_CH4 = y_CH4_in * F_total_in
F_in_H2O = y_H2O_in * F_total_in

# --- Scan parameters ---
U_S_FIXED   = 5.0e-3   # [m/s]  solid velocity (fixed)
Y_H2O_LIST  = [                                   # 0.1 % → 1.0 % in 0.1 % steps,
    0.001, 0.002, 0.003, 0.004, 0.005,            # then 1.2 % → 2.0 % in 0.2 % steps
    0.006, 0.007, 0.008, 0.009, 0.010,
    0.012, 0.014, 0.016, 0.018, 0.020,
]
T_IN_SINGLE = 280       # [°C]
N_GRID      = 200       # axial grid points in solve_mpb

print(f"MPB flux form: d={d_b*100:.0f} cm, L={L_b:.1f} m, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3, u_g_STP={u_g_STP*1e3:.1f} mm/s  |  N={N_GRID}")
print(f"  F_in_total = {F_total_in:.4f} mol/(m2·s)  |  d_p = {d_p*1e3:.1f} mm  |  "
      f"u_s = {U_S_FIXED*1e3:.1f} mm/s  |  U_a = {U_a:.0f} W/(m3·K)")
print(f"  Water fractions: {[f'{y*100:.2f}%' for y in Y_H2O_LIST]}")
# endregion


# region 2. FUNCTIONS
# =============================================================================
def P_sat_bar(T_K):
    log10_p = (29.8605 - 3.1522e3/T_K - 7.3037*np.log10(T_K)
               + 2.4247e-9*T_K + 1.8090e-6*T_K**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5

def rho_water(T_K):
    return 996.0 / (1.0 + 2.0e-3*(T_K - 298.15))

def q_star_vec(T_K, p_arr, W0, E, n):
    p    = np.asarray(p_arr, dtype=float)
    Psat = P_sat_bar(T_K)
    p_safe = np.clip(p, 1e-15, Psat*(1-1e-10))
    A_raw  = (R_gas/MW_H2O)*T_K*np.log(Psat/p_safe)
    A  = np.where((p <= 0)|(p >= Psat), 0.0, A_raw)
    W  = W0*np.exp(-np.minimum((A/E)**n, 500.0))
    qs = rho_water(T_K)/MW_H2O*W
    return np.where(p <= 0, 0.0, qs)

def K_LDF_vec(T_K, p_arr, W0, E, n):
    D_M  = 2.5e-5*(T_K/300.0)**1.75                                           # molecular diffusivity [m²/s]
    p    = np.asarray(p_arr, dtype=float)
    dp   = 1.0/1e5                                                             # 1 Pa in bar
    dqsp = (q_star_vec(T_K, p+dp, W0, E, n)
            - q_star_vec(T_K, np.maximum(p-dp, 1e-15), W0, E, n)) / 2.0      # central finite diff dq*/dp [mol/(kg·Pa)]
    dqsp = np.maximum(dqsp, 1e-30)
    r_p = 0.5 * d_p
    return 15.0 * eps_p * D_M / (r_p**2 * tau_p * rho_p * R_gas * T_K * dqsp)  # Glueckauf LDF [1/s]

def K_eq_sabatier(T_K):
    return 137.0*T_K**(-3.994)*np.exp(158700.0/(R_gas*T_K))

def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    vH    = lambda dH: np.exp(-dH/R_gas*(1.0/T_K - 1.0/T_ref_K))
    k     = k_ref*np.exp(-Ea_k/R_gas*(1.0/T_K - 1.0/T_ref_K))
    K_OH  = A_OH*vH(dH_OH);  K_H2 = A_H2*vH(dH_H2);  K_mix = A_mix*vH(dH_mix)
    K_eq  = K_eq_sabatier(T_K)
    p_CO2_s = np.maximum(p_CO2, P_FLOOR);  p_H2_s = np.maximum(p_H2, P_FLOOR)
    beta  = (p_CH4*p_H2O**2)/(K_eq*p_CO2_s*p_H2_s**4)
    f_eq  = np.maximum(1.0 - np.where(np.isfinite(beta), beta, 1e10), 0.0)
    DEN   = (1.0 + K_OH*np.maximum(p_H2O, 0)/p_H2_s**0.5
             + K_H2*p_H2_s**0.5 + K_mix*p_CO2_s**0.5)
    return k*(p_CO2_s*p_H2_s)**0.5*f_eq/DEN**2*1000.0

def q_star(T_K, p_H2O):
    return q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

_K_LDF_MAX = 20

def K_LDF(T_K, p_H2O):
    return np.minimum(K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA), _K_LDF_MAX)

def equilibrium_conversion(T_K_val):
    K = K_eq_sabatier(T_K_val)
    def f(X):
        d = 1.0 - 2*y_CO2_in*X
        n_CO2 = y_CO2_in*(1-X)/d
        n_H2  = (y_H2_in - 4*y_CO2_in*X)/d
        n_CH4 = (y_CH4_in + y_CO2_in*X)/d
        n_H2O = (y_H2O_in + 2*y_CO2_in*X)/d
        return n_CH4*n_H2O**2/(n_CO2*max(n_H2,1e-30)**4 + 1e-100) - K
    try:
        return brentq(f, 1e-9, 1-1e-9)*100.0
    except Exception:
        return 100.0

def _gas_cap(F_CO2, F_H2, F_CH4, F_H2O):
    """Thermal flux of gas phase [W/(m²·K)] = Σ F_i·Cp_i."""
    return F_CO2*Cp_CO2 + F_H2*Cp_H2 + F_CH4*Cp_CH4 + F_H2O*Cp_H2O

def _partial_pressures(F_CO2, F_H2, F_CH4, F_H2O):
    """Convert molar fluxes to partial pressures [bar] via mole fractions."""
    F_tot = F_CO2 + F_H2 + F_CH4 + F_H2O
    if F_tot < 1e-30:
        return 0.0, 0.0, 0.0, 0.0
    return (F_CO2/F_tot*P_bar, F_H2/F_tot*P_bar,
            F_CH4/F_tot*P_bar, F_H2O/F_tot*P_bar)
# endregion


# region 3. DECOUPLED SOLVER
# =============================================================================
def solve_mpb(u_s, T_K, T_wall=None, max_iter=400, tol=1e-5, N=100, q_init=None):
    """
    Counter-current MPB — molar flux form, lightly cooled, regime-switching.

    Uses global F_in_* inlet fluxes (reassigned per run in the solve loop).
    """
    if T_wall is None:
        T_wall = T_K

    solid_cap  = u_s * rho_bed_tot * Cp_cat
    gas_cap_in = _gas_cap(F_in_CO2, F_in_H2, F_in_CH4, F_in_H2O)
    gas_dominates = (solid_cap < gas_cap_in)

    z_grid = np.linspace(0.0, L_b, N)
    if q_init is not None:
        q_prof = np.interp(z_grid, np.linspace(0, L_b, len(q_init)), q_init)
    else:
        q_prof = np.zeros(N)
    T_prof = T_K * np.ones(N)

    converged = False
    err = 1.0

    for it in range(max_iter):
        q_fn = interp1d(z_grid, q_prof, kind='linear',
                        bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))
        T_fn = interp1d(z_grid, T_prof, kind='linear',
                        bounds_error=False, fill_value=(T_prof[0], T_prof[-1]))

        if gas_dominates:
            # ── GAS IVP: state = [F_CO2, F_H2, F_CH4, F_H2O, T] ─────────────
            def gas_rhs(z, y):
                F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
                F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
                T_l     = max(y[4], 200.0)
                q_l     = max(float(q_fn(z)), 0.0)

                p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                    F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)

                r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                              np.array([p_CH4]), np.array([p_H2O]))[0])
                qs  = float(q_star(T_l, np.array([p_H2O]))[0])
                Kl  = float(K_LDF(T_l,  np.array([p_H2O]))[0])
                ads = Kl*(qs - q_l)

                gas_cap_l = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
                denom     = gas_cap_l - solid_cap

                Q_rxn  = (-dH_r)   * rho_bed_cat * r
                Q_ads  = (-dH_ads) * rho_bed_ads * ads
                Q_wall = U_a * (T_l - T_wall)
                dTdz   = (Q_rxn + Q_ads - Q_wall) / denom

                return [
                    -rho_bed_cat * r,                        # dF_CO2/dz
                    -4.0*rho_bed_cat * r,                    # dF_H2/dz
                    +rho_bed_cat * r,                        # dF_CH4/dz
                    2.0*rho_bed_cat*r - rho_bed_ads*ads,     # dF_H2O/dz
                    dTdz,
                ]

            gs = solve_ivp(gas_rhs, [0.0, L_b],
                           [F_in_CO2, F_in_H2, F_in_CH4, F_in_H2O, T_K],
                           method='BDF', rtol=1e-4,
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8, 1e-2]),
                           t_eval=z_grid, dense_output=False)
            if not gs.success:
                return None

            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)
            F_CH4_prof = np.maximum(gs.y[2], 0.0);  F_H2O_prof = np.maximum(gs.y[3], 0.0)
            T_prof_new = np.maximum(gs.y[4], 200.0)

            F_tot_prof  = np.maximum(F_CO2_prof + F_H2_prof + F_CH4_prof + F_H2O_prof, 1e-30)
            p_H2O_prof  = F_H2O_prof / F_tot_prof * P_bar
            p_H2O_fn    = interp1d(z_grid, p_H2O_prof, kind='linear',
                                   bounds_error=False,
                                   fill_value=(p_H2O_prof[0], p_H2O_prof[-1]))
            T_fn_new    = interp1d(z_grid, T_prof_new, kind='linear',
                                   bounds_error=False,
                                   fill_value=(T_prof_new[0], T_prof_new[-1]))

            # ── SOLID IVP: state = [q] ────────────────────────────────────────
            def solid_rhs(zeta, q_arr):
                z_pos    = L_b - float(zeta)
                T_local  = float(T_fn_new(z_pos))
                p_H2O_l  = max(float(p_H2O_fn(z_pos)), 0.0)
                qs  = float(q_star(T_local, np.array([p_H2O_l]))[0])
                Kl  = float(K_LDF(T_local,  np.array([p_H2O_l]))[0])
                q_val = max(float(q_arr[0]), 0.0)
                return [Kl*(qs - q_val)/u_s]

            ss = solve_ivp(solid_rhs, [0.0, L_b], [0.0],
                           method='BDF', rtol=1e-4, atol=1e-8,
                           max_step=L_b/N,
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                return None

            z_from_zeta = L_b - ss.t
            q_from_zeta = np.maximum(ss.y[0], 0.0)
            sort_idx    = np.argsort(z_from_zeta)
            q_new = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])

            q_prof_new = 0.5*q_prof + 0.5*q_new
            T_prof     = 0.5*T_prof + 0.5*T_prof_new

            scale = max(np.max(q_prof_new), 1e-8)
            err   = np.max(np.abs(q_prof_new - q_prof)) / scale
            q_prof = q_prof_new

        else:
            # ── GAS IVP: state = [F_CO2, F_H2, F_CH4, F_H2O]  (T frozen) ────
            def gas_rhs_no_T(z, y):
                F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
                F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
                T_l     = max(float(T_fn(z)), 200.0)
                q_l     = max(float(q_fn(z)), 0.0)

                p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                    F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)

                r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                              np.array([p_CH4]), np.array([p_H2O]))[0])
                qs  = float(q_star(T_l, np.array([p_H2O]))[0])
                Kl  = float(K_LDF(T_l,  np.array([p_H2O]))[0])
                ads = Kl*(qs - q_l)

                return [
                    -rho_bed_cat * r,
                    -4.0*rho_bed_cat * r,
                    +rho_bed_cat * r,
                    2.0*rho_bed_cat*r - rho_bed_ads*ads,
                ]

            gs = solve_ivp(gas_rhs_no_T, [0.0, L_b],
                           [F_in_CO2, F_in_H2, F_in_CH4, F_in_H2O],
                           method='BDF', rtol=1e-4,
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8]),
                           t_eval=z_grid, dense_output=False)
            if not gs.success:
                return None

            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)
            F_CH4_prof = np.maximum(gs.y[2], 0.0);  F_H2O_prof = np.maximum(gs.y[3], 0.0)
            F_tot_prof = np.maximum(F_CO2_prof + F_H2_prof + F_CH4_prof + F_H2O_prof, 1e-30)

            _make_fn = lambda p: interp1d(z_grid, p, kind='linear',
                                          bounds_error=False, fill_value=(p[0], p[-1]))
            F_CO2_fn = _make_fn(F_CO2_prof);  F_H2_fn  = _make_fn(F_H2_prof)
            F_CH4_fn = _make_fn(F_CH4_prof);  F_H2O_fn = _make_fn(F_H2O_prof)
            F_tot_fn = _make_fn(F_tot_prof)

            # ── SOLID IVP: state = [q, T] ─────────────────────────────────────
            def solid_rhs_with_T(zeta, y_arr):
                q_val = max(float(y_arr[0]), 0.0)
                T_val = max(float(y_arr[1]), 200.0)
                z_pos = L_b - float(zeta)

                F_CO2_l = max(float(F_CO2_fn(z_pos)), 0.0)
                F_H2_l  = max(float(F_H2_fn(z_pos)),  0.0)
                F_CH4_l = max(float(F_CH4_fn(z_pos)), 0.0)
                F_H2O_l = max(float(F_H2O_fn(z_pos)), 0.0)
                F_tot_l = max(float(F_tot_fn(z_pos)), 1e-30)

                p_CO2 = F_CO2_l/F_tot_l*P_bar;  p_H2  = F_H2_l /F_tot_l*P_bar
                p_CH4 = F_CH4_l/F_tot_l*P_bar;  p_H2O = F_H2O_l/F_tot_l*P_bar

                r   = float(reaction_rate_SI(T_val, np.array([p_CO2]), np.array([p_H2]),
                                              np.array([p_CH4]), np.array([p_H2O]))[0])
                qs  = float(q_star(T_val, np.array([p_H2O]))[0])
                Kl  = float(K_LDF(T_val, np.array([p_H2O]))[0])
                ads = Kl*(qs - q_val)

                gas_cap_l   = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
                solid_denom = solid_cap - gas_cap_l

                Q_rxn  = (-dH_r)   * rho_bed_cat * r
                Q_ads  = (-dH_ads) * rho_bed_ads * ads
                Q_wall = U_a * (T_val - T_wall)

                return [Kl*(qs - q_val)/u_s,
                        (Q_rxn + Q_ads - Q_wall) / solid_denom]

            ss = solve_ivp(solid_rhs_with_T, [0.0, L_b], [0.0, T_K],
                           method='BDF', rtol=1e-4, atol=np.array([1e-8, 1.0]),
                           max_step=L_b/N,
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                return None

            z_from_zeta  = L_b - ss.t
            q_from_zeta  = np.maximum(ss.y[0], 0.0)
            T_from_zeta  = np.maximum(ss.y[1], 200.0)
            sort_idx     = np.argsort(z_from_zeta)
            q_new  = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])
            T_new  = np.interp(z_grid, z_from_zeta[sort_idx], T_from_zeta[sort_idx])

            q_prof_new = 0.85*q_prof + 0.15*q_new
            T_prof_new = 0.85*T_prof + 0.15*T_new

            err_q = np.max(np.abs(q_prof_new - q_prof)) / max(np.max(q_prof_new), 1e-8)
            err_T = np.max(np.abs(T_prof_new - T_prof)) / T_K
            err   = max(err_q, err_T)

            q_prof = q_prof_new
            T_prof = T_prof_new

        if err < tol:
            converged = True
            break

    # ── Final recompute on fine grid ─────────────────────────────────────────
    z_fine = np.linspace(0.0, L_b, 300)
    q_fn_f = interp1d(z_grid, q_prof, kind='linear',
                      bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))
    T_fn_f = interp1d(z_grid, T_prof, kind='linear',
                      bounds_error=False, fill_value=(T_prof[0], T_prof[-1]))

    if gas_dominates:
        def gas_rhs_final(z, y):
            F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
            F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
            T_l     = max(y[4], 200.0)
            q_l     = max(float(q_fn_f(z)), 0.0)
            p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
            r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                          np.array([p_CH4]), np.array([p_H2O]))[0])
            qs  = float(q_star(T_l, np.array([p_H2O]))[0])
            Kl  = float(K_LDF(T_l, np.array([p_H2O]))[0])
            ads = Kl*(qs - q_l)
            gas_cap_l = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
            denom = gas_cap_l - solid_cap
            Q_rxn = (-dH_r)*rho_bed_cat*r;  Q_ads = (-dH_ads)*rho_bed_ads*ads
            Q_wall = U_a*(T_l - T_wall)
            return [
                -rho_bed_cat*r,
                -4.0*rho_bed_cat*r,
                +rho_bed_cat*r,
                2.0*rho_bed_cat*r - rho_bed_ads*ads,
                (Q_rxn + Q_ads - Q_wall)/denom,
            ]
        gf = solve_ivp(gas_rhs_final, [0.0, L_b],
                       [F_in_CO2, F_in_H2, F_in_CH4, F_in_H2O, T_K],
                       method='BDF', rtol=1e-6,
                       atol=np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-3]),
                       t_eval=z_fine, dense_output=False)
        F_CO2f = np.maximum(gf.y[0], 0.0);  F_H2f  = np.maximum(gf.y[1], 0.0)
        F_CH4f = np.maximum(gf.y[2], 0.0);  F_H2Of = np.maximum(gf.y[3], 0.0)
        T_fine = np.maximum(gf.y[4], 200.0)

    else:
        def gas_rhs_final_no_T(z, y):
            F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
            F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
            T_l     = max(float(T_fn_f(z)), 200.0)
            q_l     = max(float(q_fn_f(z)), 0.0)
            p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
            r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                          np.array([p_CH4]), np.array([p_H2O]))[0])
            qs  = float(q_star(T_l, np.array([p_H2O]))[0])
            Kl  = float(K_LDF(T_l, np.array([p_H2O]))[0])
            ads = Kl*(qs - q_l)
            return [
                -rho_bed_cat*r,
                -4.0*rho_bed_cat*r,
                +rho_bed_cat*r,
                2.0*rho_bed_cat*r - rho_bed_ads*ads,
            ]
        gf = solve_ivp(gas_rhs_final_no_T, [0.0, L_b],
                       [F_in_CO2, F_in_H2, F_in_CH4, F_in_H2O],
                       method='BDF', rtol=1e-6,
                       atol=np.array([1e-10, 1e-10, 1e-10, 1e-10]),
                       t_eval=z_fine, dense_output=False)
        F_CO2f = np.maximum(gf.y[0], 0.0);  F_H2f  = np.maximum(gf.y[1], 0.0)
        F_CH4f = np.maximum(gf.y[2], 0.0);  F_H2Of = np.maximum(gf.y[3], 0.0)
        T_fine = np.interp(z_fine, z_grid, T_prof)

    q_fine   = np.interp(z_fine, z_grid, q_prof)
    F_totf   = np.maximum(F_CO2f + F_H2f + F_CH4f + F_H2Of, 1e-30)
    p_CO2f   = F_CO2f/F_totf*P_bar;  p_H2f  = F_H2f /F_totf*P_bar
    p_CH4f   = F_CH4f/F_totf*P_bar;  p_H2Of = F_H2Of/F_totf*P_bar
    r_fine   = reaction_rate_SI(T_fine, p_CO2f, p_H2f, p_CH4f, p_H2Of)
    X_CO2    = np.clip(1.0 - F_CO2f/F_in_CO2, 0.0, 1.0)

    u_g_fine = F_totf * R_gas * T_fine / P_Pa
    C_CO2f   = F_CO2f / u_g_fine
    C_H2f    = F_H2f  / u_g_fine
    C_CH4f   = F_CH4f / u_g_fine
    C_H2Of   = F_H2Of / u_g_fine

    return dict(z=z_fine, C_CO2=C_CO2f, C_H2=C_H2f, C_CH4=C_CH4f,
                C_H2O=C_H2Of, q=q_fine, T=T_fine, r=r_fine, X_CO2=X_CO2,
                converged=converged, n_iter=it+1, conv_err=float(err),
                gas_dominates=gas_dominates,
                F_CO2_out=float(F_CO2f[-1]), F_H2_out=float(F_H2f[-1]),
                F_CH4_out=float(F_CH4f[-1]), F_H2O_out=float(F_H2Of[-1]))
# endregion


# region 4. SOLVE LOOP
# =============================================================================
def _compute_noSE(T_K, T_wall, N=300):
    """Fixed-bed (u_s=0) reference with wall cooling; uses global F_in_* fluxes."""
    def rhs(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        T_l     = max(y[4], 200.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        gas_cap = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        Q_rxn  = (-dH_r) * rho_bed_cat * r
        Q_wall = U_a * (T_l - T_wall)
        return [
            -rho_bed_cat*r, -4.0*rho_bed_cat*r,
            +rho_bed_cat*r,  2.0*rho_bed_cat*r,
            (Q_rxn - Q_wall) / gas_cap,
        ]
    z_grid = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, F_in_H2O, T_K],
                    method='BDF', rtol=1e-5,
                    atol=np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-2]), t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    T_f     = np.maximum(sol.y[4], 200.0)
    F_tot_f = np.maximum(sol.y[0] + sol.y[1] + sol.y[2] + sol.y[3], 1e-30)
    p_H2O_f = np.maximum(sol.y[3], 0.0)/F_tot_f * P_bar
    p_CO2_f = F_CO2_f / F_tot_f * P_bar
    p_H2_f  = np.maximum(sol.y[1], 0.0)/F_tot_f * P_bar
    p_CH4_f = np.maximum(sol.y[2], 0.0)/F_tot_f * P_bar
    r_f     = reaction_rate_SI(T_f, p_CO2_f, p_H2_f, p_CH4_f, p_H2O_f)
    X_f     = np.clip(1.0 - F_CO2_f/F_in_CO2, 0.0, 1.0)
    u_g_f   = F_tot_f * R_gas * T_f / P_Pa
    return dict(X_CO2_noSE=float(X_f[-1]),
                profile=dict(z=z_grid,
                             C_CO2=F_CO2_f/u_g_f,
                             C_H2O=np.maximum(sol.y[3], 0.0)/u_g_f,
                             q=q_star(T_f, p_H2O_f), r=r_f, X_CO2=X_f, T=T_f))

def _q_physics_init(T_K, N=150):
    """Isothermal, no-adsorption gas pass to build initial q profile; uses global F_in_*."""
    def rhs_noads(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        return [
            -rho_bed_cat*r, -4.0*rho_bed_cat*r,
            +rho_bed_cat*r,  2.0*rho_bed_cat*r,
        ]
    z_grid = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs_noads, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, F_in_H2O],
                    method='BDF', rtol=1e-4, atol=1e-8, t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    F_tot_f = np.maximum(sol.y[0] + sol.y[1] + sol.y[2] + sol.y[3], 1e-30)
    p_H2O_f = np.maximum(sol.y[3], 0.0)/F_tot_f * P_bar
    return dict(q=q_star(T_K, p_H2O_f),
                X_CO2_noSE=float(np.clip(1.0 - F_CO2_f[-1]/F_in_CO2, 0.0, 1.0)))

def _fmt_seconds(s):
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"

# ── Main solve loop (iterate over inlet water fractions) ─────────────────────
all_results = {}   # keyed by y_H2O value

T_K    = T_IN_SINGLE + 273.15
T_wall = T_K

print(f"\n{'='*60}")
print(f"  T_in = {T_IN_SINGLE} C  |  u_s = {U_S_FIXED*1e3:.1f} mm/s  |  "
      f"U_a = {U_a:.0f} W/(m3·K)  |  d_p = {d_p*1e3:.1f} mm")
print(f"{'='*60}")

t_run_start = time.perf_counter()
q_warm = None   # warm-start q profile carried from one run to the next

for y_H2O in Y_H2O_LIST:
    # Adjust CH4 to maintain sum of mole fractions = 1; CO2 and H2 stay fixed
    y_CH4 = 1.0 - y_CO2_in - y_H2_in - y_H2O

    # Reassign module-level globals so solve_mpb and helpers see updated inlet fluxes
    F_in_CO2 = y_CO2_in * F_total_in
    F_in_H2  = y_H2_in  * F_total_in
    F_in_CH4 = y_CH4    * F_total_in
    F_in_H2O = y_H2O    * F_total_in

    if q_warm is not None:
        q_init = q_warm
    else:
        _phys = _q_physics_init(T_K)
        F_H2O_prod_est = 2.0 * _phys['X_CO2_noSE'] * F_in_CO2
        q_out_est = (F_in_H2O + F_H2O_prod_est) / max(U_S_FIXED * rho_bed_ads, 1e-30)
        q_init = np.linspace(q_out_est, 0.0, len(_phys['q']))

    t0  = time.perf_counter()
    res = solve_mpb(U_S_FIXED, T_K, T_wall=T_wall, q_init=q_init, N=N_GRID)
    dt  = time.perf_counter() - t0

    # Pass converged q profile as warm start for the next run
    if res is not None and res['converged']:
        q_warm = res['q']

    all_results[y_H2O] = {'res': res, 'y_H2O': y_H2O, 'y_CH4': y_CH4}

    if res is not None:
        X_out  = float(res['X_CO2'][-1]) * 100
        T_max  = float(np.max(res['T'])) - 273.15
        regime = "gas" if res['gas_dominates'] else "solid"
        tag    = "ok" if res['converged'] else "not-conv"
        print(f"  y_H2O={y_H2O*100:.2f}%  X={X_out:.1f}%  T_max={T_max:.1f} C"
              f"  [{regime}-dom, {tag}, {res['n_iter']} iter, err={res['conv_err']:.2e}]"
              f"  ({dt:.1f}s)")
        print(f"    Inlet:   CO2={y_CO2_in*100:.3f}%  H2={y_H2_in*100:.3f}%"
              f"  CH4={y_CH4*100:.3f}%  H2O={y_H2O*100:.3f}%")
        F_tot_out = res['F_CO2_out'] + res['F_H2_out'] + res['F_CH4_out'] + res['F_H2O_out']
        y_CO2_out = res['F_CO2_out'] / F_tot_out * 100
        y_H2_out  = res['F_H2_out']  / F_tot_out * 100
        y_CH4_out = res['F_CH4_out'] / F_tot_out * 100
        y_H2O_out = res['F_H2O_out'] / F_tot_out * 100
        print(f"    Outlet:  CO2={y_CO2_out:.3f}%  H2={y_H2_out:.3f}%"
              f"  CH4={y_CH4_out:.3f}%  H2O={y_H2O_out:.3f}%")
        rxn_integral    = float(np.trapz(res['r'], res['z'])) * rho_bed_cat  # mol/(m²·s) CO2 converted
        F_H2O_rxn       = 2.0 * rxn_integral
        F_H2O_solid_out = float(res['q'][0]) * U_S_FIXED * rho_bed_ads
        lhs = F_in_H2O + F_H2O_rxn
        rhs = res['F_H2O_out'] + F_H2O_solid_out
        err_pct = abs(lhs - rhs) / max(lhs, 1e-30) * 100
        print(f"    H2O balance:  in+rxn={lhs:.4f}  =  gas_out={res['F_H2O_out']:.4f}"
              f"  +  solid_out={F_H2O_solid_out:.4f}  =  {rhs:.4f}  |  err={err_pct:.1f}%")
    else:
        print(f"  y_H2O={y_H2O*100:.2f}%  FAILED  ({dt:.1f}s)")

print(f"\nAll done.  Total: {_fmt_seconds(time.perf_counter() - t_run_start)}")
# endregion


# region 5. POST-PROCESSING HELPERS
# =============================================================================
def get_metrics(entry):
    res = entry['res']
    if res is None:
        return None
    X_out       = float(res['X_CO2'][-1])
    q_out       = float(res['q'][0])
    T_max       = float(np.max(res['T']))
    return dict(X_CO2=X_out, q_out=q_out, T_max=T_max)
# endregion


# region 6. PLOTS
# =============================================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def _savefig(filename):
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=150, bbox_inches='tight')

pal = plt.cm.plasma(np.linspace(0.1, 0.85, len(Y_H2O_LIST)))

# ── Plot 1: Axial profiles — all water fractions on same axes ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f'MPB flux form  |  T_in = {T_IN_SINGLE} C  |  u_s = {U_S_FIXED*1e3:.1f} mm/s  |  '
             f'd_p = {d_p*1e3:.1f} mm  |  U_a = {U_a:.0f} W/(m³·K)  — effect of inlet H2O',
             fontsize=10)

for k, y_H2O in enumerate(Y_H2O_LIST):
    entry = all_results.get(y_H2O)
    if entry is None or entry['res'] is None:
        continue
    res = entry['res']
    ls  = '-' if res['converged'] else '--'
    lbl = f"{y_H2O*100:.2f}% H2O{'' if res['converged'] else ' (nc)'}"
    clr = pal[k]

    # p_H2O [mbar] from ideal gas: C [mol/m³] * R * T [K] = Pa → /100 → mbar
    p_H2O_mbar = res['C_H2O'] * R_gas * res['T'] / 100.0

    axes[0, 0].plot(res['z'], res['X_CO2'] * 100,  color=clr, lw=2, ls=ls, label=lbl)
    axes[0, 1].plot(res['z'], res['T'] - 273.15,   color=clr, lw=2, ls=ls, label=lbl)
    axes[1, 0].plot(res['z'], res['q'],             color=clr, lw=2, ls=ls, label=lbl)
    axes[1, 1].plot(res['z'], p_H2O_mbar,           color=clr, lw=2, ls=ls, label=lbl)

axes[0, 1].axhline(T_IN_SINGLE, color='grey', lw=1.5, ls=':', alpha=0.7,
                   label=f'T_wall = {T_IN_SINGLE} °C')

labels_units = [
    ('CO2 conversion [%]',    'CO2 conversion along bed'),
    ('T [°C]',                'Temperature profile'),
    ('q [mol/kg]',            'Solid H2O loading'),
    ('p_H2O [mbar]',          'Gas-phase H2O partial pressure'),
]
for ax, (ylabel, title) in zip(axes.flat, labels_units):
    ax.set_xlabel('z [m]', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(0,   color='tab:blue',   lw=1, ls=':', alpha=0.4)
    ax.axvline(L_b, color='tab:orange', lw=1, ls=':', alpha=0.4)

plt.tight_layout()
_savefig('water_plot1_axial_profiles.png')
plt.show()

# ── Plot 2: Summary — outlet conversion and ΔT_max vs inlet y_H2O ─────────────
fig, (ax_X, ax_T) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'MPB flux form  |  T_in = {T_IN_SINGLE} C  |  u_s = {U_S_FIXED*1e3:.1f} mm/s  |  '
             f'd_p = {d_p*1e3:.1f} mm  |  U_a = {U_a:.0f} W/(m³·K)', fontsize=10)

y_pct_list, X_list, dT_list = [], [], []
for k, y_H2O in enumerate(Y_H2O_LIST):
    entry = all_results.get(y_H2O)
    if entry is None or entry['res'] is None:
        continue
    res = entry['res']
    m   = 'o' if res['converged'] else '^'
    X_out = float(res['X_CO2'][-1]) * 100
    dT    = float(np.max(res['T'])) - 273.15 - T_IN_SINGLE
    y_pct_list.append(y_H2O * 100)
    X_list.append(X_out)
    dT_list.append(dT)
    ax_X.scatter(y_H2O * 100, X_out, color=pal[k], s=80, marker=m, zorder=3)
    ax_T.scatter(y_H2O * 100, dT,    color=pal[k], s=80, marker=m, zorder=3)

if y_pct_list:
    ax_X.plot(y_pct_list, X_list,  'k-', lw=1.5, alpha=0.4)
    ax_T.plot(y_pct_list, dT_list, 'k-', lw=1.5, alpha=0.4)

ax_X.set_xlabel('y_H2O in feed [%]', fontsize=11)
ax_X.set_ylabel('Outlet CO2 conversion [%]', fontsize=11)
ax_X.set_title('CO2 conversion vs inlet water fraction')
ax_X.grid(True, alpha=0.3)

ax_T.set_xlabel('y_H2O in feed [%]', fontsize=11)
ax_T.set_ylabel('ΔT_max = T_peak − T_in  [K]', fontsize=11)
ax_T.set_title('Peak temperature rise vs inlet water fraction')
ax_T.grid(True, alpha=0.3)

plt.tight_layout()
_savefig('water_plot2_summary.png')
plt.show()

# ── Plot 3: Inlet vs Outlet mole fractions ────────────────────────────────────
fig, axes3 = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f'MPB flux form  |  T_in = {T_IN_SINGLE} C  |  u_s = {U_S_FIXED*1e3:.1f} mm/s  — '
             f'Inlet vs outlet mole fractions', fontsize=10)

y_pct_vals = [y * 100 for y in Y_H2O_LIST
              if all_results.get(y) and all_results[y]['res'] is not None]
y_in_dict  = {s: [] for s in ('CO2', 'H2', 'CH4', 'H2O')}
y_out_dict = {s: [] for s in ('CO2', 'H2', 'CH4', 'H2O')}

for y_H2O in Y_H2O_LIST:
    entry = all_results.get(y_H2O)
    if entry is None or entry['res'] is None:
        continue
    res    = entry['res']
    y_CH4_ = entry['y_CH4']
    y_in_dict['CO2'].append(y_CO2_in * 100)
    y_in_dict['H2'].append(y_H2_in * 100)
    y_in_dict['CH4'].append(y_CH4_ * 100)
    y_in_dict['H2O'].append(y_H2O * 100)
    F_tot_out = res['F_CO2_out'] + res['F_H2_out'] + res['F_CH4_out'] + res['F_H2O_out']
    y_out_dict['CO2'].append(res['F_CO2_out'] / F_tot_out * 100)
    y_out_dict['H2'].append(res['F_H2_out']  / F_tot_out * 100)
    y_out_dict['CH4'].append(res['F_CH4_out'] / F_tot_out * 100)
    y_out_dict['H2O'].append(res['F_H2O_out'] / F_tot_out * 100)

x     = np.arange(len(y_pct_vals))
width = 0.35

for ax, sp in zip(axes3.flat, ('CO2', 'H2', 'CH4', 'H2O')):
    ax.bar(x - width/2, y_in_dict[sp],  width, label='Inlet',  color='steelblue', alpha=0.8)
    ax.bar(x + width/2, y_out_dict[sp], width, label='Outlet', color='tomato',    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:.2f}%' for v in y_pct_vals], rotation=30, ha='right')
    ax.set_xlabel('y_H2O in feed [mol%]', fontsize=10)
    ax.set_ylabel(f'y_{sp} [mol%]', fontsize=10)
    ax.set_title(f'{sp} mole fraction', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
_savefig('water_plot3_molfrac_in_out.png')
plt.show()
# endregion
