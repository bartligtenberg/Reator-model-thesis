"""
Moving Packed Bed (MPB) Reactor — 5 Operating Points
=====================================================
Five capacity levels (10 / 25 / 50 / 75 / 100 %) with 100 % = GHSV = 0.75 NL/g_ads/h.

GHSV  [NL/g_ads/h] : 0.075 | 0.1875 | 0.375 | 0.5625 | 0.75
Capacity      [%]  :   10  |   25   |  50   |   75   | 100

H2 spec: ≤ 0.5 mol% H2 in dry product  →  X_CO2 ≥ 97.4 %
u_s chosen to keep u_s / u_s* ≈ constant across all loads.
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
d_b   = 0.050
L_b   = 2.000
A_b   = np.pi / 4 * d_b**2
V_bed = A_b * L_b
eps_b = 0.4

M_cat = 0.064
M_ads = 1.22
rho_bed_cat = M_cat / V_bed
rho_bed_ads = M_ads / V_bed
rho_bed_tot = (M_cat + M_ads) / V_bed

d_p   = 2.5e-3
eps_p = 0.615
tau_p = 3.0
rho_p = 1400

W0_DA = 190.00e-6
E_DA  = 1190e3
n_DA  = 1.55

T_ref_K = 555.0
k_ref   = 3.46e-4
Ea_k    = 77.5e3
A_OH    = 0.50;  dH_OH  =  22.4e3
A_H2    = 0.44;  dH_H2  =  -6.2e3
A_mix   = 0.88;  dH_mix = -10.0e3
P_FLOOR = 1e-4

dH_r   = -165.0e3
dH_ads =  -45.0e3
Cp_cat = 1100.0
Cp_CO2 = 45.4;  Cp_H2 = 29.3;  Cp_CH4 = 46.9;  Cp_H2O = 34.2

U_a    = 2000.0
R_gas  = 8.314
MW_H2O = 0.018015

P_bar = 1.0
P_Pa  = P_bar * 1e5
y_CO2_in = 0.04
y_H2_in  = 0.16
y_CH4_in = 0.80
T_STP    = 273.15

# 5 capacity levels: 10 / 25 / 50 / 75 / 100 %  (100 % ≡ GHSV = 0.75 NL/g_ads/h)
GHSV_LIST = [0.5625, 0.75]
CAPACITY  = {0.075: 10, 0.1875: 25, 0.375: 50, 0.5625: 75, 0.75: 100}
T_IN_LIST = [280]
U_S_FIXED = 5.0e-3   # m/s — same solid velocity (5 mm/s) for every GHSV
FORCE_SOLID = True   # force the solid-dominated energy balance for every point

H2_SPEC   = 0.5   # mol% H2 in dry product
X_MIN_SPEC = 1.0 - (H2_SPEC/100.0) / (y_H2_in - H2_SPEC/100.0 * y_H2_in)  # ≈ 0.9737

# Placeholders — overwritten per GHSV in the solve loop
F_total_in = 0.0
F_in_CO2   = 0.0
F_in_H2    = 0.0
F_in_CH4   = 0.0
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
    p      = np.asarray(p_arr, dtype=float)
    Psat   = P_sat_bar(T_K)
    p_safe = np.clip(p, 1e-15, Psat*(1-1e-10))
    A_raw  = (R_gas/MW_H2O)*T_K*np.log(Psat/p_safe)
    A      = np.where((p <= 0)|(p >= Psat), 0.0, A_raw)
    W      = W0*np.exp(-np.minimum((A/E)**n, 500.0))
    qs     = rho_water(T_K)/MW_H2O*W
    return np.where(p <= 0, 0.0, qs)

def K_LDF_vec(T_K, p_arr, W0, E, n):
    D_M  = 2.5e-5*(T_K/300.0)**1.75
    p    = np.asarray(p_arr, dtype=float)
    dp   = 1.0/1e5
    dqsp = (q_star_vec(T_K, p+dp, W0, E, n)
            - q_star_vec(T_K, np.maximum(p-dp, 1e-15), W0, E, n)) / 2.0
    dqsp = np.maximum(dqsp, 1e-30)
    r_p  = 0.5 * d_p
    return 15.0 * eps_p * D_M / (r_p**2 * tau_p * rho_p * R_gas * T_K * dqsp)

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
        d = 1.0 - 0.08*X
        return ((0.80+0.04*X)/d * (0.08*X/d)**2
                / ((0.04*(1-X)/d)*(0.16*(1-X)/d)**4 + 1e-100) - K)
    try:
        return brentq(f, 1e-9, 1-1e-9)*100.0
    except Exception:
        return 100.0

def h2_mol_pct_dry(X):
    """Dry-basis H2 mol% in product at CO2 conversion X (fraction 0–1)."""
    return 100.0 * y_H2_in * (1.0 - X) / (1.0 - y_H2_in * X)

def _gas_cap(F_CO2, F_H2, F_CH4, F_H2O):
    return F_CO2*Cp_CO2 + F_H2*Cp_H2 + F_CH4*Cp_CH4 + F_H2O*Cp_H2O

def _partial_pressures(F_CO2, F_H2, F_CH4, F_H2O):
    F_tot = F_CO2 + F_H2 + F_CH4 + F_H2O
    if F_tot < 1e-30:
        return 0.0, 0.0, 0.0, 0.0
    return (F_CO2/F_tot*P_bar, F_H2/F_tot*P_bar,
            F_CH4/F_tot*P_bar, F_H2O/F_tot*P_bar)
# endregion


# region 3. DECOUPLED SOLVER
# =============================================================================
def solve_mpb(u_s, T_K, T_wall=None, max_iter=400, tol=1e-4, N=100,
              q_init=None, T_init=None, force_solid=False):
    if T_wall is None:
        T_wall = T_K

    solid_cap     = u_s * rho_bed_tot * Cp_cat
    gas_cap_in    = _gas_cap(F_in_CO2, F_in_H2, F_in_CH4, 0.0)
    gas_dominates = (solid_cap < gas_cap_in) and not force_solid

    z_grid = np.linspace(0.0, L_b, N)
    if q_init is not None:
        q_prof = np.interp(z_grid, np.linspace(0, L_b, len(q_init)), q_init)
    else:
        q_prof = np.zeros(N)
    if T_init is not None:
        T_prof = np.interp(z_grid, np.linspace(0, L_b, len(T_init)), T_init)
    else:
        T_prof = T_K * np.ones(N)

    converged = False
    err       = 1.0

    for it in range(max_iter):
        q_fn = interp1d(z_grid, q_prof, kind='linear',
                        bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))
        T_fn = interp1d(z_grid, T_prof, kind='linear',
                        bounds_error=False, fill_value=(T_prof[0], T_prof[-1]))

        if gas_dominates:
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
                return [-rho_bed_cat*r, -4.0*rho_bed_cat*r, +rho_bed_cat*r,
                        2.0*rho_bed_cat*r - rho_bed_ads*ads,
                        (Q_rxn + Q_ads - Q_wall)/denom]

            gs = solve_ivp(gas_rhs, [0.0, L_b],
                           [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],
                           method='BDF', rtol=1e-4,
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8, 1e-2]),
                           t_eval=z_grid, dense_output=False)
            if not gs.success:
                return None

            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)
            F_CH4_prof = np.maximum(gs.y[2], 0.0);  F_H2O_prof = np.maximum(gs.y[3], 0.0)
            T_prof_new = np.maximum(gs.y[4], 200.0)

            F_tot_prof = np.maximum(F_CO2_prof+F_H2_prof+F_CH4_prof+F_H2O_prof, 1e-30)
            p_H2O_prof = F_H2O_prof / F_tot_prof * P_bar
            p_H2O_fn   = interp1d(z_grid, p_H2O_prof, kind='linear',
                                   bounds_error=False,
                                   fill_value=(p_H2O_prof[0], p_H2O_prof[-1]))
            T_fn_new   = interp1d(z_grid, T_prof_new, kind='linear',
                                   bounds_error=False,
                                   fill_value=(T_prof_new[0], T_prof_new[-1]))

            def solid_rhs(zeta, q_arr):
                z_pos   = L_b - float(zeta)
                T_local = float(T_fn_new(z_pos))
                p_H2O_l = max(float(p_H2O_fn(z_pos)), 0.0)
                qs  = float(q_star(T_local, np.array([p_H2O_l]))[0])
                Kl  = float(K_LDF(T_local,  np.array([p_H2O_l]))[0])
                q_val = max(float(q_arr[0]), 0.0)
                return [Kl*(qs - q_val)/u_s]

            ss = solve_ivp(solid_rhs, [0.0, L_b], [0.0],
                           method='BDF', rtol=1e-4, atol=1e-8,
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                return None

            z_from_zeta = L_b - ss.t
            q_from_zeta = np.maximum(ss.y[0], 0.0)
            sort_idx    = np.argsort(z_from_zeta)
            q_new = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])

            q_prof_new = 0.5*q_prof + 0.5*q_new
            T_prof     = 0.5*T_prof + 0.5*T_prof_new
            scale      = max(np.max(q_prof_new), 1e-8)
            err        = np.max(np.abs(q_prof_new - q_prof)) / scale
            q_prof     = q_prof_new

        else:
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
                return [-rho_bed_cat*r, -4.0*rho_bed_cat*r, +rho_bed_cat*r,
                        2.0*rho_bed_cat*r - rho_bed_ads*ads]

            gs = solve_ivp(gas_rhs_no_T, [0.0, L_b],
                           [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                           method='BDF', rtol=1e-4,
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8]),
                           t_eval=z_grid, dense_output=False)
            if not gs.success:
                return None

            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)
            F_CH4_prof = np.maximum(gs.y[2], 0.0);  F_H2O_prof = np.maximum(gs.y[3], 0.0)
            F_tot_prof = np.maximum(F_CO2_prof+F_H2_prof+F_CH4_prof+F_H2O_prof, 1e-30)

            _mk = lambda p: interp1d(z_grid, p, kind='linear',
                                     bounds_error=False, fill_value=(p[0], p[-1]))
            F_CO2_fn = _mk(F_CO2_prof);  F_H2_fn  = _mk(F_H2_prof)
            F_CH4_fn = _mk(F_CH4_prof);  F_H2O_fn = _mk(F_H2O_prof)
            F_tot_fn = _mk(F_tot_prof)

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
                        (Q_rxn + Q_ads - Q_wall)/solid_denom]

            ss = solve_ivp(solid_rhs_with_T, [0.0, L_b], [0.0, T_K],
                           method='BDF', rtol=1e-4, atol=np.array([1e-8, 1.0]),
                           max_step=1e-3,
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                return None

            z_from_zeta = L_b - ss.t
            q_from_zeta = np.maximum(ss.y[0], 0.0)
            T_from_zeta = np.maximum(ss.y[1], 200.0)
            sort_idx    = np.argsort(z_from_zeta)
            q_new  = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])
            T_new  = np.interp(z_grid, z_from_zeta[sort_idx], T_from_zeta[sort_idx])

            q_prof_new = 0.5*q_prof + 0.5*q_new
            T_prof_new = 0.5*T_prof + 0.5*T_new
            err_q = np.max(np.abs(q_prof_new - q_prof)) / max(np.max(q_prof_new), 1e-8)
            err_T = np.max(np.abs(T_prof_new - T_prof)) / T_K
            err   = max(err_q, err_T)
            q_prof = q_prof_new
            T_prof = T_prof_new

        if err < tol:
            converged = True
            break

    # Final recompute on fine grid
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
            return [-rho_bed_cat*r, -4.0*rho_bed_cat*r, +rho_bed_cat*r,
                    2.0*rho_bed_cat*r - rho_bed_ads*ads,
                    (Q_rxn + Q_ads - Q_wall)/denom]

        gf = solve_ivp(gas_rhs_final, [0.0, L_b],
                       [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],
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
            return [-rho_bed_cat*r, -4.0*rho_bed_cat*r, +rho_bed_cat*r,
                    2.0*rho_bed_cat*r - rho_bed_ads*ads]

        gf = solve_ivp(gas_rhs_final_no_T, [0.0, L_b],
                       [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                       method='BDF', rtol=1e-6,
                       atol=np.array([1e-10, 1e-10, 1e-10, 1e-10]),
                       t_eval=z_fine, dense_output=False)
        F_CO2f = np.maximum(gf.y[0], 0.0);  F_H2f  = np.maximum(gf.y[1], 0.0)
        F_CH4f = np.maximum(gf.y[2], 0.0);  F_H2Of = np.maximum(gf.y[3], 0.0)
        T_fine = np.interp(z_fine, z_grid, T_prof)

    q_fine  = np.interp(z_fine, z_grid, q_prof)
    F_totf  = np.maximum(F_CO2f+F_H2f+F_CH4f+F_H2Of, 1e-30)
    p_CO2f  = F_CO2f/F_totf*P_bar;  p_H2f  = F_H2f /F_totf*P_bar
    p_CH4f  = F_CH4f/F_totf*P_bar;  p_H2Of = F_H2Of/F_totf*P_bar
    r_fine  = reaction_rate_SI(T_fine, p_CO2f, p_H2f, p_CH4f, p_H2Of)
    X_CO2   = np.clip(1.0 - F_CO2f/F_in_CO2, 0.0, 1.0)
    u_g_fine = F_totf * R_gas * T_fine / P_Pa

    return dict(z=z_fine,
                C_CO2=F_CO2f/u_g_fine, C_H2=F_H2f/u_g_fine,
                C_CH4=F_CH4f/u_g_fine, C_H2O=F_H2Of/u_g_fine,
                q=q_fine, T=T_fine, r=r_fine, X_CO2=X_CO2,
                F_H2O_out=float(F_H2Of[-1]),
                converged=converged, n_iter=it+1, conv_err=float(err),
                gas_dominates=gas_dominates)
# endregion


# region 4. HELPERS
# =============================================================================
def _compute_noSE(T_K, T_wall, N=300):
    def rhs(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        T_l     = max(y[4], 200.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        gc     = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        Q_rxn  = (-dH_r)*rho_bed_cat*r
        Q_wall = U_a*(T_l - T_wall)
        return [-rho_bed_cat*r, -4.0*rho_bed_cat*r,
                +rho_bed_cat*r,  2.0*rho_bed_cat*r,
                (Q_rxn - Q_wall)/gc]
    z_g = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],
                    method='BDF', rtol=1e-5,
                    atol=np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-2]), t_eval=z_g)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    T_f     = np.maximum(sol.y[4], 200.0)
    F_tot_f = np.maximum(sol.y[0]+sol.y[1]+sol.y[2]+sol.y[3], 1e-30)
    X_f     = np.clip(1.0 - F_CO2_f/F_in_CO2, 0.0, 1.0)
    return float(X_f[-1])

def _q_physics_init(T_K, N=150):
    def rhs_noads(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        return [-rho_bed_cat*r, -4.0*rho_bed_cat*r,
                +rho_bed_cat*r,  2.0*rho_bed_cat*r]
    z_g = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs_noads, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                    method='BDF', rtol=1e-4, atol=1e-8, t_eval=z_g)
    F_tot_f = np.maximum(sol.y[0]+sol.y[1]+sol.y[2]+sol.y[3], 1e-30)
    p_H2O_f = np.maximum(sol.y[3], 0.0)/F_tot_f*P_bar
    return q_star(T_K, p_H2O_f)

def _fmt_seconds(s):
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"
# endregion


# region 5. SOLVE LOOP
# =============================================================================
all_results  = {}
noSE_results = {}
n_total      = len(GHSV_LIST) * len(T_IN_LIST)
n_done       = 0
t_run_start  = time.perf_counter()
prev_res     = None   # warm start: previous converged (q, T) profile

for ghsv in GHSV_LIST:
    cap        = CAPACITY[ghsv]
    Q_STP      = ghsv * M_ads / 3600.0
    u_g_STP_g  = Q_STP / A_b
    F_total_in = u_g_STP_g * P_Pa / (R_gas * T_STP)
    F_in_CO2   = y_CO2_in * F_total_in
    F_in_H2    = y_H2_in  * F_total_in
    F_in_CH4   = y_CH4_in * F_total_in

    print(f"\n{'='*70}")
    print(f"  {cap}% capacity  |  GHSV = {ghsv:.4f} NL/g_ads/h  |  u_g_STP = {u_g_STP_g*1e3:.1f} mm/s")
    print(f"  F_in_total = {F_total_in:.4f} mol/(m²·s)  "
          f"[CO2: {F_in_CO2:.4f}  H2: {F_in_H2:.4f}  CH4: {F_in_CH4:.4f}]")
    print(f"{'='*70}")

    for T_C in T_IN_LIST:
        T_K    = T_C + 273.15
        T_wall = T_K

        gas_cap_in = _gas_cap(F_in_CO2, F_in_H2, F_in_CH4, 0.0)
        u_s_star   = gas_cap_in / (rho_bed_tot * Cp_cat)
        u_s_opt    = U_S_FIXED
        noSE_X     = _compute_noSE(T_K, T_wall)
        noSE_results[(ghsv, T_C)] = noSE_X
        print(f"  T_in = {T_C} C  |  u_s* = {u_s_star*1e3:.3f} mm/s  |  "
              f"u_s = {u_s_opt*1e3:.3f} mm/s  |  non-SE = {noSE_X*100:.1f}%")

        if prev_res is not None:
            q_init = prev_res['q']      # warm start from previous GHSV solution
            T_init = prev_res['T']
        else:
            q_init = _q_physics_init(T_K)[::-1]
            T_init = None
        t0      = time.perf_counter()
        res     = solve_mpb(u_s_opt, T_K, T_wall=T_wall, max_iter=400,
                            q_init=q_init, T_init=T_init, force_solid=FORCE_SOLID)
        if res is not None:
            prev_res = res
        dt      = time.perf_counter() - t0
        n_done += 1
        elapsed = time.perf_counter() - t_run_start
        eta     = elapsed / n_done * (n_total - n_done)

        if res is not None:
            X_out   = float(res['X_CO2'][-1])
            q_out   = float(res['q'][0])
            T_max   = float(np.max(res['T'])) - 273.15
            h2_pct  = h2_mol_pct_dry(X_out)
            spec_ok = "OK" if h2_pct <= H2_SPEC else "FAIL"
            regime  = "gas"  if res['gas_dominates'] else "solid"
            tag     = "ok"   if res['converged']     else "not-conv"
            h2o_prod = 2.0 * F_in_CO2 * X_out
            h2o_gas  = res['F_H2O_out']
            h2o_ads  = u_s_opt * rho_bed_ads * q_out
            h2o_err  = (h2o_gas + h2o_ads - h2o_prod) / max(h2o_prod, 1e-30) * 100.0
            print(f"  X={X_out*100:.1f}%  H2={h2_pct:.2f}% [{spec_ok}]  "
                  f"q(0)={q_out:.3f}  T_max={T_max:.1f} C  "
                  f"[{regime}-dom, {tag}, {res['n_iter']} iter, err={res['conv_err']:.2e}]"
                  f"  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
            print(f"    H2O: prod={h2o_prod:.4e}  gas={h2o_gas:.4e}  "
                  f"ads={h2o_ads:.4e}  err={h2o_err:+.1f}%")
        else:
            print(f"  FAILED  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")

        all_results[(ghsv, T_C)] = {'res': res, 'u_s': u_s_opt,
                                     'T_K': T_K, 'T_wall': T_wall,
                                     'ghsv': ghsv, 'cap': cap}

print(f"\nAll done.  Total: {_fmt_seconds(time.perf_counter() - t_run_start)}")
# endregion


# region 6. SUMMARY TABLE
# =============================================================================
T_C_MAIN = T_IN_LIST[0]
print(f"\n{'='*70}")
print(f"  SUMMARY  —  best operating point per capacity level  (T_in = {T_C_MAIN} C)")
print(f"  {'Cap':>5}  {'GHSV':>7}  {'u_s':>7}  {'X':>7}  {'H2 dry':>8}  {'spec':>5}  {'T_max':>7}  {'status'}")
print(f"  {'[%]':>5}  {'[NL/g/h]':>7}  {'[mm/s]':>7}  {'[%]':>7}  {'[mol%]':>8}  {'':>5}  {'[°C]':>7}")
print(f"  {'-'*65}")

for ghsv in GHSV_LIST:
    cap = CAPACITY[ghsv]
    e   = all_results.get((ghsv, T_C_MAIN))
    if e and e['res'] is not None:
        res    = e['res']
        u_s    = e['u_s']
        X_out  = float(res['X_CO2'][-1])
        T_max  = float(np.max(res['T'])) - 273.15
        h2_pct = h2_mol_pct_dry(X_out)
        tag    = "ok" if res['converged'] else "not-conv"
        spec   = "OK" if h2_pct <= H2_SPEC else "FAIL"
        print(f"  {cap:>5}  {ghsv:>7.4f}  {u_s*1e3:>7.3f}  {X_out*100:>7.1f}  "
              f"{h2_pct:>8.2f}  {spec:>5}  {T_max:>7.1f}  {tag}")
    else:
        print(f"  {cap:>5}  {ghsv:>7.4f}  {'—':>7}  {'—':>7}  {'—':>8}  {'—':>5}  {'—':>7}  FAILED")

print(f"  {'-'*65}")
print(f"  H2 spec: ≤ {H2_SPEC} mol%  (requires X_CO2 ≥ {X_MIN_SPEC*100:.1f}%)")
# endregion


# region 7. PLOTS
# =============================================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def _savefig(name):
    plt.savefig(os.path.join(SAVE_DIR, name), dpi=150, bbox_inches='tight')

pal     = plt.cm.plasma(np.linspace(0.1, 0.85, len(GHSV_LIST)))
markers = ['o', 's', '^', 'D', 'v']

# Plot 1: Axial profiles (X and T) for all 5 capacity levels
fig1, axes1 = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
fig1.suptitle(f'Axial profiles  |  T_in = {T_C_MAIN} °C  |  u_s = {U_S_FIXED*1e3:.1f} mm/s (fixed)',
              fontsize=11)
for j, ghsv in enumerate(GHSV_LIST):
    e = all_results.get((ghsv, T_C_MAIN))
    if e is None or e['res'] is None:
        continue
    res = e['res']
    cap = CAPACITY[ghsv]
    lbl = f"{cap}%  u_s={e['u_s']*1e3:.2f} mm/s"
    axes1[0].plot(res['z'], res['X_CO2']*100, color=pal[j], lw=2, label=lbl)
    axes1[1].plot(res['z'], res['T']-273.15,  color=pal[j], lw=2, label=lbl)
axes1[0].axhline(X_MIN_SPEC*100, color='red', lw=1.5, ls='--',
                 label=f'X = {X_MIN_SPEC*100:.1f}% (H₂ spec)')
axes1[1].axhline(T_C_MAIN, color='grey', lw=1.5, ls='--', alpha=0.8,
                 label=f'T_wall = {T_C_MAIN} °C')
axes1[0].set_ylabel('CO₂ conversion  [%]', fontsize=10)
axes1[1].set_ylabel('Temperature  [°C]', fontsize=10)
axes1[1].set_xlabel('z  [m]', fontsize=10)
for ax in axes1:
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('op_plot1_axial_profiles.png');  plt.close()

# Plot 2: Summary bar chart — X and H2% per capacity level
caps_ok, X_vals, h2_vals, Tmax_vals = [], [], [], []
for ghsv in GHSV_LIST:
    e = all_results.get((ghsv, T_C_MAIN))
    if e and e['res'] is not None:
        X = float(e['res']['X_CO2'][-1])
        caps_ok.append(CAPACITY[ghsv])
        X_vals.append(X * 100)
        h2_vals.append(h2_mol_pct_dry(X))
        Tmax_vals.append(float(np.max(e['res']['T'])) - 273.15)

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 5))
fig2.suptitle(f'Operating point summary  |  T_in = {T_C_MAIN} °C  |  u_s = {U_S_FIXED*1e3:.1f} mm/s (fixed)',
              fontsize=11)
axes2[0].bar(caps_ok, X_vals, color=pal[:len(caps_ok)], edgecolor='k', linewidth=0.5)
axes2[0].axhline(X_MIN_SPEC*100, color='red', lw=1.5, ls='--', label=f'spec limit ({X_MIN_SPEC*100:.1f}%)')
axes2[0].set_xlabel('Capacity  [%]');  axes2[0].set_ylabel('CO₂ conversion  [%]')
axes2[0].set_ylim(0, 105);  axes2[0].legend(fontsize=8);  axes2[0].grid(True, alpha=0.3, axis='y')

axes2[1].bar(caps_ok, h2_vals, color=pal[:len(caps_ok)], edgecolor='k', linewidth=0.5)
axes2[1].axhline(H2_SPEC, color='red', lw=1.5, ls='--', label=f'spec ({H2_SPEC} mol%)')
axes2[1].set_xlabel('Capacity  [%]');  axes2[1].set_ylabel('H₂ in dry product  [mol%]')
axes2[1].legend(fontsize=8);  axes2[1].grid(True, alpha=0.3, axis='y')

axes2[2].bar(caps_ok, Tmax_vals, color=pal[:len(caps_ok)], edgecolor='k', linewidth=0.5)
axes2[2].axhline(T_C_MAIN, color='grey', lw=1.5, ls='--', alpha=0.8, label=f'T_wall ({T_C_MAIN} °C)')
axes2[2].set_xlabel('Capacity  [%]');  axes2[2].set_ylabel('T_peak  [°C]')
axes2[2].legend(fontsize=8);  axes2[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
_savefig('op_plot2_summary_bars.png');  plt.close()
# endregion
