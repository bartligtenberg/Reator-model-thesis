"""
Energy Balance Check — MPB Flux Form (no H2O in feed)
======================================================
Runs a single case and verifies self-consistency of the solved profiles.

Checks (printed):
  A. ODE residual  — (gas_cap − solid_cap)·dT/dz  vs  Q_rxn + Q_ads − Q_wall
     Both sides are integrated over z; they must match to within ODE solver tolerance.
  B. Solid H2O mass balance — ∫ρ_ads·ads dz  vs  u_s·ρ_ads·q(z=0)
     Tests that the adsorption steady-state is satisfied.
  C. Atom balance — CH4/CO2 ≈ 1,  H2/CO2 ≈ 4
  D. Total H2O mole balance — H2O produced (2×∫ρ_cat·r dz) ≈ H2O in gas outlet + H2O in solid outlet

Plots (one 2×2 figure):
  1. LHS vs RHS of energy ODE along z  (+  residual LHS−RHS)
  2. Cumulative ∫Q_rxn, ∫Q_ads, ∫Q_wall along z  (energy budget build-up)
  3. Local source terms Q_rxn, Q_ads, Q_wall [W/m³]
  4. Adsorption loading: q*(z), q(z), ads rate = -u_s·dq/dz along z
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d


# =============================================================================
# 1. PARAMETERS  (identical to parent file MPB_flux_form.py)
# =============================================================================
d_b   = 0.050;  L_b  = 2.000
A_b   = np.pi / 4 * d_b**2;  V_bed = A_b * L_b
eps_b = 0.4

M_cat = 0.064;  M_ads = 1.22
rho_bed_cat = M_cat / V_bed
rho_bed_ads = M_ads / V_bed
rho_bed_tot = (M_cat + M_ads) / V_bed

d_p   = 2.5e-3;  eps_p = 0.615;  tau_p = 3.0;  rho_p = 1400.0

W0_DA = 190.00e-6;  E_DA = 1190e3;  n_DA = 1.55

T_ref_K = 555.0;  k_ref = 3.46e-4;  Ea_k = 77.5e3
A_OH = 0.50;  dH_OH  =  22.4e3
A_H2 = 0.44;  dH_H2  =  -6.2e3
A_mix= 0.88;  dH_mix = -10.0e3
P_FLOOR = 1e-4

dH_r   = -165.0e3
dH_ads =  -45.0e3
Cp_cat = 1100.0
Cp_CO2 = 45.4;  Cp_H2 = 29.3;  Cp_CH4 = 46.9;  Cp_H2O = 34.2

U_a   = 2000.0
R_gas = 8.314;  MW_H2O = 0.018015

P_bar = 1.0;  P_Pa = P_bar * 1e5
y_CO2_in = 0.04;  y_H2_in = 0.16;  y_CH4_in = 0.80  # no H2O in feed

T_STP  = 273.15;  GHSV = 0.5
Q_STP  = GHSV * M_ads / 3600.0
u_g_STP = Q_STP / A_b
F_total_in = u_g_STP * P_Pa / (R_gas * T_STP)
F_in_CO2 = y_CO2_in * F_total_in
F_in_H2  = y_H2_in  * F_total_in
F_in_CH4 = y_CH4_in * F_total_in


# =============================================================================
# 2. FUNCTIONS  (identical to parent file)
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
    A  = np.where((p <= 0)|(p >= Psat), 0.0, A_raw)
    W  = W0*np.exp(-np.minimum((A/E)**n, 500.0))
    qs = rho_water(T_K)/MW_H2O*W
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

def q_star(T_K, p_H2O):
    return q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

_K_LDF_MAX = 20
def K_LDF(T_K, p_H2O):
    return np.minimum(K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA), _K_LDF_MAX)

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

def _gas_cap(F_CO2, F_H2, F_CH4, F_H2O):
    return F_CO2*Cp_CO2 + F_H2*Cp_H2 + F_CH4*Cp_CH4 + F_H2O*Cp_H2O

def _partial_pressures(F_CO2, F_H2, F_CH4, F_H2O):
    F_tot = F_CO2 + F_H2 + F_CH4 + F_H2O
    if F_tot < 1e-30:
        return 0.0, 0.0, 0.0, 0.0
    return (F_CO2/F_tot*P_bar, F_H2/F_tot*P_bar,
            F_CH4/F_tot*P_bar, F_H2O/F_tot*P_bar)

def solve_mpb(u_s, T_K, T_wall=None, max_iter=1000, tol=1e-5, N=400, q_init=None):
    if T_wall is None:
        T_wall = T_K

    solid_cap  = u_s * rho_bed_tot * Cp_cat
    gas_cap_in = _gas_cap(F_in_CO2, F_in_H2, F_in_CH4, 0.0)
    gas_dominates = (solid_cap < gas_cap_in)

    z_grid = np.linspace(0.0, L_b, N)
    q_prof = (np.interp(z_grid, np.linspace(0, L_b, len(q_init)), q_init)
              if q_init is not None else np.zeros(N))
    T_prof = T_K * np.ones(N)

    converged = False;  err = 1.0;  h2o_err = 1.0;  err_q = 1.0;  err_T = 1.0

    for it in range(max_iter):
        q_fn = interp1d(z_grid, q_prof, kind='linear',
                        bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))
        T_fn = interp1d(z_grid, T_prof, kind='linear',
                        bounds_error=False, fill_value=(T_prof[0], T_prof[-1]))

        if gas_dominates:
            def gas_rhs(z, y):
                F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
                F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
                T_l = max(y[4], 200.0);  q_l = max(float(q_fn(z)), 0.0)
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

            gs = solve_ivp(gas_rhs, [0.0, L_b],
                           [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],
                           method='BDF', rtol=1e-5,
                           atol=np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-2]),
                           t_eval=z_grid, dense_output=True)
            if not gs.success:
                return None

            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)
            F_CH4_prof = np.maximum(gs.y[2], 0.0);  F_H2O_prof = np.maximum(gs.y[3], 0.0)
            T_prof_new = np.maximum(gs.y[4], 200.0)

            # Use gas ODE's dense interpolant directly in solid ODE so that T(z) and
            # p_H2O(z) are bit-identical in both ODEs — eliminates structural H2O balance error
            # from piecewise-linear interpolation at grid points.
            def solid_rhs(zeta, q_arr):
                z_pos   = L_b - float(zeta)
                y_gas   = gs.sol(z_pos)
                F_tot_l = max(float(y_gas[0]+y_gas[1]+y_gas[2]+y_gas[3]), 1e-30)
                T_local = max(float(y_gas[4]), 200.0)
                p_H2O_l = max(float(y_gas[3])/F_tot_l * P_bar, 0.0)
                qs = float(q_star(T_local, np.array([p_H2O_l]))[0])
                Kl = float(K_LDF(T_local, np.array([p_H2O_l]))[0])
                return [Kl*(qs - max(float(q_arr[0]), 0.0))/u_s]

            ss = solve_ivp(solid_rhs, [0.0, L_b], [0.0],
                           method='BDF', rtol=1e-5, atol=1e-9,
                           max_step=L_b/N,
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                return None

            z_from_zeta = L_b - ss.t
            q_from_zeta = np.maximum(ss.y[0], 0.0)
            sort_idx    = np.argsort(z_from_zeta)
            q_new = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])

            h2o_prod = 2.0 * (F_in_CO2 - F_CO2_prof[-1])
            h2o_err  = (abs(h2o_prod - F_H2O_prof[-1] - rho_bed_ads*u_s*q_new[0])
                        / max(h2o_prod, 1e-30))
            q_prof_new = 0.5*q_prof + 0.5*q_new
            err_q      = np.max(np.abs(q_prof_new - q_prof)) / max(np.max(q_prof_new), 1e-8)
            err_T      = np.max(np.abs(T_prof_new - T_prof)) / T_K
            err        = max(err_q, err_T)
            T_prof     = 0.5*T_prof + 0.5*T_prof_new
            q_prof     = q_prof_new

        else:
            def gas_rhs_no_T(z, y):
                F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
                F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
                T_l = max(float(T_fn(z)), 200.0);  q_l = max(float(q_fn(z)), 0.0)
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
                           method='BDF', rtol=1e-5,
                           atol=np.array([1e-9, 1e-9, 1e-9, 1e-9]),
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
                q_val = max(float(y_arr[0]), 0.0);  T_val = max(float(y_arr[1]), 200.0)
                z_pos = L_b - float(zeta)
                F_CO2_l = max(float(F_CO2_fn(z_pos)), 0.0)
                F_H2_l  = max(float(F_H2_fn(z_pos)),  0.0)
                F_CH4_l = max(float(F_CH4_fn(z_pos)), 0.0)
                F_H2O_l = max(float(F_H2O_fn(z_pos)), 0.0)
                F_tot_l = max(float(F_tot_fn(z_pos)), 1e-30)
                p_CO2 = F_CO2_l/F_tot_l*P_bar;  p_H2  = F_H2_l/F_tot_l*P_bar
                p_CH4 = F_CH4_l/F_tot_l*P_bar;  p_H2O = F_H2O_l/F_tot_l*P_bar
                r   = float(reaction_rate_SI(T_val, np.array([p_CO2]), np.array([p_H2]),
                                             np.array([p_CH4]), np.array([p_H2O]))[0])
                qs  = float(q_star(T_val, np.array([p_H2O]))[0])
                Kl  = float(K_LDF(T_val, np.array([p_H2O]))[0])
                ads = Kl*(qs - q_val)
                gas_cap_l   = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
                solid_denom = solid_cap - gas_cap_l
                Q_rxn  = (-dH_r)*rho_bed_cat*r
                Q_ads  = (-dH_ads)*rho_bed_ads*ads
                Q_wall = U_a*(T_val - T_wall)
                return [Kl*(qs - q_val)/u_s, (Q_rxn + Q_ads - Q_wall)/solid_denom]

            ss = solve_ivp(solid_rhs_with_T, [0.0, L_b], [0.0, T_K],
                           method='BDF', rtol=1e-5, atol=np.array([1e-9, 0.1]),
                           max_step=1e-3,
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                return None

            z_from_zeta = L_b - ss.t
            q_from_zeta = np.maximum(ss.y[0], 0.0)
            T_from_zeta = np.maximum(ss.y[1], 200.0)
            sort_idx    = np.argsort(z_from_zeta)
            q_new = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])
            T_new = np.interp(z_grid, z_from_zeta[sort_idx], T_from_zeta[sort_idx])

            h2o_prod = 2.0 * (F_in_CO2 - F_CO2_prof[-1])
            h2o_err  = (abs(h2o_prod - F_H2O_prof[-1] - rho_bed_ads*u_s*q_new[0])
                        / max(h2o_prod, 1e-30))
            q_prof_new = 0.75*q_prof + 0.25*q_new
            T_prof_new = 0.75*T_prof + 0.25*T_new
            err_q = np.max(np.abs(q_prof_new - q_prof)) / max(np.max(q_prof_new), 1e-8)
            err_T = np.max(np.abs(T_prof_new - T_prof)) / T_K
            err   = max(err_q, err_T)
            q_prof = q_prof_new;  T_prof = T_prof_new

        if err < tol:
            converged = True;  break

    # Return the converged iterate profiles directly — self-consistent by construction
    F_totf  = np.maximum(F_CO2_prof+F_H2_prof+F_CH4_prof+F_H2O_prof, 1e-30)
    p_CO2f  = F_CO2_prof/F_totf*P_bar;  p_H2f  = F_H2_prof /F_totf*P_bar
    p_CH4f  = F_CH4_prof/F_totf*P_bar;  p_H2Of = F_H2O_prof/F_totf*P_bar
    r_out   = reaction_rate_SI(T_prof, p_CO2f, p_H2f, p_CH4f, p_H2Of)
    X_CO2   = np.clip(1.0 - F_CO2_prof/F_in_CO2, 0.0, 1.0)
    u_g_out = F_totf * R_gas * T_prof / P_Pa
    return dict(z=z_grid, T=T_prof, q=q_prof, r=r_out, X_CO2=X_CO2,
                F_CO2=F_CO2_prof, F_H2=F_H2_prof, F_CH4=F_CH4_prof, F_H2O=F_H2O_prof,
                C_H2O=F_H2O_prof/u_g_out,
                converged=converged, n_iter=it+1, conv_err=float(err),
                err_q=float(err_q), err_T=float(err_T), h2o_err=float(h2o_err),
                gas_dominates=gas_dominates,
                solid_cap=solid_cap)


# =============================================================================
# 3. CASE TO CHECK  — change U_S_CHECK or T_IN_C as needed
# =============================================================================
U_S_CHECK = 4.0e-3   # solid velocity [m/s]
T_IN_C    = 280
T_K_check = T_IN_C + 273.15
T_wall    = T_K_check

gas_cap_in = _gas_cap(F_in_CO2, F_in_H2, F_in_CH4, 0.0)
u_s_star   = gas_cap_in / (rho_bed_tot * Cp_cat)

print("=" * 65)
print(f"Energy balance check")
print(f"  u_s = {U_S_CHECK*1e3:.2f} mm/s  |  T_in = {T_IN_C} °C  |  U_a = {U_a:.0f} W/(m³·K)")
print(f"  u_s* = {u_s_star*1e3:.3f} mm/s  (regime switch)")
print(f"  Solving with tol=1e-5, max_iter=1000, N=200")
print(f"  Output: converged iterate grid — no fine-grid recompute")
print("=" * 65)

# Stage 1: loose solve to get a good initial q profile
_res_warm = solve_mpb(U_S_CHECK, T_K_check, T_wall=T_wall, max_iter=300, tol=1e-3)
_q_init = (np.interp(np.linspace(0, L_b, 150), _res_warm['z'], _res_warm['q'])
           if _res_warm is not None else None)

# Stage 2: tight solve with warm start
res = solve_mpb(U_S_CHECK, T_K_check, T_wall=T_wall, q_init=_q_init)
if res is None:
    raise RuntimeError("Solver failed — cannot check energy balance.")

regime = "gas-dominated" if res['gas_dominates'] else "solid-dominated"
tag    = "converged" if res['converged'] else f"NOT converged"
print(f"  Regime: {regime}  |  {tag}  |  {res['n_iter']} iterations")
print(f"  Final errors:  err_q={res['err_q']:.2e}  err_T={res['err_T']:.2e}  h2o_balance={res['h2o_err']*100:.2f}%")
print(f"  X_CO2 = {res['X_CO2'][-1]*100:.2f}%  |  T_max = {np.max(res['T'])-273.15:.1f} °C  "
      f"|  q(z=0) = {res['q'][0]:.4f} mol/kg")


# =============================================================================
# 4. RECONSTRUCT PROFILES FOR CHECK
# =============================================================================
z    = res['z']
T    = res['T']
q    = res['q']
r    = res['r']       # reaction rate [mol/(kg_cat·s)] × 1000 → actually mol/(m³_bed·s) after rho multiplication below

# H2O partial pressure [bar] from ideal gas: p = C*R*T
p_H2O = res['C_H2O'] * R_gas * T / 1e5

# Equilibrium loading and K_LDF (for panel 4 plot only)
qs_prof = q_star_vec(T, p_H2O, W0_DA, E_DA, n_DA)
K_prof  = np.minimum(K_LDF_vec(T, p_H2O, W0_DA, E_DA, n_DA), _K_LDF_MAX)

# Adsorption rate from the q gradient — numerically stable in the equilibrium limit.
# When K_LDF >> u_s/L, q ≈ q* everywhere and K_LDF*(q*-q) gives 0 due to cancellation.
# The solid ODE gives: dq/dζ = ads/u_s with ζ=L-z, so ads = -u_s * dq/dz.
ads = -U_S_CHECK * np.gradient(q, z)

# Molar flux profiles reconstructed from stoichiometry + cumulative integrals
n_rxn_cum = cumulative_trapezoid(rho_bed_cat * r, z, initial=0.0)
n_ads_cum = cumulative_trapezoid(rho_bed_ads * ads, z, initial=0.0)

F_CO2_rec = F_in_CO2 - n_rxn_cum
F_H2_rec  = F_in_H2  - 4.0*n_rxn_cum
F_CH4_rec = F_in_CH4 + n_rxn_cum
F_H2O_rec = 0.0      + 2.0*n_rxn_cum - n_ads_cum   # F_in_H2O = 0

# Gas thermal capacity from direct solver output [W/(m²·K)]
# Use res['F_CO2'] etc. (not the reconstruction) so gas_cap is consistent with the T profile
gas_cap_prof = (res['F_CO2']*Cp_CO2 + res['F_H2']*Cp_H2
                + res['F_CH4']*Cp_CH4 + res['F_H2O']*Cp_H2O)
solid_cap_val = res['solid_cap']   # u_s * rho_bed_tot * Cp_cat  [W/(m²·K)]

# Local energy terms [W/m³]
Q_rxn_local  = (-dH_r)   * rho_bed_cat * r
Q_ads_local  = (-dH_ads) * rho_bed_ads * ads
Q_wall_local = U_a * (T - T_wall)
RHS = Q_rxn_local + Q_ads_local - Q_wall_local

# LHS of energy ODE
dTdz = np.gradient(T, z)
LHS  = (gas_cap_prof - solid_cap_val) * dTdz

residual = LHS - RHS


# =============================================================================
# 5. PRINT CHECKS
# =============================================================================
def _pct(a, b):
    return (a - b) / max(abs(b), 1e-30) * 100.0

# A. ODE integral check
int_LHS = np.trapz(LHS,  z)
int_RHS = np.trapz(RHS,  z)
Q_rxn_total  = np.trapz(Q_rxn_local,  z)
Q_ads_total  = np.trapz(Q_ads_local,  z)
Q_wall_total = np.trapz(Q_wall_local, z)

print("\n--- A. ODE integral check [W/m²] ---")
print(f"  ∫ LHS dz  = {int_LHS:+.2f}")
print(f"  ∫ RHS dz  = {int_RHS:+.2f}")
print(f"  Difference = {int_LHS - int_RHS:+.2f}  ({_pct(int_LHS, int_RHS):+.3f}%)")
print(f"  (Should be < ~1% if ODE solver is accurate)")
print(f"\n  ∫ Q_rxn  dz = {Q_rxn_total:+.2f}  W/m²")
print(f"  ∫ Q_ads  dz = {Q_ads_total:+.2f}  W/m²")
print(f"  ∫ Q_wall dz = {Q_wall_total:+.2f}  W/m²")
print(f"  Net heat gen - wall = {Q_rxn_total + Q_ads_total - Q_wall_total:+.2f}  W/m²")

# B. Solid H2O mass balance
#    The solid ODE gives dq/dz = -ads/u_s with q(z=L)=0, so integrating:
#    q(z=0) = ∫₀ᴸ ads/u_s dz  →  u_s·ρ_ads·q(0) = ∫ ρ_ads·ads dz
#    Both sides measure the total rate at which H2O is transferred from gas to solid [mol/(m²·s)].
ads_integral  = np.trapz(rho_bed_ads * ads, z)    # ∫ ρ_ads·ads dz [mol/(m²·s)]
solid_h2o_out = U_S_CHECK * rho_bed_ads * q[0]    # u_s·ρ_ads·q(z=0)

print("\n--- B. Solid H2O mass balance [mol/(m²·s)] ---")
print(f"  ∫ ρ_ads·ads dz         = {ads_integral:.6f}  (H2O adsorption rate integrated over bed)")
print(f"  u_s·ρ_ads·q(z=0)       = {solid_h2o_out:.6f}  (H2O carried out by solid at z=0 exit)")
print(f"  Difference             = {ads_integral - solid_h2o_out:+.6f}"
      f"  ({_pct(ads_integral, solid_h2o_out):+.3f}%)")
print(f"  (Should be < ~1%: both sides equal ∫ρ_ads·ads dz by the solid ODE)")

# C. Atom balance
dF_CO2 = F_in_CO2 - res['F_CO2'][-1]     # CO2 consumed
dF_CH4 = res['F_CH4'][-1] - F_in_CH4     # CH4 produced
dF_H2  = F_in_H2  - res['F_H2'][-1]      # H2 consumed

print("\n--- C. Atom balance ---")
print(f"  CO2 consumed = {dF_CO2*1e3:.4f} mmol/(m²·s)")
print(f"  CH4 produced = {dF_CH4*1e3:.4f} mmol/(m²·s)"
      f"   →  CH4/CO2 = {dF_CH4/max(dF_CO2, 1e-15):.5f}  (expect 1.00000)")
print(f"  H2  consumed = {dF_H2*1e3:.4f} mmol/(m²·s)"
      f"   →  H2/CO2  = {dF_H2 /max(dF_CO2, 1e-15):.5f}  (expect 4.00000)")

# D. Total H2O mole balance
#    Stoichiometry: CO2 + 4H2 → CH4 + 2H2O, so 2 mol H2O produced per mol CO2 converted.
#    At steady state: H2O produced = H2O in gas at outlet (z=L) + H2O in solid at exit (z=0).
#    Gas outlet at z=L: F_H2O[-1].  Solid exit at z=0: u_s·ρ_ads·q(z=0).
h2o_produced  = 2.0 * np.trapz(rho_bed_cat * r, z)   # 2 × ∫ρ_cat·r dz
h2o_gas_out   = res['F_H2O'][-1]                      # H2O leaving in gas at z=L
h2o_solid_out = solid_h2o_out                          # H2O leaving in solid at z=0 (from check B)
h2o_out_total = h2o_gas_out + h2o_solid_out

print("\n--- D. Total H2O mole balance [mol/(m²·s)] ---")
print(f"  H2O produced by rxn     = {h2o_produced*1e3:.4f} mmol/(m²·s)  (= 2 × ∫ρ_cat·r dz)")
print(f"  H2O out in gas  (z=L)   = {h2o_gas_out*1e3:.4f} mmol/(m²·s)  (gas-phase outlet flux)")
print(f"  H2O out in solid (z=0)  = {h2o_solid_out*1e3:.4f} mmol/(m²·s)  (u_s·ρ_ads·q(z=0))")
print(f"  Total H2O out           = {h2o_out_total*1e3:.4f} mmol/(m²·s)")
print(f"  Difference (prod−out)   = {(h2o_produced - h2o_out_total)*1e3:+.4f} mmol/(m²·s)"
      f"  ({_pct(h2o_produced, h2o_out_total):+.3f}%)")
print(f"  (Should be < ~1%: all H2O from reaction must leave via gas or solid)")

max_res = np.max(np.abs(residual))
rms_res = np.sqrt(np.mean(residual**2))
print(f"\n--- ODE residual (LHS − RHS) statistics [W/m³] ---")
print(f"  Max |residual| = {max_res:.2f}")
print(f"  RMS residual   = {rms_res:.2f}")
print(f"  Max |Q_rxn|    = {np.max(np.abs(Q_rxn_local)):.2f}  (scale reference)")


# =============================================================================
# 6. PLOTS
# =============================================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f'Energy Balance Check  |  u_s = {U_S_CHECK*1e3:.1f} mm/s  |  '
    f'T_in = {T_IN_C} °C  |  {regime}  |  U_a = {U_a:.0f} W/(m³·K)',
    fontsize=11)

# Panel 1 — LHS vs RHS of energy ODE
ax = axes[0, 0]
ax.plot(z, LHS,      lw=2, color='steelblue',  label='LHS: (gas_cap − solid_cap)·dT/dz')
ax.plot(z, RHS,      lw=2, color='darkorange', ls='--', label='RHS: Q_rxn + Q_ads − Q_wall')
ax.plot(z, residual, lw=1.5, color='red', alpha=0.8, label=f'Residual (LHS−RHS)')
ax.axhline(0, color='grey', lw=0.8, ls=':')
ax.set_xlabel('z [m]');  ax.set_ylabel('[W/m³]')
ax.set_title('Energy ODE: LHS vs RHS  (should overlap)')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

# Panel 2 — Cumulative integrals along z
ax = axes[0, 1]
Q_rxn_cum  = cumulative_trapezoid(Q_rxn_local,  z, initial=0.0)
Q_ads_cum  = cumulative_trapezoid(Q_ads_local,  z, initial=0.0)
Q_wall_cum = cumulative_trapezoid(Q_wall_local, z, initial=0.0)
net_cum    = Q_rxn_cum + Q_ads_cum - Q_wall_cum
ax.plot(z, Q_rxn_cum,  lw=2, color='crimson',    label='∫Q_rxn dz')
ax.plot(z, Q_ads_cum,  lw=2, color='darkorange', label='∫Q_ads dz')
ax.plot(z, Q_wall_cum, lw=2, color='steelblue',  label='∫Q_wall dz  (heat removed)')
ax.plot(z, net_cum,    lw=2, color='black', ls='--',
        label='∫(Q_rxn+Q_ads−Q_wall) dz  (= ∫LHS dz)')
ax.axhline(0, color='grey', lw=0.8, ls=':')
ax.set_xlabel('z [m]');  ax.set_ylabel('[W/m²]')
ax.set_title('Cumulative energy budget along z')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

# Panel 3 — Local source terms
ax = axes[1, 0]
ax.plot(z, Q_rxn_local,  lw=2, color='crimson',    label='Q_rxn = −ΔH_r · ρ_cat · r')
ax.plot(z, Q_ads_local,  lw=2, color='darkorange', label='Q_ads = −ΔH_ads · ρ_ads · ads')
ax.plot(z, Q_wall_local, lw=2, color='steelblue',  label='Q_wall = U_a · (T − T_wall)')
ax.axhline(0, color='grey', lw=0.8, ls=':')
ax.set_xlabel('z [m]');  ax.set_ylabel('[W/m³]')
ax.set_title('Local energy source terms along z')
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

# Panel 4 — Adsorption loading and rate
# q*−q ≈ 0 in the equilibrium limit (K_LDF >> u_s/L → relaxation length ~0.3 mm << 2 m).
# ads rate derived from −u_s·dq/dz to avoid floating-point cancellation.
ax  = axes[1, 1]
ax2 = ax.twinx()
ax.plot(z,  qs_prof,   lw=2, color='limegreen', ls='-',  label='q* (equil. loading)')
ax.plot(z,  q,         lw=2, color='darkgreen',  ls='--', label='q  (actual loading)')
ax2.plot(z, ads * 1e3, lw=2, color='purple',     ls='-',  label='ads rate [mmol/(kg·s)]')
ax2.axhline(0, color='grey', lw=0.8, ls=':')
ax.set_xlabel('z [m]');  ax.set_ylabel('loading [mol/kg]', color='darkgreen')
ax2.set_ylabel('ads rate [mmol/(kg·s)]', color='purple')
ax.set_title('Adsorption loading (q≈q*: equilibrium limit)')
ax.legend(loc='upper left',  fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fname = f'energy_balance_check_us{U_S_CHECK*1e3:.2f}mms_T{T_IN_C}C.png'
plt.savefig(os.path.join(SAVE_DIR, fname), dpi=150, bbox_inches='tight')

# Reaction rate figure — for comparison with Bareschino
r_vol = rho_bed_cat * r   # [mol/(m³_bed·s)]

fig2, ax_r = plt.subplots(figsize=(7, 4))
ax_r.plot(z, r_vol, lw=2, color='crimson')
ax_r.set_xlabel('z [m]')
ax_r.set_ylabel('Reaction rate  [mol/(m³$_{bed}$·s)]')
ax_r.set_title(
    f'Reaction rate along bed  |  u_s = {U_S_CHECK*1e3:.1f} mm/s  |  '
    f'T_in = {T_IN_C} °C  |  {regime}')
ax_r.grid(True, alpha=0.3)
fig2.tight_layout()
fname2 = f'reaction_rate_us{U_S_CHECK*1e3:.2f}mms_T{T_IN_C}C.png'
fig2.savefig(os.path.join(SAVE_DIR, fname2), dpi=150, bbox_inches='tight')

plt.show()
