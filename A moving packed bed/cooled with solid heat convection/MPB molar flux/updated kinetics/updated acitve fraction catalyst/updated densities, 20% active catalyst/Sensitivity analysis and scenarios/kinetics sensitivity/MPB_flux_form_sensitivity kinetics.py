"""
Moving Packed Bed (MPB) Reactor Model — Kinetics Sensitivity Sweep
==========================================================================================
Same reference operating point as the Sec. 7.1 deep dive:
    u_s   = 5.84 mm/s
    GHSV  = 1.5 m3_STP/(kg_ads.h)
    T_in  = 280 C
    dilution = y_CH4_in = 0.80

but instead of a single solve, the LHHW rate expression is multiplied by a
global factor KIN_MULT swept over KIN_MULT_LIST = [0.5, 1.0, 2.0] (i.e. the
catalyst is assumed 2x slower / nominal / 2x faster than the Koschany et al.
(2016) rate law used elsewhere in this thesis). Everything downstream (heat
release, adsorption coupling, solver) responds self-consistently to the
scaled rate -- only the reaction-rate prefactor itself is touched.

Physics core (parameters, isotherm/kinetics functions, decoupled Gauss-Seidel
solver, non-SE fixed-bed reference) is copied unmodified from the Sec. 7.1
script, except:
  - reaction_rate_SI() is scaled by the module-level global KIN_MULT, which
    the sweep loop (region 5) sets before each solve_mpb / _compute_noSE call.
  - regions 5-7 (solve, figures, console output) are restructured into a
    run_case() helper called once per KIN_MULT, with each of the 9 original
    figures re-expressed as either an overlay of all 3 cases (same axes) or
    a 1x3 small-multiple panel (one subplot per case), whichever keeps the
    original figure readable. See the per-figure comments in region 6.
  - the maximum local reaction rate observed is printed per case and overall.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# region 1. PARAMETERS
# =============================================================================

# --- Bed geometry (Bareschino lab setup) ---
d_b   = 0.050                # [m]   reactor inner (tube) diameter — Bareschino et al. (2023)
L_b   = 2.000                # [m]   bed length — Bareschino et al. (2023)
A_b   = np.pi / 4 * d_b**2   # [m²]  bed cross-sectional area
V_bed = A_b * L_b            # [m³]  total bed volume
eps_b = 0.4                  # [-]   inter-particle (bed) void fraction — typical packed-bed value (not from Bareschino table)

# --- Particle properties (13X zeolite pellets) ---
d_p   = 2.5e-3   # [m]      particle diameter — this study's pellet size (Same as Bareschino et al. 2023. Wei used 0.75 mm)
eps_p = 0.242    # [-]      intraparticle void fraction — calclated with Wei's pore volume and density (Bareschino)
tau_p = 4.0      # [-]      pore tortuosity factor — Mette et al. (2015)
rho_p = 1400     # [kg/m³]  particle (skeletal) density of sorbent — Bareschino et al. (2023)

# --- Catalyst and sorbent loading ---
bifuctional_mass = 0.8   # [kg]  mass of bifunctional 5%Ni-2.5%Ce/13X material
M_zeolite_added  = 0     # [kg]  additional pure 13X zeolite mixed in — 100% sorbent-active, 0% catalytically active (no Ni)

M_ads = bifuctional_mass * 0.925 + M_zeolite_added   # [kg]  sorbent mass: 92.5% of the bifunctional material acts as sorbent, plus all of the added pure zeolite

active_fraction = 0.20   # [-]  fraction of the bifunctional material's mass that is catalytically active
M_cat_active = bifuctional_mass * active_fraction   # [kg]  active catalyst mass — only the bifunctional material carries Ni; the added zeolite contributes none

M_solid_physical = bifuctional_mass + M_zeolite_added   # [kg]  true physical solid mass present (catalyst+sorbent material), before filler

# --- Inert filler (thermal buffering / dilution / flow aid) ---
M_filler = (1 - eps_b) * V_bed * rho_p - M_solid_physical   # [kg]

rho_bed_cat = M_cat_active / V_bed   # [kg_cat/m³_bed]  catalyst bulk density (reaction terms)
rho_bed_ads = M_ads / V_bed          # [kg_ads/m³_bed]  sorbent bulk density (adsorption terms)
rho_bed_tot = (M_solid_physical + M_filler) / V_bed  # [kg_solid/m³_bed]  total solids bulk density (cat+ads material+filler)

# --- Dubinin-Astakhov isotherm (H2O on 13X) --- fitted based on Wei et al. (2021)
W0_DA = 190.00e-6   # [m³/kg_sorbent]  limiting micropore volume
E_DA  = 1192e3      # [J/kg]           characteristic adsorption energy
n_DA  = 1.55        # [-]              DA heterogeneity parameter

# --- LHHW kinetics (Koschany et al. 2016, Table 6) ---
T_ref_K = 555.0      # [K]                    reference temperature (282 °C) — Koschany et al. (2016)
k_ref   = 3.46e-4    # [mol/(g_cat·s·bar)]    rate constant at T_ref — Koschany et al. (2016)
Ea_k    = 77.5e3     # [J/mol]                activation energy — Koschany et al. (2016)
A_OH    = 0.50;  dH_OH  =  22.4e3   # [bar^-0.5], [J/mol]
A_H2    = 0.44;  dH_H2  =  -6.2e3   # [bar^-0.5], [J/mol]
A_mix   = 0.88;  dH_mix = -10.0e3   # [bar^-0.5], [J/mol]
P_FLOOR = 1e-4       # [bar]

# --- Kinetics sensitivity sweep ---
KIN_MULT_LIST = [0.5, 1.0, 2.0]   # [-] multiplicative factors applied to the LHHW rate expression
KIN_MULT = 1.0                     # [-] active multiplier; mutated by the sweep loop (region 5) before each solve

# --- Thermochemistry ---
dH_r   = -165.0e3   # [J/mol_CO2]   Sabatier reaction enthalpy — NIST
dH_ads =  -45.0e3   # [J/mol_H2O]   isosteric heat of H2O adsorption on 13X — Bareschino et al. (2023) Table 3
Cp_cat = 1100.0      # [J/(kg·K)]    catalyst/sorbent heat capacity — Bareschino et al. (2023) Table 3
Cp_CO2 = 45.4;  Cp_H2 = 29.3;  Cp_CH4 = 46.9;  Cp_H2O = 34.2   # [J/(mol·K)]  gas heat capacities — NIST at ~550 K

# --- Wall heat transfer ---
U_a    = 2000.0   # [W/(m³_bed·K)]

# --- Physical constants ---
R_gas  = 8.314       # [J/(mol·K)]
MW_H2O = 0.018015    # [kg/mol]

# --- Operating conditions ---
P_bar = 1.0                # [bar]
P_Pa  = P_bar * 1e5         # [Pa]
y_CO2_in = 0.04             # [-]    CO2 inlet mole fraction — Bareschino et al. (2022/2023)
y_H2_in  = 0.16             # [-]    H2 inlet mole fraction (H2/CO2 = 4, stoichiometric)
y_CH4_in = 0.80             # [-]    CH4 inlet mole fraction (background/diluent)

# --- Inlet molar fluxes [mol/(m²·s)] ---
T_STP   = 273.15                  # [K]
GHSV    = 1.5                     # [m³_STP/(kg_ads·h)]
Q_STP   = GHSV * M_ads / 3600.0   # [m³_STP/s]
u_g_STP = Q_STP / A_b             # [m/s]
F_total_in = u_g_STP * P_Pa / (R_gas * T_STP)  # [mol/(m²·s)]
F_in_CO2   = y_CO2_in * F_total_in
F_in_H2    = y_H2_in  * F_total_in
F_in_CH4   = y_CH4_in * F_total_in

# --- Reference-case operating point ---
u_s_case = 0.00584   # [m/s]  solid (sorbent/catalyst) velocity
T_C_case = 280       # [C]    inlet temperature

print(f"MPB flux form — kinetics sensitivity sweep: d={d_b*100:.0f} cm, L={L_b:.1f} m, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3")
print(f"  GHSV = {GHSV:.1f} m3_STP/(kg_ads.h)  ->  u_g_STP = {u_g_STP*1e3:.3f} mm/s, "
      f"F_in_total = {F_total_in:.4f} mol/(m2.s)")
print(f"  Dilution (background/diluent): y_CH4_in = {y_CH4_in:.0%}")
print(f"  u_s = {u_s_case*1e3:.1f} mm/s,  T_in = {T_C_case} C")
print(f"  M_cat(active) = {M_cat_active*1000:.1f} g,  M_ads = {M_ads*1000:.1f} g,  U_a = {U_a:.0f} W/(m3.K)")
print(f"  Kinetics multipliers swept: KIN_MULT = {KIN_MULT_LIST}")
# endregion


# region 2. FUNCTIONS
# =============================================================================
def P_sat_bar(T_K):
    log10_p = (29.8605 - 3.1522e3/T_K - 7.3037*np.log10(T_K)
               + 2.4247e-9*T_K + 1.8090e-6*T_K**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5   # [mmHg] -> [bar]

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
    D_M = 3.36e-9 * T_K**1.75
    p    = np.asarray(p_arr, dtype=float)
    dp   = 1.0/1e5
    dqsp = (q_star_vec(T_K, p+dp, W0, E, n)
            - q_star_vec(T_K, np.maximum(p-dp, 1e-15), W0, E, n)) / 2.0
    dqsp = np.maximum(dqsp, 1e-30)
    r_p = 0.5 * d_p
    return 15.0 * eps_p * D_M / (r_p**2 * tau_p * rho_p * R_gas * T_K * dqsp)

def K_eq_sabatier(T_K):
    return 137.0*T_K**(-3.994)*np.exp(158700.0/(R_gas*T_K))

def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    """LHHW rate [mol/(kg_cat·s)], scaled by the global kinetics multiplier KIN_MULT."""
    vH    = lambda dH: np.exp(-dH/R_gas*(1.0/T_K - 1.0/T_ref_K))
    k     = k_ref*np.exp(-Ea_k/R_gas*(1.0/T_K - 1.0/T_ref_K))
    K_OH  = A_OH*vH(dH_OH);  K_H2 = A_H2*vH(dH_H2);  K_mix = A_mix*vH(dH_mix)
    K_eq  = K_eq_sabatier(T_K)
    p_CO2_s = np.maximum(p_CO2, P_FLOOR);  p_H2_s = np.maximum(p_H2, P_FLOOR)
    beta  = (p_CH4*p_H2O**2)/(K_eq*p_CO2_s*p_H2_s**4)
    f_eq  = np.maximum(1.0 - np.where(np.isfinite(beta), beta, 1e10), 0.0)
    DEN   = (1.0 + K_OH*np.maximum(p_H2O, 0)/p_H2_s**0.5
             + K_H2*p_H2_s**0.5 + K_mix*p_CO2_s**0.5)
    return KIN_MULT * k*(p_CO2_s*p_H2_s)**0.5*f_eq/DEN**2*1000.0

def q_star(T_K, p_H2O):
    return q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

def K_LDF(T_K, p_H2O):
    return K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

def equilibrium_conversion(T_K_val):
    """Thermodynamic equilibrium conversion — independent of kinetics, computed once."""
    K = K_eq_sabatier(T_K_val)
    def f(X):
        d = 1.0 - 0.08*X
        return ((0.80+0.04*X)/d * (0.08*X/d)**2
                / ((0.04*(1-X)/d)*(0.16*(1-X)/d)**4 + 1e-100) - K)
    try:
        return brentq(f, 1e-9, 1-1e-9)*100.0
    except Exception:
        return 100.0

def _gas_cap(F_CO2, F_H2, F_CH4, F_H2O):
    """Thermal flux of gas phase [W/(m²·K)] = Σ F_i·Cp_i."""
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
def solve_mpb(u_s, T_K, T_wall=None, max_iter=1000, tol=1e-5, N=400, q_init=None):
    """Counter-current MPB — molar flux form, lightly cooled, regime-switching.
    Reads reaction_rate_SI(), which is scaled by the module-level KIN_MULT in
    effect at call time -- callers must set KIN_MULT before calling solve_mpb.
    """
    if T_wall is None:
        T_wall = T_K

    solid_cap  = u_s * rho_bed_tot * Cp_cat
    gas_cap_in = _gas_cap(F_in_CO2, F_in_H2, F_in_CH4, 0.0)
    gas_dominates = (solid_cap < gas_cap_in)

    z_grid = np.linspace(0.0, L_b, N)
    if q_init is not None:
        q_prof = np.interp(z_grid, np.linspace(0, L_b, len(q_init)), q_init)
    else:
        q_prof = np.zeros(N)
    T_prof = T_K * np.ones(N)

    converged = False
    err = 1.0
    _solid_denom_min = [np.inf]

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
                dTdz   = (Q_rxn + Q_ads - Q_wall) / denom

                return [
                    -rho_bed_cat * r,
                    -4.0*rho_bed_cat * r,
                    +rho_bed_cat * r,
                    2.0*rho_bed_cat*r - rho_bed_ads*ads,
                    dTdz,
                ]

            gs = solve_ivp(gas_rhs, [0.0, L_b],
                           [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],
                           method='BDF', rtol=1e-4,
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8, 1e-2]),
                           t_eval=z_grid, dense_output=False)
            if not gs.success:
                print(f"    [solve_ivp FAILED] gas IVP (gas-dominant): {gs.message}  "
                      f"(z_last={gs.t[-1]:.4f} m)")
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
                           max_step=1e-3,
                           t_eval=np.linspace(0.0, L_b, N),
                           dense_output=False)
            if not ss.success:
                print(f"    [solve_ivp FAILED] solid IVP (gas-dominant): {ss.message}  "
                      f"(zeta_last={ss.t[-1]:.4f} m)")
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
                           [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                           method='BDF', rtol=1e-4,
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8]),
                           t_eval=z_grid, dense_output=False)
            if not gs.success:
                print(f"    [solve_ivp FAILED] gas IVP (solid-dominant): {gs.message}  "
                      f"(z_last={gs.t[-1]:.4f} m)")
                return None

            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)
            F_CH4_prof = np.maximum(gs.y[2], 0.0);  F_H2O_prof = np.maximum(gs.y[3], 0.0)
            F_tot_prof = np.maximum(F_CO2_prof + F_H2_prof + F_CH4_prof + F_H2O_prof, 1e-30)

            _make_fn = lambda p: interp1d(z_grid, p, kind='linear',
                                          bounds_error=False, fill_value=(p[0], p[-1]))
            F_CO2_fn = _make_fn(F_CO2_prof);  F_H2_fn  = _make_fn(F_H2_prof)
            F_CH4_fn = _make_fn(F_CH4_prof);  F_H2O_fn = _make_fn(F_H2O_prof)
            F_tot_fn = _make_fn(F_tot_prof)

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
                _solid_denom_min[0] = min(_solid_denom_min[0], solid_denom)

                Q_rxn  = (-dH_r)   * rho_bed_cat * r
                Q_ads  = (-dH_ads) * rho_bed_ads * ads
                Q_wall = U_a * (T_val - T_wall)

                return [Kl*(qs - q_val)/u_s,
                        (Q_rxn + Q_ads - Q_wall) / solid_denom]

            ss = solve_ivp(solid_rhs_with_T, [0.0, L_b], [0.0, T_K],
                           method='BDF', rtol=1e-4, atol=np.array([1e-8, 0.1]),
                           max_step=1e-3,
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                sd_min = _solid_denom_min[0]
                sd_str = f"{sd_min:.2f}" if np.isfinite(sd_min) else "n/a"
                print(f"    [solve_ivp FAILED] solid IVP (solid-dominant): {ss.message}  "
                      f"(zeta_last={ss.t[-1]:.4f} m, min solid_denom={sd_str} W/(m2.K))")
                return None

            z_from_zeta  = L_b - ss.t
            q_from_zeta  = np.maximum(ss.y[0], 0.0)
            T_from_zeta  = np.maximum(ss.y[1], 200.0)
            sort_idx     = np.argsort(z_from_zeta)
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
            return [
                -rho_bed_cat*r,
                -4.0*rho_bed_cat*r,
                +rho_bed_cat*r,
                2.0*rho_bed_cat*r - rho_bed_ads*ads,
            ]
        gf = solve_ivp(gas_rhs_final_no_T, [0.0, L_b],
                       [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
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
                F_CO2=F_CO2f, F_H2=F_H2f, F_CH4=F_CH4f, F_H2O=F_H2Of,
                converged=converged, n_iter=it+1, conv_err=float(err),
                gas_dominates=gas_dominates)
# endregion


# region 4. NON-SE REFERENCE + WARM-START HELPER
# =============================================================================
def _compute_noSE(T_K, T_wall, N=300):
    """Fixed-bed (u_s=0) reference with wall cooling: no sorption enhancement (noSE).
    Also reads the global KIN_MULT -- computed per-case in the sweep so it
    reflects the same kinetics multiplier as its paired MPB solve."""
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
    sol = solve_ivp(rhs, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],
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
    """Isothermal, no-adsorption gas pass -> physics-motivated initial q(z) guess for warm-starting solve_mpb."""
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
    sol = solve_ivp(rhs_noads, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                    method='BDF', rtol=1e-4, atol=1e-8, t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    F_tot_f = np.maximum(sol.y[0] + sol.y[1] + sol.y[2] + sol.y[3], 1e-30)
    p_H2O_f = np.maximum(sol.y[3], 0.0)/F_tot_f * P_bar
    return dict(q=q_star(T_K, p_H2O_f),
                X_CO2_noSE=float(np.clip(1.0 - F_CO2_f[-1]/F_in_CO2, 0.0, 1.0)))

def _h2o_balance_line(u_s, res):
    """Sanity check: does H2O produced by reaction == H2O leaving in the gas + H2O leaving on the solid?"""
    T_out      = float(res['T'][-1])
    y_CO2_out  = float(res['C_CO2'][-1]) * R_gas * T_out / P_Pa
    F_CO2_out  = F_in_CO2 * (1.0 - float(res['X_CO2'][-1]))
    F_tot_out  = F_CO2_out / max(y_CO2_out, 1e-30)
    F_H2O_out  = float(res['C_H2O'][-1]) * R_gas * T_out / P_Pa * F_tot_out
    F_H2O_prod = 2.0 * F_in_CO2 * float(res['X_CO2'][-1])
    F_H2O_ads  = u_s * rho_bed_ads * float(res['q'][0])
    bal_err    = (F_H2O_out + F_H2O_ads - F_H2O_prod) / max(F_H2O_prod, 1e-30) * 100
    return (f"    H2O balance [mmol/(m²·s)]:  produced={F_H2O_prod*1e3:.3f}  "
            f"gas_out={F_H2O_out*1e3:.3f}  solid_out={F_H2O_ads*1e3:.3f}  "
            f"err={bal_err:+.1f}%")

def _z_at_conversion(res, target_pct):
    """z [m] where CO2 conversion first reaches target_pct (linear interpolation); None if never reached."""
    z_arr, X_arr = res['z'], res['X_CO2']
    target = target_pct / 100.0
    if X_arr[-1] < target:
        return None
    return float(np.interp(target, X_arr, z_arr))
# endregion


# region 5. SOLVE THE KINETICS SWEEP
# =============================================================================
T_K_case    = T_C_case + 273.15
T_wall_case = T_K_case   # adiabatic wall (T_wall = T_in)

X_eq = equilibrium_conversion(T_K_case)   # thermodynamic equilibrium — independent of kinetics, computed once

def run_case(kin_mult, q_init):
    """Solve the MPB + noSE reference for one kinetics multiplier and package
    every derived quantity that the region-6 figures / region-7 tables need."""
    global KIN_MULT
    KIN_MULT = kin_mult

    print(f"\n{'='*70}")
    print(f"  Solving case: KIN_MULT = {kin_mult:.2f}x   "
          f"(u_s = {u_s_case*1e3:.2f} mm/s, GHSV = {GHSV:.1f}, T_in = T_wall = {T_C_case} C)")
    print(f"{'='*70}")

    t0  = time.perf_counter()
    res = solve_mpb(u_s_case, T_K_case, T_wall=T_wall_case, q_init=q_init)
    attempt = 1
    while (res is None or not res['converged']) and attempt < 5:
        attempt += 1
        if res is not None:
            q_init = np.interp(np.linspace(0, L_b, 150), res['z'], res['q'])
        print(f"  Not converged on attempt {attempt-1} -> retrying (attempt {attempt}), "
              f"warm-starting from its own q(z) profile...")
        res = solve_mpb(u_s_case, T_K_case, T_wall=T_wall_case, q_init=q_init,
                         max_iter=2000, tol=1e-6)
    dt = time.perf_counter() - t0

    if res is None:
        raise RuntimeError(f"solve_mpb failed to produce a solution for KIN_MULT={kin_mult}.")

    regime = "gas-dominated" if res['gas_dominates'] else "solid-dominated"
    r_max  = float(np.max(res['r']))
    print(f"  Solved in {dt:.1f}s  [{regime}, {'converged' if res['converged'] else 'NOT CONVERGED'}, "
          f"{res['n_iter']} outer iterations, err={res['conv_err']:.2e}]")
    print(f"  X_CO2(outlet) = {res['X_CO2'][-1]*100:.2f}%   T_max = {np.max(res['T'])-273.15:.1f} C   "
          f"r_max = {r_max*1e3:.4f} mmol/(kg_cat.s)")
    print(f"  T_out (gas, z=L) = {res['T'][-1]-273.15:.1f} C   T(solid outlet, z=0) = {res['T'][0]-273.15:.1f} C")

    z_95 = _z_at_conversion(res, 95.0)
    if z_95 is not None:
        print(f"  z(95% conversion) = {z_95:.3f} m  ({z_95/L_b*100:.1f}% of bed length)")
    else:
        print(f"  z(95% conversion) = not reached (outlet X_CO2 = {res['X_CO2'][-1]*100:.2f}% < 95%)")

    noSE = _compute_noSE(T_K_case, T_wall_case)   # same KIN_MULT as this case, so it's a matched non-SE comparison
    print(f"  Non-SE fixed-bed reference conversion: {noSE['X_CO2_noSE']*100:.2f}%   "
          f"(thermodynamic equilibrium @ T_in: {X_eq:.2f}%, kinetics-independent)")
    print(_h2o_balance_line(u_s_case, res))

    # ── derived per-z quantities, needed by figs 3/6/7/8/9 and the char.-time table ──
    z = res['z']
    p_H2O_z = res['C_H2O']*R_gas*res['T']/1e5   # [bar]
    q_eq_z  = q_star(res['T'], p_H2O_z)
    Kl_f    = K_LDF(res['T'], p_H2O_z)
    ads_f   = Kl_f*(q_eq_z - res['q'])

    S_rxn   = (-dH_r)   * rho_bed_cat * res['r']
    S_ads   = (-dH_ads) * rho_bed_ads * ads_f
    S_wall  = -U_a * (res['T'] - T_wall_case)
    S_total = S_rxn + S_ads + S_wall

    dTdz_num      = np.gradient(res['T'], z)
    gas_cap_z     = _gas_cap(res['F_CO2'], res['F_H2'], res['F_CH4'], res['F_H2O'])
    solid_cap_case = u_s_case * rho_bed_tot * Cp_cat
    S_gas_sens    = gas_cap_z * dTdz_num
    S_solid_sens  = -solid_cap_case * dTdz_num

    C_gas       = cumulative_trapezoid(S_gas_sens,   z, initial=0) * A_b
    C_solid_fwd = cumulative_trapezoid(S_solid_sens, z, initial=0) * A_b
    C_solid     = C_solid_fwd[-1] - C_solid_fwd
    C_rxn  = cumulative_trapezoid(S_rxn,  z, initial=0) * A_b
    C_ads  = cumulative_trapezoid(S_ads,  z, initial=0) * A_b
    C_wall = cumulative_trapezoid(S_wall, z, initial=0) * A_b

    Q_rxn_W  = float(C_rxn[-1]);   Q_ads_W  = float(C_ads[-1]);   Q_wall_W = float(-C_wall[-1])
    Q_gasS_W = float(C_gas[-1]);   Q_solS_W = float(C_solid[0]);  Q_sensS_W = Q_gasS_W + Q_solS_W
    Q_prod   = Q_rxn_W + Q_ads_W;  Q_out    = Q_wall_W + Q_sensS_W
    mismatch_bb = (Q_prod - Q_out) / max(abs(Q_prod), 1e-30) * 100

    # ── characteristic timescales ──
    u_g_z   = (res['F_CO2']+res['F_H2']+res['F_CH4']+res['F_H2O']) * R_gas * res['T'] / P_Pa
    tau_gas = float(np.trapz(1.0/u_g_z, z))
    tau_solid = L_b / u_s_case
    i_pmax  = int(np.argmax(p_H2O_z))
    Kldf_peak = float(K_LDF(res['T'][i_pmax], np.array([p_H2O_z[i_pmax]]))[0])
    inv_Kldf_peak = 1.0/Kldf_peak
    u_g_in   = F_total_in * R_gas * T_K_case / P_Pa
    C_CO2_in = F_in_CO2 / u_g_in
    r_in = float(reaction_rate_SI(T_K_case, np.array([y_CO2_in*P_bar]), np.array([y_H2_in*P_bar]),
                                   np.array([y_CH4_in*P_bar]), np.array([0.0]))[0])
    tau_rxn = C_CO2_in / (rho_bed_cat * r_in)

    return dict(kin_mult=kin_mult, res=res, noSE=noSE, regime=regime, r_max=r_max,
                p_H2O_z=p_H2O_z, q_eq_z=q_eq_z, Kl_f=Kl_f, i_pmax=i_pmax,
                S_rxn=S_rxn, S_ads=S_ads, S_wall=S_wall, S_total=S_total,
                C_gas=C_gas, C_solid=C_solid,
                Q_rxn_W=Q_rxn_W, Q_ads_W=Q_ads_W, Q_wall_W=Q_wall_W,
                Q_gasS_W=Q_gasS_W, Q_solS_W=Q_solS_W, Q_sensS_W=Q_sensS_W,
                Q_prod=Q_prod, Q_out=Q_out, mismatch_bb=mismatch_bb,
                tau_gas=tau_gas, tau_solid=tau_solid, inv_Kldf_peak=inv_Kldf_peak,
                tau_rxn=tau_rxn, r_in=r_in, dt=dt, z_95=z_95)


_phys  = _q_physics_init(T_K_case)   # KIN_MULT is still 1.0 here (module default) for the initial guess
q_init = _phys['q'][::-1]

cases = []
for _mult in KIN_MULT_LIST:
    _case = run_case(_mult, q_init)
    q_init = np.interp(np.linspace(0, L_b, 150), _case['res']['z'], _case['res']['q'])   # warm-start next case
    cases.append(_case)

r_max_overall  = max(c['r_max'] for c in cases)
mult_at_rmax   = next(c['kin_mult'] for c in cases if c['r_max'] == r_max_overall)
print(f"\n{'='*70}")
print(f"  Max local reaction rate observed across the sweep: "
      f"r_max = {r_max_overall*1e3:.4f} mmol/(kg_cat.s)  (at KIN_MULT = {mult_at_rmax:.2f}x)")
print(f"{'='*70}")
# endregion


# region 6. FIGURES
# =============================================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def _savefig(filename):
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=150, bbox_inches='tight')

CASE_COLORS = ['tab:blue', 'black', 'tab:red']   # one per entry of KIN_MULT_LIST
CASE_LABELS = [f'KIN_MULT = {m:g}x' for m in KIN_MULT_LIST]
i_ref = KIN_MULT_LIST.index(1.0) if 1.0 in KIN_MULT_LIST else 0   # baseline case used for the single noSE reference curve

# ── Figure 1: Conversion vs z (overlay) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for c, col, lab in zip(cases, CASE_COLORS, CASE_LABELS):
    ax.plot(c['res']['z'], c['res']['X_CO2']*100, color=col, lw=2.5, label=f'MPB, {lab}')
    print(f"  Fig.1 [{lab}]: SE outlet conversion = {c['res']['X_CO2'][-1]*100:.2f}%, "
          f"non-SE fixed-bed outlet = {c['noSE']['X_CO2_noSE']*100:.2f}%")
noSE_ref = cases[i_ref]['noSE']
ax.plot(noSE_ref['profile']['z'], noSE_ref['profile']['X_CO2']*100, color='k', lw=2, ls='--',
        label=f'Fixed bed, no SE ($u_s$=0, {CASE_LABELS[i_ref]})')
ax.axhline(X_eq, color='grey', lw=1.5, ls=':', label=f'Thermodynamic equilibrium ({T_C_case} C)')
ax.set_xlabel('z [m]');  ax.set_ylabel('CO2 conversion [%]')
ax.set_title(f'CO2 conversion vs bed position — kinetics sensitivity  |  $T_{{in}}$={T_C_case} C')
ax.legend(fontsize=10);  ax.grid(True, alpha=0.3);  ax.set_ylim(0, 105)
plt.tight_layout()
_savefig('kinsens_fig1_conversion_vs_z.png');  plt.show()

# ── Figure 2: Gas-phase composition vs z (1x3 small multiples, one per case) ─
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
for ax1, c, lab in zip(axes, cases, CASE_LABELS):
    res_c = c['res']
    C_tot = res_c['C_CO2'] + res_c['C_H2'] + res_c['C_CH4'] + res_c['C_H2O']
    y_CO2 = res_c['C_CO2']/C_tot;  y_H2 = res_c['C_H2']/C_tot
    y_CH4 = res_c['C_CH4']/C_tot;  y_H2O = res_c['C_H2O']/C_tot
    ax1.plot(res_c['z'], y_CH4*100, color='tab:green', lw=2.0, label='CH4')
    ax1.set_xlabel('z [m]');  ax1.set_ylabel('y_CH4 [mol%]', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax2 = ax1.twinx()
    ax2.plot(res_c['z'], y_CO2*100, color='tab:blue',   lw=1.8, label='CO2')
    ax2.plot(res_c['z'], y_H2*100,  color='tab:orange', lw=1.8, label='H2')
    ax2.plot(res_c['z'], y_H2O*100, color='tab:cyan',   lw=1.8, label='H2O')
    ax2.set_ylabel('minor species [mol%]')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper right')
    ax1.set_title(lab, fontsize=10)
    ax1.grid(True, alpha=0.3)
fig.suptitle('Gas-phase composition vs z — kinetics sensitivity')
plt.tight_layout()
_savefig('kinsens_fig2_gas_composition_vs_z.png');  plt.show()

# ── Figure 3: Solid loading vs equilibrium loading (1x3 small multiples) ─────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
for ax, c, lab in zip(axes, cases, CASE_LABELS):
    res_c = c['res'];  q_eq_z = c['q_eq_z']
    ax.plot(res_c['z'], res_c['q'], color='tab:blue', lw=2.2, label='q(z) actual')
    ax.plot(res_c['z'], q_eq_z,     color='tab:red',  lw=1.8, ls='--', label='q*(z) DA equilibrium')
    ax.fill_between(res_c['z'], res_c['q'], q_eq_z, where=(q_eq_z >= res_c['q']),
                    color='tab:blue', alpha=0.15, label='adsorption driving force')
    ax.fill_between(res_c['z'], res_c['q'], q_eq_z, where=(q_eq_z < res_c['q']),
                    color='tab:red', alpha=0.15, label='desorption driving force')
    ax.set_xlabel('z [m]');  ax.set_title(lab, fontsize=10)
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
axes[0].set_ylabel('H2O loading [mol/kg_ads]')
fig.suptitle('Solid H2O loading vs equilibrium — kinetics sensitivity')
plt.tight_layout()
_savefig('kinsens_fig3_loading_vs_equilibrium.png');  plt.show()

# ── Figure 4: Temperature profile (overlay) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for c, col, lab in zip(cases, CASE_COLORS, CASE_LABELS):
    res_c = c['res']
    i_max = int(np.argmax(res_c['T']))
    z_hot = float(res_c['z'][i_max]);  T_max_C = float(res_c['T'][i_max]) - 273.15
    ax.plot(res_c['z'], res_c['T']-273.15, color=col, lw=2.2, label=f'{lab}  ({c["regime"]})')
    ax.plot(z_hot, T_max_C, 'o', color=col, ms=6)
ax.axhline(T_C_case, color='grey', lw=1.5, ls='--', label=f'$T_{{in}}$ = $T_{{wall}}$ = {T_C_case} C')
ax.set_xlabel('z [m]');  ax.set_ylabel('T [C]')
ax.set_title('Temperature profile — kinetics sensitivity')
ax.legend(fontsize=10);  ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('kinsens_fig4_temperature_vs_z.png');  plt.show()

# ── Figure 5: Local rate and cumulative conversion (twin-axis, overlay) ─────
fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()
for c, col, lab in zip(cases, CASE_COLORS, CASE_LABELS):
    res_c = c['res']
    ax1.plot(res_c['z'], res_c['r']*1e3, color=col, lw=2.2, ls='-', label=f'r(z), {lab}')
    ax2.plot(res_c['z'], res_c['X_CO2']*100, color=col, lw=1.8, ls='--', label=f'$X_{{CO2}}$(z), {lab}')
ax2.plot(noSE_ref['profile']['z'], noSE_ref['profile']['X_CO2']*100, color='k', lw=1.8, ls=':',
         label=f'Fixed bed, no SE ({CASE_LABELS[i_ref]})')
ax2.axhline(X_eq, color='grey', lw=1.5, ls=':', label=f'Thermodynamic equilibrium ({T_C_case} C)')
ax1.set_xlabel('z [m]');  ax1.set_ylabel('r [mmol/(kg_cat·s)]  (solid, primary axis)')
ax2.set_ylabel('CO2 conversion [%]  (dashed, secondary axis)')
ax2.set_ylim(0, 105)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=8.5, loc='center right')
ax1.set_title('Local reaction rate and cumulative conversion — kinetics sensitivity')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('kinsens_fig5_rate_and_conversion_vs_z.png');  plt.show()

# ── Figure 6: Energy budget — local generation/removal terms (1x3 small multiples) ──
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=False)
for axA, c, lab in zip(axes, cases, CASE_LABELS):
    res_c = c['res']
    axA.plot(res_c['z'], c['S_rxn']/1e3,   color='tab:red',    lw=1.8, label='reaction')
    axA.plot(res_c['z'], c['S_ads']/1e3,   color='tab:orange', lw=1.8, label='adsorption')
    axA.plot(res_c['z'], c['S_wall']/1e3,  color='tab:blue',   lw=1.8, label='wall removal')
    axA.plot(res_c['z'], c['S_total']/1e3, color='k', lw=1.8, ls='--', label='total')
    axA.axhline(0, color='k', lw=0.8, ls=':')
    axA.set_xlabel('z [m]');  axA.set_title(lab, fontsize=10)

    axA2 = axA.twinx()
    axA2.plot(res_c['z'], res_c['T']-273.15, color='tab:green', lw=1.6, ls='-.', label='T(z)')
    axA2.set_ylabel('T [C]', color='tab:green')
    axA2.tick_params(axis='y', labelcolor='tab:green')

    linesA, labelsA = axA.get_legend_handles_labels()
    linesA2, labelsA2 = axA2.get_legend_handles_labels()
    axA.legend(linesA+linesA2, labelsA+labelsA2, fontsize=7.5)
    axA.grid(True, alpha=0.3)
axes[0].set_ylabel('local volumetric rate [kW/m³]')
fig.suptitle(f'Energy budget: local generation/removal terms — kinetics sensitivity  |  $T_{{in}}$={T_C_case} C', fontsize=10)
plt.tight_layout()
_savefig('kinsens_fig6_energy_budget.png');  plt.show()

# ── Figure 7: Energy budget — cumulative sensible heat, gas vs solid (overlay) ──
fig, axB = plt.subplots(figsize=(9, 5.5))
for c, col, lab in zip(cases, CASE_COLORS, CASE_LABELS):
    z = c['res']['z']
    axB.plot(z, c['C_gas'],   color=col, lw=2.0, ls='-',  label=f'gas, {lab}')
    axB.plot(z, c['C_solid'], color=col, lw=2.0, ls='--', label=f'solid, {lab}')
axB.axhline(0, color='k', lw=0.8, ls=':')
axB.set_xlabel('z [m]');  axB.set_ylabel('cumulative sensible heat picked up [W]')
axB.set_title('Energy budget: cumulative sensible heat, gas (solid) vs solid (dashed) — kinetics sensitivity', fontsize=10)
axB.legend(fontsize=8.5, ncol=2);  axB.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('kinsens_fig7_sensible_heat_cumulative.png');  plt.show()

print(f"\n{'='*70}")
print("  Energy closure @ z=L (per case)")
print(f"{'='*70}")
for c, lab in zip(cases, CASE_LABELS):
    print(f"  [{lab}]  Q_gen(rxn+ads)={c['Q_prod']:.3f} W   Q_wall_removed={c['Q_wall_W']:.3f} W   "
          f"Q_sensible_out(gas+solid)={c['Q_sensS_W']:.3f} W   mismatch={c['mismatch_bb']:+.2f}%")

# ── Figure 8: Whole-bed heat balance — Q produced vs Q removed, grouped by case ──
print(f"\n{'='*70}")
print("  Reactor heat balance -- Q produced vs. Q removed [W]  (per case)")
print(f"{'='*70}")
for c, lab in zip(cases, CASE_LABELS):
    print(f"  [{lab}]")
    print(f"    Q produced:  reaction = {c['Q_rxn_W']:8.3f} W   adsorption = {c['Q_ads_W']:8.3f} W"
          + ("   (net desorption -> sink)" if c['Q_ads_W'] < 0 else "")
          + f"   total = {c['Q_prod']:8.3f} W")
    print(f"    Q removed:   wall = {c['Q_wall_W']:8.3f} W   sensible(gas+solid) = {c['Q_sensS_W']:8.3f} W"
          f"   total = {c['Q_out']:8.3f} W")
    print(f"    mismatch = {c['mismatch_bb']:+.2f}%   "
          f"(gas sensible = {c['Q_gasS_W']:8.3f} W, solid sensible = {c['Q_solS_W']:8.3f} W)")

fig, ax = plt.subplots(figsize=(9, 6.5))
n_cases = len(cases)
group_w = 0.7
bar_w   = group_w / 2
group_centers = np.arange(n_cases) * 1.6

_prod_key = [('reaction heat',   'Q_rxn_W', 'tab:red'),
             ('adsorption heat', 'Q_ads_W', 'tab:orange')]
_out_key  = [('wall cooling',              'Q_wall_W',  'tab:blue'),
             ('sensible heat (gas+solid)', 'Q_sensS_W', 'tab:brown')]

for i, (c, lab) in enumerate(zip(cases, CASE_LABELS)):
    x_prod = group_centers[i] - bar_w/2
    x_out  = group_centers[i] + bar_w/2
    bottom = 0.0
    for seg_lab, key, col in _prod_key:
        val = c[key]
        ax.bar(x_prod, val, bottom=bottom, width=bar_w*0.9, color=col, edgecolor='k', linewidth=0.7)
        if abs(val) > 0.01 * max(abs(c['Q_prod']), abs(c['Q_out']), 1e-9):
            ax.text(x_prod, bottom + val/2, f'{val:.2f}', ha='center', va='center', fontsize=7)
        bottom += val
    bottom = 0.0
    for seg_lab, key, col in _out_key:
        val = c[key]
        ax.bar(x_out, val, bottom=bottom, width=bar_w*0.9, color=col, edgecolor='k', linewidth=0.7)
        if abs(val) > 0.01 * max(abs(c['Q_prod']), abs(c['Q_out']), 1e-9):
            ax.text(x_out, bottom + val/2, f'{val:.2f}', ha='center', va='center', fontsize=7)
        bottom += val

ax.set_xticks(group_centers, [f'{lab}\n(prod | removed)' for lab in CASE_LABELS])
ax.set_ylabel('heat rate [W]')
ax.axhline(0, color='k', lw=0.8)
handles = [plt.Rectangle((0,0),1,1, color=col) for _,_,col in _prod_key+_out_key]
labels  = [seg_lab for seg_lab,_,_ in _prod_key+_out_key]
ax.legend(handles, labels, fontsize=8.5, loc='upper right')
ax.set_title(f'Whole-bed heat balance vs kinetics multiplier  |  $T_{{in}}$={T_C_case} C', fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
_savefig('kinsens_fig8_heat_balance_bars.png');  plt.show()

# ── Figure 9: K_LDF vs z (overlay) ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for c, col, lab in zip(cases, CASE_COLORS, CASE_LABELS):
    res_c = c['res']
    ax.plot(res_c['z'], c['Kl_f'], color=col, lw=2.2, label=f'$K_{{LDF}}(z)$, {lab}')
    ax.plot(res_c['z'][c['i_pmax']], c['Kl_f'][c['i_pmax']], 'o', color=col, ms=6)
ax.set_xlabel('z [m]');  ax.set_ylabel('$K_{LDF}$ [1/s]')
ax.set_yscale('log')
ax.set_title('Solid-side LDF mass-transfer coefficient vs bed position — kinetics sensitivity')
ax.legend(fontsize=9);  ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
_savefig('kinsens_fig9_K_LDF_vs_z.png');  plt.show()
# endregion


# region 7. CONSOLE OUTPUT — SPECIES / WATER BALANCE + CHARACTERISTIC QUANTITIES
# =============================================================================
for c, lab in zip(cases, CASE_LABELS):
    res_c = c['res']
    F_CO2_out  = float(res_c['F_CO2'][-1]);   F_H2_out   = float(res_c['F_H2'][-1])
    F_CH4_out  = float(res_c['F_CH4'][-1]);   F_H2Og_out = float(res_c['F_H2O'][-1])
    F_H2Oa_out = u_s_case * rho_bed_ads * float(res_c['q'][0])
    X_CO2_out  = float(res_c['X_CO2'][-1])

    print(f"\n{'='*70}")
    print(f"  Species / water balance table  [{lab}]  [mol/(m2.s)]  (outlet in brackets: mmol/(m2.s))")
    print(f"{'='*70}")
    print(f"  {'species':<16}{'in':>10}{'out':>10}   {'out [mmol]':>12}")
    for name, fin, fout in [('F_CO2',       F_in_CO2, F_CO2_out),
                             ('F_H2',        F_in_H2,  F_H2_out),
                             ('F_CH4',       F_in_CH4, F_CH4_out),
                             ('F_H2O (gas)', 0.0,      F_H2Og_out),
                             ('F_H2O (ads)', 0.0,      F_H2Oa_out)]:
        print(f"  {name:<16}{fin:10.5f}{fout:10.5f}   {fout*1e3:12.4f}")
    print(f"  {'X_CO2 (outlet)':<16}{'':>10}{'':>10}   {X_CO2_out*100:11.2f}%")

    print(f"\n  Characteristic quantities [{lab}]")
    print(f"  gas residence time      tau_gas   = {c['tau_gas']:6.2f} s")
    print(f"  solid residence time     tau_solid = {c['tau_solid']:6.2f} s")
    print(f"  1/K_LDF @ peak p_H2O     = {c['inv_Kldf_peak']:6.2f} s")
    print(f"  characteristic reaction time  tau_rxn = {c['tau_rxn']:6.2f} s   "
          f"(r_in={c['r_in']*1e3:.4f} mmol/(kg_cat.s) at fresh dry feed, T_in)")

# ── Summary table across the sweep ───────────────────────────────────────────
print(f"\n{'='*70}")
print("  Kinetics sensitivity summary")
print(f"{'='*70}")
print(f"  {'KIN_MULT':>10}{'X_CO2 out [%]':>16}{'T_max [C]':>12}{'r_max [mmol/kg.s]':>20}"
      f"{'tau_rxn [s]':>14}{'z(95% X) [m]':>15}")
for c, mult in zip(cases, KIN_MULT_LIST):
    z95_str = f"{c['z_95']:.3f}" if c['z_95'] is not None else "not reached"
    print(f"  {mult:>10.2f}{c['res']['X_CO2'][-1]*100:>16.2f}"
          f"{np.max(c['res']['T'])-273.15:>12.1f}{c['r_max']*1e3:>20.4f}{c['tau_rxn']:>14.2f}{z95_str:>15}")
print(f"\n  Max local reaction rate observed across the sweep: "
      f"r_max = {r_max_overall*1e3:.4f} mmol/(kg_cat.s)  (at KIN_MULT = {mult_at_rmax:.2f}x)")
# endregion
