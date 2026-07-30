"""
Moving Packed Bed (MPB) Reactor Model — W0_DA (DA limiting micropore volume) Sensitivity Analysis
==========================================================================================
Same single operating point as the Sec. 7.1 reference-case deep dive
("MPB_flux_form_7.1_reference_case.py"), except u_s is doubled here to keep
Lambda (solid sorption capacity flux / stoichiometric H2O generation flux,
see the "lambda threshold determination" folder) above 2.5 for every case in
the W0 sweep below, including the 0.5x (lowest-capacity) case:
    u_s   = 11.68 mm/s   (2x the 5.84 mm/s reference-case value)
    GHSV  = 1.5 m3_STP/(kg_ads.h)
    T_in  = T_wall = 280 C
    dilution = script default (y_CH4_in = 0.80)

Instead of a single W0_DA (the Dubinin-Astakhov limiting micropore volume),
the isotherm's W0 is scaled by a uniform multiplier and solved three times:
    W0_MULTIPLIERS = [1.0, 0.5, 2.0]   (baseline / half capacity / double capacity)

Physics core (parameters, isotherm/kinetics functions, decoupled Gauss-Seidel
solver, non-SE fixed-bed reference) copied unmodified from
"MPB_flux_form_7.1_reference_case.py", except:
  - q_star() and K_LDF() now evaluate the DA isotherm at W0_DA * W0_MULTIPLIER
    (module-level global) instead of the fixed W0_DA. Both functions depend on
    it -- K_LDF via the numerical isotherm slope dqs/dp used in its LDF
    coefficient -- so setting W0_MULTIPLIER once before a solve_mpb call scales
    the whole case consistently (solve_mpb itself is unmodified: every
    internal rate/adsorption evaluation already calls q_star()/K_LDF()).
  - _q_physics_init's warm-start q(z) guess is now recomputed per case (it
    calls q_star, which depends on the active W0_MULTIPLIER), unlike the
    K_LDF sensitivity script where it could be shared across cases.
  - region 5 solves the reference case once per W0_MULTIPLIERS entry
    (instead of once), storing every per-case diagnostic needed for the
    figures immediately after that case's solve (while W0_MULTIPLIER is
    still set correctly for it) -- including the requested peak p_H2O and
    solid outlet loading.
  - region 6 remakes all 8 of the reference-case figures, each now overlaying
    the three W0_DA cases (one colour per multiplier) instead of a single
    curve, so the sensitivity is visible directly on each plot.
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

# --- Inlet molar fluxes [mol/(m²·s)] — same derivation as the reference-case script ---
T_STP   = 273.15                  # [K]
GHSV    = 1.5                     # [m³_STP/(kg_ads·h)]   Sec. 7.1 reference case
Q_STP   = GHSV * M_ads / 3600.0   # [m³_STP/s]
u_g_STP = Q_STP / A_b             # [m/s]
F_total_in = u_g_STP * P_Pa / (R_gas * T_STP)  # [mol/(m²·s)]
F_in_CO2   = y_CO2_in * F_total_in
F_in_H2    = y_H2_in  * F_total_in
F_in_CH4   = y_CH4_in * F_total_in

# --- Reference-case operating point (held fixed across the W0_DA sweep) ---
u_s_case = 0.01168   # [m/s]  solid (sorbent/catalyst) velocity — 2x the 5.84 mm/s reference-case value, to keep Lambda > 2.5 across the W0 sweep (incl. the 0.5x case)
T_C_case = 280       # [C]    inlet temperature

# --- W0_DA sensitivity sweep ---
W0_MULTIPLIERS = [1.0, 0.5, 2.0]   # [-]  uniform multipliers applied to W0_DA
W0_COLORS = {1.0: 'tab:purple', 0.5: 'tab:blue', 2.0: 'tab:red'}
W0_LABELS = {1.0: '1x $W_0$ (baseline)', 0.5: '0.5x $W_0$', 2.0: '2x $W_0$'}
W0_MULTIPLIER = 1.0   # [-]  active multiplier; q_star()/K_LDF() read this module-level global on every call, set per-case in region 5

print(f"MPB flux form — W0_DA sensitivity analysis: d={d_b*100:.0f} cm, L={L_b:.1f} m, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3")
print(f"  GHSV = {GHSV:.1f} m3_STP/(kg_ads.h)  ->  u_g_STP = {u_g_STP*1e3:.3f} mm/s, "
      f"F_in_total = {F_total_in:.4f} mol/(m2.s)")
print(f"  u_s = {u_s_case*1e3:.1f} mm/s,  T_in = {T_C_case} C")
print(f"  M_cat(active) = {M_cat_active*1000:.1f} g,  M_ads = {M_ads*1000:.1f} g,  U_a = {U_a:.0f} W/(m3.K)")
print(f"  W0_DA = {W0_DA*1e6:.1f} cm3/kg baseline,  multipliers swept: {W0_MULTIPLIERS}")
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
    """DA equilibrium loading, evaluated at W0_DA * W0_MULTIPLIER (module-level
    global, set per-case in region 5 below). solve_mpb calls this function
    directly, so it always sees whichever multiplier is currently active --
    no changes needed inside solve_mpb itself."""
    return q_star_vec(T_K, p_H2O, W0_DA * W0_MULTIPLIER, E_DA, n_DA)

def K_LDF(T_K, p_H2O):
    """LDF mass-transfer coefficient, evaluated at the BASELINE W0_DA (not
    W0_MULTIPLIER-scaled). K_LDF's isotherm slope dqs/dp would otherwise
    scale as 1/W0_MULTIPLIER (since q_star is linear in W0), so leaving the
    multiplier in here would silently change the mass-transfer kinetics
    every time the capacity is swept -- confounding "less capacity" with
    "faster equilibration". Freezing the slope at baseline isolates the
    capacity effect alone, which is both the physically intended sweep and
    removes the tighter gas/solid coupling that made the low-W0 case hard to
    converge."""
    return K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

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
    Unchanged from the Sec. 7.1 reference-case script. Every rate/adsorption
    evaluation below calls q_star()/K_LDF(), which apply the module-level
    W0_MULTIPLIER -- so scaling W0_DA for a whole solve is done purely by
    setting that global before calling this function (see region 5).
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
    """Fixed-bed (u_s=0) reference with wall cooling: no sorption enhancement
    (noSE). No adsorption term at all, so it does NOT depend on W0_DA /
    W0_MULTIPLIER — computed once and reused across all three W0_DA cases."""
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
    """Isothermal, no-adsorption gas pass -> physics-motivated initial q(z)
    guess for warm-starting solve_mpb. Calls q_star, which now depends on the
    active W0_MULTIPLIER (unlike the K_LDF sensitivity script), so this is
    recomputed per case in region 5 below rather than shared."""
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
    return dict(F_H2O_prod=F_H2O_prod, F_H2O_out=F_H2O_out, F_H2O_ads=F_H2O_ads, bal_err=bal_err,
                line=(f"    H2O balance [mmol/(m²·s)]:  produced={F_H2O_prod*1e3:.3f}  "
                      f"gas_out={F_H2O_out*1e3:.3f}  solid_out={F_H2O_ads*1e3:.3f}  "
                      f"err={bal_err:+.1f}%"))
# endregion


# region 5. SOLVE EACH W0_DA CASE
# =============================================================================
T_K_case    = T_C_case + 273.15
T_wall_case = T_K_case   # adiabatic wall (T_wall = T_in), same convention as the reference-case script

print(f"\n{'='*70}")
print(f"  W0_DA sensitivity: u_s = {u_s_case*1e3:.2f} mm/s, GHSV = {GHSV:.1f}, "
      f"T_in = T_wall = {T_C_case} C")
print(f"{'='*70}")

noSE = _compute_noSE(T_K_case, T_wall_case)          # independent of W0_DA -- computed once
X_eq = equilibrium_conversion(T_K_case)               # thermodynamic, independent of W0_DA
print(f"  Non-SE fixed-bed reference conversion: {noSE['X_CO2_noSE']*100:.2f}%")
print(f"  Thermodynamic equilibrium conversion @ T_in: {X_eq:.2f}%")

cases = {}   # keyed by W0_DA multiplier -> dict of every per-case diagnostic needed downstream

for mult in W0_MULTIPLIERS:
    W0_MULTIPLIER = mult   # module-level assignment (top-level script scope): q_star()/K_LDF() read this on every call inside solve_mpb

    print(f"\n{'-'*70}")
    print(f"  Solving W0_DA case: {W0_LABELS[mult]}  (multiplier = {mult:g}x, "
          f"W0 = {W0_DA*mult*1e6:.1f} cm3/kg)")
    print(f"{'-'*70}")

    _phys       = _q_physics_init(T_K_case)   # depends on active W0_MULTIPLIER -> recomputed per case
    q_init_case = _phys['q'][::-1]            # flip: solid enters unloaded at z=L, builds up toward z=0

    t0  = time.perf_counter()
    res = solve_mpb(u_s_case, T_K_case, T_wall=T_wall_case, q_init=q_init_case)
    dt = time.perf_counter() - t0

    if res is None:
        raise RuntimeError(f"solve_mpb failed to produce a solution for W0_DA multiplier {mult:g}x.")
    if not res['converged']:
        raise RuntimeError(f"solve_mpb did not converge for W0_DA multiplier {mult:g}x "
                            f"(err={res['conv_err']:.2e} after {res['n_iter']} iterations).")

    regime = "gas-dominated" if res['gas_dominates'] else "solid-dominated"
    print(f"  Solved in {dt:.1f}s  [{regime}, {'converged' if res['converged'] else 'NOT CONVERGED'}, "
          f"{res['n_iter']} outer iterations, err={res['conv_err']:.2e}]")
    print(f"  X_CO2(outlet) = {res['X_CO2'][-1]*100:.2f}%   T_max = {np.max(res['T'])-273.15:.1f} C")

    z = res['z']

    # ── per-case energy-budget terms (Fig.6/7/8 + closure check), computed
    #    NOW while W0_MULTIPLIER is still correctly set for this case ──
    p_H2O_z = res['C_H2O']*R_gas*res['T']/1e5   # [bar]
    q_eq_z  = q_star(res['T'], p_H2O_z)
    qs_f    = q_star(res['T'], p_H2O_z)
    Kl_f    = K_LDF(res['T'], p_H2O_z)
    ads_f   = Kl_f*(qs_f - res['q'])            # [mol/(kg_ads·s)]

    S_rxn   = (-dH_r)   * rho_bed_cat * res['r']
    S_ads   = (-dH_ads) * rho_bed_ads * ads_f
    S_wall  = -U_a * (res['T'] - T_wall_case)
    S_total = S_rxn + S_ads + S_wall

    dTdz_num      = np.gradient(res['T'], z)
    gas_cap_z     = _gas_cap(res['F_CO2'], res['F_H2'], res['F_CH4'], res['F_H2O'])
    solid_cap_case = u_s_case * rho_bed_tot * Cp_cat
    S_gas_sens    = gas_cap_z * dTdz_num
    S_solid_sens  = -solid_cap_case * dTdz_num

    C_rxn       = cumulative_trapezoid(S_rxn,  z, initial=0) * A_b
    C_ads       = cumulative_trapezoid(S_ads,  z, initial=0) * A_b
    C_wall      = cumulative_trapezoid(S_wall, z, initial=0) * A_b
    C_gas       = cumulative_trapezoid(S_gas_sens,   z, initial=0) * A_b
    C_solid_fwd = cumulative_trapezoid(S_solid_sens, z, initial=0) * A_b
    C_solid     = C_solid_fwd[-1] - C_solid_fwd
    C_sens      = C_gas + C_solid_fwd

    Q_gen      = C_rxn[-1] + C_ads[-1]
    Q_removed  = -C_wall[-1]
    Q_sens_out = C_sens[-1]
    mismatch   = (Q_gen - Q_removed - Q_sens_out) / max(abs(Q_gen), 1e-30) * 100

    Q_rxn_W   = float(C_rxn[-1]);   Q_ads_W  = float(C_ads[-1])
    Q_wall_W  = float(-C_wall[-1])
    Q_gasS_W  = float(C_gas[-1]);   Q_solS_W = float(C_solid[0])
    Q_sensS_W = Q_gasS_W + Q_solS_W
    Q_prod    = Q_rxn_W + Q_ads_W
    Q_out     = Q_wall_W + Q_sensS_W
    mismatch_bb = (Q_prod - Q_out) / max(abs(Q_prod), 1e-30) * 100

    h2o_bal = _h2o_balance_line(u_s_case, res)

    # ── characteristic quantities ──
    u_g_z   = (res['F_CO2']+res['F_H2']+res['F_CH4']+res['F_H2O']) * R_gas * res['T'] / P_Pa
    tau_gas = float(np.trapz(1.0/u_g_z, z))
    tau_solid = L_b / u_s_case
    i_pmax = int(np.argmax(p_H2O_z))
    Kldf_peak = float(K_LDF(res['T'][i_pmax], np.array([p_H2O_z[i_pmax]]))[0])
    inv_Kldf_peak = 1.0/Kldf_peak
    u_g_in   = F_total_in * R_gas * T_K_case / P_Pa
    C_CO2_in = F_in_CO2 / u_g_in
    r_in = float(reaction_rate_SI(T_K_case, np.array([y_CO2_in*P_bar]), np.array([y_H2_in*P_bar]),
                                   np.array([y_CH4_in*P_bar]), np.array([0.0]))[0])
    tau_rxn = C_CO2_in / (rho_bed_cat * r_in)

    # ── requested: peak p_H2O along the bed, and the solid outlet loading ──
    p_H2O_max = float(p_H2O_z[i_pmax])     # [bar], peak gas-phase H2O partial pressure
    q_outlet  = float(res['q'][0])         # [mol/kg_ads], solid loading at its outlet (z=0, where the solid leaves)

    print(h2o_bal['line'])
    print(f"    Energy closure @ z=L:  Q_gen={Q_gen:.3f} W  Q_wall={Q_removed:.3f} W  "
          f"Q_sens={Q_sens_out:.3f} W  mismatch={mismatch:+.2f}%")
    print(f"    tau_gas={tau_gas:.2f}s  tau_solid={tau_solid:.2f}s  "
          f"1/K_LDF_peak={inv_Kldf_peak:.2f}s  tau_rxn={tau_rxn:.2f}s")
    print(f"    p_H2O max = {p_H2O_max*1e3:.2f} mbar (at z={z[i_pmax]:.3f} m)   "
          f"q_outlet (solid, z=0) = {q_outlet:.3f} mol/kg_ads")

    cases[mult] = dict(
        res=res, z=z, p_H2O_z=p_H2O_z, q_eq_z=q_eq_z, ads_f=ads_f,
        S_rxn=S_rxn, S_ads=S_ads, S_wall=S_wall, S_total=S_total,
        S_gas_sens=S_gas_sens, S_solid_sens=S_solid_sens,
        C_gas=C_gas, C_solid=C_solid,
        Q_rxn_W=Q_rxn_W, Q_ads_W=Q_ads_W, Q_wall_W=Q_wall_W, Q_sensS_W=Q_sensS_W,
        Q_prod=Q_prod, Q_out=Q_out, mismatch_bb=mismatch_bb,
        h2o_bal=h2o_bal, tau_gas=tau_gas, tau_solid=tau_solid,
        Kldf_peak=Kldf_peak, inv_Kldf_peak=inv_Kldf_peak, tau_rxn=tau_rxn,
        p_H2O_max=p_H2O_max, q_outlet=q_outlet,
    )

W0_MULTIPLIER = 1.0   # restore baseline after the sweep (harmless -- nothing below calls solve_mpb again)
print(f"\nAll {len(W0_MULTIPLIERS)} W0_DA cases solved.")
# endregion


# region 6. FIGURES
# =============================================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def _savefig(filename):
    stem, ext = os.path.splitext(filename)
    filename  = f'{stem}_W0_DA_sensitivity{ext}'
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=150, bbox_inches='tight')

# ── Figure 1: Conversion vs z ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for mult in W0_MULTIPLIERS:
    c = cases[mult]
    ax.plot(c['z'], c['res']['X_CO2']*100, color=W0_COLORS[mult], lw=2.5, label=W0_LABELS[mult])
ax.plot(noSE['profile']['z'], noSE['profile']['X_CO2']*100, color='k', lw=2, ls='--',
        label='Fixed bed, no SE ($u_s$=0)')
ax.axhline(X_eq, color='tab:red', lw=1.5, ls=':', label=f'Thermodynamic equilibrium ({T_C_case} C)')
ax.set_xlabel('z [m]');  ax.set_ylabel('CO2 conversion [%]')
ax.set_title(f'CO2 conversion vs bed position — $W_0$ sensitivity  |  $T_{{in}}$={T_C_case} C')
ax.legend(fontsize=10);  ax.grid(True, alpha=0.3);  ax.set_ylim(0, 105)
plt.tight_layout()
_savefig('sec7_1_fig1_conversion_vs_z.png');  plt.show()

# ── Figure 2: Gas-phase composition vs z (one panel per W0_DA case) ─────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
for ax1, mult in zip(axes, W0_MULTIPLIERS):
    r = cases[mult]['res']
    C_tot = r['C_CO2'] + r['C_H2'] + r['C_CH4'] + r['C_H2O']
    y_CO2 = r['C_CO2']/C_tot;  y_H2 = r['C_H2']/C_tot
    y_CH4 = r['C_CH4']/C_tot;  y_H2O = r['C_H2O']/C_tot

    ax1.plot(cases[mult]['z'], y_CH4*100, color='tab:green', lw=2.2, label='CH4')
    ax1.set_xlabel('z [m]')
    ax1.set_ylabel('y_CH4 [mol%]', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:green')

    ax2 = ax1.twinx()
    ax2.plot(cases[mult]['z'], y_CO2*100, color='tab:blue',   lw=2, label='CO2')
    ax2.plot(cases[mult]['z'], y_H2*100,  color='tab:orange', lw=2, label='H2')
    ax2.plot(cases[mult]['z'], y_H2O*100, color='tab:cyan',   lw=2, label='H2O')
    ax2.set_ylabel('minor species [mol%]')

    ax1.set_title(W0_LABELS[mult], fontsize=10)
    ax1.grid(True, alpha=0.3)

    if mult == W0_MULTIPLIERS[-1]:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper center',
                   bbox_to_anchor=(0.5, 0.02), ncol=4)
fig.suptitle('Gas-phase composition vs z — $W_0$ sensitivity', fontsize=11)
plt.tight_layout(rect=[0, 0.06, 1, 1])
_savefig('sec7_1_fig2_gas_composition_vs_z.png');  plt.show()

# ── Figure 3: Solid loading vs equilibrium loading ───────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for mult in W0_MULTIPLIERS:
    c = cases[mult]
    ax.plot(c['z'], c['res']['q'], color=W0_COLORS[mult], lw=2.5,
            label=f"{W0_LABELS[mult]} — q(z)")
    ax.plot(c['z'], c['q_eq_z'],   color=W0_COLORS[mult], lw=1.5, ls='--', alpha=0.6,
            label=f"{W0_LABELS[mult]} — q*(z)")
ax.set_xlabel('z [m]');  ax.set_ylabel('H2O loading [mol/kg_ads]')
ax.set_title('Solid H2O loading vs equilibrium — $W_0$ sensitivity')
ax.legend(fontsize=8.5);  ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('sec7_1_fig3_loading_vs_equilibrium.png');  plt.show()

# ── Figure 4: Temperature profile ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
for mult in W0_MULTIPLIERS:
    c = cases[mult]
    ax.plot(c['z'], c['res']['T']-273.15, color=W0_COLORS[mult], lw=2.5, label=W0_LABELS[mult])
    i_max = int(np.argmax(c['res']['T']))
    ax.plot(c['z'][i_max], c['res']['T'][i_max]-273.15, 'o', color=W0_COLORS[mult], ms=6)
ax.axhline(T_C_case, color='grey', lw=1.5, ls='--', label=f'$T_{{in}}$ = $T_{{wall}}$ = {T_C_case} C')
ax.set_xlabel('z [m]');  ax.set_ylabel('T [C]')
ax.set_title('Temperature profile — $W_0$ sensitivity')
ax.legend(fontsize=10);  ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('sec7_1_fig4_temperature_vs_z.png');  plt.show()

# ── Figure 5: Local rate and cumulative conversion (twin-axis) ──────────────
fig, ax1 = plt.subplots(figsize=(9, 5.5))
for mult in W0_MULTIPLIERS:
    c = cases[mult]
    ax1.plot(c['z'], c['res']['r']*1e3, color=W0_COLORS[mult], lw=1.5, alpha=0.6,
             label=f"{W0_LABELS[mult]} — r(z)")
ax1.set_xlabel('z [m]');  ax1.set_ylabel('r [mmol/(kg_cat·s)]')

ax2 = ax1.twinx()
for mult in W0_MULTIPLIERS:
    c = cases[mult]
    ax2.plot(c['z'], c['res']['X_CO2']*100, color=W0_COLORS[mult], lw=2.5,
             label=f"{W0_LABELS[mult]} — $X_{{CO2}}$(z)")
ax2.plot(noSE['profile']['z'], noSE['profile']['X_CO2']*100, color='k', lw=2, ls='--',
         label='Fixed bed, no SE ($u_s$=0)')
ax2.axhline(X_eq, color='tab:red', lw=1.5, ls=':', label=f'Thermodynamic equilibrium ({T_C_case} C)')
ax2.set_ylabel('CO2 conversion [%]')
ax2.set_ylim(0, 105)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=8.5, loc='center right')
ax1.set_title('Local reaction rate and cumulative conversion — $W_0$ sensitivity')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('sec7_1_fig5_rate_and_conversion_vs_z.png');  plt.show()

# ── Figure 6: Energy budget — local generation/removal terms (one panel per case) ──
fig, axesA = plt.subplots(1, 3, figsize=(17, 5.5), sharey=False)
for axA, mult in zip(axesA, W0_MULTIPLIERS):
    c = cases[mult];  z = c['z']
    axA.plot(z, c['S_rxn']/1e3,   color='tab:red',    lw=2, label='reaction heat')
    axA.plot(z, c['S_ads']/1e3,   color='tab:orange', lw=2, label='adsorption heat')
    axA.plot(z, c['S_wall']/1e3,  color='tab:blue',   lw=2, label='wall removal')
    axA.plot(z, c['S_total']/1e3, color='k', lw=2, ls='--', label='total')
    axA.axhline(0, color='k', lw=0.8, ls=':')
    axA.set_xlabel('z [m]')
    axA.set_ylabel('local volumetric rate [kW/m³]')

    T_z = c['res']['T'] - 273.15
    axA2 = axA.twinx()
    axA2.plot(z, T_z, color='tab:green', lw=2, ls='-.', label='T(z)')
    axA2.set_ylabel('T [C]', color='tab:green')
    axA2.tick_params(axis='y', labelcolor='tab:green')

    axA.set_title(W0_LABELS[mult], fontsize=10)
    axA.grid(True, alpha=0.3)
    if mult == W0_MULTIPLIERS[-1]:
        linesA, labelsA = axA.get_legend_handles_labels()
        linesA2, labelsA2 = axA2.get_legend_handles_labels()
        fig.legend(linesA+linesA2, labelsA+labelsA2, fontsize=9, loc='upper center',
                   bbox_to_anchor=(0.5, 0.02), ncol=5)
fig.suptitle(f'Energy budget: local generation/removal terms — $W_0$ sensitivity  |  $T_{{in}}$={T_C_case} C',
             fontsize=11)
plt.tight_layout(rect=[0, 0.07, 1, 1])
_savefig('sec7_1_fig6_energy_budget.png');  plt.show()

# ── Figure 7: Energy budget — cumulative sensible heat carried by gas vs. solid ──
fig, axB = plt.subplots(figsize=(9.5, 5.5))
for mult in W0_MULTIPLIERS:
    c = cases[mult]
    axB.plot(c['z'], c['C_gas'],   color=W0_COLORS[mult], lw=2.2,
             label=f"{W0_LABELS[mult]} — gas")
    axB.plot(c['z'], c['C_solid'], color=W0_COLORS[mult], lw=2.2, ls='--',
             label=f"{W0_LABELS[mult]} — solid")
axB.axhline(0, color='k', lw=0.8, ls=':')
axB.set_xlabel('z [m]');  axB.set_ylabel('cumulative sensible heat picked up [W]')
axB.set_title('Cumulative sensible heat carried by gas vs. solid — $W_0$ sensitivity', fontsize=10)
axB.legend(fontsize=8.5);  axB.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('sec7_1_fig7_sensible_heat_cumulative.png');  plt.show()

# ── Figure 8: whole-bed heat balance — Q produced vs. Q removed, grouped by case ──
fig, ax = plt.subplots(figsize=(10, 6.5))
n_cases = len(W0_MULTIPLIERS)
group_x = np.arange(n_cases)
bar_w   = 0.32

for i, mult in enumerate(W0_MULTIPLIERS):
    c = cases[mult]
    _prod_segs = [('reaction',    c['Q_rxn_W'],  'tab:red'),
                  ('adsorption',  c['Q_ads_W'],  'tab:orange')]
    _out_segs  = [('wall',        c['Q_wall_W'], 'tab:blue'),
                  ('sensible',    c['Q_sensS_W'],'tab:brown')]
    for x, segs in [(group_x[i]-bar_w/2, _prod_segs), (group_x[i]+bar_w/2, _out_segs)]:
        bottom = 0.0
        for lab, val, col in segs:
            ax.bar(x, val, bottom=bottom, width=bar_w, color=col, edgecolor='k', linewidth=0.7)
            if abs(val) > 0.02 * max(abs(c['Q_prod']), abs(c['Q_out']), 1e-9):
                ax.text(x, bottom + val/2, f'{val:.2f} W', ha='center', va='center', fontsize=7.5)
            bottom += val
    ax.text(group_x[i]-bar_w/2, -0.06*max(c['Q_prod'], c['Q_out']), 'produced',
            ha='center', va='top', fontsize=7.5, rotation=0)
    ax.text(group_x[i]+bar_w/2, -0.06*max(c['Q_prod'], c['Q_out']), 'removed',
            ha='center', va='top', fontsize=7.5, rotation=0)

from matplotlib.patches import Patch
legend_handles = [Patch(facecolor='tab:red', edgecolor='k', label='reaction heat'),
                  Patch(facecolor='tab:orange', edgecolor='k', label='adsorption heat'),
                  Patch(facecolor='tab:blue', edgecolor='k', label='wall cooling'),
                  Patch(facecolor='tab:brown', edgecolor='k', label='sensible heat (gas+solid)')]
ax.legend(handles=legend_handles, fontsize=9, loc='upper right')

ax.set_xticks(group_x, [f"{W0_LABELS[m]}\n(mismatch {cases[m]['mismatch_bb']:+.1f}%)" for m in W0_MULTIPLIERS],
              fontsize=8.5)
ax.set_ylabel('heat rate [W]')
ax.axhline(0, color='k', lw=0.8)
ax.set_title(f'Whole-bed heat balance — $W_0$ sensitivity  |  $T_{{in}}$={T_C_case} C', fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
_savefig('sec7_1_fig8_heat_balance_bars.png');  plt.show()
# endregion


# region 7. CONSOLE OUTPUT — COMPARISON TABLE
# =============================================================================
print(f"\n{'='*100}")
print("  W0_DA sensitivity — comparison table")
print(f"{'='*100}")
header = (f"  {'case':<20}{'X_out[%]':>10}{'T_max[C]':>10}{'p_H2O_max[mbar]':>17}{'q_outlet':>10}"
          f"{'H2O err[%]':>12}{'E closure[%]':>14}{'tau_gas[s]':>12}{'1/K_LDF_pk[s]':>15}")
print(header)
for mult in W0_MULTIPLIERS:
    c   = cases[mult];  r = c['res']
    X_out = float(r['X_CO2'][-1])*100
    T_max = float(np.max(r['T']))-273.15
    print(f"  {W0_LABELS[mult]:<20}{X_out:10.2f}{T_max:10.1f}{c['p_H2O_max']*1e3:17.2f}{c['q_outlet']:10.3f}"
          f"{c['h2o_bal']['bal_err']:12.1f}{c['mismatch_bb']:14.2f}{c['tau_gas']:12.2f}{c['inv_Kldf_peak']:15.3f}")
print(f"\n  Non-SE fixed-bed reference conversion: {noSE['X_CO2_noSE']*100:.2f}%")
print(f"  Thermodynamic equilibrium conversion @ T_in: {X_eq:.2f}%")
# endregion
