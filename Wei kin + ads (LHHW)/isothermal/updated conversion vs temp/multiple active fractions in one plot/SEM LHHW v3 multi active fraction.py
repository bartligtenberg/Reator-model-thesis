"""
Coupled SEM Column Model  —  1D Isothermal Transient — multiple active fractions
=================================================================================

Same physics as "SEM LHHW v3 with active fraction.py" (one directory up), but
instead of fixing a single active_fraction, this script sweeps a LIST of
active fractions and overlays the resulting CO2-conversion-vs-temperature
curves (SE and non-SE) on one combined plot, together with Wei's measured
data points.

Reaction
--------
    CO2  +  4 H2  →  CH4  +  2 H2O       (Sabatier)

active_fraction = fraction of the total bed mass (m_cat_total) treated as
catalytically active. The sorbent mass for adsorption is always the FULL bed
mass (m_cat_total), regardless of active_fraction — only the reaction terms
scale with it. Because active_fraction changes rho_bed_cat, BOTH the SE
(sorption-enhanced) curve and the non-SE (reaction-only) curve change with it,
so both are computed and plotted for every active fraction in ACTIVE_FRACTIONS.

This file is self-contained and does NOT modify pfr_simulation.py,
adsorption_simulation.py, or the original "SEM LHHW v3 with active
fraction.py" — all are left completely unchanged. It reads
wpd_datasets.csv from the parent folder (read-only).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# region 1. PARAMETERS
# =============================================================================
# 1. PARAMETERS
# =============================================================================

# --------------- Bed geometry  (identical to adsorption_simulation.py) -------
d_b     = 0.010             # bed diameter                        [m]
L_b     = 0.100             # bed length                          [m]
A_b     = np.pi / 4 * d_b**2   # cross-sectional area            [m²]
V_bed   = A_b * L_b             # total bed volume                [m³]
m_cat_total = 6.5e-3        # total bed material mass             [kg]  (Wei Fig. 5.3, 6.5 g — GHSV basis)
rho_bed_ads = m_cat_total / V_bed   # sorbent bulk density   [kg_sorb/m³_bed] (adsorption terms — always 100% of mass)
eps_b   = 0.40              # void fraction between particles      [-]

# --------------- Active fractions to sweep and overlay -----------------------
ACTIVE_FRACTIONS = [0.05, 0.20, 1.00]   # fraction of m_cat_total that is catalytically active [-]

# --------------- Particle properties  (identical to adsorption_simulation.py)
d_p   = 0.75e-3             # particle diameter                   [m]
eps_p = 0.242                 # intraparticle void fraction          [-]
tau_p = 4.0                 # tortuosity factor                    [-]

# --------------- Adsorbed-phase density (liquid water) -----------------------
rho_ads = 791             # liquid water density                 [kg/m³]

# --------------- DA isotherm parameters: Mette (2014) ------------------------
W0_DA = 190.00e-6           # micropore volume                     [m³/kg_sorbent]
E_DA  = 1192.25e3           # characteristic adsorption energy     [J/kg]
n_DA  = 1.55                # DA heterogeneity parameter           [-]

# --------------- Operating conditions ----------------------------------------
T_LIST = [180, 210, 240, 270, 300, 330, 360]   # temperatures [°C]  (Wei Fig. 5.3 range)
P_bar = 1.0                 # total pressure                       [bar]
P_Pa  = P_bar * 1e5         # total pressure                       [Pa]

# --------------- Feed composition: Wei (2022) experimental inlet -------------
y_CO2_in = 0.025            # CO2 mole fraction                    [-]
y_H2_in  = 0.10             # H2 mole fraction  (H2/CO2 = 4 → stoichiometric)  [-]
y_CH4_in = 0.815            # CH4 mole fraction (large background, inert here)  [-]
y_N2_in  = 0.06             # N2 mole fraction  (inert tracer, not tracked)     [-]

# --------------- Gas flow ---------------------------------------------------
Q_STP = 100e-6 / 60         # volumetric flow at STP               [m³/s]
T_STP = 273.15              # STP temperature                      [K]
u_STP = Q_STP / A_b         # superficial velocity at STP          [m/s]

# --------------- Physical constants ------------------------------------------
R_gas  = 8.314              # universal gas constant               [J/(mol·K)]
MW_H2O = 0.018015           # molar mass of water                  [kg/mol]

# --------------- Kinetic parameters  (Koschany et al. 2016, Table 6) ----------
T_ref_K = 555.0             # reference temperature                [K]  (282 °C)
k_ref   = 3.46e-4           # rate constant at T_ref              [mol/(g_cat·s·bar)]
Ea_k    = 77.5e3            # activation energy                   [J/mol]
A_OH    = 0.50              # K_OH pre-exponential                [bar^-0.5]
dH_OH   = 22.4e3            # K_OH van 't Hoff parameter          [J/mol]
A_H2    = 0.44              # K_H2 pre-exponential                [bar^-0.5]
dH_H2   = -6.2e3            # K_H2 van 't Hoff parameter         [J/mol]
A_mix   = 0.88              # K_mix pre-exponential               [bar^-0.5]
dH_mix  = -10.0e3           # K_mix van 't Hoff parameter        [J/mol]

# Floor partial pressures for BDF numerical Jacobian stability.
P_FLOOR = 1e-4              # [bar]

# Total-molar-flux closure safeguards (node velocity is derived, not solved for
# directly — clip the implied total flux to keep it positive and bounded so a
# transient BDF/Newton trial iterate can't produce backflow or a blow-up)
U_FLOOR_FRAC = 1e-3         # floor on F_tot_rightface, as a fraction of F_total_in [-]
U_CEIL_FRAC  = 10.0         # defensive ceiling, same units                        [-]

# --------------- Spatial discretisation --------------------------------------
N  = 50                    # number of axial nodes
dz = L_b / (N - 1)          # node spacing                         [m]


# endregion

# region 2. THERMODYNAMIC AND KINETIC FUNCTIONS
# =============================================================================
# 2. THERMODYNAMIC AND KINETIC FUNCTIONS  (unchanged from the single-fraction file)
# =============================================================================

def P_sat_bar(T_K):
    """Saturation vapour pressure of water [bar]  —  Antoine equation."""
    log10_p = (29.8605 - 3.1522e3/T_K - 7.3037*np.log10(T_K)
               + 2.4247e-9*T_K + 1.8090e-6*T_K**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5


def q_star_vec(T_K, p_arr, W0, E, n):
    """Equilibrium H2O loading [mol/kg]  —  Dubinin-Astakhov (DA) isotherm."""
    p    = np.asarray(p_arr, dtype=float)
    Psat = P_sat_bar(T_K)

    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_raw  = (R_gas / MW_H2O) * T_K * np.log(Psat / p_safe)   # [J/kg]

    A  = np.where((p <= 0.0) | (p >= Psat), 0.0, A_raw)

    W  = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    qs = rho_ads / MW_H2O * W

    return np.where(p <= 0.0, 0.0, qs)


def K_LDF_vec(T_K, p_arr, W0, E, n):
    """LDF mass-transfer coefficient [1/s]."""
    D_M = 3.36e-9 * T_K**1.75
    p      = np.asarray(p_arr, dtype=float)
    dp_bar = 1.0 / 1e5

    p_lo = np.maximum(p - dp_bar, 1e-15)
    p_hi = p + dp_bar

    dqstar_dp = (q_star_vec(T_K, p_hi, W0, E, n)
                 - q_star_vec(T_K, p_lo, W0, E, n)) / 2.0
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)

    return  (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_ads * R_gas * T_K * dqstar_dp))


def K_eq_sabatier(T_K):
    """Equilibrium constant for CO2 + 4H2 → CH4 + 2H2O  [dimensionless, p in bar]."""
    return 137.0 * T_K**(-3.994) * np.exp(158700.0 / (R_gas * T_K))


def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    """LHHW CO2 methanation rate  [mol / (kg_cat · s)]."""
    vH  = lambda dH: np.exp(-dH / R_gas * (1.0 / T_K - 1.0 / T_ref_K))
    k     = k_ref * np.exp(-Ea_k / R_gas * (1.0 / T_K - 1.0 / T_ref_K))
    K_OH  = A_OH  * vH(dH_OH)
    K_H2  = A_H2  * vH(dH_H2)
    K_mix = A_mix * vH(dH_mix)
    K_eq  = K_eq_sabatier(T_K)

    p_CO2_s = np.maximum(p_CO2, P_FLOOR)
    p_H2_s  = np.maximum(p_H2,  P_FLOOR)

    beta = (p_CH4 * p_H2O**2) / (K_eq * p_CO2_s * p_H2_s**4)
    f_eq = np.maximum(1.0 - beta, 0.0)

    DEN = (1.0
           + K_OH  * np.maximum(p_H2O, 0.0) / p_H2_s**0.5
           + K_H2  * p_H2_s**0.5
           + K_mix * p_CO2_s**0.5)

    r_g_s = k * (p_CO2_s * p_H2_s)**0.5 * f_eq / DEN**2  # mol/(g_cat·s)
    return r_g_s * 1000.0                               # → mol/(kg_cat·s)


# endregion

# region 3. TOTAL-MOLAR-FLUX CLOSURE  (replaces the old constant superficial velocity)
# =============================================================================
# 3. TOTAL-MOLAR-FLUX CLOSURE
# =============================================================================
# CO2 + 4H2 -> CH4 + 2H2O is mole-reducing (5 -> 3 mol), and H2O is further
# removed from the gas phase by adsorption, so total gas moles shrink as the
# front moves down the column. Since the column is isothermal and isobaric,
# ideal-gas law pins the total gas concentration C_total = P/(RT) constant in
# z and t — so local velocity must drop to compensate, via a quasi-steady
# total-molar-flux closure. A constant-u model silently violates this
# conservation law at high conversion/adsorption; this restores it as an
# (approximately, up to solver tolerance) exact invariant of the ODE system.

def _compute_fields(T_K, C_CO2, C_H2, C_CH4, C_H2O, q, F_total_in, C_total,
                     se_on, rho_bed_cat, rho_bed_ads):
    """Local reaction/adsorption rates, plus the node-local gas velocity
    implied by them (rectangle-rule cumsum to exactly match the first-order-
    upwind finite-volume scheme used in rhs_sem)."""
    p_CO2 = C_CO2 * R_gas * T_K / 1e5
    p_H2  = C_H2  * R_gas * T_K / 1e5
    p_CH4 = C_CH4 * R_gas * T_K / 1e5
    p_H2O = C_H2O * R_gas * T_K / 1e5

    r    = reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O)  # mol/(kg_cat·s)
    qs   = q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)         # equilibrium loading [mol/kg]
    Kl   = K_LDF_vec( T_K, p_H2O, W0_DA, E_DA, n_DA)         # LDF rate constant   [1/s]
    dqdt = Kl * (qs - q) if se_on else np.zeros(N)           # adsorption rate     [mol/(kg·s)]

    S_tot = rho_bed_cat * (-2.0) * r - rho_bed_ads * dqdt    # net gas-mole source [mol/(m3_bed·s)]

    F_tot_rightface = F_total_in + np.cumsum(S_tot) * dz
    F_tot_rightface = np.clip(F_tot_rightface,
                               U_FLOOR_FRAC * F_total_in,
                               U_CEIL_FRAC  * F_total_in)

    u_node    = F_tot_rightface / C_total
    u_node_up = np.concatenate([[F_total_in / C_total], u_node[:-1]])

    return r, dqdt, S_tot, u_node, u_node_up


# endregion


# region 4. RIGHT-HAND SIDE OF THE ODE SYSTEM
# =============================================================================
# 4. RIGHT-HAND SIDE OF THE ODE SYSTEM
# =============================================================================
# rho_bed_cat and rho_bed_ads are passed in explicitly (rather than read from
# module-level globals) because they change with active_fraction. Node
# velocities are no longer constant — see _compute_fields.

def rhs_sem(t, y, se_on, T_K, F_total_in, C_total, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O,
            rho_bed_cat, rho_bed_ads):
    """Time derivatives for the coupled SEM column (5N equations)."""

    C_CO2 = np.maximum(y[0*N : 1*N], 0.0)
    C_H2  = np.maximum(y[1*N : 2*N], 0.0)
    C_CH4 = np.maximum(y[2*N : 3*N], 0.0)
    C_H2O = np.maximum(y[3*N : 4*N], 0.0)
    q     = np.maximum(y[4*N : 5*N], 0.0)

    r, dqdt, S_tot, u_node, u_node_up = _compute_fields(
        T_K, C_CO2, C_H2, C_CH4, C_H2O, q, F_total_in, C_total,
        se_on, rho_bed_cat, rho_bed_ads)

    C_CO2_up = np.concatenate([[C_in_CO2], C_CO2[:-1]])
    C_H2_up  = np.concatenate([[C_in_H2],  C_H2[:-1]])
    C_CH4_up = np.concatenate([[C_in_CH4], C_CH4[:-1]])
    C_H2O_up = np.concatenate([[C_in_H2O], C_H2O[:-1]])

    # u is now node-local (u_node/u_node_up), not a single constant — see _compute_fields.
    dCdt_CO2 = (-(u_node * C_CO2 - u_node_up * C_CO2_up) / dz  +  rho_bed_cat * (-1) * r) / eps_b
    dCdt_H2  = (-(u_node * C_H2  - u_node_up * C_H2_up ) / dz  +  rho_bed_cat * (-4) * r) / eps_b
    dCdt_CH4 = (-(u_node * C_CH4 - u_node_up * C_CH4_up) / dz  +  rho_bed_cat * (+1) * r) / eps_b
    dCdt_H2O = (-(u_node * C_H2O - u_node_up * C_H2O_up) / dz  +  rho_bed_cat * (+2) * r
                                                                  -  rho_bed_ads * dqdt) / eps_b

    return np.concatenate([dCdt_CO2, dCdt_H2, dCdt_CH4, dCdt_H2O, dqdt])


# endregion

# region 5. SOLVE
# =============================================================================
# 5. SOLVE  —  outer loop over active_fraction, inner loop over temperature
# =============================================================================

# p_H2O_max, and hence q_at_max/t_sat_est/t_end, depend only on the feed and
# P_bar (NOT on active_fraction, since the sorbent mass is always
# m_cat_total). Precompute per-temperature timing/feed data once.
p_H2O_max = 2 * y_CO2_in * P_bar / (1 - 2 * y_CO2_in)

T_data = {}   # T_C -> {'T_K','u','C_total','F_total_in','C_in_CO2','C_in_H2','C_in_CH4','C_in_H2O','t_end','q_at_max','t_sat_est'}
for T_C in T_LIST:
    T_K        = T_C + 273.15
    u          = u_STP * (T_K / T_STP)          # inlet superficial velocity (boundary condition) [m/s]
    C_total    = P_Pa / (R_gas * T_K)           # total molar concentration at T, P [mol/m³] — fixed (isothermal, isobaric)
    F_total_in = u * C_total                    # inlet total molar flux [mol/(m²·s)] — boundary condition
    C_in_CO2   = y_CO2_in * C_total
    C_in_H2    = y_H2_in  * C_total
    C_in_CH4   = y_CH4_in * C_total
    C_in_H2O   = 0.0

    q_at_max  = float(q_star_vec(T_K, np.array([p_H2O_max]), W0_DA, E_DA, n_DA)[0])
    F_CO2_in  = C_in_CO2 * u * A_b
    t_sat_est = q_at_max * m_cat_total / (2.0 * F_CO2_in)
    t_end     = min(2.5 * t_sat_est, 7200.0)

    T_data[T_C] = dict(T_K=T_K, u=u, C_total=C_total, F_total_in=F_total_in,
                        C_in_CO2=C_in_CO2, C_in_H2=C_in_H2,
                        C_in_CH4=C_in_CH4, C_in_H2O=C_in_H2O,
                        t_end=t_end, q_at_max=q_at_max, t_sat_est=t_sat_est)

all_results = {}   # (active_fraction, T_C) -> {'results': {True/False: sol}}

for af in ACTIVE_FRACTIONS:
    m_cat       = m_cat_total * af
    rho_bed_cat = m_cat / V_bed

    print("=" * 60)
    print(f"  Active fraction = {af:.0%}  (m_cat = {m_cat*1000:.2f} g of {m_cat_total*1000:.1f} g total)")

    for T_C in T_LIST:
        d = T_data[T_C]
        T_K, C_total, F_total_in = d['T_K'], d['C_total'], d['F_total_in']
        C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O = d['C_in_CO2'], d['C_in_H2'], d['C_in_CH4'], d['C_in_H2O']
        t_end = d['t_end']

        y0 = np.zeros(5 * N)
        y0[0*N : 1*N] = C_in_CO2
        y0[1*N : 2*N] = C_in_H2
        y0[2*N : 3*N] = C_in_CH4

        results = {}
        for se_on in [True, False]:
            tag = "SE on " if se_on else "SE off"
            print(f"    T = {T_C:>3d} °C, {tag} ...", end="", flush=True)
            sol = solve_ivp(
                rhs_sem,
                t_span=[0.0, t_end],
                y0=y0,
                args=(se_on, T_K, F_total_in, C_total, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O,
                      rho_bed_cat, rho_bed_ads),
                method='BDF',
                rtol=1e-4,
                atol=1e-8,
                dense_output=True,
            )
            print(f"  {'OK' if sol.success else 'FAILED — ' + sol.message}")
            if not sol.success:
                raise RuntimeError(f"ODE solver did not converge (af={af:.0%}, T={T_C}°C, {tag}): {sol.message}")
            results[se_on] = sol

        all_results[(af, T_C)] = {'results': results}

print("=" * 60)


# endregion

# region 6. POST-PROCESSING
# =============================================================================
# 6. POST-PROCESSING
# =============================================================================

def extract_outlet(sol, T_K_loc, C_in_CO2_loc, F_total_in_loc, C_total_loc,
                    se_on_loc, rho_bed_cat_loc, rho_bed_ads_loc):
    """Extract outlet CO2 conversion and H2O partial pressure time series.

    Conversion is flux-based (F_CO2_out = u_out * C_CO2_out), not
    concentration-based, since the outlet velocity now differs from the
    inlet velocity once total gas moles change along the bed (see
    _compute_fields) — a concentration ratio alone would silently assume
    u_out = u_in, which is exactly the assumption removed here.
    """
    t_arr     = sol.t
    y_arr     = sol.sol(t_arr)
    C_CO2_out = np.maximum(y_arr[1*N - 1, :], 0.0)
    C_H2O_out = np.maximum(y_arr[4*N - 1, :], 0.0)

    u_out = np.empty(len(t_arr))
    for i in range(len(t_arr)):
        C_CO2_i = np.maximum(y_arr[0*N : 1*N, i], 0.0)
        C_H2_i  = np.maximum(y_arr[1*N : 2*N, i], 0.0)
        C_CH4_i = np.maximum(y_arr[2*N : 3*N, i], 0.0)
        C_H2O_i = np.maximum(y_arr[3*N : 4*N, i], 0.0)
        q_i     = np.maximum(y_arr[4*N : 5*N, i], 0.0)
        _, _, _, u_node_i, _ = _compute_fields(
            T_K_loc, C_CO2_i, C_H2_i, C_CH4_i, C_H2O_i, q_i,
            F_total_in_loc, C_total_loc, se_on_loc, rho_bed_cat_loc, rho_bed_ads_loc)
        u_out[i] = u_node_i[-1]

    u_in      = F_total_in_loc / C_total_loc
    F_CO2_in  = C_in_CO2_loc * u_in
    F_CO2_out = u_out * C_CO2_out
    X_CO2     = np.clip(1.0 - F_CO2_out / F_CO2_in, 0.0, 1.0)
    p_H2O_mbar = C_H2O_out * R_gas * T_K_loc / 1e5 * 1000
    return t_arr, X_CO2, p_H2O_mbar


def equilibrium_conversion(T_K_val):
    """Equilibrium CO2 conversion for the Wei feed (2.5% CO2, 10% H2, 81.5% CH4) at 1 bar."""
    K = K_eq_sabatier(T_K_val)
    def f(X):
        return ((0.815 + 0.025*X) * 0.0025 * X**2 * (1 - 0.05*X)**2
                / (2.5e-6 * (1 - X)**5) - K)
    try:
        return brentq(f, 1e-9, 1 - 1e-9) * 100
    except Exception:
        return 100.0


T_arr = np.array(T_LIST, dtype=float)

# results_by_af[af] = {'X_off_ss': [...], 'X_on_ini': [...]}  (one value per T_LIST entry)
results_by_af = {}
for af in ACTIVE_FRACTIONS:
    rho_bed_cat_af = (m_cat_total * af) / V_bed
    X_off_ss = []
    X_on_ini = []
    for T_C in T_LIST:
        d = T_data[T_C]
        T_K_row, C_in_CO2_row, t_sat_row = d['T_K'], d['C_in_CO2'], d['t_sat_est']
        F_total_in_row, C_total_row = d['F_total_in'], d['C_total']
        results = all_results[(af, T_C)]['results']

        t_off, X_off, _ = extract_outlet(results[False], T_K_row, C_in_CO2_row,
                                          F_total_in_row, C_total_row,
                                          False, rho_bed_cat_af, rho_bed_ads)
        t_on,  X_on,  _ = extract_outlet(results[True],  T_K_row, C_in_CO2_row,
                                          F_total_in_row, C_total_row,
                                          True, rho_bed_cat_af, rho_bed_ads)

        # Non-SE: reaches kinetic/thermodynamic steady state within seconds.
        # Take mean of second half of the time series to capture the plateau.
        mid = max(1, len(X_off) // 2)
        X_off_ss.append(float(np.mean(X_off[mid:])) * 100)

        # SE: take mean of the 10%-40% window of t_sat_est (fresh-sorbent regime).
        mask = (t_on >= 0.10 * t_sat_row) & (t_on <= 0.40 * t_sat_row)
        if mask.sum() == 0:
            mask = np.ones(len(t_on), dtype=bool)
        X_on_ini.append(float(np.mean(X_on[mask])) * 100)

    results_by_af[af] = {'X_off_ss': X_off_ss, 'X_on_ini': X_on_ini}

# Equilibrium curve (independent of active_fraction)
T_fine = np.linspace(170, 370, 120)
X_eq   = [equilibrium_conversion(T + 273.15) for T in T_fine]


# endregion

# region 7. PLOT — CO2 conversion vs temperature, all active fractions overlaid
# =============================================================================
# 7. PLOT — CO2 conversion vs temperature (Wei Fig. 5.3 style), multi active fraction
# =============================================================================

# --- Digitized experimental data (Wei Fig. 5.3, via WebPlotDigitizer) ---------
# Read from the parent folder (read-only) so the data is not duplicated.
wpd_path = os.path.join(os.path.dirname(__file__), '..', 'wpd_datasets.csv')
wpd = pd.read_csv(wpd_path, header=1)
wpd.columns = ['X_nonSE', 'Y_nonSE', 'X_SE', 'Y_SE']

fig, ax = plt.subplots(figsize=(10, 7.5))

ax.plot(T_fine, X_eq, 'k--', lw=1.5, label='Equilibrium', zorder=1)

ax.scatter(wpd['X_nonSE'], wpd['Y_nonSE'], marker='o', s=50,
           facecolors='none', edgecolors='red', linewidths=1.5,
           label='Non-SE (Wei, measured)', zorder=3)
ax.scatter(wpd['X_SE'], wpd['Y_SE'], marker='^', s=50,
           facecolors='none', edgecolors='red', linewidths=1.5,
           label='SE (Wei, measured)', zorder=3)

colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(ACTIVE_FRACTIONS)))
for i, af in enumerate(ACTIVE_FRACTIONS):
    color = colors[i]
    data  = results_by_af[af]
    ax.plot(T_arr, data['X_on_ini'], color=color, marker='^', ls='-', lw=2.0, ms=8,
             label=f'SE, active fraction={af:.0%}', zorder=2)
    ax.plot(T_arr, data['X_off_ss'], color=color, marker='o', ls='--', lw=1.5, ms=7,
             label=f'Non-SE, active fraction={af:.0%}', zorder=2)

ax.set_xlabel('Temperature [°C]', fontsize=12)
ax.set_ylabel('CO₂ conversion [%]', fontsize=12)
ax.set_title('SEM column — CO₂ conversion vs temperature, multiple active fractions')
ax.set_xlim(170, 370)
ax.set_ylim(0, 105)
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'Figure_1_multi_active_fraction.png'),
            dpi=300, bbox_inches='tight')
plt.show()
# endregion
