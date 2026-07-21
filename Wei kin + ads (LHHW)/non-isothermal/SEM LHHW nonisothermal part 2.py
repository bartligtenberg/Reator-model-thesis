"""
Coupled SEM Column Model  —  1D Non-Isothermal Transient
=========================================================

Non-isothermal extension of SEM LHHW.py.  Temperature T(z,t) is added as a
6th block of N state variables and evolves via a 1D energy balance.

The exothermic Sabatier reaction (ΔH_r ≈ −165 kJ/mol) heats the bed above the
inlet setpoint.  This hot spot raises the local K_eq denominator, shifts the
approach-to-equilibrium factor β towards 1, and reduces conversion — most
visible at low inlet temperatures (180–240 °C) where the kinetics are fast
relative to the contact time.

State vector  (6 × N values — one value per axial node)
---------------------------------------------------------
    y[0 : N]     C_CO2  [mol/m³]   gas-phase CO2 concentration
    y[N : 2N]    C_H2   [mol/m³]   gas-phase H2 concentration
    y[2N : 3N]   C_CH4  [mol/m³]   gas-phase CH4 concentration
    y[3N : 4N]   C_H2O  [mol/m³]   gas-phase H2O concentration
    y[4N : 5N]   q      [mol/kg]   solid-phase H2O loading
    y[5N : 6N]   T      [K]        local bed temperature

Energy balance (per unit bed volume):
    (ρ_b·Cp_cat + ε_b·ρ_g·Cp_g) · dT/dt =
        − u · ρ_g · Cp_g · (T − T_up) / Δz   ← convective heat transport
        + (−ΔH_r) · ρ_b · r                   ← reaction heat release
        + (−ΔH_ads) · ρ_b · dq/dt             ← adsorption heat release (SE on only)
        − U_a · (T − T_wall)                   ← wall heat exchange

Simplification: superficial velocity u is computed at the inlet temperature
and held constant along the bed (same as the isothermal version).  Local gas
expansion from heating is neglected — a standard first-model approximation.

The isothermal file (SEM LHHW.py) is unchanged.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# region 1. PARAMETERS
# =============================================================================
# 1. PARAMETERS
# =============================================================================

# --------------- Bed geometry  (identical to SEM LHHW.py) --------------------
d_b     = 0.010             # bed diameter                        [m]
L_b     = 0.100             # bed length                          [m]
A_b     = np.pi / 4 * d_b**2   # cross-sectional area            [m²]
V_bed   = A_b * L_b             # total bed volume                [m³]
m_cat_total     = 6.5e-3    # total bed material mass (catalyst = sorbent, same particles) [kg]  (Wei Fig. 5.3)
active_fraction = 0.20       # fraction of m_cat_total treated as catalytically active (reaction only) [-]
m_cat       = m_cat_total * active_fraction   # active catalyst mass   [kg]  (reaction rate basis)
rho_bed_cat = m_cat       / V_bed             # catalyst bulk density  [kg_cat / m³_bed]  (reaction terms)
rho_bed_ads = m_cat_total / V_bed             # sorbent bulk density   [kg_sorb/m³_bed] (adsorption + heat-capacity terms — always 100% of mass)
eps_b   = 0.40              # void fraction between particles      [-]

# --------------- Particle properties -----------------------------------------
d_p   = 0.75e-3             # particle diameter                   [m]
eps_p = 0.242                 # intraparticle void fraction          [-]
tau_p = 4.0                 # tortuosity factor                    [-]

# --------------- Adsorbed-phase density (liquid water) -----------------------
rho_ads = 998.2             # liquid water density                 [kg/m³]

# --------------- DA isotherm parameters: Mette (2014) ------------------------
W0_DA = 190.00e-6           # micropore volume                     [m³/kg_sorbent]
E_DA  = 1192.25e3           # characteristic adsorption energy     [J/kg]
n_DA  = 1.55                # DA heterogeneity parameter           [-]

# --------------- Operating conditions ----------------------------------------
T_LIST = [180, 210, 240, 270, 300, 330, 360]   # inlet temperatures [°C]
P_bar = 1.0                 # total pressure                       [bar]
P_Pa  = P_bar * 1e5         # total pressure                       [Pa]

# --------------- Feed composition (Wei 2022, Fig. 5.3 experiment) -------------
y_CO2_in = 0.025
y_H2_in  = 0.10
y_CH4_in = 0.815
y_N2_in  = 0.06

# --------------- Gas flow (Wei Fig. 5.3: 100 mL/min, 6.5 g → GHSV = 923) ----
Q_STP = 100e-6 / 60         # volumetric flow at STP               [m³/s]
T_STP = 273.15              # STP temperature                      [K]
u_STP = Q_STP / A_b         # superficial velocity at STP          [m/s]

# --------------- Physical constants ------------------------------------------
R_gas  = 8.314
MW_H2O = 0.018015

# --------------- LHHW kinetic parameters (Koschany et al. 2016, Table 6) -----
T_ref_K = 555.0
k_ref   = 3.46e-4
Ea_k    = 77.5e3
A_OH    = 0.50;  dH_OH  =  22.4e3
A_H2    = 0.44;  dH_H2  =  -6.2e3
A_mix   = 0.88;  dH_mix = -10.0e3
P_FLOOR = 1e-4

# --------------- Thermal parameters ------------------------------------------
# Heat of Sabatier reaction: CO2 + 4H2 → CH4 + 2H2O(g), ΔH ≈ −165 kJ/mol_CO2
dH_r   = -165.0e3           # [J / mol_CO2]

# Isosteric heat of H2O adsorption on 13X zeolite — Bareschino et al. (2023) Table 3
dH_ads = -45.0e3            # [J / mol_H2O]

# Heat capacity of the solid catalyst/sorbent (zeolite-dominated) — Bareschino et al. (2023) Table 3
Cp_cat = 1100.0              # [J / (kg · K)]

# Molar heat capacities of gas species at ~300 °C  [J / (mol · K)]
Cp_CO2 = 37.1
Cp_H2  = 29.3
Cp_CH4 = 38.7
Cp_H2O = 33.6
Cp_N2  = 29.1

# Wall heat-transfer coefficient × bed interfacial area per bed volume [W/(m³·K)]
# U_a = 0  →  adiabatic (maximum hot-spot effect, lower bound on conversion)
# Large U_a → nearly isothermal (converges to SEM LHHW.py result)
# For a 10-mm tube in a furnace: h_w ~ 100–500 W/(m²·K), 4/d_b ~ 400 m²/m³
# → U_a up to ~200 000.  Start with 0 to see the full hot-spot.
U_a = 200000                  # [W / (m³ · K)]

# --------------- Spatial discretisation --------------------------------------
N  = 50
dz = L_b / (N - 1)
z_cm = np.linspace(0, L_b, N) * 100   # [cm] for plots


# endregion

# region 2. THERMODYNAMIC AND KINETIC FUNCTIONS
# =============================================================================
# 2. THERMODYNAMIC AND KINETIC FUNCTIONS
#    Identical to SEM LHHW.py — all functions vectorise naturally when T_K is
#    passed as an array of length N instead of a scalar.
# =============================================================================

def P_sat_bar(T_K):
    """
    Saturation vapour pressure of water [bar]. log10(P/mmHg) = D + E/T + F*log10(T) + G*T + H*T^2, with
    D=29.8605, E=-3.1522e3, F=-7.3037, G=2.4247e-9, H=1.8090e-6 — Eq. S.17 in Bareschino et al. (2023)
    supplementary material, credited there to Kowalska & Ambrozek (2017).
    """
    log10_p = (29.8605 - 3.1522e3/T_K - 7.3037*np.log10(T_K)
               + 2.4247e-9*T_K + 1.8090e-6*T_K**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5   # [mmHg] -> [bar]


def q_star_vec(T_K, p_arr, W0, E, n):
    """Equilibrium H2O loading [mol/kg] — Dubinin-Astakhov isotherm."""
    p    = np.asarray(p_arr, dtype=float)
    Psat = P_sat_bar(T_K)
    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_raw  = (R_gas / MW_H2O) * T_K * np.log(Psat / p_safe)
    A  = np.where((p <= 0.0) | (p >= Psat), 0.0, A_raw)
    W  = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    qs = rho_ads / MW_H2O * W
    return np.where(p <= 0.0, 0.0, qs)


def K_LDF_vec(T_K, p_arr, W0, E, n):
    """LDF mass-transfer coefficient [1/s]."""
    D_M    = 3.36e-9 * T_K**1.75   # Chapman-Enskog power-law, Bareschino et al. (2023) SI
    p      = np.asarray(p_arr, dtype=float)
    dp_bar = 1.0 / 1e5
    p_lo = np.maximum(p - dp_bar, 1e-15)
    p_hi = p + dp_bar
    dqstar_dp = (q_star_vec(T_K, p_hi, W0, E, n)
                 - q_star_vec(T_K, p_lo, W0, E, n)) / 2.0
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)
    return (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_ads * R_gas * T_K * dqstar_dp))


def K_eq_sabatier(T_K):
    """
    Equilibrium constant for CO2 + 4H2 → CH4 + 2H2O [dimensionless, p in bar].
    Koschany et al. (2016):  K_eq = 137 · T^(−3.994) · exp(158.7 kJ/mol / RT)
    """
    return 137.0 * T_K**(-3.994) * np.exp(158700.0 / (R_gas * T_K))


def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    """
    LHHW CO2 methanation rate [mol / (kg_cat · s)].
    Koschany et al. (2016).  T_K may be a scalar or an array of length N.
    """
    vH    = lambda dH: np.exp(-dH / R_gas * (1.0 / T_K - 1.0 / T_ref_K))
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

    r_g_s = k * (p_CO2_s * p_H2_s)**0.5 * f_eq / DEN**2
    return r_g_s * 1000.0 # mol/(g·s) → mol/(kg·s) with effectiveness factor


# endregion

# region 4. RIGHT-HAND SIDE (NON-ISOTHERMAL)
# =============================================================================
# 4. RIGHT-HAND SIDE OF THE ODE SYSTEM  (non-isothermal, 6 × N equations)
# =============================================================================

def rhs_sem_noniso(t, y, se_on, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O,
                   T_in, T_wall):
    """
    Time derivatives for the non-isothermal SEM column (6N equations).

    Parameters
    ----------
    se_on    : bool   — True = SE mode (adsorption active)
    u        : float  — superficial gas velocity at inlet T [m/s]  (constant along bed)
    C_in_*   : float  — inlet concentrations [mol/m³] at inlet T
    T_in     : float  — inlet / feed temperature [K]  (boundary condition for T)
    T_wall   : float  — wall / furnace temperature [K]  (used for U_a term)

    Mass balances: same first-order upwind as SEM LHHW.py, but partial pressures
    are computed with the LOCAL temperature T(z,t) from the state vector.

    Energy balance (per unit bed volume):
        Cp_eff · dT/dt = −u·ρ_g·Cp_g·(T−T_up)/Δz  +  Q_rxn  +  Q_ads  +  Q_wall
    """
    # --- Unpack and clip state vector ---
    C_CO2 = np.maximum(y[0*N : 1*N], 0.0) # so this is CO2 concentraion  at all 50 nodes in z direction
    C_H2  = np.maximum(y[1*N : 2*N], 0.0)
    C_CH4 = np.maximum(y[2*N : 3*N], 0.0)
    C_H2O = np.maximum(y[3*N : 4*N], 0.0)
    q     = np.maximum(y[4*N : 5*N], 0.0)
    T     = np.maximum(y[5*N : 6*N], 200.0)   # floor at 200 K prevents divide-by-zero

    # --- Partial pressures using LOCAL temperature and ideal gas law at each node ---
    p_CO2 = C_CO2 * R_gas * T / 1e5
    p_H2  = C_H2  * R_gas * T / 1e5
    p_CH4 = C_CH4 * R_gas * T / 1e5
    p_H2O = C_H2O * R_gas * T / 1e5

    # --- Rates with LOCAL T (all rate functions accept T as an array) ---
    r    = reaction_rate_SI(T, p_CO2, p_H2, p_CH4, p_H2O)
    qs   = q_star_vec(T, p_H2O, W0_DA, E_DA, n_DA)
    Kl   = K_LDF_vec( T, p_H2O, W0_DA, E_DA, n_DA)
    dqdt = Kl * (qs - q) if se_on else np.zeros(N)

    # --- Upwind mass balances (identical structure to SEM LHHW.py) ---
    C_CO2_up = np.concatenate([[C_in_CO2], C_CO2[:-1]])
    C_H2_up  = np.concatenate([[C_in_H2],  C_H2[:-1]])
    C_CH4_up = np.concatenate([[C_in_CH4], C_CH4[:-1]])
    C_H2O_up = np.concatenate([[C_in_H2O], C_H2O[:-1]])

    dCdt_CO2 = (-u * (C_CO2 - C_CO2_up) / dz  +  rho_bed_cat * (-1) * r) / eps_b
    dCdt_H2  = (-u * (C_H2  - C_H2_up)  / dz  +  rho_bed_cat * (-4) * r) / eps_b
    dCdt_CH4 = (-u * (C_CH4 - C_CH4_up) / dz  +  rho_bed_cat * (+1) * r) / eps_b
    dCdt_H2O = (-u * (C_H2O - C_H2O_up) / dz  +  rho_bed_cat * (+2) * r
                                                 -  rho_bed_ads * dqdt) / eps_b

    # --- Energy balance --- How fast is temperatrue changing at each node?
    
    # Local gas mole fractions (p_i / P_bar = y_i at total pressure P_bar)
    y_CO2l = p_CO2 / P_bar
    y_H2l  = p_H2  / P_bar
    y_CH4l = p_CH4 / P_bar
    y_H2Ol = p_H2O / P_bar
    y_N2l  = np.maximum(1.0 - y_CO2l - y_H2l - y_CH4l - y_H2Ol, 0.0)

    # Local mixture molar heat capacity [J/(mol·K)]
    Cp_mix = (y_CO2l*Cp_CO2 + y_H2l*Cp_H2 + y_CH4l*Cp_CH4
              + y_H2Ol*Cp_H2O + y_N2l*Cp_N2)

    # Local gas molar density [mol/m³]  (ideal gas, varies with T)
    rho_g_mol = P_Pa / (R_gas * T)

    # Effective volumetric heat capacity [J/(m³·K)]  (rho_bed_ads = total physical solid density — cat and
    # sorbent are the same particles, so this is the correct basis regardless of active_fraction)
    Cp_eff = rho_bed_ads * Cp_cat + eps_b * rho_g_mol * Cp_mix

    # Upwind temperature (T_in as inlet BC)
    T_up = np.concatenate([[T_in], T[:-1]])

    # Heat source terms [W/m³]
    Q_rxn  = (-dH_r)   * rho_bed_cat * r           # exothermic reaction  (+)
    Q_ads  = (-dH_ads) * rho_bed_ads * dqdt         # exothermic adsorption (+, zero if SE off)
    Q_wall = -U_a * (T - T_wall)                # wall cooling (negative when T > T_wall)

    dTdt = (-u * rho_g_mol * Cp_mix * (T - T_up) / dz
            + Q_rxn + Q_ads + Q_wall) / Cp_eff

    return np.concatenate([dCdt_CO2, dCdt_H2, dCdt_CH4, dCdt_H2O, dqdt, dTdt])


# endregion

# region 5. SOLVE
# =============================================================================
# 5. SOLVE
# =============================================================================

all_results = {}

p_H2O_max = 2 * y_CO2_in * P_bar / (1 - 2 * y_CO2_in)

# Absolute tolerances: tight for concentrations and loading, relaxed for T
atol_vec = np.concatenate([
    1e-8 * np.ones(4 * N),   # C blocks  [mol/m³]
    1e-8 * np.ones(N),        # q block   [mol/kg]
    1e-2 * np.ones(N),        # T block   [K]  — 0.01 K is more than sufficient
])

for T_C in T_LIST:
    T_K      = T_C + 273.15
    u        = u_STP * (T_K / T_STP)
    C_in_CO2 = y_CO2_in * P_Pa / (R_gas * T_K)
    C_in_H2  = y_H2_in  * P_Pa / (R_gas * T_K)
    C_in_CH4 = y_CH4_in * P_Pa / (R_gas * T_K)
    C_in_H2O = 0.0

    q_at_max  = float(q_star_vec(T_K, np.array([p_H2O_max]), W0_DA, E_DA, n_DA)[0])
    F_CO2_in  = C_in_CO2 * u * A_b
    t_sat_est = q_at_max * m_cat_total / (2.0 * F_CO2_in)   # sorbent capacity basis — always 100% of mass, independent of active_fraction
    t_end     = min(2.5 * t_sat_est, 7200.0)

    print("=" * 60)
    print(f"  Non-isothermal SEM — T_in = {T_C} °C,  P = {P_bar} bar,  U_a = {U_a:.0f} W/(m³·K)")
    print(f"  Feed:  CO2 = {y_CO2_in:.1%},  H2 = {y_H2_in:.0%},  "
          f"CH4 = {y_CH4_in:.1%},  N2 = {y_N2_in:.0%}")
    print(f"  m_cat(active) = {m_cat*1000:.1f} g  (active_fraction = {active_fraction:.0%} of {m_cat_total*1000:.1f} g)")
    print(f"  GHSV:  {Q_STP*3600*1e6 / (m_cat_total*1e3):.0f} mL/(g·h)")
    print(f"  t_sat_est ≈ {t_sat_est/60:.0f} min  →  t_end = {t_end/60:.0f} min")

    # Initial condition: bed pre-filled with feed gas at T_in, sorbent clean.
    y0 = np.zeros(6 * N)
    y0[0*N : 1*N] = C_in_CO2
    y0[1*N : 2*N] = C_in_H2
    y0[2*N : 3*N] = C_in_CH4
    # q block stays 0 (clean sorbent)
    y0[5*N : 6*N] = T_K        # bed starts uniformly at setpoint temperature

    results = {}
    for se_on in [True, False]:
        tag = "SE on  (reaction + adsorption)" if se_on else "SE off (reaction only)    "
        print(f"  Solving {tag} ...", end="", flush=True)
        y0_run = y0.copy()
        if not se_on:
            y0_run[4*N : 5*N] = q_at_max   # sorbent pre-saturated (matches Wei non-SE experiment)
        sol = solve_ivp(
            rhs_sem_noniso,
            t_span=[0.0, t_end],
            y0=y0_run,
            args=(se_on, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O, T_K, T_K),
            method='BDF',
            rtol=1e-4,
            atol=atol_vec,
            dense_output=True,
        )
        print(f"  {'OK' if sol.success else 'FAILED — ' + sol.message}")
        if not sol.success:
            raise RuntimeError(f"ODE solver did not converge ({tag}): {sol.message}")
        results[se_on] = sol

    all_results[T_C] = {
        'results':   results,
        'T_K':       T_K,
        'C_in_CO2':  C_in_CO2,
        'q_at_max':  q_at_max,
        't_sat_est': t_sat_est,
        't_end':     t_end,
    }

print("=" * 60)


# endregion

# region 6. POST-PROCESSING
# =============================================================================
# 6. POST-PROCESSING
# =============================================================================

def extract_outlet(sol, C_in_CO2_loc):
    """
    Extract outlet CO2 conversion, H2O pressure, and peak bed temperature.

    Returns
    -------
    t_arr      : (n_t,)  time points [s]
    X_CO2      : (n_t,)  outlet CO2 conversion [-]
    p_H2O_mbar : (n_t,)  outlet H2O partial pressure [mbar]
    T_max      : (n_t,)  maximum T along the bed at each time [K]
    """
    t_arr     = sol.t
    y_arr     = sol.sol(t_arr)
    C_CO2_out = np.maximum(y_arr[N - 1,     :], 0.0)   # last node of CO2 block
    C_H2O_out = np.maximum(y_arr[4*N - 1,   :], 0.0)   # last node of H2O block
    T_profile = y_arr[5*N : 6*N,            :]          # T at all nodes (N × n_t)
    T_out     = T_profile[-1, :]                        # outlet temperature (n_t,)
    T_max     = T_profile.max(axis=0)                   # peak temperature in bed (n_t,)

    X_CO2      = np.clip((C_in_CO2_loc - C_CO2_out) / C_in_CO2_loc, 0.0, 1.0)
    p_H2O_mbar = C_H2O_out * R_gas * T_out / 1e5 * 1000

    return t_arr, X_CO2, p_H2O_mbar, T_max


# endregion

# region 7. PLOT (time-series, disabled — kept for reference)
# =============================================================================
# 7. PLOT  (time-series, disabled — kept for reference)
# =============================================================================
if True:
    n_rows = len(T_LIST)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5 * n_rows), squeeze=False)
    fig.suptitle(
        f'Non-isothermal SEM Column  —  P = {P_bar} bar,  U_a = {U_a:.0f} W/(m³·K),  '
        f'active_fraction = {active_fraction:.0%}\n'
        f'Feed: {y_CO2_in:.1%} CO₂ / {y_H2_in:.0%} H₂ / {y_CH4_in:.1%} CH₄ / '
        f'{y_N2_in:.0%} N₂  —  GHSV = 923 mL/(g·h)',
        fontsize=11
    )

    for row, T_C in enumerate(T_LIST):
        data         = all_results[T_C]
        results      = data['results']
        C_in_CO2_row = data['C_in_CO2']
        q_max_row    = data['q_at_max']

        t_on,  X_on,  pH2O_on,  Tmax_on  = extract_outlet(results[True],  C_in_CO2_row)
        t_off, X_off, pH2O_off, Tmax_off = extract_outlet(results[False], C_in_CO2_row)

        sol_on   = results[True]
        sol_off  = results[False]
        t_snaps  = np.linspace(sol_on.t[1], sol_on.t[-1], 5)
        snap_col = plt.cm.plasma(np.linspace(0.15, 0.85, len(t_snaps)))

        ax1, ax2, ax3, ax4 = axes[row]

        ax1.plot(t_on  / 60, X_on  * 100, color='tab:blue',   lw=2.5, label='SE on')
        ax1.plot(t_off / 60, X_off * 100, color='tab:orange', lw=2.5, ls='--', label='SE off')
        ax1.set_xlabel('Time [min]'); ax1.set_ylabel('CO₂ conversion [%]')
        ax1.set_title(f'Outlet CO₂ conversion — T_in = {T_C} °C')
        ax1.set_ylim(0, 105); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2.plot(t_on  / 60, Tmax_on  - (T_C + 273.15), color='tab:blue',   lw=2.5, label='SE on')
        ax2.plot(t_off / 60, Tmax_off - (T_C + 273.15), color='tab:orange', lw=2.5, ls='--', label='SE off')
        ax2.set_xlabel('Time [min]'); ax2.set_ylabel('ΔT_max [K]')
        ax2.set_title(f'Peak hot-spot above T_in — {T_C} °C')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        for i, t_s in enumerate(t_snaps):
            y_s = sol_on.sol(t_s)
            q_s = np.maximum(y_s[4*N : 5*N], 0.0)
            ax3.plot(z_cm, q_s, color=snap_col[i], lw=2.0, label=f't = {t_s/60:.0f} min')
        ax3.axhline(q_max_row, color='grey', ls=':', lw=1.0, label=f'q* = {q_max_row:.2f} mol/kg')
        ax3.set_xlabel('Bed position z [cm]'); ax3.set_ylabel('q  [mol/kg]')
        ax3.set_title(f'Solid-phase H₂O loading (SE on) — {T_C} °C')
        ax3.legend(fontsize=8, loc='upper left'); ax3.grid(True, alpha=0.3)

        # --- Outlet species concentrations ---
        y_on_arr  = sol_on.sol(sol_on.t)
        y_off_arr = sol_off.sol(sol_off.t)
        T_out_on  = y_on_arr[6*N - 1, :]
        T_out_off = y_off_arr[6*N - 1, :]
        to_mbar = lambda C, T_loc: np.maximum(C, 0.0) * R_gas * T_loc / 1e5 * 1000

        for blk, label, col in [(0,'CO₂','tab:blue'), (1,'H₂','tab:green'),
                                 (3,'H₂O','tab:purple')]:
            C_on  = y_on_arr[ (blk+1)*N - 1, :]
            C_off = y_off_arr[(blk+1)*N - 1, :]
            ax4.plot(sol_on.t  / 60, to_mbar(C_on,  T_out_on),  color=col, lw=2.0,
                     label=f'{label} (SE on)')
            ax4.plot(sol_off.t / 60, to_mbar(C_off, T_out_off), color=col, lw=2.0,
                     ls='--', label=f'{label} (SE off)')
        ax4.set_xlabel('Time [min]')
        ax4.set_ylabel('Outlet partial pressure [mbar]')
        ax4.set_title(f'Outlet species — T_in = {T_C} °C')
        ax4.legend(fontsize=7, ncol=2)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    plt.savefig(os.path.join(os.path.dirname(__file__),
                              f'Figure_7_axial_profiles_active_fraction_{active_fraction:.0%}.png'), dpi=300)
    plt.show()
# endregion

# region 7b. PLOT — CO2 conversion + hot-spot vs temperature
# =============================================================================
# 7b. PLOT — CO2 conversion vs temperature  +  peak ΔT vs temperature
# =============================================================================
from scipy.optimize import brentq


def equilibrium_conversion(T_K_val):
    """Equilibrium CO2 conversion for Wei's feed at 1 bar (reference for isothermal limit)."""
    K = K_eq_sabatier(T_K_val)
    def f(X):
        return ((0.815 + 0.025*X) * 0.0025 * X**2 * (1 - 0.05*X)**2
                / (2.5e-6 * (1 - X)**5) - K)
    try:
        return brentq(f, 1e-9, 1 - 1e-9) * 100
    except Exception:
        return 100.0


T_arr    = np.array(T_LIST, dtype=float)
X_off_ss = []
X_on_ini = []
dT_off   = []   # peak hot-spot ΔT for non-SE case
dT_on    = []   # peak hot-spot ΔT for SE case

for T_C in T_LIST:
    data         = all_results[T_C]
    results      = data['results']
    T_K_row      = data['T_K']
    C_in_CO2_row = data['C_in_CO2']
    t_sat_row    = data['t_sat_est']

    t_off, X_off, _, Tmax_off = extract_outlet(results[False], C_in_CO2_row)
    t_on,  X_on,  _, Tmax_on  = extract_outlet(results[True],  C_in_CO2_row)

    # Non-SE steady state: mean of second half
    mid = max(1, len(X_off) // 2)
    X_off_ss.append(float(np.mean(X_off[mid:])) * 100)
    dT_off.append(float(np.mean(Tmax_off[mid:])) - T_K_row)

    # SE fresh-sorbent plateau: mean of 10%–40% of t_sat window
    mask = (t_on >= 0.10 * t_sat_row) & (t_on <= 0.40 * t_sat_row)
    if mask.sum() == 0:
        mask = np.ones(len(t_on), dtype=bool)
    X_on_ini.append(float(np.mean(X_on[mask])) * 100)
    dT_on.append(float(np.mean(Tmax_on[mask])) - T_K_row)

# Equilibrium curve (isothermal reference)
T_fine = np.linspace(170, 370, 120)
X_eq   = [equilibrium_conversion(T + 273.15) for T in T_fine]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
fig.suptitle(
    '5%Ni2.5%Ce13X  —  Non-isothermal SEM model\n'
    f'GHSV = {Q_STP*3600*1e6/(m_cat_total*1e3):.0f} mL/(g·h),  P = {P_bar} bar,  '
    f'U_a = {U_a:.0f} W/(m³·K)  ({"adiabatic" if U_a == 0 else "cooled"}),  '
    f'active_fraction = {active_fraction:.0%}\n'
    f'Feed: {y_CO2_in:.1%} CO₂ / {y_H2_in:.0%} H₂ / {y_CH4_in:.1%} CH₄',
    fontsize=10
)

# --- Top panel: CO2 conversion ---
ax_top.plot(T_fine, X_eq,     'k--',  lw=1.5, label='Equilibrium (isothermal ref.)')
ax_top.plot(T_arr,  X_off_ss, 'ko--', lw=2.0, ms=7, label='Non-SE (steady state)')
ax_top.plot(T_arr,  X_on_ini, 'r^-',  lw=2.0, ms=7, label='SE (fresh sorbent)')
ax_top.set_ylabel('CO₂ conversion [%]', fontsize=12)
ax_top.set_ylim(0, 105)
ax_top.legend(fontsize=10)
ax_top.grid(True, alpha=0.3)

# --- Bottom panel: peak hot-spot temperature rise ---
ax_bot.plot(T_arr, dT_off, 'ko--', lw=2.0, ms=7, label='Non-SE')
ax_bot.plot(T_arr, dT_on,  'r^-',  lw=2.0, ms=7, label='SE (fresh sorbent)')
ax_bot.axhline(0, color='grey', lw=0.8, ls=':')
ax_bot.set_xlabel('Inlet temperature [°C]', fontsize=12)
ax_bot.set_ylabel('Peak ΔT  (T_max − T_in)  [K]', fontsize=12)
ax_bot.set_title('Hot-spot magnitude')
ax_bot.legend(fontsize=10)
ax_bot.grid(True, alpha=0.3)

ax_bot.set_xlim(170, 370)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__),
                          f'Figure_7b_conversion_hotspot_active_fraction_{active_fraction:.0%}.png'), dpi=300)
plt.show()
# endregion

# region 8. ADSORPTION CAPACITY AND BREAKTHROUGH TIMES
# =============================================================================
# 8. ADSORPTION CAPACITY AND BREAKTHROUGH TIMES PER TEMPERATURE
# =============================================================================

print("\n" + "=" * 70)
print("  Adsorption Capacity and Breakthrough Times — Non-Isothermal SEM")
print("=" * 70)
print(f"{'T_in':>6} {'q_max':>10} {'Cap':>12} {'t_bt_est':>11} {'t_bt_sim':>11}")
print(f"{'[°C]':>6} {'[mmol/g]':>10} {'[mol H2O]':>12} {'[min]':>11} {'[min]':>11}")
print("-" * 70)

cap_mol_kg    = []
cap_mol_tot   = []
t_bt_est_list = []
t_bt_sim_list = []

for T_C in T_LIST:
    data         = all_results[T_C]
    sol_on       = data['results'][True]
    C_in_CO2_row = data['C_in_CO2']
    q_max_row    = data['q_at_max']
    t_sat_row    = data['t_sat_est']

    t_pts  = sol_on.t
    y_arr  = sol_on.sol(t_pts)                   # shape (6N, n_t)
    q_out  = np.maximum(y_arr[5*N - 1, :], 0.0) # outlet solid loading [mol/kg]

    # Simulated breakthrough: first time outlet q reaches 5 % of equilibrium capacity
    idx_bt   = np.where(q_out >= 0.05 * q_max_row)[0]
    t_bt_val = float(t_pts[idx_bt[0]]) / 60 if len(idx_bt) > 0 else float(t_pts[-1]) / 60

    cap_mol_kg.append(q_max_row)
    cap_mol_tot.append(q_max_row * m_cat_total)
    t_bt_est_list.append(t_sat_row / 60)
    t_bt_sim_list.append(t_bt_val)

    print(f"{T_C:>6} {q_max_row:>10.3f} {q_max_row*m_cat_total:>12.4f} "
          f"{t_sat_row/60:>11.1f} {t_bt_val:>11.1f}")

print("=" * 70)
print("  q_max    : DA equilibrium loading at max H₂O partial pressure  [mmol H₂O / g_cat]")
print("  Cap      : total sorbent capacity  (q_max × m_cat_total)")
print("  t_bt_est : mass-balance estimate   (Cap / (2 × F_CO2_in))")
print("  t_bt_sim : simulated breakthrough  (outlet q ≥ 5 % of q_max)")

# ---- Figure: capacity and breakthrough times vs temperature -----------------
fig_bt, (ax_cap, ax_bt) = plt.subplots(1, 2, figsize=(12, 5))
fig_bt.suptitle(
    'Adsorption capacity and breakthrough times — Non-isothermal SEM\n'
    f'DA isotherm:  W₀ = {W0_DA*1e6:.0f} cm³/kg,  E = {E_DA/1e3:.0f} kJ/kg,  n = {n_DA},  '
    f'active_fraction = {active_fraction:.0%}',
    fontsize=10
)

ax_cap.plot(T_arr, cap_mol_kg, 'b^-', lw=2.0, ms=8)
ax_cap.set_xlabel('Inlet temperature [°C]', fontsize=12)
ax_cap.set_ylabel('Adsorption capacity  q* [mmol H₂O / g sorbent]', fontsize=11)
ax_cap.set_title('Equilibrium H₂O loading at max p_H₂O')
ax_cap.set_xlim(240, 320)
ax_cap.grid(True, alpha=0.3)

ax_bt.plot(T_arr, t_bt_est_list, 'ko--', lw=2.0, ms=7, label='Theoretical (mass balance)')
ax_bt.plot(T_arr, t_bt_sim_list, 'r^-',  lw=2.0, ms=7, label='Simulated (q_out ≥ 5 % q_max)')
ax_bt.set_xlabel('Inlet temperature [°C]', fontsize=12)
ax_bt.set_ylabel('Breakthrough time [min]', fontsize=12)
ax_bt.set_title('Time until adsorption front exits bed')
ax_bt.legend(fontsize=10)
ax_bt.set_xlim(240, 320)
ax_bt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__),
                          f'Figure_8_capacity_breakthrough_active_fraction_{active_fraction:.0%}.png'), dpi=300)
plt.show()
# endregion
