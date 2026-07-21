"""
Coupled SEM Column Model  —  1D Isothermal Transient
=====================================================

Combines the two validated sub-models into a single simulation:
  - LHHW CO2 methanation kinetics       (Koschany et al. 2016)
  - H2O adsorption on 5%Ni2.5%Ce/13X   (DA isotherm + LDF, from adsorption_simulation.py)

Reaction
--------
    CO2  +  4 H2  →  CH4  +  2 H2O       (Sabatier)

Sorption-enhancement mechanism
-------------------------------
Without a sorbent the Sabatier reaction is limited by thermodynamic equilibrium:
at 300 °C and 1 bar the equilibrium CO2 conversion for a 1:4 CO2/H2 feed is
roughly 85–90 %. By adsorbing the H2O product as it forms, the gas phase loses
H2O, the equilibrium shifts to the right, and conversion rises above that limit.

This model shows:
  1. How CO2 conversion at the column outlet changes over time.
  2. How the H2O front and adsorption loading build up axially.
  3. When the sorbent saturates and the sorption-enhancement effect fades.

State vector  (5 × N values — one value per axial node for each variable)
--------------------------------------------------------------------------
    y[0 : N]     C_CO2  [mol/m³]   gas-phase CO2 concentration
    y[N : 2N]    C_H2   [mol/m³]   gas-phase H2 concentration
    y[2N : 3N]   C_CH4  [mol/m³]   gas-phase CH4 concentration
    y[3N : 4N]   C_H2O  [mol/m³]   gas-phase H2O concentration
    y[4N : 5N]   q      [mol/kg]   solid-phase H2O loading

This file is self-contained. It does NOT modify pfr_simulation.py or
adsorption_simulation.py — both are left completely unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# region 1. PARAMETERS
# =============================================================================
# 1. PARAMETERS
# =============================================================================

# --------------- Bed geometry  (identical to adsorption_simulation.py) -------
d_b     = 0.010             # bed diameter                        [m]
L_b     = 0.100             # bed length                          [m]
A_b     = np.pi / 4 * d_b**2   # cross-sectional area            [m²]
V_bed   = A_b * L_b             # total bed volume                [m³]
m_cat_total    = 6.5e-3     # total bed material mass             [kg]  (Wei Fig. 5.3, 6.5 g — GHSV basis)
active_fraction = 0.20      # fraction of m_cat_total treated as catalytically active (reaction only) [-]
m_cat       = m_cat_total * active_fraction   # active catalyst mass   [kg]  (reaction rate basis)
rho_bed_cat = m_cat       / V_bed             # catalyst bulk density  [kg_cat/m³_bed]  (reaction terms)
rho_bed_ads = m_cat_total / V_bed             # sorbent bulk density   [kg_sorb/m³_bed] (adsorption terms — always 100% of mass)
eps_b   = 0.40              # void fraction between particles      [-]

# --------------- Particle properties  (identical to adsorption_simulation.py)
d_p   = 0.75e-3             # particle diameter                   [m]
eps_p = 0.242                 # intraparticle void fraction          [-]
tau_p = 4.0                 # tortuosity factor                    [-]

# --------------- Adsorbed-phase density (liquid water) -----------------------
# The DA isotherm computes the volume of adsorbed liquid per kg sorbent.
# Dividing by the liquid density and molar mass gives moles per kg sorbent.
rho_ads = 791             # liquid water density                 [kg/m³]

# --------------- DA isotherm parameters: Mette (2014) ------------------------
# These gave the best agreement with Wei's 300 °C breakthrough data in the
# adsorption-only validation (adsorption_simulation.py).
W0_DA = 190.00e-6           # micropore volume                     [m³/kg_sorbent]
E_DA  = 1192.25e3           # characteristic adsorption energy     [J/kg]
n_DA  = 1.55                # DA heterogeneity parameter           [-]

# --------------- Operating conditions ----------------------------------------
# T_LIST: list of temperatures to simulate; each gets its own subplot row.
T_LIST = [180, 210, 240, 270, 300, 330, 360]   # temperatures [°C]  (Wei Fig. 5.3 range)
P_bar = 1.0                 # total pressure                       [bar]
P_Pa  = P_bar * 1e5         # total pressure                       [Pa]

# --------------- Feed composition: Wei (2022) experimental inlet -------------
# Source: Wei thesis, adsorption breakthrough experiment (same setup as
# adsorption_simulation.py, but now reaction also occurs inside the column).
# N2 is an inert tracer and diluent; it is NOT tracked in the state vector
# because it neither reacts nor adsorbs.  Its presence is implicitly accounted
# for via the total pressure (partial pressures of the four tracked species
# will sum to less than P_bar; the remainder is N2).
y_CO2_in = 0.025            # CO2 mole fraction                    [-]
y_H2_in  = 0.10             # H2 mole fraction  (H2/CO2 = 4 → stoichiometric)  [-]
y_CH4_in = 0.815            # CH4 mole fraction (large background, inert here)  [-]
y_N2_in  = 0.06             # N2 mole fraction  (inert tracer, not tracked)     [-]
# Check: 0.025 + 0.10 + 0.815 + 0.06 = 1.000 ✓

# --------------- Gas flow ---------------------------------------------------
# Wei Fig. 5.3: 100 mL/min total flow, 6.5 g catalyst → GHSV = 923 mL/(g·h).
Q_STP = 100e-6 / 60         # volumetric flow at STP               [m³/s]
T_STP = 273.15              # STP temperature                      [K]
u_STP = Q_STP / A_b         # superficial velocity at STP          [m/s]
# u = u_STP * (T_K / T_STP) is recomputed per temperature in the solve loop.

# Note: u is kept constant along the bed.  Strictly the total molar flow
# decreases as CO2 + 4 H2 (5 mol) converts to CH4 + 2 H2O (3 mol in gas,
# or 1 mol if H2O is fully adsorbed).  Treating u as constant is a standard
# first-model simplification; the error is modest at partial conversion.

# --------------- Physical constants ------------------------------------------
R_gas  = 8.314              # universal gas constant               [J/(mol·K)]
MW_H2O = 0.018015           # molar mass of water                  [kg/mol]

# --------------- Kinetic parameters  (Koschany et al. 2016, Table 6) ----------
# LHHW rate for CO2 methanation on Ni catalyst, validated 180–360 °C, 1–15 bar.
# Rate form:
#   r = k · (p_CO2 · p_H2)^0.5 · (1 − β) / DEN²   [mol/(g_cat·s)]
#   β   = p_CH4 · p_H2O² / (K_eq · p_CO2 · p_H2⁴)
#   DEN = 1 + K_OH·p_H2O/p_H2^0.5 + K_H2·p_H2^0.5 + K_mix·p_CO2
# All K_i use van 't Hoff temperature dependence referenced to T_ref_K.
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
# K_OH · p_H2O / p_H2^0.5 → ∞ as p_H2 → 0; P_FLOOR on p_H2 prevents this.
# Also guards the β denominator (p_CO2 · p_H2^4) from dividing by zero.
P_FLOOR = 1e-4              # [bar]

# --------------- Spatial discretisation --------------------------------------
N  = 50                    # number of axial nodes
dz = L_b / (N - 1)          # node spacing                         [m]
z_cm = np.linspace(0, L_b, N) * 100   # node positions for plots   [cm]


# endregion

# region 2. THERMODYNAMIC AND KINETIC FUNCTIONS
# =============================================================================
# 2. THERMODYNAMIC AND KINETIC FUNCTIONS
# =============================================================================

def P_sat_bar(T_K):
    """
    Saturation vapour pressure of water [bar]  —  Antoine equation.
    Copied unchanged from adsorption_simulation.py.
    """
    log10_p = (29.8605 - 3.1522e3/T_K - 7.3037*np.log10(T_K)
               + 2.4247e-9*T_K + 1.8090e-6*T_K**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5


def q_star_vec(T_K, p_arr, W0, E, n):
    """
    Equilibrium H2O loading [mol/kg]  —  Dubinin-Astakhov (DA) isotherm.
    Copied unchanged from adsorption_simulation.py.

    p_arr : partial pressure of H2O [bar], array-like.

    The DA isotherm relates adsorption potential A to the fraction of micropore
    volume W that is filled.  A is zero at saturation (p = P_sat) and rises as
    p falls below P_sat, driving more adsorption.
    """
    p    = np.asarray(p_arr, dtype=float)
    Psat = P_sat_bar(T_K)

    # Clip to safe range before taking log (log(0) = -inf, log(>Psat) is wrong).
    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_raw  = (R_gas / MW_H2O) * T_K * np.log(Psat / p_safe)   # [J/kg]

    # Force A = 0 at the boundaries (no adsorption if p ≤ 0 or p ≥ P_sat).
    A  = np.where((p <= 0.0) | (p >= Psat), 0.0, A_raw)

    # Cap exponent to prevent overflow (exp(-500) ≈ 0 in any case).
    W  = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    qs = rho_ads / MW_H2O * W

    return np.where(p <= 0.0, 0.0, qs)


def K_LDF_vec(T_K, p_arr, W0, E, n):
    """
    LDF mass-transfer coefficient [1/s].
    Copied unchanged from adsorption_simulation.py.

    Controls how fast the solid loading q approaches equilibrium q*.
    A large K_LDF means fast adsorption (equilibrium-controlled);
    a small K_LDF means slow adsorption (kinetically limited).

    The formula accounts for molecular diffusion into the pores and the local
    steepness of the isotherm.  A steep isotherm (large dq*/dp) means each
    extra bit of pressure drives a lot of adsorption, making diffusion the
    bottleneck  →  smaller K_LDF.
    """
    D_M = 3.36e-9 * T_K**1.75                                                  # molecular diffusivity [m²/s], power-law T-dependence (Chapman-Enskog) from supplementary material of Bareschino et al. (2023)
    p      = np.asarray(p_arr, dtype=float)
    dp_bar = 1.0 / 1e5          # 1 Pa expressed in bar (central-difference step)

    p_lo = np.maximum(p - dp_bar, 1e-15)
    p_hi = p + dp_bar

    # Central-difference derivative of the isotherm (matches adsorption_simulation.py).
    dqstar_dp = (q_star_vec(T_K, p_hi, W0, E, n)
                 - q_star_vec(T_K, p_lo, W0, E, n)) / 2.0
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)   # prevent division by zero on flat isotherm

    return  (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_ads * R_gas * T_K * dqstar_dp))


def K_eq_sabatier(T_K):
    """
    Equilibrium constant for CO2 + 4H2 → CH4 + 2H2O  [dimensionless, p in bar].

    Source: Koschany et al. (2016) Applied Catalysis B, same paper as the LHHW
    kinetic parameters.  Form:
        K_eq = 137 · T^(−3.994) · exp(158.7 kJ/mol / (R·T))

    The T^(−3.994) term is a Kirchhoff (heat-capacity) correction: ΔH° of the
    Sabatier reaction changes with temperature, so integrating the van 't Hoff
    equation properly yields an extra T^n factor on top of the usual exp(−ΔH/RT).
    This makes the formula more accurate over the full 180–360 °C range than a
    simple two-parameter fit.

    Using this formula (rather than a different source's fit) is important for
    internal consistency: the LHHW rate parameters were fitted assuming this K_eq
    in the approach-to-equilibrium factor β = Q / K_eq.

    At 300 °C (573 K): K_eq ≈ 4 × 10⁵.
    """
    return 137.0 * T_K**(-3.994) * np.exp(158700.0 / (R_gas * T_K))


def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    """
    LHHW CO2 methanation rate  [mol / (kg_cat · s)].

    Source: Koschany et al. (2016) section 4.3.2.
    Rate form:
        r = k · (p_CO2 · p_H2)^0.5 · (1 − β) / DEN²

    where  β = p_CH4 · p_H2O² / (K_eq · p_CO2 · p_H2⁴)   (approach-to-equilibrium)
    and  DEN = 1 + K_OH·p_H2O/p_H2^0.5 + K_H2·p_H2^0.5 + K_mix·p_CO2

    When β = 1 (at equilibrium) the rate is zero; when β < 1 the reaction runs
    forward; β > 1 would give a negative rate, so it is clipped to zero — this is
    the thermodynamic brake that the power-law lacks.

    Parameters
    ----------
    T_K  : temperature [K]
    p_i  : partial pressure of species i [bar], numpy array of length N
    """
    # --- Van 't Hoff temperature corrections (referenced to T_ref_K = 555 K) ---
    vH  = lambda dH: np.exp(-dH / R_gas * (1.0 / T_K - 1.0 / T_ref_K))
    k     = k_ref * np.exp(-Ea_k / R_gas * (1.0 / T_K - 1.0 / T_ref_K))
    K_OH  = A_OH  * vH(dH_OH) #how strongle OH and H2O groups occypy active sites. gets smaller at high temp.
    K_H2  = A_H2  * vH(dH_H2)
    K_mix = A_mix * vH(dH_mix)
    K_eq  = K_eq_sabatier(T_K)

    # --- Floor p_CO2 and p_H2 to prevent division-by-zero in β and K_OH term ---
    p_CO2_s = np.maximum(p_CO2, P_FLOOR)
    p_H2_s  = np.maximum(p_H2,  P_FLOOR)

    # --- Approach-to-equilibrium factor (clip so rate never goes negative) ------
    beta = (p_CH4 * p_H2O**2) / (K_eq * p_CO2_s * p_H2_s**4)
    f_eq = np.maximum(1.0 - beta, 0.0)

    # --- Inhibition denominator -------------------------------------------------
    # assumes the surface has a finite number of sites and that there is competiion between species. 
    DEN = (1.0
           + K_OH  * np.maximum(p_H2O, 0.0) / p_H2_s**0.5
           + K_H2  * p_H2_s**0.5
           + K_mix * p_CO2_s**0.5)

    r_g_s = k * (p_CO2_s * p_H2_s)**0.5 * f_eq / DEN**2  # mol/(g_cat·s)
    return r_g_s * 1000.0                               # → mol/(kg_cat·s) with effectivity factor of 0.8


# endregion


# region 4. RIGHT-HAND SIDE OF THE ODE SYSTEM
# =============================================================================
# 4. RIGHT-HAND SIDE OF THE ODE SYSTEM
# =============================================================================

def rhs_sem(t, y, se_on, T_K, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O):
    """
    Time derivatives for the coupled SEM column (5N equations).

    Parameters
    ----------
    se_on    : bool   — True = sorption-enhanced (adsorption active),
                        False = reaction-only (sorbent inert, dq/dt = 0).
    T_K      : float  — temperature [K]
    u        : float  — superficial gas velocity [m/s]
    C_in_*   : float  — inlet concentrations [mol/m³] for each species

    Gas-phase balance (plug flow  +  reaction  +  adsorption):
    -----------------------------------------------------------
        ε_b · dC_i/dt = −u · (C_i − C_upstream) / Δz     ← convection
                       + ρ_bed · ν_i · r                   ← reaction source/sink
                      [− ρ_bed · dq/dt]                    ← adsorption sink (H2O only, se_on=True)

    Stoichiometry of CO2 + 4H2 → CH4 + 2H2O:
        ν_CO2 = −1,  ν_H2 = −4,  ν_CH4 = +1,  ν_H2O = +2

    Solid-phase balance (Linear Driving Force):
    -------------------------------------------
        dq/dt = K_LDF · (q* − q)   [se_on=True]
        dq/dt = 0                   [se_on=False, sorbent inactive]

    Spatial scheme: first-order upwind differences (C_upstream = inlet BC at
    node 0, previous node everywhere else).  Upwind is numerically stable for
    convection-dominated problems.
    """

    # --- Unpack and clip state vector (negatives can appear from ODE numerics) ---
    C_CO2 = np.maximum(y[0*N : 1*N], 0.0)
    C_H2  = np.maximum(y[1*N : 2*N], 0.0)
    C_CH4 = np.maximum(y[2*N : 3*N], 0.0)
    C_H2O = np.maximum(y[3*N : 4*N], 0.0)
    q     = np.maximum(y[4*N : 5*N], 0.0)

    # --- Convert concentrations to partial pressures [bar] ---
    p_CO2 = C_CO2 * R_gas * T_K / 1e5
    p_H2  = C_H2  * R_gas * T_K / 1e5
    p_CH4 = C_CH4 * R_gas * T_K / 1e5
    p_H2O = C_H2O * R_gas * T_K / 1e5

    # --- Reaction rate at every axial node [mol/(kg_cat·s)] ---
    r = reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O)

    # --- Adsorption: equilibrium loading and LDF rate ---
    qs   = q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)   # [mol/kg]
    Kl   = K_LDF_vec( T_K, p_H2O, W0_DA, E_DA, n_DA)   # [1/s]
    # When SE is off, adsorption is disabled: sorbent is inert, q stays at zero.
    dqdt = Kl * (qs - q) if se_on else np.zeros(N)       # [mol/(kg·s)]

    # --- Build upstream concentrations for first-order upwind scheme ---
    # Node 0 gets the inlet boundary condition; every other node gets its left neighbour.
    C_CO2_up = np.concatenate([[C_in_CO2], C_CO2[:-1]])
    C_H2_up  = np.concatenate([[C_in_H2],  C_H2[:-1]])
    C_CH4_up = np.concatenate([[C_in_CH4], C_CH4[:-1]])
    C_H2O_up = np.concatenate([[C_in_H2O], C_H2O[:-1]])

    # --- Gas-phase time derivatives ---
    # All four species share the same advection structure:
    #   dC/dt = [−u·(C − C_up)/Δz  +  ρ_bed·ν·r] / ε_b
    # H2O gets an extra adsorption sink before dividing by ε_b.
    dCdt_CO2 = (-u * (C_CO2 - C_CO2_up) / dz  +  rho_bed_cat * (-1) * r) / eps_b
    dCdt_H2  = (-u * (C_H2  - C_H2_up)  / dz  +  rho_bed_cat * (-4) * r) / eps_b
    dCdt_CH4 = (-u * (C_CH4 - C_CH4_up) / dz  +  rho_bed_cat * (+1) * r) / eps_b
    dCdt_H2O = (-u * (C_H2O - C_H2O_up) / dz  +  rho_bed_cat * (+2) * r
                                                 -  rho_bed_ads * dqdt) / eps_b
    # Solid-phase derivative — returned as-is (not divided by ε_b).
    # The LDF equation is written per unit sorbent mass, not per unit bed volume.

    return np.concatenate([dCdt_CO2, dCdt_H2, dCdt_CH4, dCdt_H2O, dqdt])


# endregion

# region 5. SOLVE
# =============================================================================
# 5. SOLVE
# =============================================================================

# Loop over all temperatures.  For each T_C we compute T_K, u, and inlet
# concentrations, run both SE cases, and store the solutions for plotting.
all_results = {}   # T_C (int) → {'results': {True/False: sol}, 'T_K', 'C_in_CO2'}

# p_H2O_max is the same for every temperature (depends only on feed and P_bar).
# Upper bound: assume 100% CO2 conversion → all CO2 becomes 2 H2O.
# n_out = 1 − 2·y_CO2_in  (2 moles of H2O replace 5 moles of CO2+H2, net −2)
p_H2O_max = 2 * y_CO2_in * P_bar / (1 - 2 * y_CO2_in)

for T_C in T_LIST:
    T_K      = T_C + 273.15
    u        = u_STP * (T_K / T_STP)
    C_in_CO2 = y_CO2_in * P_Pa / (R_gas * T_K)
    C_in_H2  = y_H2_in  * P_Pa / (R_gas * T_K)
    C_in_CH4 = y_CH4_in * P_Pa / (R_gas * T_K)
    C_in_H2O = 0.0

    # Estimate end time: fill the sorbent at the max possible H2O loading.
    q_at_max  = float(q_star_vec(T_K, np.array([p_H2O_max]), W0_DA, E_DA, n_DA)[0])
    F_CO2_in  = C_in_CO2 * u * A_b          # molar CO2 inlet flow  [mol/s]
    t_sat_est = q_at_max * m_cat_total / (2.0 * F_CO2_in)
    t_end     = min(2.5 * t_sat_est, 7200.0)

    print("=" * 60)
    print(f"  SEM column — T = {T_C} °C,  P = {P_bar} bar")
    print(f"  Feed:  CO2 = {y_CO2_in:.1%},  H2 = {y_H2_in:.0%},  "
          f"CH4 = {y_CH4_in:.1%},  N2 = {y_N2_in:.0%}")
    print(f"  Bed:   L = {L_b*100:.0f} cm,  d = {d_b*100:.1f} cm,  "
          f"m_cat(active) = {m_cat*1000:.1f} g  (active_fraction = {active_fraction:.0%} of {m_cat_total*1000:.1f} g, "
          f"sorbent kept at 100% = {m_cat_total*1000:.1f} g)")
    print(f"  GHSV:  {Q_STP*3600*1e6 / (m_cat_total*1e3):.0f} mL/(g·h)")
    print(f"  q* at p_H2O_max = {p_H2O_max:.3f} bar:  {q_at_max:.2f} mol/kg")
    print(f"  t_sat_est ≈ {t_sat_est/60:.0f} min  →  t_end = {t_end/60:.0f} min")

    # Initial condition: bed pre-filled with feed gas, sorbent clean.
    y0 = np.zeros(5 * N)
    y0[0*N : 1*N] = C_in_CO2
    y0[1*N : 2*N] = C_in_H2
    y0[2*N : 3*N] = C_in_CH4

    results = {}
    for se_on in [True, False]:
        tag = "SE on  (reaction + adsorption)" if se_on else "SE off (reaction only)    "
        print(f"  Solving {tag} ...", end="", flush=True)
        sol = solve_ivp(
            rhs_sem,
            t_span=[0.0, t_end],
            y0=y0,
            args=(se_on, T_K, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O),
            method='BDF',
            rtol=1e-4,
            atol=1e-8,
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

def extract_outlet(sol, T_K_loc, C_in_CO2_loc):
    """Extract outlet CO2 conversion and H2O partial pressure time series."""
    t_arr     = sol.t
    y_arr     = sol.sol(t_arr)
    C_CO2_out = np.maximum(y_arr[1*N - 1, :], 0.0)   # last node of CO2 block
    C_H2O_out = np.maximum(y_arr[4*N - 1, :], 0.0)   # last node of H2O block
    X_CO2     = np.clip((C_in_CO2_loc - C_CO2_out) / C_in_CO2_loc, 0.0, 1.0)
    p_H2O_mbar = C_H2O_out * R_gas * T_K_loc / 1e5 * 1000   # bar → mbar
    return t_arr, X_CO2, p_H2O_mbar


# endregion

# region 7. PLOT (time-series, disabled — kept for reference)
# =============================================================================
# 7. PLOT  (one row per temperature, three columns: conversion / H2O / loading)
# =============================================================================
if False:   # disabled: run section 7b instead
    n_rows = len(T_LIST)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows), squeeze=False)
    fig.suptitle(
        f'SEM Column — Isothermal,  P = {P_bar} bar\n'
        f'Feed: {y_CO2_in:.1%} CO₂ / {y_H2_in:.0%} H₂ / {y_CH4_in:.1%} CH₄ / '
        f'{y_N2_in:.0%} N₂  —  GHSV = 923 mL/(g·h),  Mette (2014) DA isotherm',
        fontsize=11
    )

    for row, T_C in enumerate(T_LIST):
        data         = all_results[T_C]
        results      = data['results']
        T_K_row      = data['T_K']
        C_in_CO2_row = data['C_in_CO2']
        q_max_row    = data['q_at_max']

        t_on,  X_on,  pH2O_on  = extract_outlet(results[True],  T_K_row, C_in_CO2_row)
        t_off, X_off, pH2O_off = extract_outlet(results[False], T_K_row, C_in_CO2_row)

        sol_on   = results[True]
        t_snaps  = np.linspace(sol_on.t[1], sol_on.t[-1], 5)
        snap_col = plt.cm.plasma(np.linspace(0.15, 0.85, len(t_snaps)))

        ax1, ax2, ax3 = axes[row]

        ax1.plot(t_on  / 60, X_on  * 100, color='tab:blue',   lw=2.5,
                 label='SE on  (+ adsorption)')
        ax1.plot(t_off / 60, X_off * 100, color='tab:orange', lw=2.5, ls='--',
                 label='SE off (reaction only)')
        ax1.set_xlabel('Time [min]')
        ax1.set_ylabel('CO₂ conversion at outlet [%]')
        ax1.set_title(f'Outlet CO₂ conversion — {T_C} °C')
        ax1.set_ylim(0, 105)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        for i, t_s in enumerate(t_snaps):
            ax1.axvline(t_s / 60, color=snap_col[i], ls=':', lw=1.0, alpha=0.6)

        ax2.plot(t_on  / 60, pH2O_on,  color='tab:blue',   lw=2.5,
                 label='SE on  (+ adsorption)')
        ax2.plot(t_off / 60, pH2O_off, color='tab:orange', lw=2.5, ls='--',
                 label='SE off (reaction only)')
        ax2.set_xlabel('Time [min]')
        ax2.set_ylabel('p_{H₂O} at outlet [mbar]')
        ax2.set_title(f'Outlet H₂O pressure — {T_C} °C')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        for i, t_s in enumerate(t_snaps):
            y_s = sol_on.sol(t_s)
            q_s = np.maximum(y_s[4*N : 5*N], 0.0)
            ax3.plot(z_cm, q_s, color=snap_col[i], lw=2.0,
                     label=f't = {t_s/60:.0f} min')
        ax3.axhline(q_max_row, color='grey', ls=':', lw=1.0,
                    label=f'q* = {q_max_row:.2f} mol/kg')
        ax3.set_xlabel('Bed position z [cm]')
        ax3.set_ylabel('q  [mol / kg]')
        ax3.set_title(f'Solid-phase H₂O loading (SE on) — {T_C} °C')
        ax3.legend(fontsize=8, loc='upper left')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    plt.show()
# endregion

# region 7b. PLOT — CO2 conversion vs temperature  (Wei Fig. 5.3 style)
# =============================================================================
# 7b. PLOT — CO2 conversion vs temperature
# =============================================================================
from scipy.optimize import brentq

def equilibrium_conversion(T_K_val):
    """Equilibrium CO2 conversion for the Wei feed (2.5% CO2, 10% H2, 81.5% CH4) at 1 bar."""
    K = K_eq_sabatier(T_K_val)
    # β = 1 at equilibrium:
    # K = (0.815+0.025X) · 0.0025X² · (1−0.05X)² / (2.5e-6 · (1−X)⁵)
    def f(X):
        return ((0.815 + 0.025*X) * 0.0025 * X**2 * (1 - 0.05*X)**2
                / (2.5e-6 * (1 - X)**5) - K)
    try:
        return brentq(f, 1e-9, 1 - 1e-9) * 100
    except Exception:
        return 100.0

T_arr    = np.array(T_LIST, dtype=float)
X_off_ss = []   # non-SE steady-state conversion at each temperature
X_on_ini = []   # SE initial (fresh sorbent) conversion at each temperature

for T_C in T_LIST:
    data         = all_results[T_C]
    results      = data['results']
    T_K_row      = data['T_K']
    C_in_CO2_row = data['C_in_CO2']
    t_sat_row    = data['t_sat_est']

    t_off, X_off, _ = extract_outlet(results[False], T_K_row, C_in_CO2_row)
    t_on,  X_on,  _ = extract_outlet(results[True],  T_K_row, C_in_CO2_row)

    # Non-SE: column reaches kinetic/thermodynamic steady state within seconds.
    # Take mean of second half of the time series to capture the plateau.
    mid = max(1, len(X_off) // 2)
    X_off_ss.append(float(np.mean(X_off[mid:])) * 100)

    # SE: take mean of the 10%–40% window of t_sat_est.
    # By 10% of t_sat the reaction has equilibrated; sorbent is still fresh.
    mask = (t_on >= 0.10 * t_sat_row) & (t_on <= 0.40 * t_sat_row)
    if mask.sum() == 0:
        mask = np.ones(len(t_on), dtype=bool)
    X_on_ini.append(float(np.mean(X_on[mask])) * 100)

# Equilibrium curve
T_fine = np.linspace(170, 370, 120)
X_eq   = [equilibrium_conversion(T + 273.15) for T in T_fine]

# --- Digitized experimental data (Wei Fig. 5.3, via WebPlotDigitizer) ---------
import os
import pandas as pd

wpd_path = os.path.join(os.path.dirname(__file__), 'wpd_datasets.csv')
wpd = pd.read_csv(wpd_path, header=1)
wpd.columns = ['X_nonSE', 'Y_nonSE', 'X_SE', 'Y_SE']

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(T_fine,  X_eq,     'k--',  lw=1.5, label='Equilibrium')
ax.plot(T_arr,   X_off_ss, 'ko--', lw=2.0, ms=7, label='Non-SE (steady state)')
ax.plot(T_arr,   X_on_ini, 'r^-',  lw=2.0, ms=7, label='SE (fresh sorbent)')
ax.scatter(wpd['X_nonSE'], wpd['Y_nonSE'], marker='o', s=50,
           facecolors='none', edgecolors='black', linewidths=1.5,
           label='Non-SE (Wei, measured)')
ax.scatter(wpd['X_SE'], wpd['Y_SE'], marker='^', s=50,
           facecolors='none', edgecolors='red', linewidths=1.5,
           label='SE (Wei, measured)')
ax.set_xlabel('Temperature [°C]', fontsize=12)
ax.set_ylabel('CO₂ conversion [%]', fontsize=12)
ax.set_xlim(170, 370)
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__),
                          f'Figure_1_active_fraction_{active_fraction:.0%}.png'), dpi=300)
plt.show()
# endregion
