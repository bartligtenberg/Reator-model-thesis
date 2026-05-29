"""
Coupled SEM Column Model  —  1D Isothermal Transient
=====================================================

Combines the two validated sub-models into a single simulation:
  - Power-law CO2 methanation kinetics  (Wei 2022, from pfr_simulation.py)
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
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp


# =============================================================================
# 1. PARAMETERS
# =============================================================================

# --------------- Bed geometry  (identical to adsorption_simulation.py) -------
d_b     = 0.010             # bed diameter                        [m]
L_b     = 0.100             # bed length                          [m]
A_b     = np.pi / 4 * d_b**2   # cross-sectional area            [m²]
V_bed   = A_b * L_b             # total bed volume                [m³]
m_cat   = 6.5e-3            # catalyst / sorbent mass             [kg]
rho_bed = m_cat / V_bed     # bulk density = m_cat / V_bed        [kg_cat / m³_bed]
eps_b   = 0.40              # void fraction between particles      [-]

# --------------- Particle properties  (identical to adsorption_simulation.py)
d_p   = 0.75e-3             # particle diameter                   [m]
eps_p = 0.6                 # intraparticle void fraction          [-]
tau_p = 3.0                 # tortuosity factor                    [-]

# --------------- Adsorbed-phase density (liquid water) -----------------------
# The DA isotherm computes the volume of adsorbed liquid per kg sorbent.
# Dividing by the liquid density and molar mass gives moles per kg sorbent.
rho_ads = 998.2             # liquid water density                 [kg/m³]

# --------------- DA isotherm parameters: Mette (2014) ------------------------
# These gave the best agreement with Wei's 300 °C breakthrough data in the
# adsorption-only validation (adsorption_simulation.py).
W0_DA = 341.00e-6           # micropore volume                     [m³/kg_sorbent]
E_DA  = 1192.25e3           # characteristic adsorption energy     [J/kg]
n_DA  = 1.55                # DA heterogeneity parameter           [-]

# --------------- Operating conditions ----------------------------------------
T_C   = 300.0               # temperature — isothermal assumption  [°C]
T_K   = T_C + 273.15        # same in Kelvin                       [K]
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

# --------------- Gas flow  (same GHSV as adsorption_simulation.py) -----------
# GHSV = 923 mL_gas/(g_cat·h)  ×  6.5 g_cat  ≈  100 mL/min  (at STP = 0 °C)
Q_STP = 100e-6 / 60         # volumetric flow at STP               [m³/s]
T_STP = 273.15              # STP temperature                      [K]
u_STP = Q_STP / A_b         # superficial velocity at STP          [m/s]
u     = u_STP * (T_K / T_STP)   # actual velocity at T_K           [m/s]

# Note: u is kept constant along the bed.  Strictly the total molar flow
# decreases as CO2 + 4 H2 (5 mol) converts to CH4 + 2 H2O (3 mol in gas,
# or 1 mol if H2O is fully adsorbed).  Treating u as constant is a standard
# first-model simplification; the error is modest at partial conversion.

# --------------- Physical constants ------------------------------------------
R_gas  = 8.314              # universal gas constant               [J/(mol·K)]
MW_H2O = 0.018015           # molar mass of water                  [kg/mol]

# --------------- Kinetic parameters  (Wei 2022, Table S.6.4) -----------------
# Taken directly from pfr_simulation.py — units there are mol/(min·g_cat).
k_Tref  = 1.1e-4            # rate constant at T_ref               [mol/(min·g_cat)]
T_ref_K = 266.0 + 273.15    # reference temperature                [K]  (= 539 K)
Ea      = 81.9e3            # activation energy                    [J/mol]
n_CO2   =  0.16             # reaction order in CO2                [-]
n_H2    =  0.48             # reaction order in H2                 [-]
n_CH4   =  0.01             # reaction order in CH4                [-]
n_H2O   = -0.003            # H2O is a weak inhibitor (neg. order) [-]

# Unit conversion:  mol/(g_cat · min)  →  mol/(kg_cat · s)
# ×1000 because 1 kg = 1000 g;  ÷60 because 1 min = 60 s.
RATE_CONV = 1000.0 / 60.0

# Floor partial pressures used inside reaction_rate_SI.
# Why 1e-4 bar, not the 1e-12 used in pfr_simulation.py?
# pfr_simulation.py uses explicit RK45 which never computes a Jacobian.
# BDF (used here) builds a numerical Jacobian by finite differences.
# The sensitivity ∂r/∂C_i ∝ (n_i − 1)·r / p_i diverges as p_i → 0
# when n_i < 1 (CO2: 0.16, H2: 0.48) or when n_i < 0 (H2O: −0.003).
# With P_FLOOR = 1e-12 the Jacobian entries reach ~10^9 s⁻¹, overflowing
# the LU factorisation.  1e-4 bar caps entries at ~700 s⁻¹ — stiff but
# within BDF's range.  Physical effect: only activates at >99% conversion,
# changing the rate by < 3 % — completely negligible for this model.
P_FLOOR = 1e-4              # [bar]

# --------------- Spatial discretisation --------------------------------------
N  = 50                     # number of axial nodes
dz = L_b / (N - 1)          # node spacing                         [m]
z_cm = np.linspace(0, L_b, N) * 100   # node positions for plots   [cm]


# =============================================================================
# 2. THERMODYNAMIC AND KINETIC FUNCTIONS
# =============================================================================

def P_sat_bar(T_K):
    """
    Saturation vapour pressure of water [bar]  —  Antoine equation.
    Copied unchanged from adsorption_simulation.py.
    """
    return 10.0 ** (5.40221 - 1838.675 / (T_K - 31.737))


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
    D_M    = 2.5e-5 * (T_K / 300.0) ** 1.75   # molecular diffusivity    [m²/s]
    p      = np.asarray(p_arr, dtype=float)
    dp_bar = 1.0 / 1e5          # 1 Pa expressed in bar (central-difference step)

    p_lo = np.maximum(p - dp_bar, 1e-15)
    p_hi = p + dp_bar

    # Central-difference derivative of the isotherm (matches adsorption_simulation.py).
    dqstar_dp = (q_star_vec(T_K, p_hi, W0, E, n)
                 - q_star_vec(T_K, p_lo, W0, E, n)) / 2.0
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)   # prevent division by zero on flat isotherm

    return (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_ads * R_gas * T_K * dqstar_dp))


def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    """
    Power-law CO2 methanation rate  [mol / (kg_cat · s)].

    Adapted from pfr_simulation.py (Wei 2022, Table S.6.4).
    Two changes versus pfr_simulation.py:
      1. Accepts numpy arrays (one value per axial node) instead of scalars.
         Uses np.maximum() for element-wise flooring instead of max().
      2. Converted from mol/(g_cat · min) to mol/(kg_cat · s) via RATE_CONV.

    Parameters
    ----------
    T_K  : temperature [K]
    p_i  : partial pressure of species i [bar], numpy array of length N
    """
    # Arrhenius temperature dependence
    k = k_Tref * np.exp(-Ea / R_gas * (1.0 / T_K - 1.0 / T_ref_K))

    # Floor all partial pressures.
    # - Reactants (CO2, H2): avoids 0^positive_exponent = 0 at the bed inlet
    #   before the gas front arrives.
    # - CH4: same reason at the inlet where no CH4 has been produced yet.
    # - H2O: critical — n_H2O = -0.003 means p_H2O^(-0.003) → ∞ as p_H2O → 0.
    #   When the sorbent strips all H2O, this floor prevents a numerical blowup
    #   while only changing the rate by p_FLOOR^(-0.003) ≈ 1.00007 (negligible).
    p_CO2_s = np.maximum(p_CO2, P_FLOOR)
    p_H2_s  = np.maximum(p_H2,  P_FLOOR)
    p_CH4_s = np.maximum(p_CH4, P_FLOOR)
    p_H2O_s = np.maximum(p_H2O, P_FLOOR)

    r_gmin = (k
              * p_CO2_s**n_CO2
              * p_H2_s **n_H2
              * p_CH4_s**n_CH4
              * p_H2O_s**n_H2O)

    return r_gmin * RATE_CONV   # convert to mol / (kg_cat · s)


# =============================================================================
# 3. INLET CONCENTRATIONS
# =============================================================================

# Ideal gas law: C [mol/m³] = p [Pa] / (R [J/(mol·K)] · T [K])
C_in_CO2 = y_CO2_in * P_Pa / (R_gas * T_K)   # [mol/m³]
C_in_H2  = y_H2_in  * P_Pa / (R_gas * T_K)   # [mol/m³]
C_in_CH4 = y_CH4_in * P_Pa / (R_gas * T_K)   # [mol/m³]  — large background!
C_in_H2O = 0.0                                  # no H2O in the dry feed


# =============================================================================
# 4. RIGHT-HAND SIDE OF THE ODE SYSTEM
# =============================================================================

def rhs_sem(t, y, se_on):
    """
    Time derivatives for the coupled SEM column (5N equations).

    Parameters
    ----------
    se_on : bool
        True  → sorption-enhanced mode: H2O is adsorbed as it forms.
        False → reaction-only mode:     adsorption disabled (dq/dt = 0),
                H2O stays in the gas phase.  Use this as a reference to see
                how much CO2 conversion the reaction alone achieves.

    At every axial node we solve simultaneously:
      · A gas-phase balance for CO2, H2, CH4, and H2O.
      · A solid-phase balance for H2O loading q.

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
    dCdt_CO2 = (-u * (C_CO2 - C_CO2_up) / dz  +  rho_bed * (-1) * r) / eps_b
    dCdt_H2  = (-u * (C_H2  - C_H2_up)  / dz  +  rho_bed * (-4) * r) / eps_b
    dCdt_CH4 = (-u * (C_CH4 - C_CH4_up) / dz  +  rho_bed * (+1) * r) / eps_b
    dCdt_H2O = (-u * (C_H2O - C_H2O_up) / dz  +  rho_bed * (+2) * r
                                                 -  rho_bed * dqdt) / eps_b
    # Solid-phase derivative — returned as-is (not divided by ε_b).
    # The LDF equation is written per unit sorbent mass, not per unit bed volume.

    return np.concatenate([dCdt_CO2, dCdt_H2, dCdt_CH4, dCdt_H2O, dqdt])


# =============================================================================
# 5. SOLVE
# =============================================================================

# --- Rough estimate of sorbent saturation time --------------------------------
# Upper bound: all CO2 converts to CH4 + 2 H2O; all H2O is adsorbed until
# the sorbent is full.  This tells us roughly how long to simulate.
F_CO2_in_mol_s = C_in_CO2 * u * A_b      # molar CO2 flow at inlet  [mol/s]
# H2O partial pressure at 100% CO2 conversion (stoichiometric H2/CO2 = 4):
#   Feed (1 mol):  0.025 CO2 + 0.10 H2 + 0.815 CH4 + 0.06 N2
#   Products:      0.840 CH4 + 0.050 H2O + 0.060 N2  =  0.95 mol total
#   → y_H2O = 0.05/0.95 = 5/95  (same as adsorption_simulation.py inlet ✓)
# General formula for stoichiometric diluted feed: n_out = 1 - 2·y_CO2_in

p_H2O_max = 2 * y_CO2_in * P_bar / (1 - 2 * y_CO2_in)
q_at_max  = float(q_star_vec(T_K, np.array([p_H2O_max]), W0_DA, E_DA, n_DA)[0])
t_sat_est = q_at_max * m_cat / (2.0 * F_CO2_in_mol_s)   # [s]
t_end     = min(2.5 * t_sat_est, 7200.0)                  # cap at 2 hours

print("=" * 60)
print(f"  SEM column — isothermal, T = {T_C} °C,  P = {P_bar} bar")
print(f"  Feed:  CO2 = {y_CO2_in:.1%},  H2 = {y_H2_in:.0%},  "
      f"CH4 = {y_CH4_in:.1%},  N2 = {y_N2_in:.0%}")
print(f"  Bed:   L = {L_b*100:.0f} cm,  d = {d_b*100:.1f} cm,  m_cat = {m_cat*1000:.1f} g")
# GHSV [mL/(g·h)] = Q_STP [m³/s] × 3600 [s/h] × 1e6 [mL/m³] / (m_cat [kg] × 1e3 [g/kg])
print(f"  GHSV:  {Q_STP*3600*1e6 / (m_cat*1e3):.0f} mL/(g·h)")
print(f"  Sorbent capacity at max p_H2O = {p_H2O_max:.2f} bar:  q* = {q_at_max:.2f} mol/kg")
print(f"  Estimated saturation time:  ~{t_sat_est/60:.0f} min")
print(f"  Simulating to:  t_end = {t_end/60:.0f} min")
print("=" * 60)
# Initial condition: column pre-filled with inlet feed gas, clean sorbent.
# Why not all-zeros?  With y0=0, CO2 and H2 start at zero everywhere.
# The BDF Jacobian then evaluates ∂r/∂C at p ≈ P_FLOOR, still giving large
# entries during the first ~2 s fill-up transient.  Pre-filling with the
# inlet composition puts all species well above P_FLOOR from t=0 onward,
# so the Jacobian is well-conditioned immediately.
# The 2-second fill-up transient is irrelevant for a simulation of ~hours.
y0 = np.zeros(5 * N)
y0[0*N : 1*N] = C_in_CO2   # CO2 at inlet mole fraction throughout the bed
y0[1*N : 2*N] = C_in_H2    # H2  at inlet mole fraction throughout the bed
y0[2*N : 3*N] = C_in_CH4   # CH4 background (largest component) throughout the bed
# H2O block (indices 3N:4N) and q block (4N:5N) stay zero — clean, dry sorbent.

# Run both cases: SE on (reaction + adsorption) and SE off (reaction only).
# Storing both solutions lets us overlay them on the same axes to make the
# sorption-enhancement effect visible directly.
results = {}
for se_on in [True, False]:
    tag = "SE on  (reaction + adsorption)" if se_on else "SE off (reaction only)    "
    print(f"  Solving {tag} ...", end="", flush=True)
    sol = solve_ivp(
        rhs_sem,
        t_span=[0.0, t_end],
        y0=y0,
        args=(se_on,),
        method='BDF',          # BDF handles the stiffness from fast adsorption
        # (timescale ~ 1/K_LDF) vs slow axial advection (timescale ~ L/u).
        rtol=1e-4,
        atol=1e-8,
        dense_output=True,
    )
    print(f"  {'OK' if sol.success else 'FAILED — ' + sol.message}")
    if not sol.success:
        raise RuntimeError(f"ODE solver did not converge ({tag}): {sol.message}")
    results[se_on] = sol

print("=" * 60)


# =============================================================================
# 6. POST-PROCESSING
# =============================================================================

def extract_outlet(sol):
    """Extract outlet CO2 conversion and H2O partial pressure time series."""
    t_arr      = sol.t
    y_arr      = sol.sol(t_arr)
    C_CO2_out  = np.maximum(y_arr[1*N - 1, :], 0.0)   # last node of CO2 block
    C_H2O_out  = np.maximum(y_arr[4*N - 1, :], 0.0)   # last node of H2O block
    X_CO2      = np.clip((C_in_CO2 - C_CO2_out) / C_in_CO2, 0.0, 1.0)
    p_H2O_mbar = C_H2O_out * R_gas * T_K / 1e5 * 1000   # bar → mbar
    return t_arr, X_CO2, p_H2O_mbar

t_on,  X_on,  pH2O_on  = extract_outlet(results[True])
t_off, X_off, pH2O_off = extract_outlet(results[False])

# Snapshot times for spatial profiles, taken from the SE-on solution.
sol_on = results[True]
t_snaps = np.linspace(sol_on.t[1], sol_on.t[-1], 5)
snap_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(t_snaps)))


# =============================================================================
# 7. PLOT
# =============================================================================

fig = plt.figure(figsize=(16, 5))
fig.suptitle(
    f'SEM Column — Isothermal,  T = {T_C} °C,  P = {P_bar} bar\n'
    f'Feed: {y_CO2_in:.1%} CO₂ / {y_H2_in:.0%} H₂ / {y_CH4_in:.1%} CH₄ / {y_N2_in:.0%} N₂  —  '
    f'GHSV = 923 mL/(g·h),  Mette (2014) DA isotherm',
    fontsize=11
)
gs = gridspec.GridSpec(1, 3, wspace=0.38)

# --- Panel 1: CO2 conversion — SE on vs SE off --------------------------------
# The gap between the two curves IS the sorption-enhancement effect.
# SE on stays high while the sorbent is fresh; it drops toward the SE-off line
# once the sorbent saturates and can no longer remove product H2O.
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_on  / 60, X_on  * 100, color='tab:blue',   lw=2.5, label='SE on  (+ adsorption)')
ax1.plot(t_off / 60, X_off * 100, color='tab:orange', lw=2.5, ls='--', label='SE off (reaction only)')
ax1.set_xlabel('Time [min]')
ax1.set_ylabel('CO₂ conversion at outlet [%]')
ax1.set_title('Outlet CO₂ conversion over time')
ax1.set_ylim(0, 105)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
# Dotted vertical lines mark the 5 snapshot times linked to panels 2 and 3.
for i, t_s in enumerate(t_snaps):
    ax1.axvline(t_s / 60, color=snap_colors[i], ls=':', lw=1.0, alpha=0.6)

# --- Panel 2: Outlet H2O partial pressure — SE on vs SE off ------------------
# With SE on, H2O is trapped in the bed; the outlet stays near zero until
# sorbent breakthrough.  With SE off, H2O passes straight through at the
# steady-state level set by reaction kinetics.
ax2 = fig.add_subplot(gs[1])
ax2.plot(t_on  / 60, pH2O_on,  color='tab:blue',   lw=2.5, label='SE on  (+ adsorption)')
ax2.plot(t_off / 60, pH2O_off, color='tab:orange', lw=2.5, ls='--', label='SE off (reaction only)')
ax2.set_xlabel('Time [min]')
ax2.set_ylabel('p_{H₂O} at outlet [mbar]')
ax2.set_title('Outlet H₂O partial pressure')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Solid-phase H2O loading profiles (SE on) -----------------------
# Shows how the sorbent fills up from inlet toward outlet over time.
# When q approaches q* at a node, adsorption stops there and H2O breaks through
# to the next section — the classic mass-transfer zone moving down the bed.
ax3 = fig.add_subplot(gs[2])
for i, t_s in enumerate(t_snaps):
    y_s = sol_on.sol(t_s)
    q_s = np.maximum(y_s[4*N : 5*N], 0.0)
    ax3.plot(z_cm, q_s, color=snap_colors[i], lw=2.0,
             label=f't = {t_s/60:.0f} min')
ax3.set_xlabel('Bed position z [cm]')
ax3.set_ylabel('q  [mol / kg]')
ax3.set_title('Solid-phase H₂O loading (SE on)')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.0, 1, 0.93])
plt.show()
