"""
Adsorption-only model — reproduce Bareschino et al. (2023) Fig. 1(a)
H₂O breakthrough capacity [mmol/g] vs temperature [°C]

Physical setup
--------------
We model a packed bed filled with the bi-functional 5%Ni/13X material.
The feed entering the column already contains H₂O from complete CO₂ methanation
(CO₂ + 4H₂ → CH₄ + 2H₂O). We assume 100% CO₂ conversion happens
instantaneously before the column inlet — this decouples the adsorption
from the reaction so we can validate the isotherm and transport parameters
independently.

Only H₂O is tracked. All other species (CH₄, N₂) are inert here.

Feed stoichiometry (100 mol basis, Bareschino Fig. 1 caption):
  In:  6 N₂ + 10 H₂ + 2.5 CO₂ + 81.5 CH₄  = 100 mol
  Out: 6 N₂ +  0 H₂ +   0 CO₂ + 84.0 CH₄  +  5 H₂O = 95 mol
  → y_H₂O = 5/95 = 0.05263,  p_H₂O = 0.0526 bar

Model equations
---------------
Gas-phase H₂O balance (1D plug flow, transient):
    ε_b (void fraction) · dC/dt (how much water accumulates over time) = -u · dC/dz (velocity times slope water concentration) -  ρ_bed · dq/dt
    

Solid-phase loading (Linear Driving Force, LDF):
    dq/dt = K_LDF · (q* - q)

  where q* is the equilibrium loading from the Dubinin-Astakhov (DA) isotherm
  and K_LDF controls how fast the solid approaches equilibrium, its the " mass transfer resistance from the fluid phase into the micro pores of the zeolite beads)
  

Numerical method: upwind finite differences in space, BDF in time (stiff solver).

Validation target: Bareschino et al. (2023) Chem. Eng. Sci. 277, Fig. 1(a)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# =============================================================================
# 1. PARAMETERS
# =============================================================================

# --- Reactor geometry (Wei Ch.5 experimental setup) ---
d_b     = 0.010           # inner tube diameter                   [m]
L_b     = 0.100           # packed bed length                     [m]
A_b     = np.pi / 4 * d_b**2   # cross-sectional area            [m²]
V_bed   = A_b * L_b             # total bed volume                [m³]

m_cat   = 6.5e-3          # mass of bi-functional material        [kg]
rho_bed = m_cat / V_bed   # bulk density = m_cat / V_bed          [kg/m³]
eps_b   = 0.40            # void fraction between particles       [-]
d_p     = 0.75e-3         # particle diameter (midpoint 0.5–1 mm) [m]

# --- Feed conditions ---
# After 100% CO₂ conversion, y_H₂O = 5/95 (see stoichiometry above)
y_H2O_in = 5.0 / 95.0    # H₂O mole fraction at column inlet    [-]
P_bar    = 1.0            # total pressure                        [bar]
P_Pa     = P_bar * 1e5    # total pressure                        [Pa]

# --- Volumetric flow rate ---
# GHSV = 923 mL_gas/(g_cat·h) × 6.5 g_cat = 5999 mL/h ≈ 100 mL/min
# Reference conditions: STP = 0°C, 1 bar  (not NTP = 20°C)
Q_STP   = 100e-6 / 60    # volumetric flow at STP                [m³/s]
T_STP   = 273.15          # STP temperature                       [K]
u_STP   = Q_STP / A_b    # superficial gas velocity at STP       [m/s]
# At operating temperature T_K, the actual velocity scales with T (ideal gas):
#   u(T) = u_STP × (T_K / T_STP)

# --- Physical constants ---
R_gas   = 8.314           # universal gas constant                [J/(mol·K)]
MW_H2O  = 0.018015        # molar mass of water                   [kg/mol]

# --- Adsorbed-phase density ---
# In the DA isotherm, W [m³/kg] is the volume of adsorbed liquid per unit
# sorbent mass. Converting to molar loading requires the liquid-phase density.
# We use the constant value for liquid water (independent of T, per Bareschino).
rho_ads = 998.2           # adsorbed-phase density (liquid water) [kg/m³]

# --- Intraparticle transport parameters (Mette 2014, via Bareschino Eq. 24) ---
eps_p   = 0.6             # intraparticle void fraction (pore volume fraction) [-]
tau_p   = 3.0             # tortuosity factor (accounts for winding pore paths) [-]

# --- Breakthrough criterion ---
# Bareschino and Walspurger define breakthrough when the outlet H₂O
# concentration reaches 10% of the inlet value.
BT_FRACTION = 0.10

# --- Spatial discretisation ---
# The bed is divided into N equally spaced nodes. More nodes = more accurate
# but slower. N=30 is sufficient for this validation at ~1% spatial error.
N  = 30
dz = L_b / (N - 1)       # node spacing                          [m]

# --- DA isotherm parameter sets (Table 1, Bareschino 2023) ---
# Two sets from different literature sources; both use the DA model.
# W0 [m³/kg]: maximum adsorption volume (pore saturation capacity)
# E  [J/kg]:  characteristic energy of adsorption (steepness of isotherm)
# n  [-]:     heterogeneity parameter (shape of the isotherm curve)
PARAM_SETS = {
    'Mette (2014)': {'W0': 341.00e-6, 'E': 1192.25e3, 'n': 1.55},
    'Kiefer (2022)':  {'W0':  90.17e-6, 'E': 1030.90e3, 'n': 1.55},
    'Ligtenberg (2026)': {'W0':  200e-6, 'E':  1192.00e3, 'n': 1.55},  # hypothetical set for testing
}

# --- Experimental reference data ---
# Digitised from Bareschino Fig. 1(a); originally from Wei et al. (2021a).
wei_exp_T   = [280, 300, 320]          # temperature               [°C]
wei_exp_cap = [1.30, 1.05, 0.85]      # breakthrough capacity      [mmol/g]

# Temperature range to simulate (matching Bareschino Fig. 1a)
temperatures_C = [280, 290, 300, 310, 320]


# =============================================================================
# 2. THERMODYNAMIC FUNCTIONS
# =============================================================================

def P_sat_bar(T_K):
    """
    Saturation vapour pressure of water [bar] using the Antoine equation.

    Antoine equation:  log10(P_sat) = A - B / (T + C)
    with T in Kelvin and P_sat in bar.
    Constants from NIST (valid ~274–441 K).
    """
    return 10.0 ** (5.40221 - 1838.675 / (T_K - 31.737))


def q_star_vec(T_K, p_arr, W0, E, n):
    """
    Equilibrium H₂O loading on the sorbent [mol/kg] from the
    Dubinin-Astakhov (DA) isotherm, evaluated at each pressure in p_arr [bar].

    DA isotherm:
        W  = W0 · exp( -(A/E)^n )          [m³_liquid / kg_sorbent]
        q* = (ρ_ads / MW_H₂O) · W          [mol / kg_sorbent]

    Adsorption potential A [J/kg]:
        A = (R/MW_H₂O) · T · ln(P_sat / p)

    A represents the free energy driving adsorption: it is zero at saturation
    (p = P_sat, no driving force) and grows as p drops below P_sat.
    The exponential in W converts this potential to an adsorbed volume.
    """
    p    = np.asarray(p_arr, dtype=float)
    Psat = P_sat_bar(T_K)

    # if p is over p_sat, water would condense rather than adsorb. When the partial pressure of a vapor in a gas mixture reaches the saturation pressure at that temperature, the vapor starts to condense.
    # Clip p to a safe range before taking log: p=0 would give log(∞),
    # and p≥P_sat means the gas is saturated so A=0 and W=W0 (full loading).
    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_raw  = (R_gas / MW_H2O) * T_K * np.log(Psat / p_safe)

    # Force A=0 at the boundaries so the isotherm returns the correct limits.
    A = np.where((p <= 0.0) | (p >= Psat), 0.0, A_raw)  

    # Cap the exponent at 500 to avoid numerical overflow (exp(-500) ≈ 0).
    W  = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    qs = rho_ads / MW_H2O * W

    # Return 0 loading where there is no H₂O present.
    return np.where(p <= 0.0, 0.0, qs)


def K_LDF_vec(T_K, p_arr, W0, E, n):
    """
    Linear Driving Force (LDF) mass transfer coefficient [1/s],
    from Bareschino (2023) Eq. 24 / Mette (2014).

    Physical meaning: K_LDF sets how fast the solid loading q approaches
    the equilibrium q*. A large K_LDF means fast adsorption (solid quickly
    saturates); a small K_LDF means slow adsorption (dispersed breakthrough).

    Formula (pore diffusion control):
        K_LDF = 15 · D_eff · MW_H₂O · ε_p
                ─────────────────────────────────────────────────
                0.5 · d_p² · τ · ρ_ads · R · T · (dq*/dP)

    where D_eff = D_M · ε_p / τ  (effective pore diffusivity),
    D_M is the molecular diffusivity of H₂O in the gas [m²/s].

    dq*/dP [mol/(kg·Pa)] is the local slope of the isotherm: it tells us
    how much extra loading we get per unit pressure increase. A steep
    isotherm (large slope) means diffusion is the bottleneck → lower K_LDF.
    We compute this slope numerically with a central-difference step of 1 Pa.
    """
    # Molecular diffusivity: Chapman-Enskog power-law approximation.
    # Scales as T^1.75 (kinetic theory); reference value 2.5e-5 m²/s at 300 K.
    D_M = 2.5e-5 * (T_K / 300.0) ** 1.75

    p      = np.asarray(p_arr, dtype=float)
    dp_bar = 1.0 / 1e5           # 1 Pa expressed in bar (central difference step)

    p_lo = np.maximum(p - dp_bar, 1e-15)   # lower point (clamp above zero)
    p_hi = p + dp_bar                       # upper point

    # Central difference: Δq* over a 2 Pa window → units [mol/kg / Pa]
    dqstar_dp = (q_star_vec(T_K, p_hi, W0, E, n)
                 - q_star_vec(T_K, p_lo, W0, E, n)) / 2.0

    # Where the isotherm is essentially flat (e.g. very low loading), the
    # slope → 0 which would blow up K_LDF. Floor it to avoid division by zero.
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)

    return (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_ads * R_gas * T_K * dqstar_dp))


# =============================================================================
# 3. COLUMN MODEL — PDE RIGHT-HAND SIDE
# =============================================================================

def rhs_column(t, y, T_K, u, C_in, W0, E, n):
    """
    Time derivatives for the 1D transient adsorption column.

    The state vector y has length 2N:
        y[:N]  = C [mol/m³]  — gas-phase H₂O concentration at each node
        y[N:]  = q [mol/kg]  — solid-phase H₂O loading at each node

    Governing equations at each spatial node:

      Gas balance (plug flow with adsorption sink):
          ε_b · dC/dt = -u · (C_i - C_{i-1}) / Δz  -  ρ_bed · dq/dt
          [void storage] = [convection in - out]      [adsorption sink]

      Solid balance (LDF):
          dq/dt = K_LDF · (q* - q)
          [loading change] = [driving force toward equilibrium]

    Spatial discretisation: first-order upwind differences (C_{i-1} is the
    upstream neighbour). Upwind is numerically stable for convection-dominated
    flow and avoids spurious oscillations near the adsorption front.
    """
    # Clip to zero: concentrations and loadings cannot be negative physically.
    C = np.maximum(y[:N], 0.0)
    q = np.maximum(y[N:], 0.0)

    # Convert gas-phase concentration to partial pressure using ideal gas law:
    # p [bar] = C [mol/m³] · R·T [J/mol] / 1e5 [Pa/bar]
    p = C * R_gas * T_K / 1e5

    # Equilibrium loading and LDF coefficient at each node's local pressure.
    qs = q_star_vec(T_K, p, W0, E, n)
    Kl = K_LDF_vec(T_K, p, W0, E, n)

    # Solid loading rate: positive when q < q* (still adsorbing).
    dqdt = Kl * (qs - q)

    # Build the upstream concentration array for the upwind scheme:
    # node 0 uses C_in (inlet boundary); node i uses C[i-1] (upstream node).
    C_up  = np.concatenate([[C_in], C[:-1]])
    dCdt  = (-u * (C - C_up) / dz - rho_bed * dqdt) / eps_b

    return np.concatenate([dCdt, dqdt])


def make_bt_event(C_in):
    """
    Create the breakthrough detection event for the ODE solver.

    solve_ivp stops integration when this function crosses zero from below,
    i.e. when the outlet concentration (last node) first reaches 10% of inlet.
    Setting terminal=True ensures the solver stops at that moment.
    """
    def event(t, y, T_K, u, C_in, W0, E, n):
        return y[N - 1] - BT_FRACTION * C_in   # = 0 when C_out = 10% · C_in

    event.terminal  = True    # stop integration when event triggers
    event.direction = 1       # only trigger on rising signal (C_out increasing)
    return event


# =============================================================================
# 4. RUN SIMULATIONS
# =============================================================================

results = {}   # stores breakthrough capacities [mmol/g] for each parameter set

for label, params in PARAM_SETS.items():
    W0, E, n = params['W0'], params['E'], params['n']
    caps = []   # capacities at each temperature for this parameter set

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  {'T [°C]':>6}  {'q* [mol/kg]':>12}  {'K_LDF [1/s]':>12}  {'capacity [mmol/g]':>18}")
    print(f"  {'-'*55}")

    for T_C in temperatures_C:
        T_K  = T_C + 273.15

        # Actual gas velocity at operating temperature (ideal gas, constant P).
        u = u_STP * (T_K / T_STP)

        # Inlet H₂O concentration from ideal gas law: C = p / (R·T).
        C_in = y_H2O_in * P_Pa / (R_gas * T_K)     # [mol/m³]

        # Inlet H₂O partial pressure and corresponding equilibrium values.
        p_in = y_H2O_in * P_bar
        qs0  = float(q_star_vec(T_K, np.array([p_in]), W0, E, n)[0])
        Kl0  = float(K_LDF_vec(T_K,  np.array([p_in]), W0, E, n)[0])

        # Estimate breakthrough time from the equilibrium front velocity.
        # The adsorption front moves at u_front ≈ u · ε_b / (ε_b + ρ_bed·dq*/dC).
        # Simplified: t_bt ≈ total moles adsorbed at equilibrium / molar flux in.
        # This gives a sensible upper bound for how long to integrate.
        t_bt_est = qs0 * m_cat / (C_in * u * A_b) if C_in > 0 else 1e5
        t_end    = min(5.0 * t_bt_est, 3e4)   # cap at 30 000 s as a safety limit

        # Initial condition: bed starts clean (no H₂O in gas or solid phase).
        y0    = np.zeros(2 * N)
        event = make_bt_event(C_in)

        # BDF (Backward Differentiation Formula) is a stiff-aware solver.
        # It is necessary here because K_LDF and the advection term create
        # very different time scales that would cause explicit solvers to
        # take extremely small time steps.
        sol = solve_ivp(
            rhs_column,
            t_span=[0.0, t_end],
            y0=y0,
            args=(T_K, u, C_in, W0, E, n),
            method='BDF',
            events=event,
            rtol=1e-4,
            atol=1e-8,
        )

        if not sol.success:
            print(f"  {T_C:>6}  SOLVER FAILED: {sol.message}")
            caps.append(np.nan)
            continue

        # Breakthrough capacity = mean solid loading across the bed at the
        # moment breakthrough is detected. This equals the total moles adsorbed
        # divided by the sorbent mass, which is what Bareschino Fig. 1(a) plots.
        # Units: mol/kg = mmol/g, so no conversion needed.
        q_final  = sol.y[N:, -1]              # solid loading at all N nodes
        capacity = float(np.mean(q_final))    # bed-average loading [mol/kg = mmol/g]
        caps.append(capacity)

        print(f"  {T_C:>6}  {qs0:>12.3f}  {Kl0:>12.4f}  {capacity:>18.3f}")

    results[label] = caps


# =============================================================================
# 5. PLOT — reproduce Bareschino et al. (2023) Fig. 1(a)
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 5))

styles = {
    'Mette (2014)':       {'color': 'tab:blue',   'marker': 's', 'ms': 8},
    'Kiefer (2022)':      {'color': 'tab:orange',  'marker': '*', 'ms': 10},
    'Ligtenberg (2026)':  {'color': 'tab:green',   'marker': '^', 'ms': 8},
}

# Plot simulated breakthrough capacities for each DA parameter set.
for label, caps in results.items():
    s = styles[label]
    ax.plot(temperatures_C, caps,
            color=s['color'], marker=s['marker'], markersize=s['ms'],
            linewidth=1.5, label=f'mod. ({label})')

# Overlay the experimental data points from Wei et al. (2021a).
ax.scatter(wei_exp_T, wei_exp_cap,
           color='k', marker='o', s=60, zorder=5,
           label='exp. (Wei et al. 2021a)')

ax.set_xlabel('Temperature [°C]')
ax.set_ylabel('H₂O breakthrough capacity [mmol$_{H_2O}$/g$_{ads}$]')
ax.set_title('Bareschino et al. (2023) Fig. 1(a)\n'
             '1D isothermal model, 100% CO₂ conversion assumed')
ax.set_xlim(270, 330)
ax.set_ylim(0, 2.5)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = __file__.replace('adsorption_simulation.py', 'bareschino_fig1a.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {out_path}")
