"""
Validation — Wei (2022) Figure 5.10
=====================================

Reproduces the sorption-enhanced CO2 methanation breakthrough curve from
Wei's thesis Fig. 5.10:
  "Typical water breakthrough capacity and duration of bifunctional
   catalyst-sorbent 5%Ni2.5%Ce13X. Experiment at 240 °C, P = 1 bar,
   GHSV = 923 mL/(g_cat·h). Feed: 10 mL/min H2, 2.5 mL/min CO2,
   81.5 mL/min CH4, 6 mL/min N2."

Same coupled SEM column model as SEM LHHW.py (LHHW kinetics + DA isotherm +
LDF mass transfer), run only for the sorption-enhanced case at 240 °C.

Plot (dual y-axis, matching Fig. 5.10):
    Left y-axis  :  H2, CO2, H2O outlet mol%          (0 – 10 %)
    Right y-axis :  CH4 outlet mol% & CO2 conversion  (−10 – 110 %)
                    CO2 equilibrium conversion (flat dashed line)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# =============================================================================
# 1. PARAMETERS  (identical to SEM LHHW.py)
# =============================================================================

d_b     = 0.010                    # bed diameter                [m]
L_b     = 0.100                    # bed length                  [m]
A_b     = np.pi / 4 * d_b**2      # cross-sectional area        [m²]
V_bed   = A_b * L_b                # total bed volume            [m³]
m_cat   = 6.5e-3                   # catalyst/sorbent mass       [kg]
rho_bed = m_cat / V_bed            # bulk density                [kg/m³]
eps_b   = 0.40                     # void fraction               [-]

d_p     = 0.75e-3                  # particle diameter           [m]
eps_p   = 0.6                      # intraparticle void fraction [-]
tau_p   = 3.0                      # tortuosity                  [-]

rho_ads = 998.2                    # liquid water density        [kg/m³]
MW_H2O  = 0.018015                 # molar mass water            [kg/mol]
R_gas   = 8.314                    # gas constant                [J/(mol·K)]

# DA isotherm parameters (LIGTENBERG 2026, metter is wo = 340), fitted to Wei 300 °C breakthrough)
W0_DA   = 175.00e-6                #             [m³/kg]
E_DA    = 1192.25e3                # characteristic energy       [J/kg]
n_DA    = 1.55     

# Operating conditions
T_C     = 240
T_K     = T_C + 273.15
P_bar   = 1.0
P_Pa    = P_bar * 1e5

# Feed composition (Wei Fig. 5.10: 10/2.5/81.5/6 mL/min, total 100 mL/min)
y_CO2_in = 0.025
y_H2_in  = 0.100
y_CH4_in = 0.815
y_N2_in  = 0.060

Q_STP   = 100e-6 / 60             # volumetric flow STP         [m³/s]
T_STP   = 273.15                   # STP temperature             [K]
u       = (Q_STP / A_b) * (T_K / T_STP)  # superficial velocity [m/s]

# LHHW kinetic parameters (Koschany et al. 2016)
T_ref_K = 555.0
k_ref   = 3.46e-4
Ea_k    = 77.5e3
A_OH    = 0.50;  dH_OH   = 22.4e3
A_H2    = 0.44;  dH_H2   = -6.2e3
A_mix   = 0.88;  dH_mix  = -10.0e3

P_FLOOR = 1e-4                     # partial-pressure floor      [bar]

# Spatial discretisation
N  = 50
dz = L_b / (N - 1)


# =============================================================================
# 2. THERMODYNAMIC AND KINETIC FUNCTIONS  (copied unchanged from SEM LHHW.py)
# =============================================================================

def P_sat_bar(T):
    return 10.0 ** (5.40221 - 1838.675 / (T - 31.737))


def q_star_vec(T, p_arr, W0, E, n):
    p      = np.asarray(p_arr, dtype=float)
    Psat   = P_sat_bar(T)
    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_raw  = (R_gas / MW_H2O) * T * np.log(Psat / p_safe)
    A      = np.where((p <= 0.0) | (p >= Psat), 0.0, A_raw)
    W      = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    return np.where(p <= 0.0, 0.0, rho_ads / MW_H2O * W)


def K_LDF_vec(T, p_arr, W0, E, n):
    D_M       = 2.5e-5 * (T / 300.0) ** 1.75
    p         = np.asarray(p_arr, dtype=float)
    dp_bar    = 1.0 / 1e5
    p_lo      = np.maximum(p - dp_bar, 1e-15)
    p_hi      = p + dp_bar
    dqstar_dp = (q_star_vec(T, p_hi, W0, E, n)
                 - q_star_vec(T, p_lo, W0, E, n)) / 2.0
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)
    return (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_ads * R_gas * T * dqstar_dp))


def K_eq_sabatier(T):
    return 137.0 * T**(-3.994) * np.exp(158700.0 / (R_gas * T))


def reaction_rate_SI(T, p_CO2, p_H2, p_CH4, p_H2O):
    vH      = lambda dH: np.exp(-dH / R_gas * (1.0 / T - 1.0 / T_ref_K))
    k       = k_ref * np.exp(-Ea_k / R_gas * (1.0 / T - 1.0 / T_ref_K))
    K_OH    = A_OH  * vH(dH_OH)
    K_H2    = A_H2  * vH(dH_H2)
    K_mix   = A_mix * vH(dH_mix)
    K_eq    = K_eq_sabatier(T)
    p_CO2_s = np.maximum(p_CO2, P_FLOOR)
    p_H2_s  = np.maximum(p_H2,  P_FLOOR)
    beta    = (p_CH4 * p_H2O**2) / (K_eq * p_CO2_s * p_H2_s**4)
    f_eq    = np.maximum(1.0 - beta, 0.0)
    DEN     = (1.0
               + K_OH  * np.maximum(p_H2O, 0.0) / p_H2_s**0.5
               + K_H2  * p_H2_s**0.5
               + K_mix * p_CO2_s**0.5)
    return k * (p_CO2_s * p_H2_s)**0.5 * f_eq / DEN**2 * 1000.0


# =============================================================================
# 3. RHS  (sorption-enhanced only — se_on always True here)
# =============================================================================

def rhs_sem(t, y, T, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O):
    C_CO2 = np.maximum(y[0*N : 1*N], 0.0)
    C_H2  = np.maximum(y[1*N : 2*N], 0.0)
    C_CH4 = np.maximum(y[2*N : 3*N], 0.0)
    C_H2O = np.maximum(y[3*N : 4*N], 0.0)
    q     = np.maximum(y[4*N : 5*N], 0.0)

    p_CO2 = C_CO2 * R_gas * T / 1e5
    p_H2  = C_H2  * R_gas * T / 1e5
    p_CH4 = C_CH4 * R_gas * T / 1e5
    p_H2O = C_H2O * R_gas * T / 1e5

    r    = reaction_rate_SI(T, p_CO2, p_H2, p_CH4, p_H2O)
    qs   = q_star_vec(T, p_H2O, W0_DA, E_DA, n_DA)
    Kl   = K_LDF_vec(T,  p_H2O, W0_DA, E_DA, n_DA)
    dqdt = Kl * (qs - q)

    C_CO2_up = np.concatenate([[C_in_CO2], C_CO2[:-1]])
    C_H2_up  = np.concatenate([[C_in_H2],  C_H2[:-1]])
    C_CH4_up = np.concatenate([[C_in_CH4], C_CH4[:-1]])
    C_H2O_up = np.concatenate([[C_in_H2O], C_H2O[:-1]])

    dCdt_CO2 = (-u * (C_CO2 - C_CO2_up) / dz + rho_bed * (-1) * r) / eps_b
    dCdt_H2  = (-u * (C_H2  - C_H2_up)  / dz + rho_bed * (-4) * r) / eps_b
    dCdt_CH4 = (-u * (C_CH4 - C_CH4_up) / dz + rho_bed * (+1) * r) / eps_b
    dCdt_H2O = (-u * (C_H2O - C_H2O_up) / dz + rho_bed * (+2) * r
                                              - rho_bed * dqdt) / eps_b

    return np.concatenate([dCdt_CO2, dCdt_H2, dCdt_CH4, dCdt_H2O, dqdt])


# =============================================================================
# 4. SOLVE
# =============================================================================

C_total  = P_Pa / (R_gas * T_K)   # total molar concentration at T, P [mol/m³]
C_in_CO2 = y_CO2_in * C_total
C_in_H2  = y_H2_in  * C_total
C_in_CH4 = y_CH4_in * C_total
C_in_H2O = 0.0

t_end = 7200.0                     # 120 min

y0 = np.zeros(5 * N)
y0[0*N : 1*N] = C_in_CO2
y0[1*N : 2*N] = C_in_H2
y0[2*N : 3*N] = C_in_CH4

GHSV = Q_STP * 3600 * 1e6 / (m_cat * 1e3)
print(f"Validating Wei Fig. 5.10:  T = {T_C} °C,  P = {P_bar} bar,  "
      f"GHSV = {GHSV:.0f} mL/(g·h)")
print(f"Solving sorption-enhanced case for {t_end/60:.0f} min ...")

sol = solve_ivp(
    rhs_sem,
    t_span=[0.0, t_end],
    y0=y0,
    args=(T_K, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O),
    method='BDF',
    rtol=1e-4,
    atol=1e-8,
    dense_output=True,
)
if not sol.success:
    raise RuntimeError(f"ODE solver failed: {sol.message}")
print("Done.")


# =============================================================================
# 5. POST-PROCESS
# =============================================================================

t_plot = np.linspace(0.0, t_end, 400)
y_out  = sol.sol(t_plot)

# Outlet concentrations: last spatial node of each species block
C_CO2_out = np.maximum(y_out[  N - 1, :], 0.0)
C_H2_out  = np.maximum(y_out[2*N - 1, :], 0.0)
C_CH4_out = np.maximum(y_out[3*N - 1, :], 0.0)
C_H2O_out = np.maximum(y_out[4*N - 1, :], 0.0)

# Mol% — N2 is inert and conserved, so its concentration is fixed at C_N2_in.
# Dividing by the actual gas-phase total (tracked species + fixed N2) gives the
# physically correct mole fractions that a GC would measure.
C_N2_in   = y_N2_in * C_total                                         # constant [mol/m³]
C_gas_out = C_CO2_out + C_H2_out + C_CH4_out + C_H2O_out + C_N2_in  # actual gas-phase total

pct_CO2 = C_CO2_out / C_gas_out * 100
pct_H2  = C_H2_out  / C_gas_out * 100
pct_CH4 = C_CH4_out / C_gas_out * 100
pct_H2O = C_H2O_out / C_gas_out * 100

# CO2 conversion
X_CO2 = np.clip((C_in_CO2 - C_CO2_out) / C_in_CO2, 0.0, 1.0) * 100

# Thermodynamic equilibrium conversion for this feed at 240 °C (gas-phase,
# no sorption).  Formula derived from Q = K_eq with 5-species mole balance.
def equilibrium_conversion_pct(T_K_val):
    K  = K_eq_sabatier(T_K_val)
    def f(X):
        return ((0.815 + 0.025*X) * 0.0025 * X**2 * (1 - 0.05*X)**2
                / (2.5e-6 * (1 - X)**5) - K)
    try:
        return brentq(f, 1e-9, 1 - 1e-9) * 100
    except Exception:
        return 100.0

X_eq  = equilibrium_conversion_pct(T_K)
t_min = t_plot / 60

print(f"CO2 equilibrium conversion at {T_C} °C: {X_eq:.1f} %")
print(f"CH4 mol% at full CO2 conversion (N2 dilution): "
      f"{(y_CH4_in + y_CO2_in) / (1 - y_CO2_in) * 100:.1f} %  "
      f"(Wei caption: 94 %)")


# =============================================================================
# 6. PLOT — matching Wei Fig. 5.10 dual-axis style
# =============================================================================

fig, ax1 = plt.subplots(figsize=(9, 6))

me = 20   # marker every N-th point (400 total → ~20 visible markers per line)

# --- Left y-axis: H2, CO2, H2O (mol%) ---
h_H2,  = ax1.plot(t_min, pct_H2,  'k-',
                   marker='D', markevery=me, ms=5, lw=1.8,
                   label='H₂ concentration')
h_CO2, = ax1.plot(t_min, pct_CO2, color='tab:red', lw=1.8,
                   marker='o', markevery=me, ms=5,
                   label='CO₂ concentration')
h_H2O, = ax1.plot(t_min, pct_H2O, color='olive', lw=2.2,
                   label='H₂O concentration')

ax1.set_xlabel('Time (min)', fontsize=12)
ax1.set_ylabel('H₂, CO₂, H₂O concentration (mol%)', fontsize=11)
ax1.set_xlim(0, 70)
ax1.set_ylim(-0.5, 10.5)
ax1.set_yticks([0, 2, 4, 6, 8, 10])

# --- Right y-axis: CH4 mol% and CO2 conversion ---
ax2 = ax1.twinx()

h_CH4, = ax2.plot(t_min, pct_CH4,
                   color='tab:blue', marker='o', markerfacecolor='none',
                   markevery=me, ms=7, lw=1.8,
                   label='CH₄ concentration')
h_Xeq, = ax2.plot([0, 70], [X_eq, X_eq],
                   color='m', lw=1.5, ls='--',
                   label=f'CO₂ equilibrium conversion ({X_eq:.0f} %)')
h_X,   = ax2.plot(t_min, X_CO2,
                   color='m', marker='s', markevery=me, ms=6, lw=1.8,
                   label='CO₂ conversion')

ax2.set_ylabel('CH₄ concentration (mol%)  /  CO₂ conversion (%)', fontsize=11)
ax2.set_ylim(-10, 110)
ax2.set_yticks([0, 20, 40, 60, 80, 100])

# --- Combined legend (left axis entries first, matching Wei's order) ---
all_handles = [h_CH4, h_Xeq, h_X, h_H2, h_CO2, h_H2O]
ax2.legend(handles=all_handles, loc='center right', fontsize=9, framealpha=0.85)

ax1.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('validation_fig5_10_wei_isothermal.png', dpi=150, bbox_inches='tight')
plt.show()
