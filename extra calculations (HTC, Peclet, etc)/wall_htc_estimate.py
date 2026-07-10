"""
Bed-to-wall heat transfer coefficient for the MPB reactor
==========================================================

Goal: estimate h_bw [W/(m2·K)] — the heat transfer coefficient between the
packed bed and the reactor tube wall — and convert it to the volumetric
coefficient U_a [W/(m3_bed·K)] used in the MPB model energy balance.

Heat flows from the bed to the coolant through three resistances in series:

    1/U_total = 1/h_bw  +  t_wall/k_metal  +  1/h_coolant
                  ^              ^                  ^
             bed-to-wall    wall conduction     coolant film
            (bottleneck)     (negligible)        (small)

PRIMARY METHOD — Bareschino et al. (2023), Eq. S.10  (Specchia-type, two-term):

    h_bw = (k_gas/d_p) * [2*eps_b + (1-eps_b) / (0.0024*(d_t/d_p)^1.58
                                                   + (1/3)*(k_gas/k_sol))]
         + 0.0835 * (k_gas/d_p) * Re_p^0.91

    Static term  — stagnant bed conduction through gas voids and solid
                   contact chains. Dominant at low Re (your case: Re_p ~ 3).
    Dynamic term — convective contribution (Specchia et al. 1980);
                   0.0835 * Re_p^0.91, valid for Re_p < 190.

COMPARISON — Peters, Schiffino & Harriott (1988):

    Nu_w = 0.19 * Re_p^0.79     (empirical, Re_p = 0.1-40, dynamic only)

    Does not include the stagnant conduction term, so it underestimates
    h_bw at low Re where conduction dominates.

Volumetric coefficient:
    U_a = h_bw * (4/d_t)   [W/(m3_bed·K)]
    — wall area per unit bed volume for a cylinder: A/V = 4/d
"""

import numpy as np

# =============================================================================
# 1. REACTOR GEOMETRY  (Bareschino lab setup, same as MPB model)
# =============================================================================
d_b    = 0.050      # [m]    reactor inner diameter (tube diameter d_t)
d_p    = 0.75e-3    # [m]    particle diameter (13X zeolite pellets)
t_wall = 0.003      # [m]    estimated wall thickness (3 mm stainless steel)
eps_b  = 0.40       # [-]    bed void fraction

# Solid thermal conductivity of 13X zeolite pellets
# Literature range: 0.1 – 0.5 W/(m·K). Pellets are porous so lower bound
# applies; 0.15 W/(m·K) is a representative value for zeolite 13X pellets.
k_sol  = 0.4       # [W/(m·K)]

a_wall = 4.0 / d_b  # [m2/m3]  wall surface per unit bed volume (= 4/d for cylinder)
print(f"Wall surface-to-volume ratio:  a = {a_wall:.1f} m2/m3")
print(f"d_t / d_p  = {d_b/d_p:.1f}  (tube-to-particle diameter ratio)")

# =============================================================================
# 2. OPERATING CONDITIONS
# =============================================================================
T_C   = 280.0           # [degC]  representative mid-range temperature
T_K   = T_C + 273.15    # [K]
P_bar = 1.0             # [bar]
P_Pa  = P_bar * 1e5     # [Pa]

# Feed composition (same as MPB model)
y_CH4 = 0.80
y_H2  = 0.16
y_CO2 = 0.04

# Gas superficial velocity from GHSV (same parameters as MPB model)
M_ads   = 1.22          # [kg]   adsorbent mass
A_b     = np.pi/4 * d_b**2   # [m2]
GHSV    = 0.5           # [m3_STP/(kg_ads·h)]
T_STP   = 273.15        # [K]
Q_STP   = GHSV * M_ads / 3600.0
u_g_STP = Q_STP / A_b                  # [m/s]  at STP
u_g     = u_g_STP * (T_K / T_STP)     # [m/s]  at operating T

print(f"\nOperating conditions: T = {T_C} degC,  P = {P_bar} bar")
print(f"u_g (STP)     = {u_g_STP*1e3:.1f} mm/s")
print(f"u_g ({T_C:.0f} degC) = {u_g*1e3:.1f} mm/s  (thermal expansion of ideal gas)")

# =============================================================================
# 3. GAS MIXTURE PROPERTIES  at operating temperature
# =============================================================================
# Thermal conductivity of pure components at ~250 degC
# CH4: ~0.045 W/(m·K),  H2: ~0.230 W/(m·K),  CO2: ~0.025 W/(m·K)
# Mixture weighted by mole fractions — approximate but sufficient for h_bw estimate.
k_CH4 = 0.045;  k_H2 = 0.230;  k_CO2 = 0.025   # [W/(m·K)]
k_gas = y_CH4*k_CH4 + y_H2*k_H2 + y_CO2*k_CO2
# H2 has exceptionally high k and pulls the mixture well above pure CH4.
print(f"\nGas thermal conductivity (mole-fraction weighted): k_gas = {k_gas:.4f} W/(m·K)")

# Dynamic viscosity of pure components at ~250 degC
mu_CH4 = 1.50e-5;  mu_H2 = 1.10e-5;  mu_CO2 = 2.00e-5   # [Pa·s]
mu_gas = y_CH4*mu_CH4 + y_H2*mu_H2 + y_CO2*mu_CO2

# Gas density (ideal gas law)
R_gas   = 8.314
MW_mix  = y_CH4*16e-3 + y_H2*2e-3 + y_CO2*44e-3   # [kg/mol]
rho_gas = (P_Pa / (R_gas * T_K)) * MW_mix

print(f"Gas dynamic viscosity:  mu_gas  = {mu_gas:.2e} Pa·s")
print(f"Gas density:            rho_gas = {rho_gas:.3f} kg/m3")
print(f"Solid conductivity:     k_sol   = {k_sol:.3f} W/(m·K)  (zeolite 13X pellet)")

# =============================================================================
# 4. PARTICLE REYNOLDS NUMBER
# =============================================================================
# Re_p = rho * u_g * d_p / mu  — based on superficial velocity and particle diameter.
# Controls how much convective radial mixing occurs. At Re_p < ~10, convection
# is negligible and the stagnant conduction term dominates h_bw.
Re_p = rho_gas * u_g * d_p / mu_gas
Pr   = mu_gas * 1500 / k_gas   # approximate Prandtl (Cp_mix ~ 1500 J/kg·K)
print(f"\nParticle Reynolds number:  Re_p = {Re_p:.2f}")
print(f"Prandtl number:            Pr   = {Pr:.2f}")
print(f"Regime: {'low Re  -> stagnant conduction dominates' if Re_p < 10 else 'moderate Re'}")

# =============================================================================
# 5. BED-TO-WALL HTC — Bareschino (2023) Eq. S.10  [Specchia-type, two-term]
# =============================================================================
# Reference: Bareschino et al. (2023), Numerical modelling of a
# sorption-enhanced methanation system, supplementary Eq. S.10.
# Originally based on Specchia, Baldi & Sicardi (1980),
# Chem. Eng. Commun. 4, 361-380.
#
# h_bw = (k_gas/d_p) * STATIC  +  (k_gas/d_p) * DYNAMIC
#
# STATIC term: stagnant bed conduction
#   accounts for heat flowing through (a) gas in void spaces and
#   (b) solid contact chains between particles, including the near-wall
#   layer where voidage is higher than in the bed interior.
#   The term (d_t/d_p)^1.58 captures the wall-geometry effect: for larger
#   d_t/d_p, fewer particles are in direct wall contact per unit area.
#
# DYNAMIC term: convective contribution from flowing gas
#   Specchia correlation: 0.0835 * Re_p^0.91, valid for Re_p < 190.
#   At Re_p ~ 3 this is small compared to the static term.

ratio_dt_dp  = d_b / d_p
denom_static = 0.0024 * ratio_dt_dp**1.58  +  (1/3) * (k_gas / k_sol)
bracket_static = 2*eps_b  +  (1 - eps_b) / denom_static

h_bw_static  = (k_gas / d_p) * bracket_static
h_bw_dynamic = 0.0835 * (k_gas / d_p) * Re_p**0.91
h_bw         = h_bw_static + h_bw_dynamic

print(f"\n--- Bareschino / Specchia correlation ---")
print(f"  d_t/d_p              = {ratio_dt_dp:.1f}")
print(f"  denom (static term)  = 0.0024*{ratio_dt_dp:.1f}^1.58 + (1/3)*(k_gas/k_sol)"
      f" = {denom_static:.3f}")
print(f"  bracket (static)     = 2*eps_b + (1-eps_b)/denom = {bracket_static:.3f}")
print(f"  h_bw static          = {h_bw_static:.1f}  W/(m2·K)  "
      f"({100*h_bw_static/h_bw:.0f}% of total)")
print(f"  h_bw dynamic         = {h_bw_dynamic:.1f}  W/(m2·K)  "
      f"({100*h_bw_dynamic/h_bw:.0f}% of total)")
print(f"  h_bw TOTAL           = {h_bw:.1f}  W/(m2·K)")

# =============================================================================
# 6. COMPARISON WITH PETERS & HARRIOTT (1988)
# =============================================================================
# Peters, Schiffino & Harriott, Ind. Eng. Chem. Res. 27(2), 226-233.
# Purely empirical fit to dynamic (flow) data only — no stagnant term.
# Underestimates h_bw at low Re_p where conduction through the stagnant
# bed is the dominant mechanism.
Nu_w_PH  = 0.19 * Re_p**0.79
h_bw_PH  = Nu_w_PH * k_gas / d_p
print(f"\n--- Peters & Harriott correlation (for comparison) ---")
print(f"  Nu_w = 0.19 * Re_p^0.79 = {Nu_w_PH:.3f}")
print(f"  h_bw = {h_bw_PH:.1f}  W/(m2·K)")
print(f"  Ratio Specchia/PH = {h_bw/h_bw_PH:.1f}x  "
      f"(PH underestimates because it has no stagnant term)")

# =============================================================================
# 7. OTHER RESISTANCES IN THE SERIES CHAIN
# =============================================================================
k_metal   = 16.0    # [W/(m·K)]  stainless steel
h_cool    = 300.0   # [W/(m2·K)] circulating Dowtherm/oil jacket (representative)
# h_cool >> h_bw, so the coolant film contributes only a small fraction.
# Any circulating liquid thermostat achieves this easily; the bed-to-wall
# resistance always dominates for gas-phase packed beds.

R_bed  = 1.0 / h_bw
R_wall = t_wall / k_metal
R_cool = 1.0 / h_cool
R_tot  = R_bed + R_wall + R_cool
U_ov   = 1.0 / R_tot

print(f"\nThermal resistance breakdown  (Specchia h_bw = {h_bw:.0f} W/m2K):")
print(f"  R_bed-wall  = {R_bed:.5f}  m2·K/W  ({100*R_bed/R_tot:.0f}%)")
print(f"  R_wall      = {R_wall:.5f}  m2·K/W  ({100*R_wall/R_tot:.0f}%)")
print(f"  R_coolant   = {R_cool:.5f}  m2·K/W  ({100*R_cool/R_tot:.0f}%)")
print(f"  U_overall   = {U_ov:.1f}  W/(m2·K)")

# =============================================================================
# 8. VOLUMETRIC COEFFICIENT  U_a  used in the MPB energy balance
# =============================================================================
# The 1-D energy balance integrates heat removal over the bed volume:
#   Q_wall / V_bed = U_a * (T - T_wall)
# where  U_a = U_overall * (4/d_t)  is the wall area per unit bed volume.

U_a_specchia = U_ov * a_wall
U_a_PH       = (1.0/(1/h_bw_PH + R_wall + R_cool)) * a_wall
U_a_model    = 2000.0   # [W/(m3·K)]  value hardcoded in MPB model

print(f"\nVolumetric cooling coefficient  U_a = U_overall * (4/d_t):")
print(f"  a_wall            = {a_wall:.1f}  m2/m3  (= 4 / d_b)")
print(f"  U_a  Specchia     = {U_a_specchia:.0f}  W/(m3·K)")
print(f"  U_a  Peters-H.    = {U_a_PH:.0f}  W/(m3·K)")
print(f"  U_a  MPB model    = {U_a_model:.0f}  W/(m3·K)  (hardcoded)")
print(f"  Model is {U_a_specchia/U_a_model:.1f}x conservative relative to Specchia estimate")

# =============================================================================
# 9. SENSITIVITY: vary temperature (200-340 degC)
# =============================================================================
print(f"\nSensitivity over operating temperature range  (Specchia formula):")
print(f"  {'T [C]':>7}  {'u_g [mm/s]':>11}  {'Re_p':>6}  "
      f"{'h_static':>10}  {'h_dynamic':>11}  {'h_bw':>8}  {'U_a':>10}")

for T_test in range(200, 341, 20):
    T_K_t   = T_test + 273.15
    u_g_t   = u_g_STP * (T_K_t / T_STP)
    rho_t   = (P_Pa / (R_gas * T_K_t)) * MW_mix
    mu_t    = mu_gas  * (T_K_t / T_K)**0.7    # Sutherland scaling ~T^0.7
    k_g_t   = k_gas   * (T_K_t / T_K)**0.8    # gas k scales ~T^0.8
    Re_t    = rho_t * u_g_t * d_p / mu_t

    # Specchia formula at temperature T_test
    denom_t   = 0.0024 * ratio_dt_dp**1.58  +  (1/3) * (k_g_t / k_sol)
    bracket_t = 2*eps_b  +  (1-eps_b) / denom_t
    h_stat_t  = (k_g_t / d_p) * bracket_t
    h_dyn_t   = 0.0835 * (k_g_t / d_p) * Re_t**0.91
    h_bw_t    = h_stat_t + h_dyn_t
    U_a_t     = (1.0 / (1/h_bw_t + t_wall/k_metal + 1/h_cool)) * a_wall

    print(f"  {T_test:>7}  {u_g_t*1e3:>11.1f}  {Re_t:>6.2f}  "
          f"{h_stat_t:>10.1f}  {h_dyn_t:>11.1f}  {h_bw_t:>8.1f}  {U_a_t:>10.0f}")
