"""
Ergun pressure drop estimate for the MPB reactor.

Parameters taken from MPB_flux_form high mass.py:
  - bed geometry: d_b = 5 cm, L = 2 m, eps_b = 0.4
  - particle:     d_p = 2.5 mm, rho_p = 1400 kg/m3 (Bareschino)
  - mass loading: derived from full bed packing (not fixed at Bareschino 1.22 kg)
  - flow:         GHSV = 0.5 m3_STP / (kg_ads * h), T = 280 C, P = 1 bar
"""

import numpy as np

# -------------------------------------------------------------------------
# Bed geometry
# -------------------------------------------------------------------------
d_b   = 0.050           # tube inner diameter              [m]
L_b   = 2.000           # bed length                       [m]
A_b   = np.pi / 4 * d_b**2    # bed cross-sectional area  [m2]
V_bed = A_b * L_b       # total bed volume                 [m3]
eps_b = 0.40            # bed void fraction (inter-particle)

# -------------------------------------------------------------------------
# Particle properties — 5%Ni 2.5%Ce / 13X zeolite bi-functional pellets
# -------------------------------------------------------------------------
d_p   = 2.5e-3          # particle diameter                [m]
rho_p = 1400            # particle (skeletal) density      [kg/m3]  Bareschino

# -------------------------------------------------------------------------
# Catalyst and sorbent masses — derived from full packing of the bed
# (this is the "high mass" consistent approach; Bareschino used M_ads = 1.22 kg)
# -------------------------------------------------------------------------
f_Ni  = 0.050           # Ni mass fraction in pellet
f_Ce  = 0.025           # Ce mass fraction in pellet
f_ads = 1.0 - f_Ni - f_Ce    # zeolite 13X fraction = 0.925

# Total solid mass that fills the bed at eps_b = 0.4
M_total = V_bed * (1 - eps_b) * rho_p   # [kg]
M_cat   = (f_Ni + f_Ce) * M_total       # [kg] catalytic metal fraction
M_ads   = f_ads * M_total               # [kg] adsorbent (zeolite 13X) fraction

print(f"Bed packing: M_total = {M_total:.3f} kg  "
      f"(M_cat = {M_cat:.3f} kg, M_ads = {M_ads:.3f} kg)")

# -------------------------------------------------------------------------
# Operating conditions
# -------------------------------------------------------------------------
T     = 553.15          # operating temperature            [K]  (280 C)
P     = 1.0e5           # total pressure                   [Pa] (1 bar)
R_gas = 8.314           # ideal gas constant               [J/(mol*K)]

# Inlet mole fractions (stoichiometric H2:CO2 = 4:1 diluted in CH4 product)
y_CO2 = 0.04
y_H2  = 0.16
y_CH4 = 0.80

# -------------------------------------------------------------------------
# Gas mixture molar mass and density (ideal gas)
# -------------------------------------------------------------------------
MW = y_CH4 * 16e-3 + y_H2 * 2e-3 + y_CO2 * 44e-3   # [kg/mol]
rho_g = P * MW / (R_gas * T)                          # [kg/m3]

# -------------------------------------------------------------------------
# Gas mixture dynamic viscosity — Wilke mixing rule
# Component values at ~280 C from wall_htc_estimate.py
# -------------------------------------------------------------------------
mu_CH4, mu_H2, mu_CO2 = 1.50e-5, 1.10e-5, 2.00e-5   # [Pa*s]
MW_CH4, MW_H2, MW_CO2 = 16e-3,   2e-3,    44e-3      # [kg/mol]

ys  = np.array([y_CH4,  y_H2,  y_CO2])
mus = np.array([mu_CH4, mu_H2, mu_CO2])
MWs = np.array([MW_CH4, MW_H2, MW_CO2])

# Wilke interaction parameter phi_ij
phi = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        phi[i, j] = ((1 + (mus[i]/mus[j])**0.5 * (MWs[j]/MWs[i])**0.25)**2
                     / (8 * (1 + MWs[i]/MWs[j]))**0.5)

mu_mix = sum(ys[i] * mus[i] / sum(ys[j] * phi[i, j] for j in range(3))
             for i in range(3))

# -------------------------------------------------------------------------
# Superficial gas velocity at operating conditions
# GHSV defined per kg adsorbent at STP (0 C, 1 bar), corrected to T via ideal gas
# -------------------------------------------------------------------------
T_STP  = 273.15         # [K]
GHSV   = 0.5            # [m3_STP / (kg_ads * h)]

Q_STP  = GHSV * M_ads / 3600   # volumetric flow at STP   [m3/s]
u_STP  = Q_STP / A_b            # superficial velocity at STP  [m/s]
u0     = u_STP * T / T_STP      # corrected to operating T (same P) [m/s]

# -------------------------------------------------------------------------
# Ergun equation:  dP/dz = A*mu*u0 + B*rho*u0^2
#
# Blake-Kozeny (viscous, low Re):   A = 150*(1-eps)^2 / (dp^2 * eps^3)
# Burke-Plummer (inertial, high Re): B = 1.75*(1-eps) / (dp * eps^3)
# -------------------------------------------------------------------------
A_visc  = 150 * (1 - eps_b)**2 / (d_p**2 * eps_b**3)   # [m^-2]
B_inert = 1.75 * (1 - eps_b)  / (d_p    * eps_b**3)    # [m^-1]

dPdz        = A_visc * mu_mix * u0 + B_inert * rho_g * u0**2   # [Pa/m]
dP_total    = dPdz * L_b                                         # [Pa]
dP_mbar     = dP_total / 100                                     # [mbar]
dP_fraction = dP_total / P * 100                                 # [% of P]

# Modified particle Reynolds number (Ergun convention: uses (1-eps) in denominator)
Re_p = rho_g * u0 * d_p / (mu_mix * (1 - eps_b))

# Fraction of dP from viscous vs inertial term
visc_frac = A_visc * mu_mix * u0 / dPdz * 100

# -------------------------------------------------------------------------
# Results
# -------------------------------------------------------------------------
print()
print("=== Ergun pressure drop estimate ===")
print(f"  d_p         = {d_p*1e3:.1f} mm")
print(f"  M_ads       = {M_ads:.3f} kg  (full bed packing, 92.5 wt% zeolite)")
print(f"  u0          = {u0*1e3:.1f} mm/s  at {T-273.15:.0f} C")
print(f"  rho_g       = {rho_g:.4f} kg/m3")
print(f"  mu_mix      = {mu_mix*1e6:.2f} uPa*s  (Wilke)")
print(f"  Re_p        = {Re_p:.2f}  (modified Ergun convention)")
print()
print(f"  dP/dz       = {dPdz:.1f} Pa/m")
print(f"  dP total    = {dP_total:.1f} Pa  =  {dP_mbar:.2f} mbar")
print(f"  dP/P        = {dP_fraction:.3f}%  of total pressure")
print()
print(f"  Viscous term = {visc_frac:.1f}%  of total dP")
print(f"  (Re_p < 2 = pure Darcy; Re_p > 1000 = pure inertial; 38 = mixed regime)")