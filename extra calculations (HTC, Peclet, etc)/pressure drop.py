"""
Ergun pressure drop estimate for the MPB reactor.

Parameters taken from "MPB_flux_form active frac.py" (updated densities, 20% active
catalyst) -- same density structure used in peclet_numbers.py:
  - bed geometry: d_b = 5 cm, L = 2 m, eps_b = 0.4
  - particle:     d_p = 2.5 mm, rho_p = 1400 kg/m3 (Bareschino)
  - mass loading: 5wt%Ni-2.5wt%Ce/13X bifunctional pellet (only active_fraction of its
                  mass is catalytically active) + inert filler topping the bed to eps_b
  - flow:         GHSV = 1.0 m3_STP / (kg_ads * h), T = 280 C, P = 1 bar
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
rho_p = 1400.0          # particle (skeletal) density      [kg/m3]  Bareschino

# -------------------------------------------------------------------------
# Catalyst / sorbent / filler masses — matches MPB_flux_form active frac.py:
# a single bifunctional pellet carries both functions; only active_fraction of its
# mass is catalytically active (Ni); inert filler tops the bed up to eps_b packing.
# -------------------------------------------------------------------------
bifunctional_mass = 0.4    # [kg]  mass of 5wt%Ni-2.5wt%Ce/13X bifunctional material
M_zeolite_added   = 0.0    # [kg]  additional pure 13X zeolite (sorbent-only, no Ni)
active_fraction   = 0.20   # [-]   fraction of bifunctional material mass that is catalytically active

M_ads        = bifunctional_mass * 0.925 + M_zeolite_added   # [kg] sorbent mass (92.5% of bifunctional material + zeolite added)
M_cat_active = bifunctional_mass * active_fraction            # [kg] active catalyst mass
M_solid_phys = bifunctional_mass + M_zeolite_added             # [kg] physical solid mass before filler
M_filler     = (1 - eps_b) * V_bed * rho_p - M_solid_phys       # [kg] inert filler, tops bed to eps_b packing at rho_p

rho_bed_cat  = M_cat_active / V_bed              # [kg_cat/m³_bed]
rho_bed_ads  = M_ads / V_bed                     # [kg_ads/m³_bed]
rho_bed_fill = M_filler / V_bed                  # [kg_fill/m³_bed]
rho_bed_tot  = (M_solid_phys + M_filler) / V_bed  # [kg_solid/m³_bed]

print(f"Bed packing: M_cat_active = {M_cat_active*1e3:.1f} g, M_ads = {M_ads*1e3:.1f} g, "
      f"M_filler = {M_filler*1e3:.1f} g  "
      f"-> rho_bed_cat={rho_bed_cat:.1f}  rho_bed_ads={rho_bed_ads:.1f}  "
      f"rho_bed_fill={rho_bed_fill:.1f}  rho_bed_tot={rho_bed_tot:.1f} kg/m3_bed")

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
GHSV   = 1.0            # [m3_STP / (kg_ads * h)]

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
print(f"  M_ads       = {M_ads:.3f} kg  (92.5 wt% of bifunctional material)")
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