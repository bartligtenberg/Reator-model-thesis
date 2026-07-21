"""
Peclet numbers for the MPB reactor — model assumption checks
Pe >> 1: convection dominates, diffusion/conduction can be neglected.

1. Axial mass Peclet (Pe_L, Delgado 2006)  — plug-flow valid?
2. Axial heat Peclet (gas + solid) — axial heat conduction negligible?
3. Radial heat Peclet + wall Biot  — radial temperature gradient significant?
4. Gas-particle Nusselt (Gunn)     — pseudohomogeneous (T_gas = T_solid) valid?
5. Weisz-Prater                    — internal (intraparticle) diffusion limitation?
"""

import numpy as np

# ── Reactor / bed ─────────────────────────────────────────────────────────────
d_b, L_b   = 0.050, 2.000          # [m] inner diameter, bed length
R_b        = d_b / 2
d_p, eps_b = 2.5e-3, 0.40         # [m], [-]
k_eff_ax   = 0.40                  # [W/(m·K)] effective axial thermal conductivity (Mette et al., 2014)
k_eff_r    = 0.40                  # [W/(m·K)] effective radial thermal conductivity (Mette et al., 2014)
V_bed      = np.pi/4 * d_b**2 * L_b

# ── Solid (MPB) — catalyst / sorbent / filler densities ───────────────────────
# Matches the density structure in "MPB_flux_form active frac.py" (updated densities,
# 20% active catalyst): a single bifunctional 5wt%Ni-2.5wt%Ce/13X pellet carries both
# the catalytic (Ni) and sorption (13X) function; only a mass fraction of it is
# catalytically active. Inert filler tops the bed up to the assumed void fraction.
eps_p   = 0.242    # [-]      intraparticle porosity — Wei's pore volume + Bareschino particle density
tau_p   = 4.0      # [-]      pore tortuosity factor — Mette et al. (2015)
rho_p   = 1400.0   # [kg/m³]  particle (skeletal) density — Bareschino et al. (2023)

bifunctional_mass = 0.4    # [kg]  mass of 5wt%Ni-2.5wt%Ce/13X bifunctional material
M_zeolite_added   = 0.0    # [kg]  additional pure 13X zeolite (sorbent-only, no Ni)
active_fraction   = 0.20   # [-]   fraction of the bifunctional material's mass that is catalytically active

M_ads        = bifunctional_mass * 0.925 + M_zeolite_added   # [kg] sorbent mass (92.5% of bifunctional material + all added zeolite)
M_cat_active = bifunctional_mass * active_fraction            # [kg] active catalyst mass
M_solid_phys = bifunctional_mass + M_zeolite_added             # [kg] physical solid mass before filler
M_filler     = (1 - eps_b) * V_bed * rho_p - M_solid_phys       # [kg] inert filler, tops bed to eps_b packing at rho_p

rho_bed_cat  = M_cat_active / V_bed              # [kg_cat/m³_bed]   catalyst bulk density (reaction terms)
rho_bed_ads  = M_ads / V_bed                     # [kg_ads/m³_bed]   sorbent bulk density (adsorption terms)
rho_bed_fill = M_filler / V_bed                  # [kg_fill/m³_bed]  filler bulk density
rho_bed_tot  = (M_solid_phys + M_filler) / V_bed  # [kg_solid/m³_bed] total solids bulk density (heat capacity basis)
Cp_cat       = 1100.0             # [J/(kg·K)]
u_s_min, u_s_max = 1e-4, 5e-3     # [m/s] solid velocity scan range

print(f"Densities | M_cat_active={M_cat_active*1e3:.1f}g  M_ads={M_ads*1e3:.1f}g  M_filler={M_filler*1e3:.1f}g  "
      f"-> rho_bed_cat={rho_bed_cat:.1f}  rho_bed_ads={rho_bed_ads:.1f}  rho_bed_fill={rho_bed_fill:.1f}  "
      f"rho_bed_tot={rho_bed_tot:.1f} kg/m3_bed\n")

# ── Operating conditions (280 °C, 1 bar) ─────────────────────────────────────
T_K  = 280.0 + 273.15
P_Pa = 1e5
R_gas = 8.314
y_CH4, y_H2, y_CO2 = 0.80, 0.16, 0.04

GHSV  = 1.0                        # [m³_STP/(kg_ads·h)]
A_b   = np.pi/4 * d_b**2
u_g   = (GHSV * M_ads / 3600.0) / A_b * (T_K / 273.15)   # GHSV=1.0 m³_STP/(kg_ads·h)

k_gas  = y_CH4*0.045  + y_H2*0.230  + y_CO2*0.025        # [W/(m·K)]
mu_gas = y_CH4*1.50e-5 + y_H2*1.10e-5 + y_CO2*2.00e-5   # [Pa·s]
rho_gas = (P_Pa / (R_gas * T_K)) * (y_CH4*16e-3 + y_H2*2e-3 + y_CO2*44e-3)
Cp_mix  = 1500.0                                          # [J/(kg·K)]

Re_p = rho_gas * u_g * d_p / mu_gas
Pr   = mu_gas * Cp_mix / k_gas
Sc   = mu_gas / (rho_gas * (1.0 / (y_H2/1.50e-4 + y_CH4/2.50e-5)))

print(f"MPB Peclet analysis | d_b={d_b*100:.0f}cm  L={L_b:.1f}m  d_p={d_p*1e3:.2f}mm  "
      f"T=280C  u_g={u_g*1e3:.1f}mm/s  Re_p={Re_p:.2f}  Pr={Pr:.2f}  Sc={Sc:.1f}\n")

# ── 1. Axial mass Peclet (Pe_L, Delgado 2006 — plug-flow check) ──────────────
# Two-asymptote form for dispersion in packed beds (Delgado 2006, "A critical review of
# dispersion in packed beds", Heat Mass Transfer 42(4):279-310, Eq. 12), rather than the
# Wen & Fan (1975) three-constant intermediate-regime fit -- easier to source/verify, and
# per Delgado, conservative for gas flow in this Re*Sc range (measured values run somewhat
# higher than this predicts). tau = sqrt(2) is the tortuosity factor for diffusion.
tau     = np.sqrt(2.0)
Re_Sc   = Re_p * Sc
inv_PeL = (eps_b / tau) / Re_Sc + 0.5
Pe_L    = 1.0 / inv_PeL
Bo_bed  = Pe_L * (L_b / d_p)
print(f"1. Axial mass Peclet (plug-flow, Delgado 2006)")
print(f"   Pe_L = {Pe_L:.2f}   D_ax = {u_g*d_p/Pe_L:.2e} m2/s   Bo_bed = {Bo_bed:.0f}"
      f"  -> {'VALID' if Bo_bed > 100 else 'QUESTIONABLE'}\n")

# ── 1b. Solid-phase axial mass Peclet (order-of-magnitude, no back-mixing check) ─
# No validated dispersion correlation exists for solids in a moving packed bed (unlike
# Delgado/Wen & Fan-type correlations for the gas above); D_ax,solid ~ d_p*u_s is a common
# order-of-magnitude scaling for granular axial mixing (dispersion coefficient of order one
# particle diameter times the local solid velocity). u_s cancels, so Pe_ax,solid = L/d_p
# regardless of solid throughput -- this is a rough estimate, not a fitted correlation,
# flag in text.
Pe_ax_solid = L_b / d_p
print(f"1b. Solid axial mass Peclet (no back-mixing, order-of-magnitude)")
print(f"   D_ax,solid ~ d_p*u_s (scaling estimate)   Pe_ax,solid = L/d_p = {Pe_ax_solid:.0f}"
      f"  -> {'VALID' if Pe_ax_solid > 100 else 'QUESTIONABLE'}\n")

# ── 2. Axial heat Peclet (gas convection vs conduction; solid convection vs conduction) ──
# Threshold set to 100, matching the mass-dispersion Bo_bed criterion (Fogler 2006,
# Levenspiel 1999, via Su et al. 2021, "Scale-up of micro- and milli-reactors", Chem. Eng.
# J. Advances, 2021) -- no independently-established heat-specific threshold was found in
# the literature, so the mass-dispersion criterion is reused here as a proxy.
Pe_gas_bed = rho_gas * Cp_mix * u_g * L_b / k_eff_ax
Pe_sol_min = u_s_min * rho_bed_tot * Cp_cat * L_b / k_eff_ax
Pe_sol_max = u_s_max * rho_bed_tot * Cp_cat * L_b / k_eff_ax
print(f"2. Axial heat Peclet (conduction negligible?)")
print(f"   Gas   Pe_h,bed = {Pe_gas_bed:.0f}"
      f"  -> {'VALID' if Pe_gas_bed > 100 else 'QUESTIONABLE'}")
print(f"   Solid Pe_h,bed = {Pe_sol_min:.0f} - {Pe_sol_max:.0f}"
      f"  (u_s = {u_s_min*1e3:.1f}-{u_s_max*1e3:.1f} mm/s)"
      f"  -> {'VALID' if Pe_sol_min > 100 else 'QUESTIONABLE'}\n")

# ── 3. Radial heat Peclet + wall Biot (1D vs 2D check) ───────────────────────
r_peak_active = 0.02             # [mol/(kg_cat_active·s)] representative peak reaction rate
h_bw      = 25.0                  # [W/(m²·K)] from Specchia formula
Bi_wall   = h_bw * R_b / k_eff_r
Q_peak    = 165000.0 * rho_bed_cat * r_peak_active   # dH_r * rho_bed_cat * r_peak [W/m³]
dT_radial = Q_peak * R_b**2 / (4 * k_eff_r)
dT_center = Q_peak * R_b**2 / (8 * k_eff_r)
print(f"3. Radial heat (1D vs 2D)")
print(f"   Bi_wall = {Bi_wall:.1f}   dT_wall-to-center = {dT_radial:.1f} K"
      f"   dT_center-above-avg = {dT_center:.1f} K")
print(f"   -> {'1D ADEQUATE' if Bi_wall < 2 else 'RADIAL GRADIENT SIGNIFICANT — 1D is conservative'}\n")

# ── 4. Gas-particle Nusselt (Gunn — pseudohomogeneous check) ─────────────────
Nu_sg = ((7 - 10*eps_b + 5*eps_b**2) * (1 + 0.7*Re_p**0.2 * Pr**(1/3))
       + (1.33 - 2.4*eps_b + 1.2*eps_b**2) * Re_p**0.7 * Pr**(1/3))
h_sg  = Nu_sg * k_gas / d_p
a_p   = 6*(1 - eps_b) / d_p
NTU   = h_sg * a_p * L_b / (rho_gas * Cp_mix * u_g)
dT_gs = Q_peak / (h_sg * a_p)
print(f"4. Gas-particle heat (pseudohomogeneous)")
print(f"   Nu_sg = {Nu_sg:.2f}   h_sg = {h_sg:.0f} W/(m2·K)   NTU = {NTU:.0f}"
      f"   dT_gas-solid = {dT_gs:.3f} K")
print(f"   -> {'VALID' if NTU > 100 else 'QUESTIONABLE'}\n")

# ── 5. Weisz-Prater (internal / intraparticle diffusion limitation check) ────
# Binary gas diffusivity of CO2 in the bulk background (80 % CH4) via the Fuller
# correlation (Fuller, Ensley & Giddings, 1969) — a standard estimate for
# gas-phase binary diffusivities when no measured value is available.
v_CO2, v_CH4 = 26.9, 24.42                          # [-] Fuller diffusion volumes
M_CO2_g, M_CH4_g = 44.01, 16.04                     # [g/mol]
M_AB  = 2.0 / (1.0/M_CO2_g + 1.0/M_CH4_g)           # [g/mol]
P_atm = P_Pa / 101325.0
D_CO2_CH4 = (0.00143 * T_K**1.75
             / (P_atm * np.sqrt(M_AB) * (v_CO2**(1/3) + v_CH4**(1/3))**2)) * 1e-4   # [cm2/s] -> [m2/s]
D_eff = D_CO2_CH4 * eps_p / tau_p                    # [m2/s] effective intraparticle diffusivity

R_p    = d_p / 2
L_char = R_p / 3                                     # [m] Weisz-Prater/Bischoff characteristic length V_p/S_p for a sphere
                                                      # (NOT the particle radius -- Weisz & Prater (1954) and Bischoff (1967)
                                                      # both define the modulus with L = V/S; using R_p directly overstates
                                                      # C_WP by (R_p/L)^2 = 9x for a sphere)
C_CO2_s = y_CO2 * P_Pa / (R_gas * T_K)               # [mol/m3] bulk CO2 concentration (surface conc. approximation)
r_peak_particle = r_peak_active * active_fraction    # [mol/(kg_particle·s)] rescaled to whole-pellet mass basis,
                                                      # since rho_p/D_eff act on the whole pellet, not just the active fraction
n_order   = 0.5   # [-] approximate reaction order in CO2 (LHHW rate ~ sqrt(p_CO2))
# Bischoff (1967) generalizes Weisz & Prater (1954) to arbitrary reaction order via his
# Eq. (17): r_obs*L^2*g(C_obs) / [2 * integral(C_eq to C_obs) of D_e(c)*g(c) dc] < 1.
# For simple nth-order kinetics, g(c) = c^n, D_e constant, and C_eq ~ 0 (irreversible
# reaction), the integral evaluates to D_e*C_obs^(n+1)/(n+1), so:
#   r_obs*L^2*C_obs^n / [2*D_e*C_obs^(n+1)/(n+1)] < 1
#   r_obs*L^2*(n+1) / (2*D_e*C_obs) < 1                 (C_obs^n/C_obs^(n+1) = 1/C_obs)
#   r_obs*L^2 / (D_e*C_obs) < 2/(n+1)
# Check: n=1 (first order) gives 2/(1+1) = 1, recovering Weisz & Prater's original
# first-order criterion (their Eq. 13) -- confirms the general formula is consistent.
thresh_WP = 2.0 / (n_order + 1)   # [-] threshold for C_WP below

C_WP = r_peak_particle * rho_p * L_char**2 / (D_eff * C_CO2_s)
print(f"5. Weisz-Prater / Bischoff (internal diffusion)")
print(f"   D_CO2-CH4 = {D_CO2_CH4:.2e} m2/s   D_eff = {D_eff:.2e} m2/s   C_CO2 = {C_CO2_s:.2f} mol/m3   L_char = R_p/3 = {L_char*1e3:.3f} mm")
print(f"   C_WP = {C_WP:.2f}   (threshold 2/(n+1) = {thresh_WP:.2f} for n~{n_order})"
      f"  -> {'VALID' if C_WP < thresh_WP else 'NOT NEGLIGIBLE'}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print("SUMMARY")
print(f"  {'Criterion':<40} {'Value':>8}   Valid?")
print(f"  {'-'*58}")
rows = [
    ("Plug flow (gas)        Bo_bed",      Bo_bed,      Bo_bed > 100),
    ("Plug flow (solid)      Pe_ax,solid", Pe_ax_solid, Pe_ax_solid > 100),
    ("Axial cond. gas        Pe_h,bed",    Pe_gas_bed,  Pe_gas_bed > 100),
    (f"Axial cond. solid      Pe_h,bed(min,u_s={u_s_min*1e3:.1f}mm/s)", Pe_sol_min, Pe_sol_min > 100),
    ("Pseudohomogeneous      NTU",         NTU,         NTU > 100),
    ("1D radial              Bi_wall",     Bi_wall,     Bi_wall < 2),
    ("Internal diff.         C_WP",        C_WP,        C_WP < thresh_WP),
]
for label, val, ok in rows:
    flag = "OK" if ok else "CHECK <--"
    print(f"  {label:<50} {val:>8.1f}   {flag}")