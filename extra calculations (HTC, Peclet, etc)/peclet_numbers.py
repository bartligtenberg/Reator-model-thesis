"""
Peclet numbers for the MPB reactor — model assumption checks
Pe >> 1: convection dominates, diffusion/conduction can be neglected.

1. Axial mass Peclet (Bodenstein)  — plug-flow valid?
2. Axial heat Peclet (gas + solid) — axial heat conduction negligible?
3. Radial heat Peclet + wall Biot  — radial temperature gradient significant?
4. Gas-particle Nusselt (Gunn)     — pseudohomogeneous (T_gas = T_solid) valid?
"""

import numpy as np

# ── Reactor / bed ─────────────────────────────────────────────────────────────
d_b, L_b   = 0.050, 2.000          # [m] inner diameter, bed length
R_b        = d_b / 2
d_p, eps_b = 2.5e-3, 0.40         # [m], [-]
k_eff_ax   = 0.40                  # [W/(m·K)] effective axial thermal conductivity (Mette et al., 2014)
k_eff_r    = 0.40                  # [W/(m·K)] effective radial thermal conductivity (Mette et al., 2014)

# ── Solid (MPB) ───────────────────────────────────────────────────────────────
M_cat, M_ads  = 0.064, 1.22        # [kg]
V_bed         = np.pi/4 * d_b**2 * L_b
rho_bed_tot   = (M_cat + M_ads) / V_bed   # [kg/m³] bed-averaged density
rho_bed_cat   = M_cat / V_bed
Cp_cat        = 1100.0             # [J/(kg·K)]
u_s_min, u_s_max = 1e-4, 5e-3     # [m/s] solid velocity scan range

# ── Operating conditions (280 °C, 1 bar) ─────────────────────────────────────
T_K  = 280.0 + 273.15
P_Pa = 1e5
R_gas = 8.314
y_CH4, y_H2, y_CO2 = 0.80, 0.16, 0.04

A_b   = np.pi/4 * d_b**2
u_g   = (0.5 * M_ads / 3600.0) / A_b * (T_K / 273.15)   # GHSV=0.5 m³_STP/(kg_ads·h)

k_gas  = y_CH4*0.045  + y_H2*0.230  + y_CO2*0.025        # [W/(m·K)]
mu_gas = y_CH4*1.50e-5 + y_H2*1.10e-5 + y_CO2*2.00e-5   # [Pa·s]
rho_gas = (P_Pa / (R_gas * T_K)) * (y_CH4*16e-3 + y_H2*2e-3 + y_CO2*44e-3)
Cp_mix  = 1500.0                                          # [J/(kg·K)]

Re_p = rho_gas * u_g * d_p / mu_gas
Pr   = mu_gas * Cp_mix / k_gas
Sc   = mu_gas / (rho_gas * (1.0 / (y_H2/1.50e-4 + y_CH4/2.50e-5)))

print(f"MPB Peclet analysis | d_b={d_b*100:.0f}cm  L={L_b:.1f}m  d_p={d_p*1e3:.2f}mm  "
      f"T=280C  u_g={u_g*1e3:.1f}mm/s  Re_p={Re_p:.2f}  Pr={Pr:.2f}  Sc={Sc:.1f}\n")

# ── 1. Axial mass Peclet (Bodenstein — plug-flow check) ───────────────────────
Re_Sc  = Re_p * Sc
inv_Bo = 0.3/Re_Sc + 0.5/(1.0 + 3.8/Re_Sc)
Bo_p   = 1.0 / inv_Bo
Bo_bed = Bo_p * (L_b / d_p)
print(f"1. Axial mass Peclet (plug-flow)")
print(f"   Bo_p = {Bo_p:.2f}   D_ax = {u_g*d_p/Bo_p:.2e} m2/s   Bo_bed = {Bo_bed:.0f}"
      f"  -> {'VALID' if Bo_bed > 100 else 'QUESTIONABLE'}\n")

# ── 2. Axial heat Peclet (gas convection vs conduction; solid convection vs conduction) ──
Pe_gas_bed = rho_gas * Cp_mix * u_g * L_b / k_eff_ax
Pe_sol_min = u_s_min * rho_bed_tot * Cp_cat * L_b / k_eff_ax
Pe_sol_max = u_s_max * rho_bed_tot * Cp_cat * L_b / k_eff_ax
print(f"2. Axial heat Peclet (conduction negligible?)")
print(f"   Gas   Pe_h,bed = {Pe_gas_bed:.0f}"
      f"  -> {'VALID' if Pe_gas_bed > 50 else 'QUESTIONABLE'}")
print(f"   Solid Pe_h,bed = {Pe_sol_min:.0f} - {Pe_sol_max:.0f}"
      f"  (u_s = {u_s_min*1e3:.1f}-{u_s_max*1e3:.1f} mm/s)"
      f"  -> {'VALID' if Pe_sol_min > 50 else 'QUESTIONABLE'}\n")

# ── 3. Radial heat Peclet + wall Biot (1D vs 2D check) ───────────────────────
h_bw      = 25.0                  # [W/(m²·K)] from Specchia formula
Bi_wall   = h_bw * R_b / k_eff_r
Q_peak    = 165000.0 * rho_bed_cat * 0.015   # dH_r * rho_bed_cat * r_peak [W/m³]
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

# ── Summary ───────────────────────────────────────────────────────────────────
print("SUMMARY")
print(f"  {'Criterion':<40} {'Value':>8}   Valid?")
print(f"  {'-'*58}")
rows = [
    ("Plug flow              Bo_bed",      Bo_bed,      Bo_bed > 100),
    ("Axial cond. gas        Pe_h,bed",    Pe_gas_bed,  Pe_gas_bed > 50),
    (f"Axial cond. solid      Pe_h,bed(min,u_s={u_s_min*1e3:.1f}mm/s)", Pe_sol_min, Pe_sol_min > 50),
    ("Pseudohomogeneous      NTU",         NTU,         NTU > 100),
    ("1D radial              Bi_wall",     Bi_wall,     Bi_wall < 2),
]
for label, val, ok in rows:
    flag = "OK" if ok else "CHECK <--"
    print(f"  {label:<50} {val:>8.1f}   {flag}")
