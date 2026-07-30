"""
Moving Packed Bed (MPB) Reactor Model — Steady-State, Lightly Cooled, Pseudo-Homogeneous
MOLAR FLUX FORM
==========================================================================================

Counter-current flow:
    gas  : z = 0 (inlet, bottom)  ->  z = L (outlet, top)    u_g > 0
    solid: z = L (inlet, top)     ->  z = 0 (outlet, bottom)  u_s > 0 (magnitude)

State variables: F_i [mol/(m²_bed·s)] — molar flux per unit bed cross-section.

Species balance (no u_g, no ε_b):
    dF_i/dz = source_i   [mol/(m²·s) / m] = [mol/(m³_bed·s)]

    dF_CO2/dz = −ρ_bed_cat · r
    dF_H2 /dz = −4 · ρ_bed_cat · r
    dF_CH4/dz = +ρ_bed_cat · r
    dF_H2O/dz = 2·ρ_bed_cat·r − ρ_bed_ads·ads

Partial pressures from mole fractions (ideal gas, constant P):
    p_i = (F_i / F_total) · P_bar

Energy balance (pseudo-homogeneous, counter-current):
    (ΣF_i·Cp_i  −  u_s·ρ_bed·Cp_cat) · dT/dz =
        (−ΔH_r)·ρ_bed_cat·r  +  (−ΔH_ads)·ρ_bed_ads·ads  −  U_a·(T − T_wall)

    Note: ΣF_i·Cp_i = u_g·(P/RT)·Cp_mix is the gas-side thermal flux [W/(m²·K)],
    derived from F_i directly without needing u_g explicitly.

Regime-switching on u_s*:
    u_s < u_s*  (gas dominates):   denom = ΣF_i·Cp_i − solid_cap > 0
                                    T solved in GAS IVP (z = 0→L), BC: T(z=0) = T_in
    u_s > u_s*  (solid dominates): rewrite in solid direction ζ = L−z:
                                    denom = solid_cap − ΣF_i·Cp_i > 0
                                    T solved in SOLID IVP (ζ = 0→L), BC: T(ζ=0) = T_in_solid

u_s* = ΣF_in_i·Cp_i / (ρ_bed·Cp_cat)  [m/s]

Solved by decoupled Gauss-Seidel iteration (same structure as concentration form).

--------------------------------------------------------------------------------------------
FEASIBILITY + SIZING SWEEP EXTENSION (this file)
--------------------------------------------------------------------------------------------
Extends the model above with a 2D sweep over (u_s, GHSV):
  Step 1 — analytic Λ_thermo pre-filter (no solver calls) masks out cells that can never
           reach spec, so the solver is only called on cells worth solving.
  Step 2 — solve_mpb() on the surviving cells, boustrophedon (snake) traversal with
           warm starts, plus a retry pass for non-converged/failed cells.
  Step 3 — per-cell post-processing: z where X_CO2 first crosses 99.5%/97.4%, T_max.
  Step 4 — heatmaps of the above over the (u_s, GHSV) grid, with the Λ_thermo = 2.5
           contour overlaid.
  Step 5 — pickle the full per-cell results dict for later re-analysis.

L_b below is the TRIAL bed length for this sweep (2 m), physically longer than the real
lab-scale unit (L_b_REF = 1 m) — a deliberate choice so a 99.5% conversion crossing has a
chance to occur inside the domain. Catalyst/sorbent mass (bifuctional_mass) is scaled up by
L_b/L_b_REF alongside it, so rho_bed_cat/rho_bed_ads stay identical to the real 1 m design —
the trial bed is a longer bed of the same packing, not a diluted version of the real charge.
"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# region 1. PARAMETERS
# =============================================================================

# --- Bed geometry (Bareschino lab setup) ---
d_b   = 0.050                # [m]   reactor inner (tube) diameter — Bareschino et al. (2023)
L_b_REF = 1.000              # [m]   real lab-scale bed length — Bareschino et al. (2023); reference for the mass scaling below
L_b   = 2.000                # [m]   TRIAL bed length for the feasibility/sizing sweep below —
                              #       deliberately longer than the real lab-scale unit so a 99.5%
                              #       CO2-conversion crossing has a chance to occur inside the
                              #       domain. Physically longer bed, but catalyst/sorbent mass is
                              #       scaled up proportionally (see bifuctional_mass below) so
                              #       rho_bed_cat/rho_bed_ads stay IDENTICAL to the real L_b_REF=1 m
                              #       design -- L_b only changes how much bed there is, not its
                              #       packing density.
A_b   = np.pi / 4 * d_b**2   # [m²]  bed cross-sectional area
V_bed = A_b * L_b            # [m³]  total bed volume
eps_b = 0.4                  # [-]   inter-particle (bed) void fraction — typical packed-bed value (not from Bareschino table)

# --- Particle properties (13X zeolite pellets) ---
d_p   = 2.5e-3   # [m]      particle diameter — this study's pellet size (Same as Bareschino et al. 2023. Wei used 0.75 mm)
eps_p = 0.242    # [-]      intraparticle void fraction — calclated with Wei's pore volume and density (Bareschino)
tau_p = 4.0      # [-]      pore tortuosity factor — Mette et al. (2015)
rho_p = 1400     # [kg/m³]  particle (skeletal) density of sorbent — Bareschino et al. (2023)

# --- Catalyst and sorbent loading ---
# Scaled by L_b/L_b_REF so that rho_bed_cat/rho_bed_ads (mass / V_bed) stay fixed at the real
# 1 m design's values as L_b is stretched to the trial length -- i.e. a longer bed is filled
# with proportionally more of the same material, not a diluted amount of the original charge.
_len_scale       = L_b / L_b_REF
bifuctional_mass = 0.4 * _len_scale   # [kg]  mass of bifunctional 5%Ni-2.5%Ce/13X material (0.4 kg at the real L_b_REF=1 m)
M_zeolite_added  = 0   * _len_scale   # [kg]  additional pure 13X zeolite mixed in — 100% sorbent-active, 0% catalytically active (no Ni)

M_ads = bifuctional_mass * 0.925 + M_zeolite_added   # [kg]  sorbent mass: 92.5% of the bifunctional material acts as sorbent, plus all of the added pure zeolite

active_fraction = 0.20   # [-]  fraction of the bifunctional material's mass that is catalytically active
M_cat_active = bifuctional_mass * active_fraction   # [kg]  active catalyst mass — only the bifunctional material carries Ni; the added zeolite contributes none

M_solid_physical = bifuctional_mass + M_zeolite_added   # [kg]  true physical solid mass present (catalyst+sorbent material), before filler

# --- Inert filler (thermal buffering / dilution / flow aid) ---
# Assumed to share the same particle density (rho_p), heat capacity (Cp_cat) and particle
# diameter (d_p) as the bifunctional material / zeolite, so no separate filler properties
# are needed. Chemically inert: adds to total bed solids mass (rho_bed_tot) only -- never to
# rho_bed_cat (reaction) or rho_bed_ads (adsorption). Sized to fill the remaining bed volume
# at the assumed bed void fraction eps_b (tops the bed up to eps_b packing density).
M_filler = (1 - eps_b) * V_bed * rho_p - M_solid_physical   # [kg]

rho_bed_cat = M_cat_active / V_bed   # [kg_cat/m³_bed]  catalyst bulk density (reaction terms)
rho_bed_ads = M_ads / V_bed          # [kg_ads/m³_bed]  sorbent bulk density (adsorption terms)
rho_bed_tot = (M_solid_physical + M_filler) / V_bed  # [kg_solid/m³_bed]  total solids bulk density (cat+ads material+filler) — physical solid mass basis for heat capacity

# --- Dubinin-Astakhov isotherm (H2O on 13X) --- fitted myself based on Wei et al. (2021)
W0_DA = 190.00e-6   # [m³/kg_sorbent]  limiting micropore volume
E_DA  = 1192e3      # [J/kg]           characteristic adsorption energy
n_DA  = 1.55        # [-]              DA heterogeneity parameter

# --- LHHW kinetics (Koschany et al. 2016, Table 6) ---
T_ref_K = 555.0      # [K]                    reference temperature (282 °C) — Koschany et al. (2016)
k_ref   = 3.46e-4    # [mol/(g_cat·s·bar)]    rate constant at T_ref — Koschany et al. (2016)
Ea_k    = 77.5e3     # [J/mol]                activation energy — Koschany et al. (2016)
A_OH    = 0.50;  dH_OH  =  22.4e3   # [bar^-0.5], [J/mol]  K_OH pre-exponential & van 't Hoff enthalpy — Koschany et al. (2016)
A_H2    = 0.44;  dH_H2  =  -6.2e3   # [bar^-0.5], [J/mol]  K_H2 pre-exponential & van 't Hoff enthalpy — Koschany et al. (2016)
A_mix   = 0.88;  dH_mix = -10.0e3   # [bar^-0.5], [J/mol]  K_mix pre-exponential & van 't Hoff enthalpy — Koschany et al. (2016) Table 6
P_FLOOR = 1e-4       # [bar]                  minimum partial pressure floor, avoids div-by-0 / log(0) in rate & DA expressions

# --- Thermochemistry ---
dH_r   = -165.0e3   # [J/mol_CO2]   Sabatier reaction enthalpy — NIST
dH_ads =  -45.0e3   # [J/mol_H2O]   isosteric heat of H2O adsorption on 13X — Bareschino et al. (2023) Table 3
Cp_cat = 1100.0      # [J/(kg·K)]    catalyst/sorbent heat capacity — Bareschino et al. (2023) Table 3
Cp_CO2 = 45.4;  Cp_H2 = 29.3;  Cp_CH4 = 46.9;  Cp_H2O = 34.2   # [J/(mol·K)]  gas heat capacities — NIST at ~550 K

# --- Wall heat transfer ---
U_a    = 2000.0   # [W/(m³_bed·K)]  volumetric wall heat-transfer coefficient — rough estimate: h_wall ≈ 25 W/(m²·K) × (4/d_b = 80 m²/m³)

# --- Physical constants ---
R_gas  = 8.314       # [J/(mol·K)]  universal gas constant
MW_H2O = 0.018015    # [kg/mol]     molar mass of water
T_STP  = 273.15      # [K]          standard temperature (0 °C)

# --- Operating conditions ---
P_bar = 1.0                # [bar]  total pressure
P_Pa  = P_bar * 1e5         # [Pa]   total pressure
y_CO2_in = 0.04             # [-]    CO2 inlet mole fraction — Bareschino et al. (2022/2023)
y_H2_in  = 0.16             # [-]    H2 inlet mole fraction (H2/CO2 = 4, stoichiometric)  — Bareschino et al. (2022/2023)
y_CH4_in = 0.80             # [-]    CH4 inlet mole fraction (background/diluent) — Bareschino et al. (2022/2023)

# --- Inlet molar fluxes [mol/(m²·s)] — now GHSV-dependent (see compute_inlet_fluxes() /
# set_inlet_fluxes() in region 2), since the sweep below varies GHSV across the grid instead
# of holding it fixed. These globals are placeholders, overwritten before any solve_mpb call. ---
F_in_CO2 = F_in_H2 = F_in_CH4 = F_total_in = u_g_STP = 0.0

# --- Feasibility/sizing sweep grid: (u_s, GHSV) — edit these to change the sweep ---
US_MIN_MM, US_MAX_MM = 3.0, 10.0     # [mm/s]                     u_s sweep range
GHSV_MIN, GHSV_MAX   = 0.5, 3.0      # [m3_STP/(kg_ads·h)]        GHSV sweep range
N_US, N_GHSV         = 10, 10        # [-]                        grid resolution along each axis
T_SWEEP_C            = 280.0         # [°C]                       fixed inlet/wall temperature for the whole sweep
LAMBDA_MIN           = 2.5           # [-]                        Λ_thermo pre-filter threshold (cells below this are never solved)

print(f"MPB flux form (feasibility/sizing sweep): d={d_b*100:.0f} cm, L_trial={L_b:.1f} m, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3")
print(f"  U_a = {U_a:.0f} W/(m3·K)")
print(f"  bifunctional material = {bifuctional_mass*1000:.1f} g, active_fraction = {active_fraction:.0%}")
print(f"  M_cat(active)   = {M_cat_active*1000:.1f} g")
print(f"  zeolite added   = {M_zeolite_added*1000:.1f} g")
print(f"  M_ads           = {M_ads*1000:.1f} g  (92.5% of bifunctional material + zeolite added)")
print(f"  M_filler        = {M_filler*1000:.1f} g  "
      f"({M_filler/(M_solid_physical+M_filler):.1%} of total solids; "
      f"tops bed up to eps_b={eps_b:.2f} packing at rho_p={rho_p:.0f} kg/m3; "
      f"same rho_p/Cp_cat/d_p as catalyst/sorbent, chemically inert -> rho_bed_tot = {rho_bed_tot:.0f} kg/m3)")
print(f"  sweep grid: u_s in [{US_MIN_MM:.1f}, {US_MAX_MM:.1f}] mm/s ({N_US} pts), "
      f"GHSV in [{GHSV_MIN:.2f}, {GHSV_MAX:.2f}] m3_STP/(kg_ads.h) ({N_GHSV} pts), T={T_SWEEP_C:.0f} C")
# endregion


# region 2. FUNCTIONS
# =============================================================================
def P_sat_bar(T_K):
    """
    Saturation vapour pressure of water [bar]. log10(P/mmHg) = D + E/T + F*log10(T) + G*T + H*T^2, with D=29.8605, E=-3.1522e3, F=-7.3037, G=2.4247e-9, H=1.8090e-6 — Eq. S.17 in
    Bareschino et al. (2023) supplementary material, credited there to Kowalska & Ambrozek (2017). Converted to bar via 1 mmHg = 133.322e-5 bar. np.clip prevents overflow at extreme temperatures.
    """
    log10_p = (29.8605 - 3.1522e3/T_K - 7.3037*np.log10(T_K)
               + 2.4247e-9*T_K + 1.8090e-6*T_K**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5   # [mmHg] -> [bar]

def rho_water(T_K):
    """
    Temperature-corrected density of the adsorbed phase (liquid water) [kg/m³]. rho(T) = rho_20C / (1 + beta_20*(T-298.15K)), rho_20C=996 kg/m3, beta_20=2e-3 [1/K]
    — Eq. S.16 in Bareschino et al. (2023) supplementary material, credited there to Schaefer & Thess (2018).
    """
    return 996.0 / (1.0 + 2.0e-3*(T_K - 298.15))

def q_star_vec(T_K, p_arr, W0, E, n):
    """
    Equilibrium H2O loading q* [mol/kg_sorbent] — Dubinin-Astakhov (DA) isotherm.
    Matches Eqs. (21)-(23) in Bareschino et al. (2023), there credited to the DA model
    used by Mette et al. (2014); W0, E, n themselves are this file's own fit (see 'Dubinin-Astakhov isotherm' parameter block above).
    p_arr: gas-phase H2O partial pressure(s) [bar], scalar or array; the actual pressure the sorbent is exposed to.
    """
    p    = np.asarray(p_arr, dtype=float)                    # ensure p is a numpy float array for vectorised operations
    Psat = P_sat_bar(T_K)                                    # saturation vapour pressure at this temperature [bar]
    p_safe = np.clip(p, 1e-15, Psat*(1-1e-10))               # keep p strictly inside (0, Psat) so log(Psat/p) stays finite
    A_raw  = (R_gas/MW_H2O)*T_K*np.log(Psat/p_safe)         # A = (R_gas/MW_H2O)*T*ln(Psat/p)   adsorption potential [J/kg]
    A  = np.where((p <= 0)|(p >= Psat), 0.0, A_raw)          # zero potential (no adsorption) when p is non-physical or at/above saturation
    W  = W0*np.exp(-np.minimum((A/E)**n, 500.0))            # W = W0*exp(-(A/E)^n)      filled micropore volume fraction [m³/kg]
    qs = rho_water(T_K)/MW_H2O*W                            # q* = rho_water(T)/MW_H2O * W         Volume -> moles via adsorbed-phase density

    return np.where(p <= 0, 0.0, qs)                         # force q* to zero when p_H2O is zero (no water vapour -> no equilibrium loading), since qs was computed from the clipped p_safe floor, not the true zero

def K_LDF_vec(T_K, p_arr, W0, E, n):
    D_M = 3.36e-9 * T_K**1.75                                                  # molecular diffusivity [m²/s], power-law T-dependence (Chapman-Enskog) from supplementary material of Bareschino et al. (2023)
    p    = np.asarray(p_arr, dtype=float)                                      # ensure p (water pressure)is a numpy float array for vectorised operations
    dp   = 1.0/1e5                                                             # pressure step = 1 Pa expressed in bar; chosen so dividing by 2.0 [Pa] gives dq*/dp in mol/(kg·Pa)
    dqsp = (q_star_vec(T_K, p+dp, W0, E, n)
            - q_star_vec(T_K, np.maximum(p-dp, 1e-15), W0, E, n)) / 2.0      # central finite difference: dq*/dp [mol/(kg·Pa)]; np.maximum prevents zero/negative pressure in DA log
    dqsp = np.maximum(dqsp, 1e-30)                                             # guard against division by zero when isotherm slope is numerically flat
    r_p = 0.5 * d_p                                                            # particle radius [m]
    return 15.0 * eps_p * D_M / (r_p**2 * tau_p * rho_p * R_gas * T_K * dqsp)  # Glueckauf LDF coefficient from pore diffusion [1/s]: large dq*/dp → slow K_LDF

def K_eq_sabatier(T_K):
    return 137.0*T_K**(-3.994)*np.exp(158700.0/(R_gas*T_K))  # equilibrium constant of the Sabatier reaction (CO2 + 4H2 <-> CH4 + 2H2O). Koschany et al. (2016)

def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    """
    LHHW rate of the Sabatier reaction [mol/(kg_cat·s)] — Koschany et al. (2016), Table 6, Eq. (6).
    Power-law/Langmuir-Hinshelwood form fitted on a co-precipitated Ni/Al2O3 catalyst;
    here applied to the 5%Ni-2.5%Ce/13X catalyst of Bareschino et al. (2023) (same kinetic expression reused).
    """
    vH    = lambda dH: np.exp(-dH/R_gas*(1.0/T_K - 1.0/T_ref_K))               # van 't Hoff shift of an adsorption constant from T_ref to T_K
    k     = k_ref*np.exp(-Ea_k/R_gas*(1.0/T_K - 1.0/T_ref_K))                  # Arrhenius rate constant at T_K, shifted from k_ref at T_ref_K
    K_OH  = A_OH*vH(dH_OH);  K_H2 = A_H2*vH(dH_H2);  K_mix = A_mix*vH(dH_mix)   # adsorption equilibrium constants at T_K (OH*, H2, mixed CO2/CO/H2O term) — Koschany et al. (2016) Table 6
    K_eq  = K_eq_sabatier(T_K)                                                 # thermodynamic equilibrium constant of CO2 + 4H2 <-> CH4 + 2H2O at T_K
    p_CO2_s = np.maximum(p_CO2, P_FLOOR);  p_H2_s = np.maximum(p_H2, P_FLOOR)  # pressure floors avoid 0**negative-power blow-up in f_eq/DEN below
    beta  = (p_CH4*p_H2O**2)/(K_eq*p_CO2_s*p_H2_s**4)                         # beta = Q/K_eq, ratio of reaction quotient Q to equilibrium constant K_eq
    f_eq  = np.maximum(1.0 - np.where(np.isfinite(beta), beta, 1e10), 0.0)     # equilibrium approach factor (1 - beta); clipped to >=0 so rate can't reverse past equilibrium
    DEN   = (1.0 + K_OH*np.maximum(p_H2O, 0)/p_H2_s**0.5                      # Langmuir-Hinshelwood adsorption denominator: 1 + K_OH*p_H2O/sqrt(p_H2)
             + K_H2*p_H2_s**0.5 + K_mix*p_CO2_s**0.5)                         #                                            + K_H2*sqrt(p_H2) + K_mix*sqrt(p_CO2)
    return k*(p_CO2_s*p_H2_s)**0.5*f_eq/DEN**2*1000.0                         # r = k*sqrt(p_CO2*p_H2)*f_eq/DEN^2, in mol/(g_cat·s); *1000 -> mol/(kg_cat·s)

def q_star(T_K, p_H2O):
    return q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)  # q_star_vec bound to this file's fitted DA parameters, so callers don't have to pass W0/E/n each time

def K_LDF(T_K, p_H2O):
    return K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)  # K_LDF_vec bound to this file's fitted DA parameters, same pattern as q_star above

def _gas_cap(F_CO2, F_H2, F_CH4, F_H2O):
    """Thermal flux of gas phase [W/(m²·K)] = Σ F_i·Cp_i."""
    return F_CO2*Cp_CO2 + F_H2*Cp_H2 + F_CH4*Cp_CH4 + F_H2O*Cp_H2O  # gas-side counterpart of solid_cap = u_s*rho_bed*Cp_cat; their difference is the energy-balance denominator (see file header, region 1)

def _partial_pressures(F_CO2, F_H2, F_CH4, F_H2O):
    """Convert molar fluxes to partial pressures [bar] via mole fractions."""
    F_tot = F_CO2 + F_H2 + F_CH4 + F_H2O
    if F_tot < 1e-30:
        return 0.0, 0.0, 0.0, 0.0                                 # all fluxes ~0 (e.g. solver probing near z=0) -> return zero pressures instead of dividing by ~0
    return (F_CO2/F_tot*P_bar, F_H2/F_tot*P_bar,                  # ideal-gas partial pressure p_i = y_i*P = (F_i/F_tot)*P_bar
            F_CH4/F_tot*P_bar, F_H2O/F_tot*P_bar)

def compute_inlet_fluxes(GHSV):
    """
    Inlet molar fluxes [mol/(m2.s)] at a given GHSV [m3_STP/(kg_ads·h)] -- same derivation as
    the original fixed-GHSV parameter block (GHSV -> STP volumetric flow -> superficial STP
    gas velocity -> total molar flux via ideal gas law -> per-species flux via inlet mole
    fraction), generalised to accept GHSV as an argument (scalar or array) instead of reading
    a fixed module-level GHSV -- needed because the sweep below varies GHSV across the grid.
    """
    Q_STP    = GHSV * M_ads / 3600.0                    # [m3_STP/s]   volumetric feed flow at STP
    u_g_STP_ = Q_STP / A_b                              # [m/s]        superficial gas velocity at STP
    F_total_ = u_g_STP_ * P_Pa / (R_gas * T_STP)        # [mol/(m2·s)] total molar flux at inlet (ideal gas law)
    F_CO2_   = y_CO2_in * F_total_
    F_H2_    = y_H2_in  * F_total_
    F_CH4_   = y_CH4_in * F_total_
    return F_CO2_, F_H2_, F_CH4_, F_total_, u_g_STP_

def set_inlet_fluxes(GHSV):
    """Overwrite the module-level inlet-flux globals that solve_mpb/_q_physics_init read, for this GHSV."""
    global F_in_CO2, F_in_H2, F_in_CH4, F_total_in, u_g_STP
    F_in_CO2, F_in_H2, F_in_CH4, F_total_in, u_g_STP = compute_inlet_fluxes(GHSV)
# endregion


# region 3. DECOUPLED SOLVER
# =============================================================================
def solve_mpb(u_s, T_K, T_wall=None, max_iter=1000, tol=1e-5, N=400, q_init=None):
    """
    Counter-current MPB — molar flux form, lightly cooled, regime-switching.

    State: F_i [mol/(m²·s)].  No u_g or ε_b in species balance.
    Partial pressures: p_i = (F_i/F_total) · P_bar.
    Energy denominator: Σ F_i·Cp_i − u_s·ρ_bed·Cp_cat.

    Three independent numerical controls, not to be conflated:
      - N: resolution of the shared z_grid used to couple the two decoupled IVPs each
           Gauss-Seidel sweep (interpolating q(z)/T(z) between gas and solid). Does NOT
           control IVP integration accuracy -- BDF integrates adaptively between these points.
      - solve_ivp's rtol/atol (set inside the loop below, not exposed as args here):
           control the accuracy of each IVP's own continuous integration, independent of N.
      - tol / max_iter: control when the outer Gauss-Seidel loop is considered converged
           (relative change in q, and T where solved, between successive sweeps).
    """
    if T_wall is None:
        T_wall = T_K                                                # default: adiabatic wall (no cooling driving force) when no explicit T_wall is given

    solid_cap  = u_s * rho_bed_tot * Cp_cat                          # solid thermal-capacity flux [W/(m²·K)] = u_s*rho_bed*Cp_cat
    gas_cap_in = _gas_cap(F_in_CO2, F_in_H2, F_in_CH4, 0.0)          # gas thermal-capacity flux [W/(m²·K)] at the (dry) inlet composition
    gas_dominates = (solid_cap < gas_cap_in)                         # regime switch: u_s below u_s* -> gas capacity wins -> solve T as a gas-side IVP (see file header)

    z_grid = np.linspace(0.0, L_b, N)                                # coarse axial grid used during Gauss-Seidel iteration (refined to z_fine after convergence)
    if q_init is not None:
        q_prof = np.interp(z_grid, np.linspace(0, L_b, len(q_init)), q_init)  # warm-start: resample a previous run's q(z) profile onto this grid
    else:
        q_prof = np.zeros(N)                                        # cold-start: assume the sorbent starts empty everywhere
    T_prof = T_K * np.ones(N)                                        # cold-start temperature guess: isothermal at the inlet temperature

    converged = False
    err = 1.0                                                        # forces at least one Gauss-Seidel pass before the err < tol check can trigger
    _solid_denom_min = [np.inf]                                      # tracks min solid_denom seen in solid_rhs_with_T (solid-dominant branch) across this call, for FAILED diagnostics

    for it in range(max_iter):                                      # Gauss-Seidel iteration loop: solve gas IVP, then solid IVP, then check convergence
        q_fn = interp1d(z_grid, q_prof, kind='linear',                # q(z) from the previous iteration, frozen for use inside this iteration's gas IVP
                        bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))    #interp1d makes discrete q(z) into a continuous function for the ODE solver to call. BDF uses adaptive step sizes, so it will call q(z) at arbitrary z values, not just the discrete z_grid points.
        T_fn = interp1d(z_grid, T_prof, kind='linear',                # T(z) from the previous iteration, frozen for use inside this iteration's solid IVP (solid-dominant branch)
                        bounds_error=False, fill_value=(T_prof[0], T_prof[-1]))

        if gas_dominates:
            # ── GAS IVP: state = [F_CO2, F_H2, F_CH4, F_H2O, T] ─────────────     Solves the partial derivatives in Z. LHS: Dfi/Dz
            def gas_rhs(z, y):
                F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)   # clip fluxes to >=0 (BDF can overshoot slightly negative near depletion)
                F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
                T_l     = max(y[4], 200.0)                             # floor T so property correlations (Arrhenius, DA isotherm) stay physical during solver transients
                q_l     = max(float(q_fn(z)), 0.0)                     # sorbent loading at this z, taken from the frozen (previous-iteration) profile

                p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                    F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)

                r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                              np.array([p_CH4]), np.array([p_H2O]))[0])  # Sabatier rate [mol/(kg_cat·s)] at local T, p_i
                qs  = float(q_star(T_l, np.array([p_H2O]))[0])         # DA equilibrium H2O loading at local T, p_H2O
                Kl  = float(K_LDF(T_l,  np.array([p_H2O]))[0])         # LDF mass-transfer coefficient at local T, p_H2O
                ads = Kl*(qs - q_l)                                    # LDF adsorption rate [mol/(kg_ads·s)]: drives q toward qs

                gas_cap_l = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
                denom     = gas_cap_l - solid_cap                      # energy-balance denominator; positive here since gas_dominates

                Q_rxn  = (-dH_r)   * rho_bed_cat * r                   # volumetric heat release from reaction [W/m³]
                Q_ads  = (-dH_ads) * rho_bed_ads * ads                 # volumetric heat release from adsorption [W/m³]
                Q_wall = U_a * (T_l - T_wall)                          # volumetric heat loss to the wall [W/m³]
                dTdz   = (Q_rxn + Q_ads - Q_wall) / denom              # gas-side energy balance, integrated forward in z (co-current with the gas)

                return [
                    -rho_bed_cat * r,                        # dF_CO2/dz [mol/(m3.s)]: kg_cat/m3_bed * mol/(kg_cat.s); CO2 consumed 1:1 with r
                    -4.0*rho_bed_cat * r,                    # dF_H2/dz  [mol/(m3.s)]: same units, x4 for the H2:CO2 stoichiometric ratio
                    +rho_bed_cat * r,                        # dF_CH4/dz [mol/(m3.s)]: CH4 produced 1:1 with CO2 consumed
                    2.0*rho_bed_cat*r - rho_bed_ads*ads,     # dF_H2O/dz [mol/(m3.s)]: 2x production (2 H2O per CO2) minus adsorption uptake (rho_bed_ads * ads, same units)
                    dTdz,                                     # dT/dz [K/m]: (Q_rxn+Q_ads-Q_wall)[W/m3] / denom[W/(m2.K)]
                ]

            gs = solve_ivp(gas_rhs, [0.0, L_b],                        # gs = "gas solve" (this file's naming); solve_ivp always returns an OdeResult with fixed attrs .t (independent var) and .y (state array), regardless of what they physically represent
                           [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],    # BCs at z=0: F_CO2(0)=F_in_CO2, F_H2(0)=F_in_H2, F_CH4(0)=F_in_CH4, F_H2O(0)=0 (dry feed), T(0)=T_K (gas inlet temperature)
                           method='BDF', rtol=1e-4,                    # BDF: implicit stiff solver, needed since reaction/adsorption rates are highly sensitive to T and p_H2O
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8, 1e-2]),  # per-component absolute tolerance: tight (1e-8) for fluxes ~1e-3-1e-1 mol/(m2.s), looser (1e-2 K) for T ~500 K
                           t_eval=z_grid, dense_output=False)           # report solution on the shared z_grid so profiles line up across Gauss-Seidel iterations
            if not gs.success:
                print(f"    [solve_ivp FAILED] gas IVP (gas-dominant): {gs.message}  "
                      f"(z_last={gs.t[-1]:.4f} m)")
                return None                                            # BDF failed to integrate the interval (e.g. RHS blew up) -> caller reports this u_s as FAILED

            # gs.y has shape (5, N): 5 state rows (F_CO2, F_H2, F_CH4, F_H2O, T) x N points on z_grid
            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)  # gs.y[i] = solved profile of state i across z_grid; clip to >=0 (undoes any BDF undershoot not caught inside gas_rhs)
            F_CH4_prof = np.maximum(gs.y[2], 0.0);  F_H2O_prof = np.maximum(gs.y[3], 0.0)  # clipping to make sure no negative values (non-physical)
            T_prof_new = np.maximum(gs.y[4], 200.0)                                        # same 200 K floor as inside gas_rhs, applied to the returned profile; "_new" = this iteration's T, blended with old T_prof later

            F_tot_prof  = np.maximum(F_CO2_prof + F_H2_prof + F_CH4_prof + F_H2O_prof, 1e-30)
            p_H2O_prof  = F_H2O_prof / F_tot_prof * P_bar             # p_H2O(z) from the freshly-solved gas profile, needed to drive the solid IVP below
            p_H2O_fn    = interp1d(z_grid, p_H2O_prof, kind='linear',    #again interpolated to get continuous curve for the ODE
                                   bounds_error=False,
                                   fill_value=(p_H2O_prof[0], p_H2O_prof[-1]))
            T_fn_new    = interp1d(z_grid, T_prof_new, kind='linear',  # T(z) just solved, used (not the stale T_fn) so the solid sees this iteration's temperature
                                   bounds_error=False,
                                   fill_value=(T_prof_new[0], T_prof_new[-1]))

            # ── SOLID IVP: state = [q] ────────────────────────────────────────
            def solid_rhs(zeta, q_arr):
                z_pos    = L_b - float(zeta)                           # solid flows z: L->0, so zeta (0->L) is distance travelled from the solid's inlet at z=L
                T_local  = float(T_fn_new(z_pos))                      # gas temperature at this position, from the profile just solved above
                p_H2O_l  = max(float(p_H2O_fn(z_pos)), 0.0)
                qs  = float(q_star(T_local, np.array([p_H2O_l]))[0])
                Kl  = float(K_LDF(T_local,  np.array([p_H2O_l]))[0])
                q_val = max(float(q_arr[0]), 0.0)
                return [Kl*(qs - q_val)/u_s]                           # dq/dzeta = ads_rate/u_s: sorbent loading builds up as it travels through the bed

            ss = solve_ivp(solid_rhs, [0.0, L_b], [0.0],               # ss = "solid solve" (this file's naming); same fixed .t/.y attrs as gs, here .t is zeta not z. BC: q(zeta=0) = 0, i.e. fresh (unloaded) sorbent enters at z=L
                           method='BDF', rtol=1e-4, atol=1e-8,
                           max_step=1e-3,                              # cap step size to resolve the steep q(zeta) front that forms near zeta=0 at low u_s (dq/dzeta ~ 1/u_s); matches solid_rhs_with_T below
                           t_eval=np.linspace(0.0, L_b, N),            # report on N points evenly spaced over zeta in [0, L_b] -- same values as z_grid, just rebuilt here for the zeta axis
                           dense_output=False)
            if not ss.success:
                print(f"    [solve_ivp FAILED] solid IVP (gas-dominant): {ss.message}  "
                      f"(zeta_last={ss.t[-1]:.4f} m)")
                return None

            z_from_zeta = L_b - ss.t                                   # map solid's zeta coordinate back to the shared bed coordinate z. ss.t is the zeta values at which the solver returned q(zeta). L_b - zeta = z.
            q_from_zeta = np.maximum(ss.y[0], 0.0)
            sort_idx    = np.argsort(z_from_zeta)                      # z_from_zeta is descending (zeta increasing -> z decreasing); resort ascending for np.interp
            q_new = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])

            q_prof_new = 0.5*q_prof + 0.5*q_new                        # under-relaxed Gauss-Seidel update (damps oscillation between the two coupled IVPs)
            T_prof     = 0.5*T_prof + 0.5*T_prof_new

            scale = max(np.max(q_prof_new), 1e-8)                      # normalise by the peak loading so err is a relative (not absolute) measure
            err   = np.max(np.abs(q_prof_new - q_prof)) / scale        # convergence metric: max relative change in q(z) between successive iterations
            q_prof = q_prof_new

        else:
            # ── solid dominates (u_s > u_s*): T is carried by the solid, not the gas ──
            # ── GAS IVP: state = [F_CO2, F_H2, F_CH4, F_H2O]  (T frozen) ────
            def gas_rhs_no_T(z, y):                                    # z = current pos in bed, y = [F_CO2, F_H2, F_CH4, F_H2O] at this z. T is NOT part of the state vector here: it's frozen from the previous iteration's solid-IVP result (T_fn built at top of the outer loop)
                F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)   # clip fluxes to >=0, same reasoning as gas_rhs above
                F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
                T_l     = max(float(T_fn(z)), 200.0)                  # T is NOT solved here: it's frozen from the previous iteration's solid-IVP result (T_fn built at top of the outer loop)
                q_l     = max(float(q_fn(z)), 0.0)                    # q also frozen from the previous iteration, same as in the gas-dominant branch

                p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                    F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)

                r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                              np.array([p_CH4]), np.array([p_H2O]))[0])  # rate at the frozen T_l (not yet updated this iteration)
                qs  = float(q_star(T_l, np.array([p_H2O]))[0])
                Kl  = float(K_LDF(T_l,  np.array([p_H2O]))[0])
                ads = Kl*(qs - q_l)                                   # adsorption rate, needed only to consume H2O from the gas phase (dF_H2O/dz below) -- q itself is solved separately, in the solid IVP

                return [                                    # returns the dF_i/dz vector for the gas IVP, same units as in gas_rhs above (mol/(m3.s)) as function of z throughout bed
                    -rho_bed_cat * r,                        # dF_CO2/dz [mol/(m3.s)] -- same species balance as gas_rhs, but no dTdz row: T isn't part of this state vector
                    -4.0*rho_bed_cat * r,                    # dF_H2/dz
                    +rho_bed_cat * r,                        # dF_CH4/dz
                    2.0*rho_bed_cat*r - rho_bed_ads*ads,     # dF_H2O/dz
                ]

            gs = solve_ivp(gas_rhs_no_T, [0.0, L_b],                  # same gas-side BCs as the gas-dominant branch: fixed inlet fluxes at z=0
                           [F_in_CO2, F_in_H2, F_in_CH4, 0.0],          # these are the BC's. F_h20 is zero at the inlet, since the feed is dry. T is not part of the state vector here, so it's not included in the initial condition.
                           method='BDF', rtol=1e-4,
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8]),   # only 4 tolerances now (no T entry), since T isn't a state variable in this IVP
                           t_eval=z_grid, dense_output=False)
            if not gs.success:
                print(f"    [solve_ivp FAILED] gas IVP (solid-dominant): {gs.message}  "
                      f"(z_last={gs.t[-1]:.4f} m)")
                return None

            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)  # clip outputs to >=0, same reasoning as the gas-dominant branch, below zero is non-physical and can cause blow-up in the solid IVP
            F_CH4_prof = np.maximum(gs.y[2], 0.0);  F_H2O_prof = np.maximum(gs.y[3], 0.0)
            F_tot_prof = np.maximum(F_CO2_prof + F_H2_prof + F_CH4_prof + F_H2O_prof, 1e-30)

            _make_fn = lambda p: interp1d(z_grid, p, kind='linear',   # shared helper: builds the same kind of clamped-linear interpolant as q_fn/T_fn above, reused for 5 different profiles below
                                          bounds_error=False, fill_value=(p[0], p[-1]))
            F_CO2_fn = _make_fn(F_CO2_prof);  F_H2_fn  = _make_fn(F_H2_prof)   # continuous F_i(z) lookups, needed inside solid_rhs_with_T since the solid IVP is integrated in zeta, not z
            F_CH4_fn = _make_fn(F_CH4_prof);  F_H2O_fn = _make_fn(F_H2O_prof)
            F_tot_fn = _make_fn(F_tot_prof)

            # ── SOLID IVP: state = [q, T] ─────────────────────────────────────
            def solid_rhs_with_T(zeta, y_arr):
                q_val = max(float(y_arr[0]), 0.0)                     # sorbent loading, solved forward in zeta (0 at the solid's own inlet, z=L_b)
                T_val = max(float(y_arr[1]), 200.0)                    # temperature -- HERE it's part of the solid's own state, unlike gas_rhs_no_T above
                z_pos = L_b - float(zeta)                              # map zeta back to the shared bed coordinate z, to look up the (already-solved) gas composition

                F_CO2_l = max(float(F_CO2_fn(z_pos)), 0.0)             # gas fluxes at this position, taken from the gas IVP just solved above (fixed for the rest of this solid integration)
                F_H2_l  = max(float(F_H2_fn(z_pos)),  0.0)
                F_CH4_l = max(float(F_CH4_fn(z_pos)), 0.0)
                F_H2O_l = max(float(F_H2O_fn(z_pos)), 0.0)
                F_tot_l = max(float(F_tot_fn(z_pos)), 1e-30)

                p_CO2 = F_CO2_l/F_tot_l*P_bar;  p_H2  = F_H2_l /F_tot_l*P_bar   # ideal-gas partial pressures at z_pos, same formula as _partial_pressures but done inline here
                p_CH4 = F_CH4_l/F_tot_l*P_bar;  p_H2O = F_H2O_l/F_tot_l*P_bar

                r   = float(reaction_rate_SI(T_val, np.array([p_CO2]), np.array([p_H2]),
                                              np.array([p_CH4]), np.array([p_H2O]))[0])  # rate uses T_val (the solid IVP's own evolving T), not a frozen T
                qs  = float(q_star(T_val, np.array([p_H2O]))[0])
                Kl  = float(K_LDF(T_val, np.array([p_H2O]))[0])
                ads = Kl*(qs - q_val)                                  # adsorption rate driving both dq/dzeta below and the heat-of-adsorption term Q_ads

                gas_cap_l   = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
                solid_denom = solid_cap - gas_cap_l                    # energy-balance denominator; positive here since solid_cap dominates (opposite sign convention to "denom" in the gas-dominant branch)
                _solid_denom_min[0] = min(_solid_denom_min[0], solid_denom)  # track weakest point of the energy-balance denominator, for FAILED diagnostics below

                Q_rxn  = (-dH_r)   * rho_bed_cat * r                   # volumetric heat release from reaction [W/m3], same as gas_rhs
                Q_ads  = (-dH_ads) * rho_bed_ads * ads                 # volumetric heat release from adsorption [W/m3]
                Q_wall = U_a * (T_val - T_wall)                        # volumetric heat loss to the wall [W/m3]

                return [Kl*(qs - q_val)/u_s,                          # dq/dzeta [mol/(kg.m)]: same LDF-loading-per-distance balance as solid_rhs above
                        (Q_rxn + Q_ads - Q_wall) / solid_denom]        # dT/dzeta [K/m]: solid-side energy balance, integrated in the solid's own direction of travel (zeta, i.e. z: L_b->0)

            ss = solve_ivp(solid_rhs_with_T, [0.0, L_b], [0.0, T_K],   # BCs at zeta=0 (solid's own inlet, z=L_b): q=0 (fresh sorbent), T=T_K -- assumes the solid enters at the same inlet temperature as the gas
                           method='BDF', rtol=1e-4, atol=np.array([1e-8, 0.1]),  # atol for T is 1.0 K here (looser than the 1e-2 K used in the gas-dominant branch), since T now spans the full reactor's exotherm as a state variable rather than a small correction
                           max_step=1e-3,                              # caps BDF's internal step to 1e-3 m: without this the stiff T equation could take steps too large to resolve sharp reaction/adsorption fronts
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                sd_min = _solid_denom_min[0]
                sd_str = f"{sd_min:.2f}" if np.isfinite(sd_min) else "n/a"
                print(f"    [solve_ivp FAILED] solid IVP (solid-dominant): {ss.message}  "
                      f"(zeta_last={ss.t[-1]:.4f} m, min solid_denom={sd_str} W/(m2.K))")
                return None

            z_from_zeta  = L_b - ss.t                                  # flip zeta back onto the shared z coordinate, same as in the gas-dominant branch's solid IVP
            q_from_zeta  = np.maximum(ss.y[0], 0.0)
            T_from_zeta  = np.maximum(ss.y[1], 200.0)                  # floor T again on the output array, mirroring the floor already applied to T_val inside solid_rhs_with_T
            sort_idx     = np.argsort(z_from_zeta)                     # re-sort ascending in z before interpolating (z_from_zeta is descending, since zeta increases as z decreases)
            q_new  = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])
            T_new  = np.interp(z_grid, z_from_zeta[sort_idx], T_from_zeta[sort_idx])

            q_prof_new = 0.5*q_prof + 0.5*q_new                        # under-relaxed Gauss-Seidel update, same damping factor as the gas-dominant branch
            T_prof_new = 0.5*T_prof + 0.5*T_new                        # T also under-relaxed here, since it's now solved (not frozen) in this branch

            err_q = np.max(np.abs(q_prof_new - q_prof)) / max(np.max(q_prof_new), 1e-8)  # relative change in q(z), same metric as the gas-dominant branch
            err_T = np.max(np.abs(T_prof_new - T_prof)) / T_K          # relative change in T(z), normalised by the inlet temperature (K, absolute scale) since T is now solved rather than frozen
            err   = max(err_q, err_T)                                  # convergence requires BOTH q and T to have stabilised, unlike the gas-dominant branch where only q is checked

            q_prof = q_prof_new
            T_prof = T_prof_new                                        # T_prof is updated here (not frozen); next iteration's q_fn/T_fn will reflect this solid-solved T

        if err < tol:                                                  # outer Gauss-Seidel convergence check: independent of N and of solve_ivp's own rtol/atol above
            converged = True
            break                                                      # exits well before max_iter in practice; max_iter is only a safety cap, not a target iteration count

    # ── Final recompute on fine grid ─────────────────────────────────────────
    # Purpose: the Gauss-Seidel loop above has converged q_prof/T_prof on the coarse z_grid; this section does ONE more gas-side solve_ivp call, at tighter tolerance
    # (rtol=1e-6 vs 1e-4, atol ~1e-10 vs 1e-8) and on a separate fixed z_fine grid (300 pts), treating the converged q(z)/T(z) as fixed (no further outer iteration).
    # It exists purely to report a cleaner final profile -- it is NOT part of the  convergence loop, so its extra accuracy doesn't feed back into anything.

    z_fine = np.linspace(0.0, L_b, 300)                                 # fixed at 300 pts regardless of N -- NOT necessarily finer than z_grid; with N>300 (e.g. this file's default N=400) this is actually a coarser axis than the iteration grid
    q_fn_f = interp1d(z_grid, q_prof, kind='linear',                    # converged q(z) profile, frozen for this one-shot recompute (mirrors q_fn inside the iteration loop)
                      bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))
    T_fn_f = interp1d(z_grid, T_prof, kind='linear',                    # converged T(z) profile, frozen (used only in the solid-dominant/else branch below, where T isn't re-solved here)
                      bounds_error=False, fill_value=(T_prof[0], T_prof[-1]))

    if gas_dominates:
        def gas_rhs_final(z, y):                                       # identical physics to gas_rhs in the iteration loop above, just re-solved once more at tighter tolerance
            F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
            F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
            T_l     = max(y[4], 200.0)
            q_l     = max(float(q_fn_f(z)), 0.0)                       # q taken from the already-converged profile, not updated further
            p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
            r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                          np.array([p_CH4]), np.array([p_H2O]))[0])
            qs  = float(q_star(T_l, np.array([p_H2O]))[0])
            Kl  = float(K_LDF(T_l, np.array([p_H2O]))[0])
            ads = Kl*(qs - q_l)
            gas_cap_l = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
            denom = gas_cap_l - solid_cap
            Q_rxn = (-dH_r)*rho_bed_cat*r;  Q_ads = (-dH_ads)*rho_bed_ads*ads
            Q_wall = U_a*(T_l - T_wall)
            return [
                -rho_bed_cat*r,
                -4.0*rho_bed_cat*r,
                +rho_bed_cat*r,
                2.0*rho_bed_cat*r - rho_bed_ads*ads,
                (Q_rxn + Q_ads - Q_wall)/denom,
            ]
        gf = solve_ivp(gas_rhs_final, [0.0, L_b],                      # gf = "gas final" (this file's naming), same BCs as the iteration-loop gas IVP
                       [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],
                       method='BDF', rtol=1e-6,                        # tightened vs the 1e-4 used during iteration: cheap to afford since this runs only once, after convergence
                       atol=np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-3]),  # correspondingly tighter absolute tolerances (T's atol also tightened, 1e-3 vs 1e-2 during iteration)
                       t_eval=z_fine, dense_output=False)
        F_CO2f = np.maximum(gf.y[0], 0.0);  F_H2f  = np.maximum(gf.y[1], 0.0)
        F_CH4f = np.maximum(gf.y[2], 0.0);  F_H2Of = np.maximum(gf.y[3], 0.0)
        T_fine = np.maximum(gf.y[4], 200.0)                             # T re-solved here (gas-dominant: T is part of this IVP's state, same as in the iteration loop)

    else:
        def gas_rhs_final_no_T(z, y):                                  # identical physics to gas_rhs_no_T in the iteration loop, T still frozen (solid-dominant branch never solves T in the gas IVP)
            F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
            F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
            T_l     = max(float(T_fn_f(z)), 200.0)                     # T looked up from the converged profile, not re-solved (matches gas_rhs_no_T's use of T_fn during iteration)
            q_l     = max(float(q_fn_f(z)), 0.0)
            p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
            r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                          np.array([p_CH4]), np.array([p_H2O]))[0])
            qs  = float(q_star(T_l, np.array([p_H2O]))[0])
            Kl  = float(K_LDF(T_l, np.array([p_H2O]))[0])
            ads = Kl*(qs - q_l)
            return [
                -rho_bed_cat*r,
                -4.0*rho_bed_cat*r,
                +rho_bed_cat*r,
                2.0*rho_bed_cat*r - rho_bed_ads*ads,
            ]
        gf = solve_ivp(gas_rhs_final_no_T, [0.0, L_b],
                       [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                       method='BDF', rtol=1e-6,
                       atol=np.array([1e-10, 1e-10, 1e-10, 1e-10]),    # no T entry here either, consistent with gas_rhs_no_T's 4-element state
                       t_eval=z_fine, dense_output=False)
        F_CO2f = np.maximum(gf.y[0], 0.0);  F_H2f  = np.maximum(gf.y[1], 0.0)
        F_CH4f = np.maximum(gf.y[2], 0.0);  F_H2Of = np.maximum(gf.y[3], 0.0)
        T_fine = np.interp(z_fine, z_grid, T_prof)                     # T here is NOT re-solved at all (unlike the gas-dominant branch) -- just interpolated from the already-converged coarse T_prof, since T lives in the solid IVP in this regime and that IVP isn't recomputed here

    q_fine   = np.interp(z_fine, z_grid, q_prof)                       # q always just interpolated from the converged coarse profile, in both regimes (never re-solved in this final section)
    F_totf   = np.maximum(F_CO2f + F_H2f + F_CH4f + F_H2Of, 1e-30)
    p_CO2f   = F_CO2f/F_totf*P_bar;  p_H2f  = F_H2f /F_totf*P_bar     # partial pressures on the fine grid, same ideal-gas formula as _partial_pressures
    p_CH4f   = F_CH4f/F_totf*P_bar;  p_H2Of = F_H2Of/F_totf*P_bar
    r_fine   = reaction_rate_SI(T_fine, p_CO2f, p_H2f, p_CH4f, p_H2Of)  # reaction rate recomputed as a plain output quantity (for plotting) -- not used inside any solve_ivp call, so this doesn't affect accuracy of the solution itself
    X_CO2    = np.clip(1.0 - F_CO2f/F_in_CO2, 0.0, 1.0)                 # CO2 conversion [-]: fraction of inlet CO2 flux consumed by z; clipped to [0,1] to guard against tiny numerical overshoot

    # Convert fluxes to concentrations for output (compatible with plot code)
    u_g_fine = F_totf * R_gas * T_fine / P_Pa                          # local superficial gas velocity [m/s] from ideal gas law (F_tot = u_g*P/(R*T)), recovered since the flux form never solves for u_g directly
    C_CO2f   = F_CO2f / u_g_fine                                       # C_i = F_i/u_g [mol/m3]: converts molar flux back to a concentration, purely so downstream plotting code (written for the concentration-form model) keeps working unchanged
    C_H2f    = F_H2f  / u_g_fine
    C_CH4f   = F_CH4f / u_g_fine
    C_H2Of   = F_H2Of / u_g_fine

    return dict(z=z_fine, C_CO2=C_CO2f, C_H2=C_H2f, C_CH4=C_CH4f,
                C_H2O=C_H2Of, q=q_fine, T=T_fine, r=r_fine, X_CO2=X_CO2,
                converged=converged, n_iter=it+1, conv_err=float(err),  # it, err are whatever the outer Gauss-Seidel loop last left them at when it stopped (converged or hit max_iter)
                gas_dominates=gas_dominates)
# endregion


# region 4. STEP 1 — ANALYTIC LAMBDA_THERMO PRE-FILTER (no solver calls)
# =============================================================================
def _fmt_seconds(s):
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"

def _q_physics_init(T_K, N=150):
    """
    Isothermal, no-adsorption gas pass to build a physics-motivated initial q profile, used
    to cold-start solve_mpb's Gauss-Seidel loop instead of starting from q=0 everywhere.
    Cheaper than a real solve_mpb call: single IVP, no energy balance (T fixed at T_K), no
    coupling loop. Returned q(z) rises with z (low near z=0, high near z=L) since it tracks
    water building up along the gas's own direction of travel; the caller flips it ([::-1])
    before use, since the solid travels z=L->0 and enters unloaded (q~0) at z=L.
    """
    def rhs_noads(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        return [
            -rho_bed_cat*r, -4.0*rho_bed_cat*r,
            +rho_bed_cat*r,  2.0*rho_bed_cat*r,
        ]
    z_grid = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs_noads, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                    method='BDF', rtol=1e-4, atol=1e-8, t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    F_tot_f = np.maximum(sol.y[0] + sol.y[1] + sol.y[2] + sol.y[3], 1e-30)
    p_H2O_f = np.maximum(sol.y[3], 0.0)/F_tot_f * P_bar
    return dict(q=q_star(T_K, p_H2O_f),
                X_CO2_noSE=float(np.clip(1.0 - F_CO2_f[-1]/F_in_CO2, 0.0, 1.0)))

T_SWEEP_K  = T_SWEEP_C + 273.15

# q*_max: DA isotherm evaluated at the water partial pressure this reactor can actually
# produce -- the outlet p_H2O at FULL CO2 conversion (Sabatier stoichiometry: 1 mol CO2 ->
# 2 mol H2O, net -2 mol gas per mol CO2 reacted), not at P_sat(T). Using P_sat(T) (tens of
# bar at 280 C) hugely overstates the reachable loading versus the ~0.05-0.1 bar water
# partial pressures actually seen in a 1 bar, CH4-diluted process stream (same distinction
# as the ~1.2 vs ~7 mol/kg gap seen against the Wei/Bareschino breakthrough validation).
p_H2O_full = (2.0*y_CO2_in) / (y_CH4_in + 3.0*y_CO2_in) * P_bar   # [bar]  y_H2O at X_CO2=1, independent of GHSV (mole-fraction ratio only)
q_star_max = float(q_star(T_SWEEP_K, np.array([p_H2O_full]))[0])  # [mol/kg]  realistic ceiling loading at T_SWEEP_C

US_GRID   = np.linspace(US_MIN_MM, US_MAX_MM, N_US) * 1e-3     # [m/s]
GHSV_GRID = np.linspace(GHSV_MIN, GHSV_MAX, N_GHSV)             # [m3_STP/(kg_ads·h)]

F_CO2_in_grid = compute_inlet_fluxes(GHSV_GRID)[0]              # vectorised over GHSV_GRID -> shape (N_GHSV,)

# Lambda_thermo[i, j] for u_s = US_GRID[i], GHSV = GHSV_GRID[j]
Lambda_thermo = (US_GRID[:, None] * rho_bed_ads * q_star_max) / (2.0 * F_CO2_in_grid[None, :])
mask_survive  = Lambda_thermo >= LAMBDA_MIN

print(f"\nStep 1: Lambda_thermo pre-filter  (q*_max={q_star_max:.3f} mol/kg @ "
      f"p_H2O_full={p_H2O_full*1e3:.1f} mbar, T={T_SWEEP_C:.0f} C)")
print(f"  grid: {N_US} x {N_GHSV} = {N_US*N_GHSV} cells, "
      f"{int(mask_survive.sum())} survive Lambda_thermo >= {LAMBDA_MIN}")
# endregion


# region 5. STEP 2 — SOLVE SURVIVING CELLS (boustrophedon warm starts)
# =============================================================================
cell_results  = {}          # (i_us, i_ghsv) -> per-cell result dict, or None for masked-out cells
n_survive     = int(mask_survive.sum())
n_solved      = 0
t_sweep_start = time.perf_counter()

set_inlet_fluxes(GHSV_GRID[0])
q_cold = _q_physics_init(T_SWEEP_K)['q'][::-1]     # cold-start guess, used only if no solved neighbour exists yet

def _nearest_col_solution(i_us, j_ghsv):
    """Nearest already-solved neighbour at the same u_s row, searching backwards through GHSV columns."""
    for j_prev in range(j_ghsv - 1, -1, -1):
        e = cell_results.get((i_us, j_prev))
        if e is not None and e.get('q') is not None:
            return e['q']
    return None

def _h2o_balance(u_s, res):
    """Sanity check: does H2O produced by reaction == H2O leaving in the gas + H2O leaving on the solid?
    Relies on F_in_CO2 (set via set_inlet_fluxes for this cell's GHSV) and rho_bed_ads, same as the
    original single-u_s sweep's version of this check. Returns the raw balance terms so they can be
    stored per cell (see cell_results below), not just printed."""
    T_out      = float(res['T'][-1])                                   # gas outlet temperature, z=L_b (last point of the fine profile)
    y_CO2_out  = float(res['C_CO2'][-1]) * R_gas * T_out / P_Pa        # recover CO2 mole fraction at outlet from concentration via ideal gas law (C = P*y/(R*T) => y = C*R*T/P)
    F_CO2_out  = F_in_CO2 * (1.0 - float(res['X_CO2'][-1]))            # CO2 molar flux remaining at outlet, from the overall conversion
    F_tot_out  = F_CO2_out / max(y_CO2_out, 1e-30)                      # total outlet molar flux, recovered from F_CO2_out/y_CO2_out (since y_CO2 = F_CO2/F_tot)
    F_H2O_out  = float(res['C_H2O'][-1]) * R_gas * T_out / P_Pa * F_tot_out  # H2O molar flux actually leaving in the gas: y_H2O_out (same ideal-gas recovery as y_CO2_out above) times F_tot_out
    F_H2O_prod = 2.0 * F_in_CO2 * float(res['X_CO2'][-1])               # H2O that SHOULD have been produced overall: 2 mol H2O per mol CO2 reacted (stoichiometry), times total CO2 converted
    F_H2O_ads  = u_s * rho_bed_ads * float(res['q'][0])                 # H2O leaving adsorbed on the solid at its own outlet (z=0): u_s[m/s] * rho_bed_ads[kg_ads/m3] * q(z=0)[mol/kg_ads] = mol/(m2.s), same units as the other two
    bal_err    = (F_H2O_out + F_H2O_ads - F_H2O_prod) / max(F_H2O_prod, 1e-30) * 100  # relative mass-balance error [%]: (gas-phase exit + solid-phase exit) vs. what reaction stoichiometry says should have been produced
    return dict(F_H2O_prod=F_H2O_prod, F_H2O_out=F_H2O_out, F_H2O_ads=F_H2O_ads, bal_err_pct=bal_err)

def _h2o_balance_line(bal):
    return (f"    H2O balance [mmol/(m²·s)]:  produced={bal['F_H2O_prod']*1e3:.3f}  "
            f"gas_out={bal['F_H2O_out']*1e3:.3f}  solid_out={bal['F_H2O_ads']*1e3:.3f}  "
            f"err={bal['bal_err_pct']:+.1f}%")

print("\nStep 2: solving surviving cells (snake traversal, warm-started)...")
for j, GHSV in enumerate(GHSV_GRID):
    set_inlet_fluxes(GHSV)
    row_order = range(N_US) if j % 2 == 0 else range(N_US - 1, -1, -1)   # alternate direction each column (snake)
    q_prev_in_col = None

    for i in row_order:
        if not mask_survive[i, j]:
            cell_results[(i, j)] = None
            continue

        u_s = US_GRID[i]
        if q_prev_in_col is not None:
            q_init = q_prev_in_col
        else:
            q_init = _nearest_col_solution(i, j)
            if q_init is None:
                q_init = q_cold

        t0  = time.perf_counter()
        res = solve_mpb(u_s, T_SWEEP_K, T_wall=T_SWEEP_K, q_init=q_init)
        dt  = time.perf_counter() - t0
        n_solved += 1

        if res is not None:
            q_prev_in_col = res['q']
            h2o_bal = _h2o_balance(u_s, res)
            cell_results[(i, j)] = dict(
                z=res['z'], X_CO2=res['X_CO2'], T=res['T'], q=res['q'],
                n_iter=res['n_iter'], converged=res['converged'],
                conv_err=res['conv_err'], solve_time=dt,
                gas_dominates=res['gas_dominates'], u_s=u_s, GHSV=GHSV,
                h2o_balance=h2o_bal,
            )
            regime  = "gas" if res['gas_dominates'] else "solid"
            tag     = f"{regime}-dom, ok" if res['converged'] else f"{regime}-dom, not-conv"
            it_str  = f"{res['n_iter']} iter, err={res['conv_err']:.2e}"
        else:
            q_prev_in_col = None       # failed cell can't warm-start the next one in this column
            cell_results[(i, j)] = dict(
                z=None, X_CO2=None, T=None, q=None, n_iter=None, converged=False,
                conv_err=np.inf, solve_time=dt, gas_dominates=None, u_s=u_s, GHSV=GHSV,
                h2o_balance=None,
            )
            tag    = "FAILED"
            it_str = "n/a"

        elapsed = time.perf_counter() - t_sweep_start
        eta     = elapsed / n_solved * (n_survive - n_solved)
        print(f"  [{n_solved}/{n_survive}] u_s={u_s*1e3:.2f} mm/s  GHSV={GHSV:.2f}  "
              f"[{tag}, {it_str}]  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
        if res is not None:
            print(_h2o_balance_line(h2o_bal))

print(f"Step 2 done. {n_solved} cell(s) solved in {_fmt_seconds(time.perf_counter() - t_sweep_start)}")

# ── Retry pass: revisit non-converged/failed survivors, warm-starting from converged neighbours ──
print("\nRetry pass for non-converged/failed surviving cells...")
n_retried = 0
for j in reversed(range(N_GHSV)):
    GHSV = GHSV_GRID[j]
    set_inlet_fluxes(GHSV)
    row_order = list(range(N_US) if j % 2 == 0 else range(N_US - 1, -1, -1))
    row_order = row_order[::-1]        # reverse of the original Step-2 sweep direction for this column
    q_retry = None

    for i in row_order:
        if not mask_survive[i, j]:
            continue
        e = cell_results.get((i, j))
        if e is not None and e['converged']:
            q_retry = e['q']
            continue
        if q_retry is None:
            q_retry = _nearest_col_solution(i, j)
        if q_retry is None:
            continue

        u_s     = US_GRID[i]
        t0      = time.perf_counter()
        res_new = solve_mpb(u_s, T_SWEEP_K, T_wall=T_SWEEP_K, q_init=q_retry)
        dt      = time.perf_counter() - t0
        n_retried += 1

        if res_new is not None:
            old_err     = e['conv_err'] if e is not None else np.inf
            improved    = res_new['converged'] or res_new['conv_err'] < old_err
            h2o_bal_new = _h2o_balance(u_s, res_new)
            if improved:
                cell_results[(i, j)] = dict(
                    z=res_new['z'], X_CO2=res_new['X_CO2'], T=res_new['T'], q=res_new['q'],
                    n_iter=res_new['n_iter'], converged=res_new['converged'],
                    conv_err=res_new['conv_err'], solve_time=dt,
                    gas_dominates=res_new['gas_dominates'], u_s=u_s, GHSV=GHSV,
                    h2o_balance=h2o_bal_new,
                )
            tag = "ok" if res_new['converged'] else "still-nc"
            print(f"  RETRY u_s={u_s*1e3:.2f} mm/s  GHSV={GHSV:.2f}  "
                  f"[{tag}, {res_new['n_iter']} iter, err={res_new['conv_err']:.2e}]  "
                  f"({'kept' if improved else 'discarded'})  ({dt:.1f}s)")
            print(_h2o_balance_line(h2o_bal_new))
            q_retry = res_new['q']
        else:
            print(f"  RETRY u_s={u_s*1e3:.2f} mm/s  GHSV={GHSV:.2f}  FAILED  ({dt:.1f}s)")
            q_retry = None

print(f"Retry pass done. {n_retried} cell(s) re-run.")
# endregion


# region 6. STEP 3 — POST-PROCESSING METRICS PER CELL
# =============================================================================
def _z_cross(z_arr, X_arr, target):
    """z where X_CO2(z) first crosses `target`; NaN if never reached within the domain."""
    if z_arr is None or np.max(X_arr) < target:
        return np.nan
    return float(np.interp(target, X_arr, z_arr))

for (i, j), e in cell_results.items():
    if e is None or e.get('z') is None:
        continue
    e['z_cross_995'] = _z_cross(e['z'], e['X_CO2'], 0.995)
    e['z_cross_974'] = _z_cross(e['z'], e['X_CO2'], 0.974)
    e['T_max']       = float(np.max(e['T']))

Z995   = np.full((N_US, N_GHSV), np.nan)   # z_cross_995 [m]
Z974   = np.full((N_US, N_GHSV), np.nan)   # z_cross_974 [m]
TMAX   = np.full((N_US, N_GHSV), np.nan)   # peak temperature [C]
NITER  = np.full((N_US, N_GHSV), np.nan)   # Gauss-Seidel iteration count
SOLVET = np.full((N_US, N_GHSV), np.nan)   # solve_mpb wall time [s]

for (i, j), e in cell_results.items():
    if e is None or e.get('z') is None:
        continue
    Z995[i, j]   = e['z_cross_995']
    Z974[i, j]   = e['z_cross_974']
    TMAX[i, j]   = e['T_max'] - 273.15
    NITER[i, j]  = e['n_iter']
    SOLVET[i, j] = e['solve_time']

n_reach_995 = int(np.sum(~np.isnan(Z995)))
print(f"\nStep 3: {n_reach_995}/{n_solved} solved cells reach X_CO2=99.5% within L_trial={L_b:.1f} m")
# endregion


# region 7. STEP 4 — PLOTS
# =============================================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def _savefig(filename):
    stem, ext = os.path.splitext(filename)
    filename  = f'{stem}_active_frac_{active_fraction:.0%}{ext}'
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=150, bbox_inches='tight')

_us_mm = US_GRID * 1e3

def _heatmap(ax, values, title, cbar_label, cmap='viridis'):
    masked   = np.ma.masked_invalid(values)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('lightgrey')
    pcm = ax.pcolormesh(GHSV_GRID, _us_mm, masked, shading='nearest', cmap=cmap_obj)
    cb  = plt.colorbar(pcm, ax=ax);  cb.set_label(cbar_label, fontsize=9)
    cs  = ax.contour(GHSV_GRID, _us_mm, Lambda_thermo, levels=[LAMBDA_MIN],
                     colors='red', linewidths=2)
    ax.clabel(cs, fmt={LAMBDA_MIN: f'Lambda={LAMBDA_MIN}'}, fontsize=8)
    ax.set_xlabel('GHSV [m3_STP/(kg_ads·h)]', fontsize=10)
    ax.set_ylabel('u_s [mm/s]', fontsize=10)
    ax.set_title(title, fontsize=10)
    return pcm

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
fig.suptitle(f'MPB feasibility/sizing sweep  |  T = {T_SWEEP_C:.0f} C  |  '
             f'L_trial = {L_b:.1f} m  |  U_a = {U_a:.0f} W/(m3·K)', fontsize=11)

_heatmap(axes[0], Z995 / L_b,
         '99.5% CO2-conversion crossing  (z_cross_995 / L_trial)',
         'z_cross_995 / L_trial [-]')
cs974 = axes[0].contour(GHSV_GRID, _us_mm, Z974 / L_b, levels=[0.25, 0.5, 0.75, 1.0],
                        colors='k', linestyles='dashed', linewidths=1)
axes[0].clabel(cs974, fmt='%.2f', fontsize=7)

_heatmap(axes[1], SOLVET, 'Solve time per cell', 'solve_mpb wall time [s]', cmap='magma')

_heatmap(axes[2], TMAX, 'Peak bed temperature', 'T_max [C]', cmap='inferno')

plt.tight_layout()
_savefig('heatmap_us_GHSV_sweep.png');  plt.show()
# endregion


# region 8. STEP 5 — SAVE RAW RESULTS
# =============================================================================
sweep_results = dict(
    US_GRID=US_GRID, GHSV_GRID=GHSV_GRID, T_SWEEP_C=T_SWEEP_C, L_b=L_b,
    LAMBDA_MIN=LAMBDA_MIN, Lambda_thermo=Lambda_thermo, mask_survive=mask_survive,
    q_star_max=q_star_max, p_H2O_full=p_H2O_full, cell_results=cell_results,
    Z995=Z995, Z974=Z974, TMAX=TMAX, NITER=NITER, SOLVET=SOLVET,
)
_pkl_path = os.path.join(SAVE_DIR, f'sweep_results_active_frac_{active_fraction:.0%}.pkl')
with open(_pkl_path, 'wb') as f:
    pickle.dump(sweep_results, f)
print(f"\nStep 5: raw sweep results saved to {_pkl_path}")
# endregion
