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
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# region 1. PARAMETERS
# =============================================================================

# --- Bed geometry (Bareschino lab setup) ---
d_b   = 0.050                # [m]   reactor inner (tube) diameter — Bareschino et al. (2023)
L_b   = 2.000                # [m]   bed length — Bareschino et al. (2023)
A_b   = np.pi / 4 * d_b**2   # [m²]  bed cross-sectional area
V_bed = A_b * L_b            # [m³]  total bed volume
eps_b = 0.4                  # [-]   inter-particle (bed) void fraction — typical packed-bed value (not from Bareschino table)

# --- Catalyst and sorbent loading ---
M_cat = 0.064   # [kg]      catalyst mass, 5%Ni-2.5%Ce/13X — Bareschino et al. (2023)
M_ads = 1.22    # [kg]      sorbent mass, 13X zeolite — Bareschino et al. (2023)
rho_bed_cat = M_cat / V_bed          # [kg_cat/m³_bed]  catalyst bulk density  ≈ 16.3 kg/m³
rho_bed_ads = M_ads / V_bed          # [kg_ads/m³_bed]  sorbent bulk density   ≈ 310.7 kg/m³
rho_bed_tot = (M_cat + M_ads) / V_bed  # [kg_solid/m³_bed]  total solids bulk density  ≈ 327.0 kg/m³  — dilute, but kept to match/compare to Bareschino

# --- Particle properties (13X zeolite pellets) ---
d_p   = 2.5e-3   # [m]      particle diameter — this study's pellet size (Same as Bareschino et al. 2023. Wei used 0.75 mm)
eps_p = 0.242    # [-]      intraparticle void fraction — calclated with Wei's pore volume and density (Bareschino)
tau_p = 4.0      # [-]      pore tortuosity factor — Mette et al. (2015) 
rho_p = 1400     # [kg/m³]  particle (skeletal) density of sorbent — Bareschino et al. (2023)

# --- Dubinin-Astakhov isotherm (H2O on 13X) --- fitted myself based on Wei et al. (2021)
W0_DA = 150.00e-6   # [m³/kg_sorbent]  limiting micropore volume
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

# --- Kinetics sensitivity multiplier ---
# Global scale factor applied to the Sabatier rate (see reaction_rate_SI below).
# Read at call time from this module-level variable, so reassigning it between
# solve_mpb() calls (region 4) changes the rate used by every subsequent call.
KINETICS_MULT = 1.0   # [-]

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

# --- Operating conditions ---
P_bar = 1.0                # [bar]  total pressure
P_Pa  = P_bar * 1e5         # [Pa]   total pressure
y_CO2_in = 0.04             # [-]    CO2 inlet mole fraction — Bareschino et al. (2022/2023)
y_H2_in  = 0.16             # [-]    H2 inlet mole fraction (H2/CO2 = 4, stoichiometric)  — Bareschino et al. (2022/2023)
y_CH4_in = 0.80             # [-]    CH4 inlet mole fraction (background/diluent) — Bareschino et al. (2022/2023)

# --- Inlet molar fluxes [mol/(m²·s)] — temperature-independent ---
T_STP   = 273.15                  # [K]                   standard temperature (0 °C)
GHSV    = 0.5                     # [m³_STP/(kg_ads·h)]   gas hourly space velocity — Bareschino et al. (2023)
Q_STP   = GHSV * M_ads / 3600.0   # [m³_STP/s]  volumetric feed flow at STP
u_g_STP = Q_STP / A_b             # [m/s]       superficial gas velocity at STP
F_total_in = u_g_STP * P_Pa / (R_gas * T_STP)  # [mol/(m²·s)]  total molar flux at inlet (ideal gas law)
F_in_CO2   = y_CO2_in * F_total_in   # [mol/(m²·s)]  CO2 inlet molar flux
F_in_H2    = y_H2_in  * F_total_in   # [mol/(m²·s)]  H2 inlet molar flux
F_in_CH4   = y_CH4_in * F_total_in   # [mol/(m²·s)]  CH4 inlet molar flux

# --- MPB scan parameters ---
U_S_LIST  = np.array([5.0]) * 1e-3   # fixed at 5 mm/s for the kinetics sensitivity scan below
T_IN_LIST = [280]

# --- Kinetics sensitivity scan ---
# Sweep the Sabatier rate multiplier at fixed u_s = 5 mm/s (see KINETICS_MULT above).
KINETICS_MULT_LIST = [0.5, 2.0, 4.0, 8.0]

print(f"MPB flux form: d={d_b*100:.0f} cm, L={L_b:.1f} m, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3, u_g_STP={u_g_STP*1e3:.1f} mm/s")
print(f"  F_in_total = {F_total_in:.4f} mol/(m2·s)  (temperature-independent)")
print(f"  U_a = {U_a:.0f} W/(m3·K)")
print(f"  Kinetics sensitivity: mult = {KINETICS_MULT_LIST}  at u_s = {U_S_LIST[0]*1e3:.1f} mm/s")
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
    return k*(p_CO2_s*p_H2_s)**0.5*f_eq/DEN**2*1000.0 * KINETICS_MULT         # r = k*sqrt(p_CO2*p_H2)*f_eq/DEN^2, in mol/(g_cat·s); *1000 -> mol/(kg_cat·s); scaled by the module-level sensitivity multiplier

def q_star(T_K, p_H2O):
    return q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)  # q_star_vec bound to this file's fitted DA parameters, so callers don't have to pass W0/E/n each time

def K_LDF(T_K, p_H2O):
    return K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)  # K_LDF_vec bound to this file's fitted DA parameters, same pattern as q_star above

def equilibrium_conversion(T_K_val):
    """
    Thermodynamic equilibrium CO2 conversion [%] at T_K_val, P = 1 bar, for the fixed
    inlet feed y_CO2=0.04 / y_H2=0.16 / y_CH4=0.80 (H2/CO2 = 4, stoichiometric).
    Solves Q(X) = K_eq for X on a 1 mol-total-inlet basis, then reports X*100.
    """
    K = K_eq_sabatier(T_K_val)                          # equilibrium constant K_eq(T) of the Sabatier reaction
    def f(X):
        d = 1.0 - 0.08*X                                 # total moles per mol inlet feed at conversion X (mole change: -1CO2-4H2+1CH4+2H2O = -2 per mol CO2 reacted)
        return ((0.80+0.04*X)/d * (0.08*X/d)**2          # reaction quotient Q(X) = p_CH4*p_H2O^2 / (p_CO2*p_H2^4), with p_i = y_i since P=1 bar
                / ((0.04*(1-X)/d)*(0.16*(1-X)/d)**4 + 1e-100) - K)  # f(X) = Q(X) - K_eq; root is the equilibrium conversion; 1e-100 avoids 0/0 as X->1
    try:
        return brentq(f, 1e-9, 1-1e-9)*100.0             # bisection for f(X)=0 on X in (0,1), converted to a percentage
    except Exception:
        return 100.0                                     # no sign change in bracket (e.g. K very large) -> equilibrium is effectively full conversion

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

                Q_rxn  = (-dH_r)   * rho_bed_cat * r                   # volumetric heat release from reaction [W/m3], same as gas_rhs
                Q_ads  = (-dH_ads) * rho_bed_ads * ads                 # volumetric heat release from adsorption [W/m3]
                Q_wall = U_a * (T_val - T_wall)                        # volumetric heat loss to the wall [W/m3]

                return [Kl*(qs - q_val)/u_s,                          # dq/dzeta [mol/(kg.m)]: same LDF-loading-per-distance balance as solid_rhs above
                        (Q_rxn + Q_ads - Q_wall) / solid_denom]        # dT/dzeta [K/m]: solid-side energy balance, integrated in the solid's own direction of travel (zeta, i.e. z: L_b->0)

            ss = solve_ivp(solid_rhs_with_T, [0.0, L_b], [0.0, T_K],   # BCs at zeta=0 (solid's own inlet, z=L_b): q=0 (fresh sorbent), T=T_K -- assumes the solid enters at the same inlet temperature as the gas
                           method='BDF', rtol=1e-4, atol=np.array([1e-8, 1.0]),  # atol for T is 1.0 K here (looser than the 1e-2 K used in the gas-dominant branch), since T now spans the full reactor's exotherm as a state variable rather than a small correction
                           max_step=1e-3,                              # caps BDF's internal step to 1e-3 m: without this the stiff T equation could take steps too large to resolve sharp reaction/adsorption fronts
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
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


# region 4. SOLVE LOOP
# =============================================================================
def _compute_noSE(T_K, T_wall, N=300):
    """
    Fixed-bed (u_s=0) reference with wall cooling: no sorption enhancement (noSE).
    No moving solid, so there's no counter-current coupling or Gauss-Seidel needed here --
    just a single co-current gas-phase IVP (species + T together), unlike solve_mpb's
    two-IVP iteration. Used as the conventional-fixed-bed baseline that the MPB curves
    (Plot 2/3) are compared against, to show what sorption enhancement actually buys you.
    """
    def rhs(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        T_l     = max(y[4], 200.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        gas_cap = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)          # no solid_cap here: with u_s=0 there's no counter-flowing solid thermal mass to weigh against the gas
        Q_rxn  = (-dH_r) * rho_bed_cat * r
        Q_wall = U_a * (T_l - T_wall)
        return [
            -rho_bed_cat*r, -4.0*rho_bed_cat*r,                       # dF_CO2/dz, dF_H2/dz -- same species balance as gas_rhs
            +rho_bed_cat*r,  2.0*rho_bed_cat*r,                       # dF_CH4/dz, dF_H2O/dz -- NOTE: no "- rho_bed_ads*ads" term here, i.e. no adsorption uptake; this is exactly what "no sorption enhancement" means
            (Q_rxn - Q_wall) / gas_cap,                                # dT/dz: only reaction heat and wall loss (no Q_ads term either, since no adsorption is happening)
        ]
    z_grid = np.linspace(0, L_b, N)                                    # N=300 default here, independent of solve_mpb's own N -- this function's own local grid, not shared with the MPB solver.
                                                                        # Every array in the returned profile dict below (z, C_CO2, C_H2O, q, r, X_CO2, T) has this same length (300 by default).
                                                                        # Note: solve_mpb's z_fine is ALSO 300 pts, but that's a coincidence of two separately hardcoded values, not a shared parameter -- so Plot 4's overlay of _p0['z'] against a real MPB profile happening to line up in point count is incidental.
    sol = solve_ivp(rhs, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],  # single solve_ivp call suffices: no coupling loop needed since there's only one phase to solve for
                    method='BDF', rtol=1e-5,
                    atol=np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-2]), t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)                                # CO2 flux profile [mol/(m2.s)]
    T_f     = np.maximum(sol.y[4], 200.0)                              # temperature profile [K]
    F_tot_f = np.maximum(sol.y[0] + sol.y[1] + sol.y[2] + sol.y[3], 1e-30)  # total flux [mol/(m2.s)], floored to avoid /0 below
    p_H2O_f = np.maximum(sol.y[3], 0.0)/F_tot_f * P_bar                # partial pressures [bar] = mole fraction * P_bar, same formula as _partial_pressures
    p_CO2_f = F_CO2_f / F_tot_f * P_bar
    p_H2_f  = np.maximum(sol.y[1], 0.0)/F_tot_f * P_bar
    p_CH4_f = np.maximum(sol.y[2], 0.0)/F_tot_f * P_bar
    r_f     = reaction_rate_SI(T_f, p_CO2_f, p_H2_f, p_CH4_f, p_H2O_f)  # rate profile [mol/(kg_cat.s)], diagnostic only (not fed back into the IVP)
    X_f     = np.clip(1.0 - F_CO2_f/F_in_CO2, 0.0, 1.0)                # CO2 conversion profile [-], clipped to [0,1]
    u_g_f   = F_tot_f * R_gas * T_f / P_Pa                              # superficial gas velocity [m/s] from ideal gas law, needed to convert fluxes -> concentrations below
    return dict(X_CO2_noSE=float(X_f[-1]),                             # single scalar: outlet CO2 conversion, the headline number plotted as the "u_s=0 (fixed bed)" reference line
                profile=dict(z=z_grid,                              # all arrays below have the same length as z_grid (N=300 by default)    
                             C_CO2=F_CO2_f/u_g_f,
                             C_H2O=np.maximum(sol.y[3], 0.0)/u_g_f,
                             q=q_star(T_f, p_H2O_f), r=r_f, X_CO2=X_f, T=T_f))  # q here is NOT real sorbent loading (nothing is actually adsorbed in this model) -- it's the DA equilibrium value implied by the un-adsorbed p_H2O. Plotted in Plot 4's "Solid H2O loading" panel as the black 'u_s=0 (fixed bed)' reference line, next to the real (mass-balanced) q(z) curves from actual MPB runs -- don't read it as the same kind of quantity

def _q_physics_init(T_K, N=150):
    """
    Isothermal, no-adsorption gas pass to build a physics-motivated initial q profile, used to warm-start solve_mpb's Gauss-Seidel loop instead of starting from q=0 everywhere.
    Cheaper than a real solve_mpb call: single IVP, no energy balance (T fixed at T_K), no coupling loop. Not physically exact (isothermal, water isn't actually removed here either).

    Returned q(z) here is LOW near z=0 and HIGH near z=L (water builds up along the gas's own direction of travel, z=0->L). The caller (region 4, right after calling this function) flips
    it with q_init_raw[::-1] before passing it to solve_mpb, so the guess actually used is HIGH near z=0 / LOW near z=L -- matching the real MPB shape (solid enters unloaded at z=L, builds
    up toward z=0). Don't be confused by this function's own q looking "backwards" in isolation. magnitude is too high, but it works well enough to warm-start the Gauss-Seidel loop, which will then converge to the correct q(z) shape and magnitude.
    """
    def rhs_noads(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),  # T_K fixed (isothermal) -- not solved, unlike _compute_noSE which does carry an energy balance
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        return [
            -rho_bed_cat*r, -4.0*rho_bed_cat*r,                    # same species balance as _compute_noSE's rhs, again with no "- rho_bed_ads*ads" term: water builds up unremoved
            +rho_bed_cat*r,  2.0*rho_bed_cat*r,                    # (no dT/dz row at all here -- only 4 states, since T isn't part of this IVP)
        ]
    z_grid = np.linspace(0, L_b, N)                                 # N=150 default: this function's own local grid, independent of solve_mpb's N and of _compute_noSE's N=300
    sol = solve_ivp(rhs_noads, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                    method='BDF', rtol=1e-4, atol=1e-8, t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    F_tot_f = np.maximum(sol.y[0] + sol.y[1] + sol.y[2] + sol.y[3], 1e-30)
    p_H2O_f = np.maximum(sol.y[3], 0.0)/F_tot_f * P_bar
    return dict(q=q_star(T_K, p_H2O_f),                             # equilibrium loading implied by this (unremoved) p_H2O(z) -- rises with z since water accumulates as the gas reacts along its own direction of travel (z=0->L)
                X_CO2_noSE=float(np.clip(1.0 - F_CO2_f[-1]/F_in_CO2, 0.0, 1.0)))  # outlet conversion for this isothermal pass; NOT the same X_CO2_noSE reported by _compute_noSE (no wall cooling here) -- unused by callers, just a byproduct

def _fmt_seconds(s):
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"

def _h2o_balance_line(u_s, res):
    """Sanity check: does H2O produced by reaction == H2O leaving in the gas + H2O leaving on the solid?"""
    T_out      = float(res['T'][-1])                                   # gas outlet temperature, z=L_b (last point of the fine profile)
    y_CO2_out  = float(res['C_CO2'][-1]) * R_gas * T_out / P_Pa        # recover CO2 mole fraction at outlet from concentration via ideal gas law (C = P*y/(R*T) => y = C*R*T/P)
    F_CO2_out  = F_in_CO2 * (1.0 - float(res['X_CO2'][-1]))            # CO2 molar flux remaining at outlet, from the overall conversion
    F_tot_out  = F_CO2_out / max(y_CO2_out, 1e-30)                      # total outlet molar flux, recovered from F_CO2_out/y_CO2_out (since y_CO2 = F_CO2/F_tot)
    F_H2O_out  = float(res['C_H2O'][-1]) * R_gas * T_out / P_Pa * F_tot_out  # H2O molar flux actually leaving in the gas: y_H2O_out (same ideal-gas recovery as y_CO2_out above) times F_tot_out
    F_H2O_prod = 2.0 * F_in_CO2 * float(res['X_CO2'][-1])               # H2O that SHOULD have been produced overall: 2 mol H2O per mol CO2 reacted (stoichiometry), times total CO2 converted
    F_H2O_ads  = u_s * rho_bed_ads * float(res['q'][0])                 # H2O leaving adsorbed on the solid at its own outlet (z=0): u_s[m/s] * rho_bed_ads[kg_ads/m3] * q(z=0)[mol/kg_ads] = mol/(m2.s), same units as the other two
    bal_err    = (F_H2O_out + F_H2O_ads - F_H2O_prod) / max(F_H2O_prod, 1e-30) * 100  # relative mass-balance error [%]: (gas-phase exit + solid-phase exit) vs. what reaction stoichiometry says should have been produced
    return (f"    H2O balance [mmol/(m²·s)]:  produced={F_H2O_prod*1e3:.3f}  "
            f"gas_out={F_H2O_out*1e3:.3f}  solid_out={F_H2O_ads*1e3:.3f}  "
            f"err={bal_err:+.1f}%")

u_s_fixed    = float(U_S_LIST[0])                                   # fixed solid velocity for the whole kinetics-multiplier scan
all_results  = {}                                                   # keyed by (T_C, i_mult): stores {'res':..., 'u_s':..., 'mult':..., 'T_K':..., 'T_wall':...} for every (T_in, mult) combination scanned
noSE_results = {}                                                   # keyed by (T_C, i_mult): stores _compute_noSE's output (X_CO2_noSE + profile) -- kinetics mult changes the noSE reference too, so it's no longer keyed by T_C alone
n_total      = len(T_IN_LIST) * len(KINETICS_MULT_LIST)             # total number of (T_in, mult) combinations to solve, for the progress/ETA printout below
n_done       = 0                                                    # running count of combinations solved so far, incremented each pass through the loop
t_run_start  = time.perf_counter()                                  # wall-clock start time, used to compute elapsed time and estimate ETA for the remaining combinations

for T_C in T_IN_LIST:                                               # outer sweep: one pass per inlet temperature in T_IN_LIST (currently just [280] C)
    T_K    = T_C + 273.15                                            # convert to Kelvin, used everywhere downstream
    T_wall = T_K                                                     # wall temperature tied to inlet temperature for this scan (no separate cooling setpoint explored here)

    print(f"\n{'='*60}")
    print(f"  T_in = {T_C} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  U_a = {U_a:.0f} W/(m3·K)")
    print(f"  Kinetics multiplier scan: {KINETICS_MULT_LIST}")
    print(f"{'='*60}")

    q_init = None                                                    # warm-start profile, carried across multipliers (recomputed fresh for the first one below)

    for i_mult, mult in enumerate(KINETICS_MULT_LIST):                # inner sweep: one solve_mpb call per kinetics multiplier, for this T_C, at fixed u_s
        KINETICS_MULT = mult                                          # reassigns the module-level global read inside reaction_rate_SI (see region 1/2)

        if q_init is None:
            _phys      = _q_physics_init(T_K)                         # physics-motivated initial guess, recomputed at this multiplier's own kinetics
            q_init_raw = _phys['q']                                   # rises with z (low near z=0, high near z=L_b), since it tracks water building up along the GAS's own direction of travel
            q_init     = q_init_raw[::-1]                              # flipped: the SOLID travels z=L_b->0, entering unloaded (q~0) at z=L_b and building up loading toward z=0 -- so the physically-correct starting guess is high near z=0, low near z=L_b, the mirror image of q_init_raw

        _noSE = _compute_noSE(T_K, T_wall)                             # fixed-bed (u_s=0) reference at this multiplier's kinetics
        noSE_results[(T_C, i_mult)] = _noSE
        print(f"  mult={mult:.1f}x  non-SE fixed-bed conversion: {_noSE['X_CO2_noSE']*100:.1f}%")

        t0  = time.perf_counter()                                     # start timer for this single solve, to report per-case runtime below
        res = solve_mpb(u_s_fixed, T_K, T_wall=T_wall, q_init=q_init)  # q_init is the _q_physics_init guess only on the FIRST multiplier here -- from the second iteration on, it's whatever the previous multiplier's converged profile was (see the update below), so the sweep warm-starts each case from its neighbour
        dt  = time.perf_counter() - t0                                 # elapsed time for just this solve_mpb call
        n_done += 1                                                    # tally toward n_total, for the ETA estimate
        elapsed = time.perf_counter() - t_run_start                     # total elapsed time since the whole T_C/mult scan started
        eta     = elapsed/n_done*(n_total - n_done)                     # simple linear ETA: average time per case so far, times cases remaining

        if res is not None:
            X_out  = float(res['X_CO2'][-1])*100
            q_out  = float(res['q'][0])
            T_max  = float(np.max(res['T'])) - 273.15
            regime = "gas" if res['gas_dominates'] else "solid"
            tag    = "ok" if res['converged'] else "not-conv"
            print(f"  mult={mult:.1f}x  X={X_out:.1f}%  "
                  f"q(0)={q_out:.3f}  T_max={T_max:.1f} C  "
                  f"[{regime}-dom, {tag}, {res['n_iter']} iter, err={res['conv_err']:.2e}]"
                  f"  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
            print(_h2o_balance_line(u_s_fixed, res))
            q_init = np.interp(np.linspace(0, L_b, 150), res['z'], res['q'])
        else:
            print(f"  mult={mult:.1f}x  FAILED"
                  f"  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
        all_results[(T_C, i_mult)] = {'res': res, 'u_s': u_s_fixed, 'mult': mult,
                                       'T_K': T_K, 'T_wall': T_wall}

print(f"\nAll done.  Total: {_fmt_seconds(time.perf_counter() - t_run_start)}")

# ── Retry pass: low mult → high mult, warm-starting from converged neighbours ──
print("\nRetry pass (backwards) for non-converged solutions...")
n_retried = 0
for T_C in T_IN_LIST:
    T_K    = T_C + 273.15
    T_wall = T_K
    q_retry = None

    for i_mult in reversed(range(len(KINETICS_MULT_LIST))):
        mult = KINETICS_MULT_LIST[i_mult]
        e    = all_results.get((T_C, i_mult))
        if e is None:
            continue
        res = e['res']

        if res is not None and res['converged']:
            q_retry = np.interp(np.linspace(0, L_b, 150), res['z'], res['q'])
            continue

        if q_retry is None:
            continue  # no converged neighbour below yet

        KINETICS_MULT = mult
        t0      = time.perf_counter()
        res_new = solve_mpb(u_s_fixed, T_K, T_wall=T_wall, q_init=q_retry)
        dt      = time.perf_counter() - t0

        if res_new is not None:
            tag = "ok" if res_new['converged'] else "still-nc"
            old_err = res['conv_err'] if res is not None else np.inf
            improved = res_new['converged'] or res_new['conv_err'] < old_err
            if improved:
                all_results[(T_C, i_mult)]['res'] = res_new
            print(f"  RETRY mult={mult:.1f}x  [{tag}, {res_new['n_iter']} iter, "
                  f"err={res_new['conv_err']:.2e}]  ({'kept' if improved else 'discarded'})  ({dt:.1f}s)")
            print(_h2o_balance_line(u_s_fixed, res_new))
            q_retry = np.interp(np.linspace(0, L_b, 150), res_new['z'], res_new['q'])
            n_retried += 1
        else:
            print(f"  RETRY mult={mult:.1f}x  FAILED  ({dt:.1f}s)")

print(f"Retry pass done. {n_retried} solution(s) re-run.")
# endregion


# region 5. POST-PROCESSING HELPERS
# =============================================================================
def get_metrics(entry):
    res = entry['res']
    if res is None:
        return None
    T_K = entry['T_K']
    X_out       = float(res['X_CO2'][-1])
    q_out       = float(res['q'][0])
    T_max       = float(np.max(res['T']))
    p_H2O_peak  = float(np.max(res['C_H2O']))*R_gas*T_K/1e5
    q_star_peak = float(q_star(T_K, np.array([max(p_H2O_peak, 1e-8)]))[0])
    return dict(X_CO2=X_out, q_out=q_out, T_max=T_max,
                sorbent_util=q_out/max(q_star_peak, 1e-10))
# endregion


# region 6. PLOTS
# =============================================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def _savefig(filename):
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=150, bbox_inches='tight')

markers = ['o', 's', '^', 'D']
cmap    = plt.cm.viridis

T_C_PROF = T_IN_LIST[0]

# Every kinetics multiplier gets its own line (only 4 values, no thinning needed).
i_plot = np.arange(len(KINETICS_MULT_LIST))
pal    = plt.cm.plasma(np.linspace(0.1, 0.85, len(i_plot)))
pal2   = cmap(np.linspace(0.1, 0.85, len(T_IN_LIST)))

# ── Plot 1: Axial profiles ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f'MPB flux form  |  T_in = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  '
             f'U_a = {U_a:.0f} W/(m³·K), counter-current  —  kinetics sensitivity', fontsize=11)
for k, i_mult in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_mult))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    ls  = '-' if r['converged'] else '--'
    lbl = f"{e['mult']:.1f}x kinetics" + ("" if r['converged'] else " (nc)")
    axes[0,0].plot(r['z'], r['C_CO2']*1e3, color=pal[k], lw=2, ls=ls, label=lbl)
    axes[0,1].plot(r['z'], r['q'],          color=pal[k], lw=2, ls=ls, label=lbl)
    axes[1,0].plot(r['z'], r['X_CO2']*100,  color=pal[k], lw=2, ls=ls, label=lbl)
    axes[1,1].plot(r['z'], r['r']*1e3,      color=pal[k], lw=2, ls=ls, label=lbl)
labels_units = [('C_CO2 [mmol/m3]', 'CO2 concentration'),
                ('q [mol/kg]',       'Solid H2O loading'),
                ('CO2 conversion [%]', 'CO2 conversion along bed'),
                ('r [mmol/(kg_cat.s)]', 'Reaction rate')]
for ax, (ylabel, title) in zip(axes.flat, labels_units):
    ax.set_xlabel('z [m]', fontsize=10);  ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10);     ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
    ax.axvline(0,   color='tab:blue',   lw=1, ls=':', alpha=0.5)
    ax.axvline(L_b, color='tab:orange', lw=1, ls=':', alpha=0.5)
plt.tight_layout()
_savefig(f'kin_plot1_axial_profiles_T{T_C_PROF}C.png');  plt.show()

# ── Plot 2: CO2 conversion vs kinetics multiplier ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    m_conv, X_conv, m_nc, X_nc = [], [], [], []
    for i_mult in range(len(KINETICS_MULT_LIST)):
        e = all_results.get((T_C, i_mult))
        if e and e['res']:
            m = get_metrics(e)
            if m:
                if e['res']['converged']:
                    m_conv.append(e['mult']);  X_conv.append(m['X_CO2']*100)
                else:
                    m_nc.append(e['mult']);    X_nc.append(m['X_CO2']*100)
    if m_conv:
        ax.semilogx(m_conv, X_conv, marker=markers[j], color=pal2[j],
                    lw=2, ms=7, label=f'{T_C} C (MPB)')
    if m_nc:
        ax.semilogx(m_nc, X_nc, marker=markers[j], color=pal2[j],
                    lw=2, ms=7, ls='--', mfc='none', mew=1.5, label=f'{T_C} C (not conv)')
    if m_conv or m_nc:
        ax.axhline(equilibrium_conversion(T_C+273.15), color=pal2[j],
                   lw=1, ls=':', alpha=0.5, label=f'{T_C} C thermo. eq.')
ax.set_xlabel('Kinetics multiplier [-]', fontsize=11);  ax.set_ylabel('CO2 conversion [%]', fontsize=11)
ax.set_title(f'MPB flux form  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  U_a = {U_a:.0f} W/(m³·K)  '
             f'— CO2 conversion vs kinetics multiplier', fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3);  ax.set_ylim(0, 105)
plt.tight_layout()
_savefig('kin_plot2_conversion_vs_mult.png');  plt.show()

# ── Plot 3: Sorbent utilisation vs kinetics multiplier ───────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    m_conv, util_conv, m_nc, util_nc = [], [], [], []
    for i_mult in range(len(KINETICS_MULT_LIST)):
        e = all_results.get((T_C, i_mult))
        if e and e['res']:
            m = get_metrics(e)
            if m:
                if e['res']['converged']:
                    m_conv.append(e['mult']);  util_conv.append(m['sorbent_util']*100)
                else:
                    m_nc.append(e['mult']);    util_nc.append(m['sorbent_util']*100)
    if m_conv:
        ax.semilogx(m_conv, util_conv, marker=markers[j], color=pal2[j],
                    lw=2, ms=7, label=f'{T_C} C')
    if m_nc:
        ax.semilogx(m_nc, util_nc, marker=markers[j], color=pal2[j],
                    lw=2, ms=7, ls='--', mfc='none', mew=1.5, label=f'{T_C} C (not conv)')
ax.axhline(100, color='grey', lw=1.5, ls='--', label='q = q* (fully saturated)')
ax.set_xlabel('Kinetics multiplier [-]', fontsize=11)
ax.set_ylabel('Sorbent utilisation  q(z=0) / q*(p_H2O_max)  [%]', fontsize=11)
ax.set_title(f'MPB flux form  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  U_a = {U_a:.0f} W/(m³·K)  '
             f'— Sorbent utilisation', fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('kin_plot3_sorbent_utilisation.png');  plt.show()

# ── Plot 4: H2O profiles ─────────────────────────────────────────────────────
T_K_prof = T_C_PROF + 273.15
fig, (ax_q, ax_h) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'H2O profiles  |  T_in = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  '
             f'U_a = {U_a:.0f} W/(m³·K)  —  kinetics sensitivity', fontsize=11)
for k, i_mult in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_mult))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    ls  = '-' if r['converged'] else '--'
    lbl = f"{e['mult']:.1f}x kinetics" + ("" if r['converged'] else " (nc)")
    ax_q.plot(r['z'], r['q'],                          color=pal[k], lw=2, ls=ls, label=lbl)
    ax_h.plot(r['z'], r['C_H2O']*R_gas*T_K_prof/1e2,  color=pal[k], lw=2, ls=ls, label=lbl)
ax_q.set_xlabel('z [m]');  ax_q.set_ylabel('q [mol/kg]')
ax_q.set_title('Solid H2O loading');  ax_q.legend(fontsize=8);  ax_q.grid(True, alpha=0.3)
ax_h.set_xlabel('z [m]');  ax_h.set_ylabel('p_H2O [mbar]')
ax_h.set_title('Gas-phase H2O partial pressure');  ax_h.legend(fontsize=8);  ax_h.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'kin_plot4_H2O_profiles_T{T_C_PROF}C.png');  plt.show()

# ── Plot 5: Temperature profiles ─────────────────────────────────────────────
fig, ax_T = plt.subplots(figsize=(9, 5))
fig.suptitle(f'Temperature profile  |  T_in = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  '
             f'U_a = {U_a:.0f} W/(m³·K)  —  kinetics sensitivity', fontsize=11)
for k, i_mult in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_mult))
    if e is None or e['res'] is None:
        continue
    r      = e['res']
    regime = 'g' if r['gas_dominates'] else 's'
    ls     = '-' if r['converged'] else '--'
    lbl    = f"{e['mult']:.1f}x kinetics ({regime})" + ("" if r['converged'] else " (nc)")
    ax_T.plot(r['z'], r['T'] - 273.15, color=pal[k], lw=2, ls=ls, label=lbl)
ax_T.axhline(T_C_PROF, color='grey', lw=1.5, ls='--', alpha=0.8, label=f'T_in = T_wall = {T_C_PROF} °C')
ax_T.set_xlabel('z [m]', fontsize=10);  ax_T.set_ylabel('T [°C]', fontsize=10)
ax_T.set_title('(g) = gas-dominated  |  (s) = solid-dominated (T from solid IVP)', fontsize=9)
ax_T.legend(fontsize=8);  ax_T.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'kin_plot5_temperature_T{T_C_PROF}C.png');  plt.show()

# ── Plot 6: Peak temperature rise vs kinetics multiplier ─────────────────────
fig, ax6 = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    m_conv, dT_conv, m_nc, dT_nc = [], [], [], []
    for i_mult in range(len(KINETICS_MULT_LIST)):
        e = all_results.get((T_C, i_mult))
        if e and e['res']:
            m = get_metrics(e)
            if m:
                if e['res']['converged']:
                    m_conv.append(e['mult']);  dT_conv.append(m['T_max'] - (T_C + 273.15))
                else:
                    m_nc.append(e['mult']);    dT_nc.append(m['T_max'] - (T_C + 273.15))
    if m_conv:
        ax6.semilogx(m_conv, dT_conv, marker=markers[j], color=pal2[j],
                     lw=2, ms=7, label=f'{T_C} C (MPB)')
    if m_nc:
        ax6.semilogx(m_nc, dT_nc, marker=markers[j], color=pal2[j],
                     lw=2, ms=7, ls='--', mfc='none', mew=1.5, label=f'{T_C} C (not conv)')
ax6.set_xlabel('Kinetics multiplier [-]', fontsize=11)
ax6.set_ylabel('ΔT_max = T_peak − T_in  [K]', fontsize=11)
ax6.set_title(f'MPB flux form  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  U_a = {U_a:.0f} W/(m³·K)  '
              f'— peak temperature rise', fontsize=10)
ax6.legend(fontsize=9);  ax6.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('kin_plot6_Tmax_vs_mult.png');  plt.show()

# ── Plot 7: H2O budget decomposition ─────────────────────────────────────────
fig, (ax_rate, ax_cumul) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'H2O budget  |  T_in = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  '
             f'U_a = {U_a:.0f} W/(m³·K)  —  kinetics sensitivity', fontsize=11)

pal_7 = plt.cm.plasma(np.linspace(0.1, 0.85, len(i_plot)))

for k, i_mult in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_mult))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    ls  = '-' if r['converged'] else '--'
    lbl = f"{e['mult']:.1f}x kinetics" + ("" if r['converged'] else " (nc)")

    p_H2O_f = r['C_H2O'] * R_gas * r['T'] / 1e5   # [bar]
    qs_f    = q_star(r['T'], p_H2O_f)
    Kl_f    = K_LDF(r['T'], p_H2O_f)
    ads_f   = Kl_f * (qs_f - r['q'])               # [mol/(kg_ads·s)]

    S_rxn = 2.0 * rho_bed_cat * r['r']             # [mol/(m³_bed·s)]
    S_ads = -rho_bed_ads * ads_f                    # [mol/(m³_bed·s)]

    F_rxn = cumulative_trapezoid(S_rxn, r['z'], initial=0)
    F_ads = cumulative_trapezoid(S_ads, r['z'], initial=0)
    F_net = F_rxn + F_ads

    ax_rate.plot(r['z'], S_rxn*1e3, color=pal_7[k], lw=2,   ls=ls,            label=lbl)
    ax_rate.plot(r['z'], S_ads*1e3, color=pal_7[k], lw=1.5, ls=ls, alpha=0.5)

    ax_cumul.plot(r['z'], F_rxn*1e3, color=pal_7[k], lw=2,   ls=ls,            label=f"{lbl} rxn")
    ax_cumul.plot(r['z'], F_ads*1e3, color=pal_7[k], lw=1.5, ls=ls, alpha=0.5, label=f"{lbl} ads")
    ax_cumul.plot(r['z'], F_net*1e3, color=pal_7[k], lw=1,   ls=ls, alpha=0.3)

ax_rate.axhline(0, color='k', lw=0.8, ls=':')
ax_rate.set_xlabel('z [m]');  ax_rate.set_ylabel('S_H2O [mmol/(m³_bed·s)]')
ax_rate.set_title('Local H2O source (+) / sink (−) rates\nsolid line = rxn,  faint = sorbent', fontsize=9)
ax_rate.legend(fontsize=8);  ax_rate.grid(True, alpha=0.3)

ax_cumul.axhline(0, color='k', lw=0.8, ls=':')
ax_cumul.set_xlabel('z [m]');  ax_cumul.set_ylabel('Cumulative H2O flux [mmol/(m²·s)]')
ax_cumul.set_title('Cumulative H2O budget\nbright = rxn,  faint = sorbent capture,  faintest = net gas', fontsize=9)
ax_cumul.legend(fontsize=7);  ax_cumul.grid(True, alpha=0.3)

plt.tight_layout()
_savefig(f'kin_plot7_H2O_budget_T{T_C_PROF}C.png');  plt.show()
# endregion
