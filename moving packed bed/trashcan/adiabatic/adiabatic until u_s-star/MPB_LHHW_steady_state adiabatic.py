"""
Moving Packed Bed (MPB) Reactor Model — Steady-State, Adiabatic, CO2 Methanation
==================================================================================

Counter-current flow:
    gas  : z = 0 (inlet, bottom)  ->  z = L (outlet, top)    u_g > 0
    solid: z = L (inlet, top)     ->  z = 0 (outlet, bottom)  u_s > 0 (magnitude)

Adiabatic: no wall heat exchange (U_a = 0).  Temperature is determined by the
full counter-current pseudo-homogeneous energy balance:

    (u_g·ρ_g_mol·Cp_mix  −  u_s·ρ_bed·Cp_cat) · dT/dz  =
        (−ΔH_r)·ρ_bed_cat·r          ← reaction heat release
      + (−ΔH_ads)·ρ_bed_ads·ads       ← adsorption heat release

For all practical solid velocities u_s·ρ_bed·Cp_cat >> u_g·ρ_g_mol·Cp_mix, so the
solid dominates heat transport.  Re-expressed in ζ = L−z (solid travel direction):

    dT/dζ = (Q_rxn + Q_ads) / (u_s·ρ_bed·Cp_cat − u_g·ρ_g_mol·Cp_mix)
                                         ↑ always > 0 when solid dominates

This is a stable forward IVP in ζ-space with BC T(ζ=0) = T_s_in (solid inlet
temperature, defaults to T_K — fresh sorbent at feed temperature).

The solid acts as the built-in coolant: higher u_s → more cold mass flow →
smaller temperature excursion.  For u_s → 0 (fixed bed), temperatures are
unbounded without wall cooling.

Key difference from non-isothermal file:
  • T is removed from the gas IVP (gas uses frozen T(z) from the solid IVP)
  • T is added as a 2nd state variable in the solid IVP alongside q
  • All 4 concentration profiles are passed to the solid IVP (needed for Q_rxn)
  • Convergence is checked on both q(z) and T(z)

Solved by decoupled Gauss-Seidel iteration:
  1. Gas IVP   (z = 0 -> L, solve_ivp BDF): fixed q(z), T(z)  → gives C(z)
  2. Solid IVP (ζ = 0 -> L, solve_ivp BDF): fixed C(z)        → gives q(z), T(z)
  3. Repeat until max|Δq|/q and max|ΔT|/ΔT_range both < tol

State variables at steady state:
    C_CO2(z), C_H2(z), C_CH4(z), C_H2O(z)  [mol/m3]
    q(z)                                      [mol/kg]
    T(z)                                      [K]       ← from solid IVP
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# region 1. PARAMETERS
# =============================================================================

# --- Bed geometry (Bareschino lab setup) ---
d_b   = 0.050          # [m]             — reactor inner diameter, Bareschino lab setup
L_b   = 2.000          # [m]             — bed length, Bareschino lab setup
A_b   = np.pi / 4 * d_b**2   # [m²]     — bed cross-sectional area, derived
V_bed = A_b * L_b      # [m³]            — bed volume, derived
eps_b = 0.4            # [-]             — bed void fraction, typical packed bed

# --- Catalyst and sorbent loading (Bareschino lab setup) ---
M_cat = 0.064          # [kg]            — catalyst (5%Ni2.5%Ce/13X) mass, Bareschino lab setup
M_ads = 1.22           # [kg]            — sorbent (13X zeolite) mass, Bareschino lab setup
rho_bed_cat = M_cat / V_bed   # [kg_cat/m³_bed] — derived
rho_bed_ads = M_ads / V_bed   # [kg_ads/m³_bed] — derived
rho_bed_tot = (M_cat + M_ads) / V_bed  # [kg/m³_bed]  — derived

# --- Particle properties (13X zeolite pellets) ---
d_p   = 0.75e-3        # [m]             — pellet diameter, 13X material spec
eps_p = 0.615          # [-]             — intra-particle porosity, 13X literature
tau_p = 3.0            # [-]             — pore tortuosity, typical zeolite pellet

# --- Dubinin-Astakhov adsorption isotherm (H2O on 13X) ---
# Fitted to Bareschino et al. (2020), Fig. 1 (see bareschino adsorption validation folder)
W0_DA = 190.00e-6      # [m³_liq/kg_sorbent] — limiting micropore volume
E_DA  = 1190e3         # [J/mol]         — characteristic adsorption energy
n_DA  = 1.55           # [-]             — isotherm heterogeneity exponent

# --- LHHW kinetics (CO2 methanation on 5%Ni/13X) ---
# Rate: r [mmol/(kg_cat·s)] = k*(p_CO2*p_H2)^0.5 * f_eq / DEN² * 1000
# All kinetic parameters from Wei et al.
T_ref_K = 555.0        # [K]             — reference temperature
k_ref   = 3.46e-4      # [mol/(kg_cat·s·bar)] — rate constant at T_ref
Ea_k    = 77.5e3       # [J/mol]         — apparent activation energy
A_OH    = 0.50;  dH_OH  =  22.4e3   # [-], [J/mol] — OH*  adsorption: pre-exp & enthalpy
A_H2    = 0.44;  dH_H2  =  -6.2e3  # [-], [J/mol] — H2   adsorption: pre-exp & enthalpy
A_mix   = 0.88;  dH_mix = -10.0e3  # [-], [J/mol] — mixed adsorption: pre-exp & enthalpy
P_FLOOR = 1e-4         # [bar]           — numerical floor to prevent log(0) singularities

# --- Thermochemistry ---
dH_r   = -165.0e3      # [J/mol_CO2]     — methanation reaction enthalpy, NIST
dH_ads =  -45.0e3      # [J/mol_H2O]     — H2O adsorption enthalpy on 13X, literature
Cp_cat = 1100.0        # [J/(kg·K)]      — catalyst/sorbent heat capacity, literature estimate
Cp_CO2 = 45.4;  Cp_H2 = 29.3;  Cp_CH4 = 46.9;  Cp_H2O = 34.2  # [J/(mol·K)] — NIST at ~550 K

# --- Physical constants ---
R_gas  = 8.314         # [J/(mol·K)]     — universal gas constant
MW_H2O = 0.018015      # [kg/mol]        — molar mass of water

# --- Operating conditions ---
P_bar = 1.0            # [bar]           — total operating pressure
P_Pa  = P_bar * 1e5    # [Pa]            — derived
y_CO2_in = 0.04        # [-]             — inlet CO2 mole fraction  (CO2:H2:CH4 = 4:16:80)
y_H2_in  = 0.16        # [-]             — inlet H2  mole fraction, stoichiometric 4:1 H2/CO2
y_CH4_in = 0.80        # [-]             — inlet CH4 mole fraction (carrier/diluent)

# --- Gas flow ---
T_STP   = 273.15       # [K]             — standard temperature for volumetric flow reference
GHSV    = 0.5          # [m³_STP/(kg_ads·h)] — gas hourly space velocity, Wei setup
Q_STP   = GHSV * M_ads / 3600.0   # [m³_STP/s]  — total volumetric flow at STP, derived
u_g_STP = Q_STP / A_b  # [m_STP/s]      — superficial gas velocity at STP, derived

# --- MPB scan parameters ---
_us_base  = np.logspace(np.log10(1e-4), np.log10(5e-3), 10)              # [m/s] 0.1 to 5 mm/s
_us_fine  = np.array([6.5e-4, 7.0e-4, 7.5e-4, 8.0e-4, 8.5e-4, 9.5e-4]) # [m/s] extra points 0.65–0.95 mm/s
U_S_LIST  = np.unique(np.concatenate([_us_base, _us_fine]))               # [m/s] sorted, ~16 points
T_IN_LIST = [280]                       # [°C]   — inlet temperatures to scan

print(f"MPB adiabatic: d={d_b*100:.0f} cm, L={L_b:.1f} m, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3, u_g_STP={u_g_STP*1e3:.1f} mm/s")
# endregion


# region 2. INHERITED FUNCTIONS
# =============================================================================
def P_sat_bar(T_K):
    """
    Saturation vapour pressure of water [bar].
    Extended Antoine equation (Wexler-Hyland form); coefficients give log10(P/mmHg),
    converted to bar via 1 mmHg = 133.322e-5 bar.
    np.clip prevents overflow at extreme temperatures.
    """
    log10_p = (29.8605 - 3.1522e3/T_K - 7.3037*np.log10(T_K)
               + 2.4247e-9*T_K + 1.8090e-6*T_K**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5   # [mmHg] -> [bar]

def rho_water(T_K):
    """
    Liquid water density [kg/m³], linear fit to NIST data (valid ~25–300 °C).
    Needed to convert DA micropore volume W [m³_liq/kg_sorb] -> molar loading [mol/kg]:
        q* = rho_water / MW_H2O * W
    """
    return 996.0 / (1.0 + 2.0e-3*(T_K - 298.15))

def q_star_vec(T_K, p_arr, W0, E, n):
    """
    Dubinin-Astakhov (DA) equilibrium isotherm: W = W0 * exp(-(A/E)^n)
    Source: fitted to Bareschino et al. (2020), Fig. 1 (H2O on 13X zeolite).

    A = (R/MW_H2O) * T * ln(P_sat/p)  is the Polanyi adsorption potential [J/kg]:
        the isothermal work to compress vapour from p to saturation.
    W  [m³_liq/kg] is the fraction of micropore volume filled.
    q* [mol/kg]    = rho_water / MW_H2O * W  (volume -> moles via liquid density).
    clip + np.where guards prevent log(0) and enforce q*=0 for p <= 0.
    """
    p    = np.asarray(p_arr, dtype=float)
    Psat = P_sat_bar(T_K)
    p_safe = np.clip(p, 1e-15, Psat*(1-1e-10))            # keep p inside (0, P_sat)
    A_raw  = (R_gas/MW_H2O)*T_K*np.log(Psat/p_safe)       # Polanyi potential [J/kg]
    A  = np.where((p <= 0)|(p >= Psat), 0.0, A_raw)        # zero outside valid range
    W  = W0*np.exp(-np.minimum((A/E)**n, 500.0))           # DA: micropore filling [-]
    qs = rho_water(T_K)/MW_H2O*W                           # convert to [mol/kg]
    return np.where(p <= 0, 0.0, qs)

def K_LDF_vec(T_K, p_arr, W0, E, n):
    """
    Linear Driving Force (LDF) mass-transfer coefficient [1/s].
    Source: Glueckauf (1955) sphere approximation — K_LDF = 15*D_eff/r_p^2.

    D_M  [m²/s]: molecular diffusivity, T^1.75 scaling from Chapman-Enskog theory.
    D_eff = D_M * eps_p / tau_p  (pore diffusion corrected for porosity & tortuosity).
    dq*/dp [mol/(kg·bar)]: local isotherm slope, computed by central finite difference.
        A steep isotherm (large dq*/dp) means each bar of pressure change moves many
        mol/kg -> apparent K_LDF is smaller (equilibration is slower per unit time).
    The denominator converts from gas-phase pressure driving force to solid-phase units.
    """
    D_M  = 2.5e-5*(T_K/300.0)**1.75                        # molecular diffusivity [m²/s]
    p    = np.asarray(p_arr, dtype=float)
    dp   = 1.0/1e5                                          # 1 Pa in bar, for central diff.
    dqsp = (q_star_vec(T_K, p+dp, W0, E, n) - q_star_vec(T_K, np.maximum(p-dp,1e-15), W0, E, n))/2.0
    dqsp = np.maximum(dqsp, 1e-30)                         # prevent division by zero
    return (15.0*D_M*MW_H2O*eps_p
            / (0.5*d_p**2*tau_p*rho_water(T_K)*R_gas*T_K*dqsp))

def K_eq_sabatier(T_K):
    """
    Thermodynamic equilibrium constant for CO2 + 4H2 <-> CH4 + 2H2O  [bar^-2].
    K_eq = p_CH4 * p_H2O^2 / (p_CO2 * p_H2^4)
    Source: Wei et al. (same reference as kinetics).
    Form A*T^B*exp(C/RT): van 't Hoff integrated with a Cp correction (T^B term).
    C = 158.7 kJ/mol reflects the strongly exothermic reaction enthalpy;
    K_eq decreases with T (higher T shifts equilibrium toward reactants).
    """
    return 137.0*T_K**(-3.994)*np.exp(158700.0/(R_gas*T_K))

def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    """
    LHHW rate for CO2 methanation [mmol/(kg_cat·s)].
    Source: Wei et al. (kinetics on Ni catalyst, adapted for 5%Ni2.5%Ce/13X).
    r = k * (p_CO2 * p_H2)^0.5 * f_eq / DEN^2 * 1000

    k(T)         : Arrhenius rate constant relative to T_ref = 555 K.
    K_OH/H2/mix  : van 't Hoff surface adsorption equilibrium constants.
    f_eq         : approach-to-equilibrium factor = max(1 - Q/K_eq, 0).
    DEN^2        : bimolecular LHHW denominator (OH* and H2 compete for active sites).
    *1000        : converts mol -> mmol to match k_ref units [mmol/(kg_cat·s·bar^0.5)].
    """
    vH    = lambda dH: np.exp(-dH/R_gas*(1.0/T_K - 1.0/T_ref_K))  # van 't Hoff factor
    k     = k_ref*np.exp(-Ea_k/R_gas*(1.0/T_K - 1.0/T_ref_K))     # Arrhenius
    K_OH  = A_OH*vH(dH_OH);  K_H2 = A_H2*vH(dH_H2);  K_mix = A_mix*vH(dH_mix)
    K_eq  = K_eq_sabatier(T_K)
    p_CO2_s = np.maximum(p_CO2, P_FLOOR);  p_H2_s = np.maximum(p_H2, P_FLOOR)
    beta  = (p_CH4*p_H2O**2)/(K_eq*p_CO2_s*p_H2_s**4)             # reaction quotient / K_eq
    f_eq  = np.maximum(1.0 - np.where(np.isfinite(beta), beta, 1e10), 0.0)
    DEN   = (1.0 + K_OH*np.maximum(p_H2O,0)/p_H2_s**0.5
             + K_H2*p_H2_s**0.5 + K_mix*p_CO2_s)
    return k*(p_CO2_s*p_H2_s)**0.5*f_eq/DEN**2*1000.0

def q_star(T_K, p_H2O):
    """Wrapper: DA equilibrium loading with the fitted 13X parameters from region 1."""
    return q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

def K_LDF(T_K, p_H2O):
    """Wrapper: LDF coefficient with the fitted 13X parameters from region 1."""
    return K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

def equilibrium_conversion(T_K_val):
    """
    Thermodynamic equilibrium CO2 conversion [%] at the given temperature.
    Solves Q(X) = K_eq via Brent's method (brentq).
    """
    K = K_eq_sabatier(T_K_val)
    def f(X):
        d = 1.0 - 0.08*X
        return ((0.80+0.04*X)/d * (0.08*X/d)**2
                / ((0.04*(1-X)/d)*(0.16*(1-X)/d)**4 + 1e-100) - K)
    try:
        return brentq(f, 1e-9, 1-1e-9)*100.0
    except Exception:
        return 100.0
# endregion


# region 3. DECOUPLED SOLVER
# =============================================================================
def _cp_mix(p_CO2, p_H2, p_CH4, p_H2O):
    """Mole-fraction weighted heat capacity [J/(mol·K)] at local gas composition."""
    y_CO2 = p_CO2/P_bar;  y_H2 = p_H2/P_bar
    y_CH4 = p_CH4/P_bar;  y_H2O = p_H2O/P_bar
    y_rest = max(1.0 - y_CO2 - y_H2 - y_CH4 - y_H2O, 0.0)
    return y_CO2*Cp_CO2 + y_H2*Cp_H2 + y_CH4*Cp_CH4 + y_H2O*Cp_H2O + y_rest*Cp_CH4


def solve_mpb(u_s, u_g, T_K, C_in_CO2, C_in_H2, C_in_CH4,
              max_iter=200, tol=1e-4, N=100, q_init=None):
    """
    Counter-current MPB at steady state, adiabatic (no wall cooling).

    T is the 5th state variable in the gas IVP (same structure as non-isothermal).
    Energy denominator uses the full adiabatic formula when gas dominates (u_s < u_s*),
    falling back to the gas-only term when solid dominates (u_s > u_s*) to keep the
    IVP stable in the gas direction.

    u_s* = u_g·ρ_g·Cp_mix / (ρ_bed·Cp_cat) ≈ 0.47 mm/s at 280 °C.

      u_s < u_s*: denom = u_g·Cp_g − u_s·Cp_s > 0  →  exact adiabatic result
      u_s > u_s*: denom = u_g·Cp_g              > 0  →  lower bound on ΔT (conservative)

    BC: T(z=0) = T_K.  Decoupled Gauss-Seidel iteration on q(z).
    """
    solid_cap = u_s * rho_bed_tot * Cp_cat   # [W/(m²·K)], constant

    z_grid = np.linspace(0.0, L_b, N)
    if q_init is not None:
        q_prof = np.interp(z_grid, np.linspace(0, L_b, len(q_init)), q_init)
    else:
        q_prof = np.zeros(N)
    T_prof = T_K * np.ones(N)

    converged = False
    err = 1.0
    for it in range(max_iter):
        q_fn = interp1d(z_grid, q_prof, kind='linear',
                        bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))

        # ── Step 1: gas IVP from z=0 to z=L ─────────────────────────────────
        def gas_rhs(z, y):
            C_CO2_l = max(y[0], 0.0);  C_H2_l  = max(y[1], 0.0)
            C_CH4_l = max(y[2], 0.0);  C_H2O_l = max(y[3], 0.0)
            T_l     = max(y[4], 200.0)
            q_l     = max(float(q_fn(z)), 0.0)

            p_CO2 = C_CO2_l*R_gas*T_l/1e5;  p_H2  = C_H2_l *R_gas*T_l/1e5
            p_CH4 = C_CH4_l*R_gas*T_l/1e5;  p_H2O = C_H2O_l*R_gas*T_l/1e5

            r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                          np.array([p_CH4]), np.array([p_H2O]))[0])
            qs  = float(q_star(T_l, np.array([p_H2O]))[0])
            Kl  = float(K_LDF(T_l,  np.array([p_H2O]))[0])
            ads = Kl*(qs - q_l)

            rho_g_mol_l = P_Pa / (R_gas * T_l)
            gas_cap     = u_g * rho_g_mol_l * _cp_mix(p_CO2, p_H2, p_CH4, p_H2O)
            full_denom  = gas_cap - solid_cap
            # Use full adiabatic denominator when gas dominates (stable, exact).
            # Fall back to gas-only when solid dominates (prevents sign flip).
            denom = full_denom if full_denom > 0 else gas_cap

            Q_rxn = (-dH_r)   * rho_bed_cat * r
            Q_ads = (-dH_ads) * rho_bed_ads * ads
            dTdz  = (Q_rxn + Q_ads) / denom

            return [
                -rho_bed_cat*r / (eps_b*u_g),
                -4.0*rho_bed_cat*r / (eps_b*u_g),
                +rho_bed_cat*r / (eps_b*u_g),
                (2.0*rho_bed_cat*r - rho_bed_ads*ads) / (eps_b*u_g),
                dTdz,
            ]

        gs = solve_ivp(gas_rhs, [0.0, L_b], [C_in_CO2, C_in_H2, C_in_CH4, 0.0, T_K],
                       method='BDF', rtol=1e-4,
                       atol=np.array([1e-8, 1e-8, 1e-8, 1e-8, 1e-2]),
                       t_eval=z_grid, dense_output=False)
        if not gs.success:
            return None

        C_H2O_prof = np.maximum(gs.y[3], 0.0)
        T_prof_new = np.maximum(gs.y[4], 200.0)
        C_H2O_fn   = interp1d(z_grid, C_H2O_prof, kind='linear',
                               bounds_error=False,
                               fill_value=(C_H2O_prof[0], C_H2O_prof[-1]))
        T_fn_new   = interp1d(z_grid, T_prof_new, kind='linear',
                               bounds_error=False,
                               fill_value=(T_prof_new[0], T_prof_new[-1]))

        # ── Step 2: solid IVP in ζ = L−z from ζ=0 (z=L) to ζ=L (z=0) ───────
        def solid_rhs(zeta, q_arr):
            z_pos   = L_b - float(zeta)
            T_local = float(T_fn_new(z_pos))
            p_H2O   = max(float(C_H2O_fn(z_pos)), 0.0)*R_gas*T_local/1e5
            qs  = float(q_star(T_local, np.array([p_H2O]))[0])
            Kl  = float(K_LDF(T_local,  np.array([p_H2O]))[0])
            q_val = max(float(q_arr[0]), 0.0)
            return [Kl*(qs - q_val)/u_s]

        zeta_eval = np.linspace(0.0, L_b, N)
        ss = solve_ivp(solid_rhs, [0.0, L_b], [0.0],
                       method='BDF', rtol=1e-4, atol=1e-8,
                       t_eval=zeta_eval, dense_output=False)
        if not ss.success:
            return None

        z_from_zeta = L_b - ss.t
        q_from_zeta = np.maximum(ss.y[0], 0.0)
        sort_idx    = np.argsort(z_from_zeta)
        q_new = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])

        q_prof_new = 0.5 * q_prof + 0.5 * q_new
        T_prof     = 0.5 * T_prof + 0.5 * T_prof_new

        scale = max(np.max(q_prof_new), 1e-8)
        err   = np.max(np.abs(q_prof_new - q_prof)) / scale
        q_prof = q_prof_new
        if err < tol:
            converged = True
            break

    # ── Final recompute on fine grid ─────────────────────────────────────────
    q_fn_f = interp1d(z_grid, q_prof, kind='linear',
                      bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))

    def gas_rhs_final(z, y):
        C_CO2_l = max(y[0], 0.0);  C_H2_l  = max(y[1], 0.0)
        C_CH4_l = max(y[2], 0.0);  C_H2O_l = max(y[3], 0.0)
        T_l     = max(y[4], 200.0)
        q_l     = max(float(q_fn_f(z)), 0.0)
        p_CO2 = C_CO2_l*R_gas*T_l/1e5;  p_H2  = C_H2_l *R_gas*T_l/1e5
        p_CH4 = C_CH4_l*R_gas*T_l/1e5;  p_H2O = C_H2O_l*R_gas*T_l/1e5
        r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                      np.array([p_CH4]), np.array([p_H2O]))[0])
        qs  = float(q_star(T_l, np.array([p_H2O]))[0])
        Kl  = float(K_LDF(T_l,  np.array([p_H2O]))[0])
        ads = Kl*(qs - q_l)
        rho_g_mol_l = P_Pa / (R_gas * T_l)
        gas_cap     = u_g * rho_g_mol_l * _cp_mix(p_CO2, p_H2, p_CH4, p_H2O)
        full_denom  = gas_cap - solid_cap
        denom = full_denom if full_denom > 0 else gas_cap
        Q_rxn = (-dH_r)   * rho_bed_cat * r
        Q_ads = (-dH_ads) * rho_bed_ads * ads
        return [
            -rho_bed_cat*r/(eps_b*u_g),
            -4.0*rho_bed_cat*r/(eps_b*u_g),
            +rho_bed_cat*r/(eps_b*u_g),
            (2.0*rho_bed_cat*r - rho_bed_ads*ads)/(eps_b*u_g),
            (Q_rxn + Q_ads) / denom,
        ]

    z_fine = np.linspace(0.0, L_b, 300)
    gf = solve_ivp(gas_rhs_final, [0.0, L_b],
                   [C_in_CO2, C_in_H2, C_in_CH4, 0.0, T_K],
                   method='BDF', rtol=1e-6,
                   atol=np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-3]),
                   t_eval=z_fine, dense_output=False)

    C_CO2f = np.maximum(gf.y[0], 0.0);  C_H2f  = np.maximum(gf.y[1], 0.0)
    C_CH4f = np.maximum(gf.y[2], 0.0);  C_H2Of = np.maximum(gf.y[3], 0.0)
    q_fine = np.interp(z_fine, z_grid, q_prof)
    T_fine = np.maximum(gf.y[4], 200.0)

    p_CO2f = C_CO2f*R_gas*T_fine/1e5;  p_H2f  = C_H2f*R_gas*T_fine/1e5
    p_CH4f = C_CH4f*R_gas*T_fine/1e5;  p_H2Of = C_H2Of*R_gas*T_fine/1e5
    r_fine = reaction_rate_SI(T_fine, p_CO2f, p_H2f, p_CH4f, p_H2Of)
    X_CO2  = np.clip(1.0 - C_CO2f/C_in_CO2, 0.0, 1.0)

    return dict(z=z_fine, C_CO2=C_CO2f, C_H2=C_H2f, C_CH4=C_CH4f,
                C_H2O=C_H2Of, q=q_fine, T=T_fine, r=r_fine, X_CO2=X_CO2,
                converged=converged, n_iter=it+1, conv_err=float(err))
# endregion


# region 4. SOLVE LOOP
# =============================================================================
def _compute_noSE_adiabatic(T_K, u_g, C_in_CO2, C_in_H2, C_in_CH4, N=300):
    """
    Fixed-bed (u_s = 0) adiabatic profile: no net adsorption, adiabatic energy balance.
    At steady state in a fixed bed the sorbent is saturated (q = q* everywhere, ads = 0),
    so only the reaction heats the gas.  denom = u_g·ρ_g·Cp_mix (gas-only, always > 0).
    Returns X_CO2_noSE (scalar) and a profile dict for plotting.
    """
    def rhs(_z, y):
        C_CO2_l = max(y[0], 0.0); C_H2_l  = max(y[1], 0.0)
        C_CH4_l = max(y[2], 0.0); C_H2O_l = max(y[3], 0.0)
        T_l     = max(y[4], 200.0)
        p_CO2 = C_CO2_l*R_gas*T_l/1e5; p_H2  = C_H2_l*R_gas*T_l/1e5
        p_CH4 = C_CH4_l*R_gas*T_l/1e5; p_H2O = C_H2O_l*R_gas*T_l/1e5
        r = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        denom = u_g * (P_Pa/(R_gas*T_l)) * _cp_mix(p_CO2, p_H2, p_CH4, p_H2O)
        return [
            -rho_bed_cat*r/(eps_b*u_g),
            -4.0*rho_bed_cat*r/(eps_b*u_g),
            +rho_bed_cat*r/(eps_b*u_g),
             2.0*rho_bed_cat*r/(eps_b*u_g),   # no adsorption sink
            (-dH_r)*rho_bed_cat*r / denom,     # adiabatic: reaction heat only
        ]
    z_grid = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs, [0, L_b], [C_in_CO2, C_in_H2, C_in_CH4, 0.0, T_K],
                    method='BDF', rtol=1e-5,
                    atol=np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-2]), t_eval=z_grid)
    C_CO2_f = np.maximum(sol.y[0], 0.0);  C_H2O_f = np.maximum(sol.y[3], 0.0)
    T_f     = np.maximum(sol.y[4], 200.0)
    p_H2O_f = C_H2O_f * R_gas * T_f / 1e5
    p_CO2_f = np.maximum(sol.y[0], 0.0) * R_gas * T_f / 1e5
    p_H2_f  = np.maximum(sol.y[1], 0.0) * R_gas * T_f / 1e5
    p_CH4_f = np.maximum(sol.y[2], 0.0) * R_gas * T_f / 1e5
    r_f     = reaction_rate_SI(T_f, p_CO2_f, p_H2_f, p_CH4_f, p_H2O_f)
    X_f     = np.clip(1.0 - C_CO2_f / C_in_CO2, 0.0, 1.0)
    X_CO2_noSE = float(X_f[-1])
    return dict(
        X_CO2_noSE=X_CO2_noSE,
        profile=dict(z=z_grid, C_CO2=C_CO2_f, C_H2O=C_H2O_f,
                     q=q_star(T_f, p_H2O_f), r=r_f, X_CO2=X_f, T=T_f)
    )


def _q_physics_init(T_K, u_g, C_in_CO2, C_in_H2, C_in_CH4, N=150):
    """Physics-based initial guess for q(z): saturated fixed-bed q*(p_H2O(z))."""
    def rhs_noads(_z, y):
        C_CO2_l = max(y[0], 0.0); C_H2_l  = max(y[1], 0.0)
        C_CH4_l = max(y[2], 0.0); C_H2O_l = max(y[3], 0.0)
        p_CO2 = C_CO2_l*R_gas*T_K/1e5; p_H2  = C_H2_l*R_gas*T_K/1e5
        p_CH4 = C_CH4_l*R_gas*T_K/1e5; p_H2O = C_H2O_l*R_gas*T_K/1e5
        r = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        return [
            -rho_bed_cat*r/(eps_b*u_g),
            -4.0*rho_bed_cat*r/(eps_b*u_g),
            +rho_bed_cat*r/(eps_b*u_g),
             2.0*rho_bed_cat*r/(eps_b*u_g),
        ]
    z_grid = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs_noads, [0, L_b], [C_in_CO2, C_in_H2, C_in_CH4, 0.0],
                    method='BDF', rtol=1e-4, atol=1e-8, t_eval=z_grid)
    p_H2O_prof = np.maximum(sol.y[3], 0.0) * R_gas * T_K / 1e5
    X_CO2_noSE = float(np.clip(1.0 - sol.y[0, -1] / C_in_CO2, 0.0, 1.0))
    return dict(q=q_star(T_K, p_H2O_prof), X_CO2_noSE=X_CO2_noSE)

def _fmt_seconds(s):
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"

all_results  = {}
noSE_results = {}

n_total     = len(T_IN_LIST) * len(U_S_LIST)
n_done      = 0
t_run_start = time.perf_counter()

for T_C in T_IN_LIST:
    T_K      = T_C + 273.15
    u_g      = u_g_STP*(T_K/T_STP)
    C_in_CO2 = y_CO2_in*P_Pa/(R_gas*T_K)
    C_in_H2  = y_H2_in *P_Pa/(R_gas*T_K)
    C_in_CH4 = y_CH4_in*P_Pa/(R_gas*T_K)

    print(f"\n{'='*60}")
    print(f"  T_in = {T_C} C  |  u_g = {u_g*1e3:.1f} mm/s  (ADIABATIC)")
    print(f"{'='*60}")

    _phys      = _q_physics_init(T_K, u_g, C_in_CO2, C_in_H2, C_in_CH4)
    q_init_raw = _phys['q']
    q_init     = q_init_raw[::-1]
    _noSE      = _compute_noSE_adiabatic(T_K, u_g, C_in_CO2, C_in_H2, C_in_CH4)
    noSE_results[T_C] = _noSE
    print(f"  non-SE fixed-bed conversion: {_noSE['X_CO2_noSE']*100:.1f}%")

    for i_us, u_s in enumerate(U_S_LIST):
        t0  = time.perf_counter()
        res = solve_mpb(u_s, u_g, T_K, C_in_CO2, C_in_H2, C_in_CH4, q_init=q_init)
        dt  = time.perf_counter() - t0
        n_done += 1
        elapsed = time.perf_counter() - t_run_start
        eta     = elapsed / n_done * (n_total - n_done)

        if res is not None:
            X_out  = float(res['X_CO2'][-1])*100
            q_out  = float(res['q'][0])
            T_max  = float(np.max(res['T'])) - 273.15
            tag    = "ok" if res['converged'] else "not-conv"
            print(f"  u_s={u_s*1e3:.4f} mm/s  X={X_out:.1f}%  "
                  f"q(0)={q_out:.3f} mol/kg  T_max={T_max:.1f} C  "
                  f"[{tag}, {res['n_iter']} iter, err={res['conv_err']:.2e}]"
                  f"  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
            q_init = np.interp(np.linspace(0, L_b, 150), res['z'], res['q'])
        else:
            print(f"  u_s={u_s*1e3:.4f} mm/s  FAILED"
                  f"  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
        all_results[(T_C, i_us)] = {'res': res, 'u_s': u_s,
                                     'T_K': T_K, 'C_in_CO2': C_in_CO2}

print(f"\nAll done.  Total: {_fmt_seconds(time.perf_counter() - t_run_start)}")
# endregion


# region 5. POST-PROCESSING HELPERS
# =============================================================================
def get_metrics(entry):
    res = entry['res']
    if res is None:
        return None
    T_K = entry['T_K']
    X_out      = float(res['X_CO2'][-1])
    q_out      = float(res['q'][0])
    T_max      = float(np.max(res['T']))
    T_out_solid = float(res['T'][0])   # solid outlet temperature at z=0
    p_H2O_peak  = float(np.max(res['C_H2O']))*R_gas*T_K/1e5
    q_star_peak = float(q_star(T_K, np.array([max(p_H2O_peak, 1e-8)]))[0])
    return dict(X_CO2=X_out, q_out=q_out, T_max=T_max, T_out_solid=T_out_solid,
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
i_plot   = np.arange(len(U_S_LIST))
pal      = plt.cm.plasma(np.linspace(0.1, 0.85, len(i_plot)))
pal2     = cmap(np.linspace(0.1, 0.85, len(T_IN_LIST)))

# Convenience: fixed-bed (u_s=0) adiabatic profile for T_C_PROF
_p0 = noSE_results.get(T_C_PROF, {}).get('profile')

# ── Plot 1: Axial profiles ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f'MPB axial profiles  |  T_in = {T_C_PROF} C  |  adiabatic, counter-current',
             fontsize=11)
for k, i_us in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_us))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    lbl = f"u_s = {e['u_s']*1e3:.3f} mm/s"
    axes[0,0].plot(r['z'], r['C_CO2']*1e3, color=pal[k], lw=2, label=lbl)
    axes[0,1].plot(r['z'], r['q'],          color=pal[k], lw=2, label=lbl)
    axes[1,0].plot(r['z'], r['X_CO2']*100,  color=pal[k], lw=2, label=lbl)
    axes[1,1].plot(r['z'], r['r']*1e3,      color=pal[k], lw=2, label=lbl)
if _p0 is not None:
    axes[0,0].plot(_p0['z'], _p0['C_CO2']*1e3, color='k', lw=2, ls='--', label='u_s = 0 (fixed bed)')
    axes[0,1].plot(_p0['z'], _p0['q'],           color='k', lw=2, ls='--', label='u_s = 0 (fixed bed)')
    axes[1,0].plot(_p0['z'], _p0['X_CO2']*100,   color='k', lw=2, ls='--', label='u_s = 0 (fixed bed)')
    axes[1,1].plot(_p0['z'], _p0['r']*1e3,       color='k', lw=2, ls='--', label='u_s = 0 (fixed bed)')
labels_units = [('C_CO2 [mmol/m3]',    'CO2 concentration'),
                ('q [mol/kg]',          'Solid H2O loading'),
                ('CO2 conversion [%]',  'CO2 conversion along bed'),
                ('r [mmol/(kg_cat.s)]', 'Reaction rate')]
for ax, (ylabel, title) in zip(axes.flat, labels_units):
    ax.set_xlabel('z [m]', fontsize=10);  ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10);     ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)
    ax.axvline(0,   color='tab:blue',   lw=1, ls=':', alpha=0.5)
    ax.axvline(L_b, color='tab:orange', lw=1, ls=':', alpha=0.5)
plt.tight_layout()
_savefig(f'plot1_axial_profiles_T{T_C_PROF}C.png');  plt.show()

# ── Plot 2: CO2 conversion vs u_s ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    us_ok, X_ok = [], []
    for i_us in range(len(U_S_LIST)):
        e = all_results.get((T_C, i_us))
        if e and e['res']:
            m = get_metrics(e)
            if m:
                us_ok.append(e['u_s']*1e3);  X_ok.append(m['X_CO2']*100)
    if us_ok:
        ax.semilogx(us_ok, X_ok, marker=markers[j], color=pal2[j],
                    lw=2, ms=6, label=f'{T_C} C (MPB adiabatic)')
        ax.axhline(equilibrium_conversion(T_C+273.15), color=pal2[j],
                   lw=1, ls=':', alpha=0.5, label=f'{T_C} C thermo. eq.')
        if T_C in noSE_results:
            ax.axhline(noSE_results[T_C]['X_CO2_noSE']*100, color=pal2[j],
                       lw=1.5, ls='--', alpha=0.8, label=f'{T_C} C u_s=0 (fixed bed)')
ax.set_xlabel('u_s [mm/s]', fontsize=11);  ax.set_ylabel('CO2 conversion [%]', fontsize=11)
ax.set_title('MPB adiabatic — CO2 conversion vs solid velocity', fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3);  ax.set_ylim(0, 105)
plt.tight_layout()
_savefig('plot2_conversion_vs_us.png');  plt.show()

# ── Plot 3: Sorbent utilisation vs u_s ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    us_ok, util_ok = [], []
    for i_us in range(len(U_S_LIST)):
        e = all_results.get((T_C, i_us))
        if e and e['res']:
            m = get_metrics(e)
            if m:
                us_ok.append(e['u_s']*1e3);  util_ok.append(m['sorbent_util']*100)
    if us_ok:
        ax.semilogx(us_ok, util_ok, marker=markers[j], color=pal2[j],
                    lw=2, ms=6, label=f'{T_C} C')
ax.axhline(100, color='grey', lw=1.5, ls='--', label='q = q* (fully saturated)')
ax.set_xlabel('u_s [mm/s]', fontsize=11)
ax.set_ylabel('Sorbent utilisation  q(z=0) / q*(p_H2O_max)  [%]', fontsize=11)
ax.set_title('MPB adiabatic — Sorbent utilisation vs solid velocity', fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('plot3_sorbent_utilisation.png');  plt.show()

# ── Plot 4: H2O profiles ─────────────────────────────────────────────────────
T_K_prof = T_C_PROF + 273.15
fig, (ax_q, ax_h) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'H2O profiles along bed  |  T_in = {T_C_PROF} C  |  adiabatic', fontsize=11)
for k, i_us in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_us))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    lbl = f"u_s = {e['u_s']*1e3:.3f} mm/s"
    ax_q.plot(r['z'], r['q'],                          color=pal[k], lw=2, label=lbl)
    ax_h.plot(r['z'], r['C_H2O']*R_gas*T_K_prof/1e2,  color=pal[k], lw=2, label=lbl)
if _p0 is not None:
    ax_q.plot(_p0['z'], _p0['q'],                            color='k', lw=2, ls='--', label='u_s = 0 (fixed bed)')
    ax_h.plot(_p0['z'], _p0['C_H2O']*R_gas*T_K_prof/1e2,   color='k', lw=2, ls='--', label='u_s = 0 (fixed bed)')
ax_q.set_xlabel('z [m]');  ax_q.set_ylabel('q [mol/kg]')
ax_q.set_title('Solid H2O loading q(z)');  ax_q.legend(fontsize=7);  ax_q.grid(True, alpha=0.3)
ax_h.set_xlabel('z [m]');  ax_h.set_ylabel('p_H2O [mbar]')
ax_h.set_title('Gas-phase H2O partial pressure');  ax_h.legend(fontsize=7);  ax_h.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'plot4_H2O_profiles_T{T_C_PROF}C.png');  plt.show()

# ── Plot 5: Temperature profiles ─────────────────────────────────────────────
fig, ax_T = plt.subplots(figsize=(9, 5))
fig.suptitle(f'Temperature profile along bed  |  T_in = {T_C_PROF} C  |  adiabatic',
             fontsize=11)
for k, i_us in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_us))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    lbl = f"u_s = {e['u_s']*1e3:.3f} mm/s"
    ax_T.plot(r['z'], r['T'] - 273.15, color=pal[k], lw=2, label=lbl)
if _p0 is not None:
    ax_T.plot(_p0['z'], _p0['T'] - 273.15, color='k', lw=2, ls='--', label='u_s = 0 (fixed bed)')
ax_T.axhline(T_C_PROF, color='grey', lw=1.5, ls='--', alpha=0.8, label='T_in')
ax_T.set_xlabel('z [m]', fontsize=10);  ax_T.set_ylabel('T [°C]', fontsize=10)
ax_T.set_title('Axial temperature profile  (adiabatic energy balance)', fontsize=9)
ax_T.legend(fontsize=7);  ax_T.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'plot5_temperature_T{T_C_PROF}C.png');  plt.show()

# ── Plot 6: Maximum temperature rise vs u_s ───────────────────────────────────
fig, ax6 = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    us_ok, dT_ok = [], []
    for i_us in range(len(U_S_LIST)):
        e = all_results.get((T_C, i_us))
        if e and e['res']:
            m = get_metrics(e)
            if m:
                us_ok.append(e['u_s']*1e3)
                dT_ok.append(m['T_max'] - (T_C + 273.15))
    if us_ok:
        ax6.semilogx(us_ok, dT_ok, marker=markers[j], color=pal2[j],
                     lw=2, ms=6, label=f'{T_C} C (MPB)')
    if T_C in noSE_results and noSE_results[T_C].get('profile') is not None:
        dT_noSE = float(np.max(noSE_results[T_C]['profile']['T'])) - (T_C + 273.15)
        ax6.axhline(dT_noSE, color=pal2[j], lw=1.5, ls='--', alpha=0.8,
                    label=f'{T_C} C u_s=0 (fixed bed)')
ax6.set_xlabel('u_s [mm/s]', fontsize=11)
ax6.set_ylabel('ΔT_max = T_peak − T_in  [K]', fontsize=11)
ax6.set_title('MPB adiabatic — peak temperature rise vs solid velocity\n'
              '(higher u_s = more cold solid = better cooling)', fontsize=10)
ax6.legend(fontsize=9);  ax6.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('plot6_Tmax_vs_us.png');  plt.show()
# endregion
