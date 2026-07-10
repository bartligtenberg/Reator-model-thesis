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
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# region 1. PARAMETERS
# =============================================================================

# --- Bed geometry (Bareschino lab setup) ---
d_b   = 0.050
L_b   = 2.000
A_b   = np.pi / 4 * d_b**2
V_bed = A_b * L_b
eps_b = 0.4

# --- Catalyst and sorbent loading ---
M_cat = 0.064
M_ads = 1.22
rho_bed_cat = M_cat / V_bed
rho_bed_ads = M_ads / V_bed
rho_bed_tot = (M_cat + M_ads) / V_bed

# --- Particle properties (13X zeolite pellets) ---
d_p   = 0.75e-3
eps_p = 0.615
tau_p = 3.0
rho_p = 1400   # [kg/m³]  particle density of sorbent (Bareschino)

# --- Dubinin-Astakhov isotherm (H2O on 13X) ---
W0_DA = 190.00e-6
E_DA  = 1190e3
n_DA  = 1.55

# --- LHHW kinetics (Wei et al., K_mix corrected to bar^-0.5) ---
T_ref_K = 555.0
k_ref   = 3.46e-4
Ea_k    = 77.5e3
A_OH    = 0.50;  dH_OH  =  22.4e3
A_H2    = 0.44;  dH_H2  =  -6.2e3
A_mix   = 0.88;  dH_mix = -10.0e3
P_FLOOR = 1e-4

# --- Thermochemistry ---
dH_r   = -165.0e3
dH_ads =  -45.0e3
Cp_cat = 1100.0
Cp_CO2 = 45.4;  Cp_H2 = 29.3;  Cp_CH4 = 46.9;  Cp_H2O = 34.2

# --- Wall heat transfer ---
U_a    = 2000.0

# --- Physical constants ---
R_gas  = 8.314
MW_H2O = 0.018015

# --- Operating conditions ---
P_bar = 1.0
P_Pa  = P_bar * 1e5
y_CO2_in = 0.04
y_H2_in  = 0.16
y_CH4_in = 0.80

# --- Inlet molar fluxes [mol/(m²·s)] — temperature-independent ---
T_STP   = 273.15
GHSV    = 0.5
Q_STP   = GHSV * M_ads / 3600.0
u_g_STP = Q_STP / A_b
F_total_in = u_g_STP * P_Pa / (R_gas * T_STP)  # total molar flux at inlet
F_in_CO2   = y_CO2_in * F_total_in
F_in_H2    = y_H2_in  * F_total_in
F_in_CH4   = y_CH4_in * F_total_in

# --- MPB scan parameters ---
U_S_LIST  = np.array([0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10, 12]) * 1e-3
T_IN_LIST = [280]

print(f"MPB flux form: d={d_b*100:.0f} cm, L={L_b:.1f} m, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3, u_g_STP={u_g_STP*1e3:.1f} mm/s")
print(f"  F_in_total = {F_total_in:.4f} mol/(m2·s)  (temperature-independent)")
print(f"  U_a = {U_a:.0f} W/(m3·K)")
# endregion


# region 2. FUNCTIONS
# =============================================================================
def P_sat_bar(T_K):
    log10_p = (29.8605 - 3.1522e3/T_K - 7.3037*np.log10(T_K)
               + 2.4247e-9*T_K + 1.8090e-6*T_K**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5

def rho_water(T_K):
    return 996.0 / (1.0 + 2.0e-3*(T_K - 298.15))

def q_star_vec(T_K, p_arr, W0, E, n):
    p    = np.asarray(p_arr, dtype=float)
    Psat = P_sat_bar(T_K)
    p_safe = np.clip(p, 1e-15, Psat*(1-1e-10))
    A_raw  = (R_gas/MW_H2O)*T_K*np.log(Psat/p_safe)
    A  = np.where((p <= 0)|(p >= Psat), 0.0, A_raw)
    W  = W0*np.exp(-np.minimum((A/E)**n, 500.0))
    qs = rho_water(T_K)/MW_H2O*W
    return np.where(p <= 0, 0.0, qs)

def K_LDF_vec(T_K, p_arr, W0, E, n):
    D_M  = 2.5e-5*(T_K/300.0)**1.75                                           # molecular diffusivity [m²/s], power-law T-dependence (Chapman-Enskog)
    p    = np.asarray(p_arr, dtype=float)                                      # ensure p is a numpy float array for vectorised operations
    dp   = 1.0/1e5                                                             # pressure step = 1 Pa expressed in bar; chosen so dividing by 2.0 gives dq*/dp in mol/(kg·Pa)
    dqsp = (q_star_vec(T_K, p+dp, W0, E, n)
            - q_star_vec(T_K, np.maximum(p-dp, 1e-15), W0, E, n)) / 2.0      # central finite difference: dq*/dp [mol/(kg·Pa)]; np.maximum prevents zero/negative pressure in DA log
    dqsp = np.maximum(dqsp, 1e-30)                                             # guard against division by zero when isotherm slope is numerically flat
    r_p = 0.5 * d_p                                                            # particle radius [m]
    return 15.0 * eps_p * D_M / (r_p**2 * tau_p * rho_p * R_gas * T_K * dqsp)  # Glueckauf LDF coefficient from pore diffusion [1/s]: large dq*/dp → slow K_LDF

def K_eq_sabatier(T_K):
    return 137.0*T_K**(-3.994)*np.exp(158700.0/(R_gas*T_K))

def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    vH    = lambda dH: np.exp(-dH/R_gas*(1.0/T_K - 1.0/T_ref_K))
    k     = k_ref*np.exp(-Ea_k/R_gas*(1.0/T_K - 1.0/T_ref_K))
    K_OH  = A_OH*vH(dH_OH);  K_H2 = A_H2*vH(dH_H2);  K_mix = A_mix*vH(dH_mix)
    K_eq  = K_eq_sabatier(T_K)
    p_CO2_s = np.maximum(p_CO2, P_FLOOR);  p_H2_s = np.maximum(p_H2, P_FLOOR)
    beta  = (p_CH4*p_H2O**2)/(K_eq*p_CO2_s*p_H2_s**4)
    f_eq  = np.maximum(1.0 - np.where(np.isfinite(beta), beta, 1e10), 0.0)
    DEN   = (1.0 + K_OH*np.maximum(p_H2O, 0)/p_H2_s**0.5
             + K_H2*p_H2_s**0.5 + K_mix*p_CO2_s**0.5)
    return k*(p_CO2_s*p_H2_s)**0.5*f_eq/DEN**2*1000.0

def q_star(T_K, p_H2O):
    return q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

_K_LDF_MAX = 20

def K_LDF(T_K, p_H2O):
    return np.minimum(K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA), _K_LDF_MAX)

def equilibrium_conversion(T_K_val):
    K = K_eq_sabatier(T_K_val)
    def f(X):
        d = 1.0 - 0.08*X
        return ((0.80+0.04*X)/d * (0.08*X/d)**2
                / ((0.04*(1-X)/d)*(0.16*(1-X)/d)**4 + 1e-100) - K)
    try:
        return brentq(f, 1e-9, 1-1e-9)*100.0
    except Exception:
        return 100.0

def _gas_cap(F_CO2, F_H2, F_CH4, F_H2O):
    """Thermal flux of gas phase [W/(m²·K)] = Σ F_i·Cp_i."""
    return F_CO2*Cp_CO2 + F_H2*Cp_H2 + F_CH4*Cp_CH4 + F_H2O*Cp_H2O

def _partial_pressures(F_CO2, F_H2, F_CH4, F_H2O):
    """Convert molar fluxes to partial pressures [bar] via mole fractions."""
    F_tot = F_CO2 + F_H2 + F_CH4 + F_H2O
    if F_tot < 1e-30:
        return 0.0, 0.0, 0.0, 0.0
    return (F_CO2/F_tot*P_bar, F_H2/F_tot*P_bar,
            F_CH4/F_tot*P_bar, F_H2O/F_tot*P_bar)
# endregion


# region 3. DECOUPLED SOLVER
# =============================================================================
def solve_mpb(u_s, T_K, T_wall=None, max_iter=400, tol=1e-4, N=100, q_init=None):
    """
    Counter-current MPB — molar flux form, lightly cooled, regime-switching.

    State: F_i [mol/(m²·s)].  No u_g or ε_b in species balance.
    Partial pressures: p_i = (F_i/F_total) · P_bar.
    Energy denominator: Σ F_i·Cp_i − u_s·ρ_bed·Cp_cat.
    """
    if T_wall is None:
        T_wall = T_K

    solid_cap  = u_s * rho_bed_tot * Cp_cat
    gas_cap_in = _gas_cap(F_in_CO2, F_in_H2, F_in_CH4, 0.0)
    gas_dominates = (solid_cap < gas_cap_in)

    z_grid = np.linspace(0.0, L_b, N)
    if q_init is not None:
        q_prof = np.interp(z_grid, np.linspace(0, L_b, len(q_init)), q_init)
    else:
        q_prof = np.zeros(N)
    T_prof = T_K * np.ones(N)

    converged = False
    err = 1.0
    balance_trace = []   # DIAG: per-sweep water-closure residual (fraction)
    alpha = 0.5          # Picard relaxation; halved when balance_trace sign flips

    for it in range(max_iter):
        q_fn = interp1d(z_grid, q_prof, kind='linear',
                        bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))
        T_fn = interp1d(z_grid, T_prof, kind='linear',
                        bounds_error=False, fill_value=(T_prof[0], T_prof[-1]))

        # Adaptive damping: halve alpha on sign-flip in balance residual (oscillation)
        if it >= 2 and balance_trace[-1] * balance_trace[-2] < 0:
            alpha = max(0.05, alpha * 0.5)

        if gas_dominates:
            # ── GAS IVP: state = [F_CO2, F_H2, F_CH4, J_H2O, T] ─────────────
            # J_H2O is a change of variables: J = F_H2O − ρ_ads·u_s·q
            # Conceptually: J = "H2O in gas" minus "H2O carried by solid past the same point".
            # When you write dJ/dz, the adsorption terms (sink in gas, source from solid) cancel
            # exactly, leaving only dJ/dz = 2·ρ_cat·r (reaction production only).
            # To recover F_H2O: F_H2O = J + ρ_ads·u_s·q  (can go negative if solid is over-loaded).
            def gas_rhs(z, y):
                F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
                F_CH4_l = max(y[2], 0.0)
                T_l     = max(y[4], 200.0)
                q_l     = max(float(q_fn(z)), 0.0)
                F_H2O_l = max(y[3] + rho_bed_ads * u_s * q_l, 0.0)

                p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                    F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)

                r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                              np.array([p_CH4]), np.array([p_H2O]))[0])
                qs  = float(q_star(T_l, np.array([p_H2O]))[0])
                Kl  = float(K_LDF(T_l,  np.array([p_H2O]))[0])
                ads = Kl*(qs - q_l)

                gas_cap_l = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
                denom     = gas_cap_l - solid_cap

                Q_rxn  = (-dH_r)   * rho_bed_cat * r
                Q_ads  = (-dH_ads) * rho_bed_ads * ads
                Q_wall = U_a * (T_l - T_wall)
                dTdz   = (Q_rxn + Q_ads - Q_wall) / denom

                return [
                    -rho_bed_cat * r,        # dF_CO2/dz
                    -4.0*rho_bed_cat * r,    # dF_H2/dz
                    +rho_bed_cat * r,        # dF_CH4/dz
                    2.0*rho_bed_cat * r,     # dJ_H2O/dz  (adsorption cancels)
                    dTdz,
                ]

            J0 = -rho_bed_ads * u_s * q_prof[0]
            gs = solve_ivp(gas_rhs, [0.0, L_b],
                           [F_in_CO2, F_in_H2, F_in_CH4, J0, T_K],
                           method='BDF', rtol=1e-4,
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8, 1e-2]),
                           t_eval=z_grid, dense_output=False)
            if not gs.success:
                return None

            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)
            F_CH4_prof = np.maximum(gs.y[2], 0.0)
            F_H2O_prof = np.maximum(gs.y[3] + rho_bed_ads * u_s * q_prof, 0.0)
            T_prof_new = np.maximum(gs.y[4], 200.0)

            F_tot_prof  = np.maximum(F_CO2_prof + F_H2_prof + F_CH4_prof + F_H2O_prof, 1e-30)
            p_H2O_prof  = F_H2O_prof / F_tot_prof * P_bar
            p_H2O_fn    = interp1d(z_grid, p_H2O_prof, kind='linear',
                                   bounds_error=False,
                                   fill_value=(p_H2O_prof[0], p_H2O_prof[-1]))
            T_fn_new    = interp1d(z_grid, T_prof_new, kind='linear',
                                   bounds_error=False,
                                   fill_value=(T_prof_new[0], T_prof_new[-1]))

            # ── SOLID IVP: state = [q] ────────────────────────────────────────
            def solid_rhs(zeta, q_arr):
                z_pos    = L_b - float(zeta)
                T_local  = float(T_fn_new(z_pos))
                p_H2O_l  = max(float(p_H2O_fn(z_pos)), 0.0)
                qs  = float(q_star(T_local, np.array([p_H2O_l]))[0])
                Kl  = float(K_LDF(T_local,  np.array([p_H2O_l]))[0])
                q_val = max(float(q_arr[0]), 0.0)
                return [Kl*(qs - q_val)/u_s]

            ss = solve_ivp(solid_rhs, [0.0, L_b], [0.0],
                           method='BDF', rtol=1e-4, atol=1e-8,
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                return None

            z_from_zeta = L_b - ss.t
            q_from_zeta = np.maximum(ss.y[0], 0.0)
            sort_idx    = np.argsort(z_from_zeta)
            q_new = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])

            q_prof_new = (1 - alpha)*q_prof + alpha*q_new
            T_prof     = (1 - alpha)*T_prof + alpha*T_prof_new

            prod_to_L  = 2.0 * np.maximum(F_CO2_prof - F_CO2_prof[-1], 0.0)
            q_max_prof = prod_to_L / max(rho_bed_ads * u_s, 1e-30)
            q_prof_new = np.minimum(q_prof_new, q_max_prof)

            scale = max(np.max(q_prof_new), 1e-8)
            err   = np.max(np.abs(q_prof_new - q_prof)) / scale
            q_prof = q_prof_new

        else:
            # ── GAS IVP: state = [F_CO2, F_H2, F_CH4, J_H2O]  (T frozen) ─────
            # J_H2O is a change of variables: J = F_H2O − ρ_ads·u_s·q
            # Conceptually: J = "H2O in gas" minus "H2O carried by solid past the same point".
            # When you write dJ/dz, the adsorption terms cancel, leaving dJ/dz = 2·ρ_cat·r only.
            # To recover F_H2O: F_H2O = J + ρ_ads·u_s·q  (can go negative if solid is over-loaded).
            def gas_rhs_no_T(z, y):
                F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
                F_CH4_l = max(y[2], 0.0)
                T_l     = max(float(T_fn(z)), 200.0)
                q_l     = max(float(q_fn(z)), 0.0)
                F_H2O_l = max(y[3] + rho_bed_ads * u_s * q_l, 0.0)

                p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                    F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)

                r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                              np.array([p_CH4]), np.array([p_H2O]))[0])

                return [
                    -rho_bed_cat * r,
                    -4.0*rho_bed_cat * r,
                    +rho_bed_cat * r,
                    2.0*rho_bed_cat * r,     # dJ_H2O/dz  (adsorption cancels)
                ]

            J0 = -rho_bed_ads * u_s * q_prof[0]
            gs = solve_ivp(gas_rhs_no_T, [0.0, L_b],
                           [F_in_CO2, F_in_H2, F_in_CH4, J0],
                           method='BDF', rtol=1e-4,
                           atol=np.array([1e-8, 1e-8, 1e-8, 1e-8]),
                           t_eval=z_grid, dense_output=False)
            if not gs.success:
                return None

            F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)
            F_CH4_prof = np.maximum(gs.y[2], 0.0)
            F_H2O_prof = np.maximum(gs.y[3] + rho_bed_ads * u_s * q_prof, 0.0)
            F_tot_prof = np.maximum(F_CO2_prof + F_H2_prof + F_CH4_prof + F_H2O_prof, 1e-30)

            _make_fn = lambda p: interp1d(z_grid, p, kind='linear',
                                          bounds_error=False, fill_value=(p[0], p[-1]))
            F_CO2_fn = _make_fn(F_CO2_prof);  F_H2_fn  = _make_fn(F_H2_prof)
            F_CH4_fn = _make_fn(F_CH4_prof);  F_H2O_fn = _make_fn(F_H2O_prof)
            F_tot_fn = _make_fn(F_tot_prof)

            # ── SOLID IVP: state = [q, T] ─────────────────────────────────────
            def solid_rhs_with_T(zeta, y_arr):
                q_val = max(float(y_arr[0]), 0.0)
                T_val = max(float(y_arr[1]), 200.0)
                z_pos = L_b - float(zeta)

                F_CO2_l = max(float(F_CO2_fn(z_pos)), 0.0)
                F_H2_l  = max(float(F_H2_fn(z_pos)),  0.0)
                F_CH4_l = max(float(F_CH4_fn(z_pos)), 0.0)
                F_H2O_l = max(float(F_H2O_fn(z_pos)), 0.0)
                F_tot_l = max(float(F_tot_fn(z_pos)), 1e-30)

                p_CO2 = F_CO2_l/F_tot_l*P_bar;  p_H2  = F_H2_l /F_tot_l*P_bar
                p_CH4 = F_CH4_l/F_tot_l*P_bar;  p_H2O = F_H2O_l/F_tot_l*P_bar

                r   = float(reaction_rate_SI(T_val, np.array([p_CO2]), np.array([p_H2]),
                                              np.array([p_CH4]), np.array([p_H2O]))[0])
                qs  = float(q_star(T_val, np.array([p_H2O]))[0])
                Kl  = float(K_LDF(T_val, np.array([p_H2O]))[0])
                ads = Kl*(qs - q_val)

                gas_cap_l   = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
                solid_denom = solid_cap - gas_cap_l

                Q_rxn  = (-dH_r)   * rho_bed_cat * r
                Q_ads  = (-dH_ads) * rho_bed_ads * ads
                Q_wall = U_a * (T_val - T_wall)

                return [Kl*(qs - q_val)/u_s,
                        (Q_rxn + Q_ads - Q_wall) / solid_denom]

            ss = solve_ivp(solid_rhs_with_T, [0.0, L_b], [0.0, T_K],
                           method='BDF', rtol=1e-4, atol=np.array([1e-8, 1.0]),
                           max_step=1e-3,
                           t_eval=np.linspace(0.0, L_b, N), dense_output=False)
            if not ss.success:
                return None

            z_from_zeta  = L_b - ss.t
            q_from_zeta  = np.maximum(ss.y[0], 0.0)
            T_from_zeta  = np.maximum(ss.y[1], 200.0)
            sort_idx     = np.argsort(z_from_zeta)
            q_new  = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])
            T_new  = np.interp(z_grid, z_from_zeta[sort_idx], T_from_zeta[sort_idx])

            q_prof_new = (1 - alpha)*q_prof + alpha*q_new
            T_prof_new = (1 - alpha)*T_prof + alpha*T_new

            prod_to_L  = 2.0 * np.maximum(F_CO2_prof - F_CO2_prof[-1], 0.0)
            q_max_prof = prod_to_L / max(rho_bed_ads * u_s, 1e-30)
            q_prof_new = np.minimum(q_prof_new, q_max_prof)

            err_q = np.max(np.abs(q_prof_new - q_prof)) / max(np.max(q_prof_new), 1e-8)
            err_T = np.max(np.abs(T_prof_new - T_prof)) / T_K
            err   = max(err_q, err_T)

            q_prof = q_prof_new
            T_prof = T_prof_new

        # ── DIAG: water-closure residual from THIS sweep's profiles ───────────
        _X_sw    = max(1.0 - F_CO2_prof[-1]/F_in_CO2, 0.0)
        _prod_sw = 2.0 * F_in_CO2 * _X_sw
        _gas_sw  = F_H2O_prof[-1]
        _sol_sw  = rho_bed_ads * u_s * q_prof[0]
        balance_trace.append((_prod_sw - _gas_sw - _sol_sw) / max(_prod_sw, 1e-30))

        if err < tol:
            converged = True
            break

    # ── Final recompute on fine grid ─────────────────────────────────────────
    z_fine = np.linspace(0.0, L_b, 300)
    q_fn_f = interp1d(z_grid, q_prof, kind='linear',
                      bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))
    T_fn_f = interp1d(z_grid, T_prof, kind='linear',
                      bounds_error=False, fill_value=(T_prof[0], T_prof[-1]))

    q_fine = np.interp(z_fine, z_grid, q_prof)

    if gas_dominates:
        def gas_rhs_final(z, y):
            F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
            F_CH4_l = max(y[2], 0.0)
            T_l     = max(y[4], 200.0)
            q_l     = max(float(q_fn_f(z)), 0.0)
            F_H2O_l = max(y[3] + rho_bed_ads * u_s * q_l, 0.0)
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
                2.0*rho_bed_cat*r,           # dJ_H2O/dz  (adsorption cancels)
                (Q_rxn + Q_ads - Q_wall)/denom,
            ]
        J0_fine = -rho_bed_ads * u_s * q_fine[0]
        gf = solve_ivp(gas_rhs_final, [0.0, L_b],
                       [F_in_CO2, F_in_H2, F_in_CH4, J0_fine, T_K],
                       method='BDF', rtol=1e-6,
                       atol=np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-3]),
                       t_eval=z_fine, dense_output=False)
        F_CO2f = np.maximum(gf.y[0], 0.0);  F_H2f  = np.maximum(gf.y[1], 0.0)
        F_CH4f = np.maximum(gf.y[2], 0.0)
        F_H2O_rawf = gf.y[3] + rho_bed_ads * u_s * q_fine   # DIAG: before clamping
        F_H2Of = np.maximum(F_H2O_rawf, 0.0)
        T_fine = np.maximum(gf.y[4], 200.0)

    else:
        def gas_rhs_final_no_T(z, y):
            F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
            F_CH4_l = max(y[2], 0.0)
            T_l     = max(float(T_fn_f(z)), 200.0)
            q_l     = max(float(q_fn_f(z)), 0.0)
            F_H2O_l = max(y[3] + rho_bed_ads * u_s * q_l, 0.0)
            p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
            r   = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                          np.array([p_CH4]), np.array([p_H2O]))[0])
            return [
                -rho_bed_cat*r,
                -4.0*rho_bed_cat*r,
                +rho_bed_cat*r,
                2.0*rho_bed_cat*r,           # dJ_H2O/dz  (adsorption cancels)
            ]
        J0_fine = -rho_bed_ads * u_s * q_fine[0]
        gf = solve_ivp(gas_rhs_final_no_T, [0.0, L_b],
                       [F_in_CO2, F_in_H2, F_in_CH4, J0_fine],
                       method='BDF', rtol=1e-6,
                       atol=np.array([1e-10, 1e-10, 1e-10, 1e-10]),
                       t_eval=z_fine, dense_output=False)
        F_CO2f = np.maximum(gf.y[0], 0.0);  F_H2f  = np.maximum(gf.y[1], 0.0)
        F_CH4f = np.maximum(gf.y[2], 0.0)
        F_H2O_rawf = gf.y[3] + rho_bed_ads * u_s * q_fine   # DIAG: before clamping
        F_H2Of = np.maximum(F_H2O_rawf, 0.0)
        T_fine = np.interp(z_fine, z_grid, T_prof)
    F_totf   = np.maximum(F_CO2f + F_H2f + F_CH4f + F_H2Of, 1e-30)
    p_CO2f   = F_CO2f/F_totf*P_bar;  p_H2f  = F_H2f /F_totf*P_bar
    p_CH4f   = F_CH4f/F_totf*P_bar;  p_H2Of = F_H2Of/F_totf*P_bar
    r_fine   = reaction_rate_SI(T_fine, p_CO2f, p_H2f, p_CH4f, p_H2Of)
    X_CO2    = np.clip(1.0 - F_CO2f/F_in_CO2, 0.0, 1.0)

    # ── DIAG: gas-side adsorption sink integrated over the bed ────────────────
    # S_gas = ∫ ρ_bed_ads · K_LDF·(q*−q) dz  [mol/(m²·s)] = water the GAS removed.
    # In the J-formulation this should match solid carry-out ρ·u_s·q(0) and
    # (produced − gas_out) by construction of J_H2O = F_H2O − ρ·u_s·q.
    qs_fine   = q_star(T_fine, p_H2Of)
    KLDF_fine = K_LDF(T_fine, p_H2Of)
    ads_fine  = KLDF_fine * (qs_fine - q_fine)
    S_gas     = float(np.trapz(rho_bed_ads * ads_fine, z_fine))

    # Convert fluxes to concentrations for output (compatible with plot code)
    u_g_fine = F_totf * R_gas * T_fine / P_Pa
    C_CO2f   = F_CO2f / u_g_fine
    C_H2f    = F_H2f  / u_g_fine
    C_CH4f   = F_CH4f / u_g_fine
    C_H2Of   = F_H2Of / u_g_fine

    return dict(z=z_fine, C_CO2=C_CO2f, C_H2=C_H2f, C_CH4=C_CH4f,
                C_H2O=C_H2Of, q=q_fine, T=T_fine, r=r_fine, X_CO2=X_CO2,
                converged=converged, n_iter=it+1, conv_err=float(err),
                gas_dominates=gas_dominates,
                F_H2O_out=float(F_H2Of[-1]),     # DIAG: actual integrated outlet flux
                F_H2O_raw_min=float(np.min(F_H2O_rawf)),  # DIAG: min F_H2O before clamping
                S_gas=S_gas,                     # DIAG: integrated gas adsorption sink
                balance_trace=balance_trace)     # DIAG: per-sweep closure residual
# endregion


# region 4. SOLVE LOOP
# =============================================================================
def _compute_noSE(T_K, T_wall, N=300):
    """Fixed-bed (u_s=0) reference with wall cooling: no sorption enhancement."""
    def rhs(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        T_l     = max(y[4], 200.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r = float(reaction_rate_SI(T_l, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        gas_cap = _gas_cap(F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        Q_rxn  = (-dH_r) * rho_bed_cat * r
        Q_wall = U_a * (T_l - T_wall)
        return [
            -rho_bed_cat*r, -4.0*rho_bed_cat*r,
            +rho_bed_cat*r,  2.0*rho_bed_cat*r,
            (Q_rxn - Q_wall) / gas_cap,
        ]
    z_grid = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0, T_K],
                    method='BDF', rtol=1e-5,
                    atol=np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-2]), t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    T_f     = np.maximum(sol.y[4], 200.0)
    F_tot_f = np.maximum(sol.y[0] + sol.y[1] + sol.y[2] + sol.y[3], 1e-30)
    p_H2O_f = np.maximum(sol.y[3], 0.0)/F_tot_f * P_bar
    p_CO2_f = F_CO2_f / F_tot_f * P_bar
    p_H2_f  = np.maximum(sol.y[1], 0.0)/F_tot_f * P_bar
    p_CH4_f = np.maximum(sol.y[2], 0.0)/F_tot_f * P_bar
    r_f     = reaction_rate_SI(T_f, p_CO2_f, p_H2_f, p_CH4_f, p_H2O_f)
    X_f     = np.clip(1.0 - F_CO2_f/F_in_CO2, 0.0, 1.0)
    u_g_f   = F_tot_f * R_gas * T_f / P_Pa
    return dict(X_CO2_noSE=float(X_f[-1]),
                profile=dict(z=z_grid,
                             C_CO2=F_CO2_f/u_g_f,
                             C_H2O=np.maximum(sol.y[3], 0.0)/u_g_f,
                             q=q_star(T_f, p_H2O_f), r=r_f, X_CO2=X_f, T=T_f))

def _q_physics_init(T_K, N=150):
    """Isothermal, no-adsorption gas pass to build initial q profile."""
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

def _fmt_seconds(s):
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"

all_results  = {}
noSE_results = {}
n_total      = len(T_IN_LIST) * len(U_S_LIST)
n_done       = 0
t_run_start  = time.perf_counter()

for T_C in T_IN_LIST:
    T_K    = T_C + 273.15
    T_wall = T_K

    gas_cap_in = _gas_cap(F_in_CO2, F_in_H2, F_in_CH4, 0.0)
    u_s_star   = gas_cap_in / (rho_bed_tot * Cp_cat)

    print(f"\n{'='*60}")
    print(f"  T_in = {T_C} C  |  u_g_STP = {u_g_STP*1e3:.1f} mm/s  |  U_a = {U_a:.0f} W/(m3·K)")
    print(f"  u_s* = {u_s_star*1e3:.3f} mm/s  (regime switch at this velocity)")
    print(f"{'='*60}")

    _phys      = _q_physics_init(T_K)
    q_init_raw = _phys['q']
    q_init     = q_init_raw[::-1]
    _noSE      = _compute_noSE(T_K, T_wall)
    noSE_results[T_C] = _noSE
    print(f"  non-SE fixed-bed conversion: {_noSE['X_CO2_noSE']*100:.1f}%")

    for i_us, u_s in enumerate(U_S_LIST):
        t0  = time.perf_counter()
        res = solve_mpb(u_s, T_K, T_wall=T_wall, q_init=q_init)
        dt  = time.perf_counter() - t0
        n_done += 1
        elapsed = time.perf_counter() - t_run_start
        eta     = elapsed/n_done*(n_total - n_done)

        if res is not None:
            X_out  = float(res['X_CO2'][-1])*100
            q_out  = float(res['q'][0])
            T_max  = float(np.max(res['T'])) - 273.15
            regime = "gas" if res['gas_dominates'] else "solid"
            tag    = "ok" if res['converged'] else "not-conv"
            X_noSE     = noSE_results[T_C]['X_CO2_noSE'] * 100
            dX         = X_out - X_noSE
            p_H2O_peak = float(np.max(res['C_H2O'])) * R_gas * T_K / 1e5
            q_star_pk  = float(q_star(T_K, np.array([max(p_H2O_peak, 1e-8)]))[0])
            util       = q_out / max(q_star_pk, 1e-10) * 100
            print(f"  u_s={u_s*1e3:.4f} mm/s  X={X_out:.1f}%  ΔX={dX:+.1f}%  "
                  f"q(0)={q_out:.3f}  util={util:.0f}%  T_max={T_max:.1f}C  "
                  f"[{regime}-dom, {tag}, {res['n_iter']} iter]"
                  f"  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
            _H2O_prod  = 2.0 * (X_out/100) * F_in_CO2
            _H2O_solid = rho_bed_ads * u_s * q_out
            _H2O_gas   = res['F_H2O_out']
            _sol_frac  = _H2O_solid / max(_H2O_prod, 1e-30) * 100
            _err_pct   = (_H2O_prod - _H2O_gas - _H2O_solid) / max(_H2O_prod, 1e-30) * 100
            print(f"    H2O: prod={_H2O_prod*1e3:.3f}  "
                  f"solid={_H2O_solid*1e3:.3f} ({_sol_frac:.0f}%)  "
                  f"gas={_H2O_gas*1e3:.3f}  balance_err={_err_pct:+.1f}%  [mmol/(m²·s)]")
            _C_tot = res['C_CO2'][-1] + res['C_H2'][-1] + res['C_CH4'][-1] + res['C_H2O'][-1]
            _y = lambda c: c / max(_C_tot, 1e-30) * 100
            print(f"    outlet [mol%]:  CO2={_y(res['C_CO2'][-1]):.2f}  "
                  f"H2={_y(res['C_H2'][-1]):.2f}  "
                  f"CH4={_y(res['C_CH4'][-1]):.2f}  "
                  f"H2O={_y(res['C_H2O'][-1]):.2f}")
            if not res['converged']:
                print(f"    NOT CONVERGED: conv_err={res['conv_err']:.2e}  "
                      f"n_iter={res['n_iter']}  "
                      f"balance_trace(last3)={[f'{v:+.3f}' for v in res['balance_trace'][-3:]]}")
            q_init = np.interp(np.linspace(0, L_b, 150), res['z'], res['q'])
        else:
            print(f"  u_s={u_s*1e3:.4f} mm/s  FAILED"
                  f"  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
        all_results[(T_C, i_us)] = {'res': res, 'u_s': u_s,
                                     'T_K': T_K, 'T_wall': T_wall}

print(f"\nAll done.  Total: {_fmt_seconds(time.perf_counter() - t_run_start)}")
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
i_plot   = np.arange(len(U_S_LIST))
pal      = plt.cm.plasma(np.linspace(0.1, 0.85, len(i_plot)))
pal2     = cmap(np.linspace(0.1, 0.85, len(T_IN_LIST)))

_p0 = noSE_results.get(T_C_PROF, {}).get('profile')

# ── Plot 1: Axial profiles ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f'MPB flux form  |  T_in = {T_C_PROF} C  |  '
             f'U_a = {U_a:.0f} W/(m³·K), counter-current', fontsize=11)
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
    axes[0,0].plot(_p0['z'], _p0['C_CO2']*1e3, color='k', lw=2, ls='--', label='u_s=0 (fixed bed)')
    axes[0,1].plot(_p0['z'], _p0['q'],          color='k', lw=2, ls='--', label='u_s=0 (fixed bed)')
    axes[1,0].plot(_p0['z'], _p0['X_CO2']*100,  color='k', lw=2, ls='--', label='u_s=0 (fixed bed)')
    axes[1,1].plot(_p0['z'], _p0['r']*1e3,      color='k', lw=2, ls='--', label='u_s=0 (fixed bed)')
labels_units = [('C_CO2 [mmol/m3]', 'CO2 concentration'),
                ('q [mol/kg]',       'Solid H2O loading'),
                ('CO2 conversion [%]', 'CO2 conversion along bed'),
                ('r [mmol/(kg_cat.s)]', 'Reaction rate')]
for ax, (ylabel, title) in zip(axes.flat, labels_units):
    ax.set_xlabel('z [m]', fontsize=10);  ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10);     ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)
    ax.axvline(0,   color='tab:blue',   lw=1, ls=':', alpha=0.5)
    ax.axvline(L_b, color='tab:orange', lw=1, ls=':', alpha=0.5)
plt.tight_layout()
_savefig(f'flux_plot1_axial_profiles_T{T_C_PROF}C.png');  plt.show()

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
                    lw=2, ms=6, label=f'{T_C} C (MPB)')
        ax.axhline(equilibrium_conversion(T_C+273.15), color=pal2[j],
                   lw=1, ls=':', alpha=0.5, label=f'{T_C} C thermo. eq.')
        if T_C in noSE_results:
            ax.axhline(noSE_results[T_C]['X_CO2_noSE']*100, color=pal2[j],
                       lw=1.5, ls='--', alpha=0.8, label=f'{T_C} C u_s=0 (fixed bed)')
ax.set_xlabel('u_s [mm/s]', fontsize=11);  ax.set_ylabel('CO2 conversion [%]', fontsize=11)
ax.set_title(f'MPB flux form  |  U_a = {U_a:.0f} W/(m³·K)  — CO2 conversion vs solid velocity',
             fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3);  ax.set_ylim(0, 105)
plt.tight_layout()
_savefig('flux_plot2_conversion_vs_us.png');  plt.show()

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
ax.set_title(f'MPB flux form  |  U_a = {U_a:.0f} W/(m³·K)  — Sorbent utilisation', fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('flux_plot3_sorbent_utilisation.png');  plt.show()

# ── Plot 4: H2O profiles ─────────────────────────────────────────────────────
T_K_prof = T_C_PROF + 273.15
fig, (ax_q, ax_h) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'H2O profiles  |  T_in = {T_C_PROF} C  |  U_a = {U_a:.0f} W/(m³·K)', fontsize=11)
for k, i_us in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_us))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    lbl = f"u_s = {e['u_s']*1e3:.3f} mm/s"
    ax_q.plot(r['z'], r['q'],                          color=pal[k], lw=2, label=lbl)
    ax_h.plot(r['z'], r['C_H2O']*R_gas*T_K_prof/1e2,  color=pal[k], lw=2, label=lbl)
if _p0 is not None:
    ax_q.plot(_p0['z'], _p0['q'],                          color='k', lw=2, ls='--', label='u_s=0 (fixed bed)')
    ax_h.plot(_p0['z'], _p0['C_H2O']*R_gas*T_K_prof/1e2,  color='k', lw=2, ls='--', label='u_s=0 (fixed bed)')
ax_q.set_xlabel('z [m]');  ax_q.set_ylabel('q [mol/kg]')
ax_q.set_title('Solid H2O loading');  ax_q.legend(fontsize=7);  ax_q.grid(True, alpha=0.3)
ax_h.set_xlabel('z [m]');  ax_h.set_ylabel('p_H2O [mbar]')
ax_h.set_title('Gas-phase H2O partial pressure');  ax_h.legend(fontsize=7);  ax_h.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'flux_plot4_H2O_profiles_T{T_C_PROF}C.png');  plt.show()

# ── Plot 5: Temperature profiles ─────────────────────────────────────────────
fig, ax_T = plt.subplots(figsize=(9, 5))
fig.suptitle(f'Temperature profile  |  T_in = {T_C_PROF} C  |  U_a = {U_a:.0f} W/(m³·K)',
             fontsize=11)
for k, i_us in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_us))
    if e is None or e['res'] is None:
        continue
    r      = e['res']
    regime = 'g' if r['gas_dominates'] else 's'
    lbl    = f"u_s = {e['u_s']*1e3:.3f} mm/s ({regime})"
    ax_T.plot(r['z'], r['T'] - 273.15, color=pal[k], lw=2, label=lbl)
if _p0 is not None:
    ax_T.plot(_p0['z'], _p0['T'] - 273.15, color='k', lw=2, ls='--', label='u_s=0 (fixed bed)')
ax_T.axhline(T_C_PROF, color='grey', lw=1.5, ls='--', alpha=0.8, label=f'T_in = T_wall = {T_C_PROF} °C')
ax_T.set_xlabel('z [m]', fontsize=10);  ax_T.set_ylabel('T [°C]', fontsize=10)
ax_T.set_title('(g) = gas-dominated  |  (s) = solid-dominated (T from solid IVP)', fontsize=9)
ax_T.legend(fontsize=7);  ax_T.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'flux_plot5_temperature_T{T_C_PROF}C.png');  plt.show()

# ── Plot 6: Peak temperature rise vs u_s ─────────────────────────────────────
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
ax6.set_title(f'MPB flux form  |  U_a = {U_a:.0f} W/(m³·K)  — peak temperature rise',
              fontsize=10)
ax6.legend(fontsize=9);  ax6.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('flux_plot6_Tmax_vs_us.png');  plt.show()
# endregion
