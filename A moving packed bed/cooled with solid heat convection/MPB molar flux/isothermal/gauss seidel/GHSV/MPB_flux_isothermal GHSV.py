"""
Moving Packed Bed (MPB) Reactor Model — Steady-State, Isothermal, Pseudo-Homogeneous
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

Temperature is fixed at T_in (isothermal). No energy balance.

Solved by decoupled Gauss-Seidel iteration:
    1. Gas IVP: [F_CO2, F_H2, F_CH4, F_H2O] from z=0 to z=L using frozen q(z)
    2. Solid IVP: [q] from ζ=0 to ζ=L (i.e. z=L to z=0) using updated p_H2O(z)
    Repeat until q converges.
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
d_p   = 3e-3 #3mm, better for convergence
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

# --- Physical constants ---
R_gas  = 8.314
MW_H2O = 0.018015

# --- Operating conditions ---
P_bar = 1.0
P_Pa  = P_bar * 1e5
y_CO2_in = 0.04
y_H2_in  = 0.16
y_CH4_in = 0.80

# --- Inlet molar fluxes recomputed per GHSV inside solve loop ---
T_STP = 273.15

# --- MPB scan parameters ---
GHSV_LIST = np.round(np.arange(0.1, 1.21, 0.1), 1)
u_s_fixed = 5e-3
T_IN_LIST = [280]

print(f"MPB flux form (isothermal): d={d_b*100:.0f} cm, L={L_b:.1f} m, "
      f"d_p={d_p*1e3:.2f} mm, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3, u_s={u_s_fixed*1e3:.1f} mm/s (fixed)")
print(f"  GHSV scan: {GHSV_LIST} m3/(kg_ads·h)")
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
def solve_mpb(u_s, T_K, max_iter=400, tol=1e-5, N=200, q_init=None):
    """
    Counter-current MPB — molar flux form, isothermal.

    State: F_i [mol/(m²·s)].  No energy balance.
    Temperature fixed at T_K throughout the reactor.
    Partial pressures: p_i = (F_i/F_total) · P_bar.
    """
    z_grid = np.linspace(0.0, L_b, N)
    if q_init is not None:
        q_prof = np.interp(z_grid, np.linspace(0, L_b, len(q_init)), q_init)
    else:
        q_prof = np.zeros(N)

    converged = False
    err = 1.0

    for it in range(max_iter):
        q_fn = interp1d(z_grid, q_prof, kind='linear',
                        bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))

        # ── GAS IVP: state = [F_CO2, F_H2, F_CH4, F_H2O] ────────────────────
        def gas_rhs(z, y):
            F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
            F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
            q_l     = max(float(q_fn(z)), 0.0)

            p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
                F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)

            r   = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                          np.array([p_CH4]), np.array([p_H2O]))[0])
            qs  = float(q_star(T_K, np.array([p_H2O]))[0])
            Kl  = float(K_LDF(T_K,  np.array([p_H2O]))[0])
            ads = Kl*(qs - q_l)

            return [
                -rho_bed_cat * r,                        # dF_CO2/dz
                -4.0*rho_bed_cat * r,                    # dF_H2/dz
                +rho_bed_cat * r,                        # dF_CH4/dz
                2.0*rho_bed_cat*r - rho_bed_ads*ads,     # dF_H2O/dz
            ]

        gs = solve_ivp(gas_rhs, [0.0, L_b],
                       [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                       method='BDF', rtol=1e-4,
                       atol=np.array([1e-8, 1e-8, 1e-8, 1e-8]),
                       t_eval=z_grid, dense_output=False)
        if not gs.success:
            return None

        F_CO2_prof = np.maximum(gs.y[0], 0.0);  F_H2_prof  = np.maximum(gs.y[1], 0.0)
        F_CH4_prof = np.maximum(gs.y[2], 0.0);  F_H2O_prof = np.maximum(gs.y[3], 0.0)

        F_tot_prof  = np.maximum(F_CO2_prof + F_H2_prof + F_CH4_prof + F_H2O_prof, 1e-30)
        p_H2O_prof  = F_H2O_prof / F_tot_prof * P_bar
        p_H2O_fn    = interp1d(z_grid, p_H2O_prof, kind='linear',
                               bounds_error=False,
                               fill_value=(p_H2O_prof[0], p_H2O_prof[-1]))

        # ── SOLID IVP: state = [q], integrated in ζ = L−z direction ─────────
        def solid_rhs(zeta, q_arr):
            z_pos    = L_b - float(zeta)
            p_H2O_l  = max(float(p_H2O_fn(z_pos)), 0.0)
            qs  = float(q_star(T_K, np.array([p_H2O_l]))[0])
            Kl  = float(K_LDF(T_K,  np.array([p_H2O_l]))[0])
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

        q_prof_new = 0.5*q_prof + 0.5*q_new

        scale = max(np.max(q_prof_new), 1e-8)
        err   = np.max(np.abs(q_prof_new - q_prof)) / scale
        q_prof = q_prof_new

        if err < tol:
            converged = True
            break

    # ── Final recompute on fine grid ─────────────────────────────────────────
    z_fine = np.linspace(0.0, L_b, 300)
    q_fn_f = interp1d(z_grid, q_prof, kind='linear',
                      bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))

    def gas_rhs_final(z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        q_l     = max(float(q_fn_f(z)), 0.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r   = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                      np.array([p_CH4]), np.array([p_H2O]))[0])
        qs  = float(q_star(T_K, np.array([p_H2O]))[0])
        Kl  = float(K_LDF(T_K, np.array([p_H2O]))[0])
        ads = Kl*(qs - q_l)
        return [
            -rho_bed_cat*r,
            -4.0*rho_bed_cat*r,
            +rho_bed_cat*r,
            2.0*rho_bed_cat*r - rho_bed_ads*ads,
        ]

    gf = solve_ivp(gas_rhs_final, [0.0, L_b],
                   [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                   method='BDF', rtol=1e-6,
                   atol=np.array([1e-10, 1e-10, 1e-10, 1e-10]),
                   t_eval=z_fine, dense_output=False)

    F_CO2f = np.maximum(gf.y[0], 0.0);  F_H2f  = np.maximum(gf.y[1], 0.0)
    F_CH4f = np.maximum(gf.y[2], 0.0);  F_H2Of = np.maximum(gf.y[3], 0.0)
    q_fine = np.interp(z_fine, z_grid, q_prof)

    F_totf = np.maximum(F_CO2f + F_H2f + F_CH4f + F_H2Of, 1e-30)
    p_CO2f = F_CO2f/F_totf*P_bar;  p_H2f  = F_H2f /F_totf*P_bar
    p_CH4f = F_CH4f/F_totf*P_bar;  p_H2Of = F_H2Of/F_totf*P_bar
    r_fine = reaction_rate_SI(T_K, p_CO2f, p_H2f, p_CH4f, p_H2Of)
    X_CO2  = np.clip(1.0 - F_CO2f/F_in_CO2, 0.0, 1.0)

    u_g_fine = F_totf * R_gas * T_K / P_Pa
    C_CO2f   = F_CO2f / u_g_fine
    C_H2f    = F_H2f  / u_g_fine
    C_CH4f   = F_CH4f / u_g_fine
    C_H2Of   = F_H2Of / u_g_fine

    return dict(z=z_fine, C_CO2=C_CO2f, C_H2=C_H2f, C_CH4=C_CH4f,
                C_H2O=C_H2Of, q=q_fine, T=T_K*np.ones(len(z_fine)),
                r=r_fine, X_CO2=X_CO2,
                converged=converged, n_iter=it+1, conv_err=float(err))
# endregion


# region 4. SOLVE LOOP
# =============================================================================
def _compute_noSE(T_K, N=300):
    """Fixed-bed (u_s=0) isothermal reference: no sorption enhancement."""
    def rhs(_z, y):
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
    sol = solve_ivp(rhs, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                    method='BDF', rtol=1e-5,
                    atol=np.array([1e-9, 1e-9, 1e-9, 1e-9]), t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    F_tot_f = np.maximum(sol.y[0] + sol.y[1] + sol.y[2] + sol.y[3], 1e-30)
    p_H2O_f = np.maximum(sol.y[3], 0.0)/F_tot_f * P_bar
    p_CO2_f = F_CO2_f / F_tot_f * P_bar
    p_H2_f  = np.maximum(sol.y[1], 0.0)/F_tot_f * P_bar
    p_CH4_f = np.maximum(sol.y[2], 0.0)/F_tot_f * P_bar
    r_f     = reaction_rate_SI(T_K, p_CO2_f, p_H2_f, p_CH4_f, p_H2O_f)
    X_f     = np.clip(1.0 - F_CO2_f/F_in_CO2, 0.0, 1.0)
    u_g_f   = F_tot_f * R_gas * T_K / P_Pa
    return dict(X_CO2_noSE=float(X_f[-1]),
                profile=dict(z=z_grid,
                             C_CO2=F_CO2_f/u_g_f,
                             C_H2O=np.maximum(sol.y[3], 0.0)/u_g_f,
                             q=q_star(T_K, p_H2O_f), r=r_f, X_CO2=X_f))

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

def _compute_perfect_ads(T_K, N=300):
    """Gas IVP with p_H2O ≡ 0: instantaneous H2O removal — pure kinetic ceiling."""
    def rhs(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l = max(y[1], 0.0);  F_CH4_l = max(y[2], 0.0)
        F_tot = max(F_CO2_l + F_H2_l + F_CH4_l, 1e-30)
        p_CO2 = F_CO2_l/F_tot * P_bar;  p_H2 = F_H2_l/F_tot * P_bar
        p_CH4 = F_CH4_l/F_tot * P_bar
        r = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([0.0]))[0])
        return [-rho_bed_cat*r, -4.0*rho_bed_cat*r, +rho_bed_cat*r]
    z_grid = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4],
                    method='BDF', rtol=1e-5, atol=1e-9, t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    return float(np.clip(1.0 - F_CO2_f[-1]/F_in_CO2, 0.0, 1.0))

def _fmt_seconds(s):
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"

all_results  = {}
noSE_results = {}
perf_results = {}
n_total      = len(T_IN_LIST) * len(GHSV_LIST)
n_done       = 0
t_run_start  = time.perf_counter()

for T_C in T_IN_LIST:
    T_K = T_C + 273.15

    print(f"\n{'='*60}")
    print(f"  T_in = {T_C} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s (fixed)  |  isothermal")
    print(f"{'='*60}")

    q_init = None

    for i_g, GHSV_val in enumerate(GHSV_LIST):
        # recompute inlet fluxes for this GHSV
        Q_STP_g    = GHSV_val * M_ads / 3600.0
        u_g_STP    = Q_STP_g / A_b
        F_total_in = u_g_STP * P_Pa / (R_gas * T_STP)
        F_in_CO2   = y_CO2_in * F_total_in
        F_in_H2    = y_H2_in  * F_total_in
        F_in_CH4   = y_CH4_in * F_total_in

        if q_init is None:
            _phys  = _q_physics_init(T_K)
            q_init = _phys['q'][::-1]

        _noSE = _compute_noSE(T_K)
        noSE_results[(T_C, i_g)] = _noSE
        perf_results[(T_C, i_g)] = _compute_perfect_ads(T_K)

        t0  = time.perf_counter()
        res = solve_mpb(u_s_fixed, T_K, q_init=q_init)
        dt  = time.perf_counter() - t0
        n_done += 1
        elapsed = time.perf_counter() - t_run_start
        eta     = elapsed/n_done*(n_total - n_done)

        if res is not None:
            X_out  = float(res['X_CO2'][-1])*100
            q_out  = float(res['q'][0])
            tag    = "ok" if res['converged'] else "not-conv"
            print(f"  GHSV={GHSV_val:.1f}  X={X_out:.1f}%  q(0)={q_out:.3f}  "
                  f"noSE={_noSE['X_CO2_noSE']*100:.1f}%  "
                  f"[{tag}, {res['n_iter']} iter, err={res['conv_err']:.2e}]"
                  f"  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
            # H2O balance diagnostic
            y_CO2_out  = float(res['C_CO2'][-1]) * R_gas * T_K / P_Pa
            F_CO2_out  = F_in_CO2 * (1.0 - float(res['X_CO2'][-1]))
            F_tot_out  = F_CO2_out / max(y_CO2_out, 1e-30)
            F_H2O_out  = float(res['C_H2O'][-1]) * R_gas * T_K / P_Pa * F_tot_out
            F_H2O_prod = 2.0 * F_in_CO2 * float(res['X_CO2'][-1])
            F_H2O_ads  = u_s_fixed * rho_bed_ads * q_out
            bal_err    = (F_H2O_out + F_H2O_ads - F_H2O_prod) / max(F_H2O_prod, 1e-30) * 100
            print(f"    H2O balance [mmol/(m²·s)]:  produced={F_H2O_prod*1e3:.3f}  "
                  f"gas_out={F_H2O_out*1e3:.3f}  solid_out={F_H2O_ads*1e3:.3f}  "
                  f"err={bal_err:+.1f}%")
            q_init = np.interp(np.linspace(0, L_b, 150), res['z'], res['q'])
        else:
            print(f"  GHSV={GHSV_val:.1f}  FAILED  ({dt:.1f}s, ETA {_fmt_seconds(eta)})")
        all_results[(T_C, i_g)] = {'res': res, 'GHSV': GHSV_val, 'T_K': T_K,
                                    'F_in_CO2': F_in_CO2}

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
    p_H2O_peak  = float(np.max(res['C_H2O']))*R_gas*T_K/1e5
    q_star_peak = float(q_star(T_K, np.array([max(p_H2O_peak, 1e-8)]))[0])
    return dict(X_CO2=X_out, q_out=q_out,
                sorbent_util=q_out/max(q_star_peak, 1e-10))
# endregion


# region 6. PLOTS
# =============================================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def _savefig(filename):
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=150, bbox_inches='tight')

import matplotlib.lines as mlines
markers = ['o', 's', '^', 'D']
cmap    = plt.cm.viridis

T_C_PROF = T_IN_LIST[0]
i_plot   = np.arange(len(GHSV_LIST))
pal      = plt.cm.plasma(np.linspace(0.1, 0.85, len(i_plot)))
pal2     = cmap(np.linspace(0.1, 0.85, len(T_IN_LIST)))

# ── Plot 1: Axial profiles ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f'MPB flux form (isothermal)  |  T = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s', fontsize=11)
T_K_plot1 = T_C_PROF + 273.15
for k, i_g in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_g))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    lbl = f"GHSV = {e['GHSV']:.1f}"
    axes[0,0].plot(r['z'], r['C_CO2']*1e3, color=pal[k], lw=2, label=lbl)
    axes[0,1].plot(r['z'], r['q'],          color=pal[k], lw=2, label=lbl)
    p_H2O_z1 = r['C_H2O'] * R_gas * T_K_plot1 / 1e5
    axes[0,1].plot(r['z'], q_star(T_K_plot1, p_H2O_z1), color=pal[k], lw=1.5, ls='--', alpha=0.6)
    axes[1,0].plot(r['z'], r['X_CO2']*100,  color=pal[k], lw=2, label=lbl)
    axes[1,1].plot(r['z'], r['r']*1e3,      color=pal[k], lw=2, label=lbl)
_qs_proxy = mlines.Line2D([], [], color='grey', lw=1.5, ls='--', alpha=0.6, label='q* (isotherm)')
labels_units = [('C_CO2 [mmol/m3]', 'CO2 concentration'),
                ('q [mol/kg]',       'Solid H2O loading  (solid=q, dashed=q*)'),
                ('CO2 conversion [%]', 'CO2 conversion along bed'),
                ('r [mmol/(kg_cat.s)]', 'Reaction rate')]
for ax, (ylabel, title) in zip(axes.flat, labels_units):
    ax.set_xlabel('z [m]', fontsize=10);  ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10);     ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)
    ax.axvline(0,   color='tab:blue',   lw=1, ls=':', alpha=0.5)
    ax.axvline(L_b, color='tab:orange', lw=1, ls=':', alpha=0.5)
handles1, labels1 = axes[0,1].get_legend_handles_labels()
axes[0,1].legend(handles=handles1 + [_qs_proxy], labels=labels1 + ['q* (isotherm)'], fontsize=7)
plt.tight_layout()
_savefig(f'ghsv_plot1_axial_profiles_T{T_C_PROF}C.png');  plt.show()

# ── Plot 2: CO2 conversion vs GHSV ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    ghsv_ok, X_ok, X_noSE_ok = [], [], []
    for i_g in range(len(GHSV_LIST)):
        e = all_results.get((T_C, i_g))
        n = noSE_results.get((T_C, i_g))
        if e and e['res'] and n:
            m = get_metrics(e)
            if m:
                ghsv_ok.append(e['GHSV'])
                X_ok.append(m['X_CO2']*100)
                X_noSE_ok.append(n['X_CO2_noSE']*100)
    if ghsv_ok:
        ax.plot(ghsv_ok, X_ok, marker=markers[j], color=pal2[j],
                lw=2, ms=6, label=f'{T_C} C (MPB, u_s={u_s_fixed*1e3:.1f} mm/s)')
        ax.axhline(equilibrium_conversion(T_C+273.15), color=pal2[j],
                   lw=1, ls=':', alpha=0.5, label=f'{T_C} C thermo. eq.')
        ax.plot(ghsv_ok, X_noSE_ok, marker=markers[j], color=pal2[j],
                lw=1.5, ls='--', alpha=0.8, ms=4, label=f'{T_C} C u_s=0 (fixed bed)')
ax.set_xlabel('GHSV [m³/(kg_ads·h)]', fontsize=11)
ax.set_ylabel('CO2 conversion [%]', fontsize=11)
ax.set_title(f'MPB flux form (isothermal)  — CO2 conversion vs GHSV  (u_s = {u_s_fixed*1e3:.1f} mm/s)', fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3);  ax.set_ylim(0, 105)
plt.tight_layout()
_savefig('ghsv_plot2_conversion_vs_GHSV.png');  plt.show()

# ── Plot 3: Sorbent utilisation vs GHSV ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    ghsv_ok, util_ok = [], []
    for i_g in range(len(GHSV_LIST)):
        e = all_results.get((T_C, i_g))
        if e and e['res']:
            m = get_metrics(e)
            if m:
                ghsv_ok.append(e['GHSV']);  util_ok.append(m['sorbent_util']*100)
    if ghsv_ok:
        ax.plot(ghsv_ok, util_ok, marker=markers[j], color=pal2[j],
                lw=2, ms=6, label=f'{T_C} C')
ax.axhline(100, color='grey', lw=1.5, ls='--', label='q = q* (fully saturated)')
ax.set_xlabel('GHSV [m³/(kg_ads·h)]', fontsize=11)
ax.set_ylabel('Sorbent utilisation  q(z=0) / q*(p_H2O_max)  [%]', fontsize=11)
ax.set_title(f'MPB flux form (isothermal)  — Sorbent utilisation  (u_s = {u_s_fixed*1e3:.1f} mm/s)', fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('ghsv_plot3_sorbent_utilisation.png');  plt.show()

# ── Plot 4: H2O profiles ─────────────────────────────────────────────────────
T_K_prof = T_C_PROF + 273.15
fig, (ax_q, ax_h) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'H2O profiles  |  T = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  isothermal', fontsize=11)
for k, i_g in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_g))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    lbl = f"GHSV = {e['GHSV']:.1f}"
    ax_q.plot(r['z'], r['q'],                         color=pal[k], lw=2, label=lbl)
    p_H2O_z4 = r['C_H2O'] * R_gas * T_K_prof / 1e5
    ax_q.plot(r['z'], q_star(T_K_prof, p_H2O_z4),    color=pal[k], lw=1.5, ls='--', alpha=0.6)
    ax_h.plot(r['z'], r['C_H2O']*R_gas*T_K_prof/1e2, color=pal[k], lw=2, label=lbl)
_qs_proxy4 = mlines.Line2D([], [], color='grey', lw=1.5, ls='--', alpha=0.6, label='q* (isotherm)')
ax_q.set_xlabel('z [m]');  ax_q.set_ylabel('q [mol/kg]')
ax_q.set_title('Solid H2O loading  (solid=q, dashed=q*)')
handles4, labels4 = ax_q.get_legend_handles_labels()
ax_q.legend(handles=handles4 + [_qs_proxy4], labels=labels4 + ['q* (isotherm)'], fontsize=7)
ax_q.grid(True, alpha=0.3)
ax_h.set_xlabel('z [m]');  ax_h.set_ylabel('p_H2O [mbar]')
ax_h.set_title('Gas-phase H2O partial pressure');  ax_h.legend(fontsize=7);  ax_h.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'ghsv_plot4_H2O_profiles_T{T_C_PROF}C.png');  plt.show()

# ── Plot 5: X vs GHSV — MPB vs fixed-bed vs perfect-adsorption ───────────────
fig, ax = plt.subplots(figsize=(9, 5))
_c_perf  = 'tab:green'
_c_mpb   = 'tab:blue'
_c_noSE  = 'tab:red'
_c_eq    = 'grey'
for j, T_C in enumerate(T_IN_LIST):
    ghsv_ok, X_mpb, X_noSE_ok, X_perf_ok = [], [], [], []
    for i_g in range(len(GHSV_LIST)):
        e = all_results.get((T_C, i_g))
        n = noSE_results.get((T_C, i_g))
        p = perf_results.get((T_C, i_g))
        if e and e['res'] and n and p is not None:
            m = get_metrics(e)
            if m:
                ghsv_ok.append(e['GHSV'])
                X_mpb.append(m['X_CO2']*100)
                X_noSE_ok.append(n['X_CO2_noSE']*100)
                X_perf_ok.append(p*100)
    if ghsv_ok:
        ax.plot(ghsv_ok, X_perf_ok, color=_c_perf, lw=2, ls='--', marker='D', ms=5,
                label=f'{T_C} C  kinetic ceiling  (p_H2O≡0)')
        ax.plot(ghsv_ok, X_mpb,     color=_c_mpb,  lw=2, ls='-',  marker='o', ms=6,
                label=f'{T_C} C  MPB  u_s={u_s_fixed*1e3:.1f} mm/s')
        ax.plot(ghsv_ok, X_noSE_ok, color=_c_noSE, lw=2, ls='-',  marker='s', ms=5,
                label=f'{T_C} C  fixed bed (u_s=0)')
        ax.axhline(equilibrium_conversion(T_C+273.15), color=_c_eq,
                   lw=1.5, ls=':', label=f'{T_C} C  thermo. eq. (no SE)')
ax.set_xlabel('GHSV [m³/(kg_ads·h)]', fontsize=11)
ax.set_ylabel('CO2 conversion [%]', fontsize=11)
ax.set_title('Limiting factor: MPB vs fixed-bed vs kinetic ceiling', fontsize=10)
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3);  ax.set_ylim(0, 105)
plt.tight_layout()
_savefig('ghsv_plot5_limiting_factor.png');  plt.show()

# ── Plot 6: f_eq(z) and H2O inhibition profiles ──────────────────────────────
fig, (ax_feq, ax_inh) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Rate decomposition  |  T = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s', fontsize=11)
T_K_d = T_C_PROF + 273.15
K_eq_d = K_eq_sabatier(T_K_d)
vH_d   = lambda dH: np.exp(-dH/R_gas*(1.0/T_K_d - 1.0/T_ref_K))
K_OH_d = A_OH*vH_d(dH_OH);  K_H2_d = A_H2*vH_d(dH_H2);  K_mix_d = A_mix*vH_d(dH_mix)

for k, i_g in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_g))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    lbl = f"GHSV = {e['GHSV']:.1f}"
    p_CO2_z = r['C_CO2'] * R_gas * T_K_d / 1e5
    p_H2_z  = r['C_H2']  * R_gas * T_K_d / 1e5
    p_CH4_z = r['C_CH4'] * R_gas * T_K_d / 1e5
    p_H2O_z = r['C_H2O'] * R_gas * T_K_d / 1e5
    p_CO2_s = np.maximum(p_CO2_z, P_FLOOR);  p_H2_s = np.maximum(p_H2_z, P_FLOOR)
    beta_z  = p_CH4_z * p_H2O_z**2 / (K_eq_d * p_CO2_s * p_H2_s**4)
    f_eq_z  = np.maximum(1.0 - beta_z, 0.0)
    DEN_z   = (1.0 + K_OH_d*p_H2O_z/p_H2_s**0.5
               + K_H2_d*p_H2_s**0.5 + K_mix_d*p_CO2_s**0.5)
    inh_z   = K_OH_d * p_H2O_z / p_H2_s**0.5 / DEN_z
    ax_feq.plot(r['z'], f_eq_z, color=pal[k], lw=2, label=lbl)
    ax_inh.plot(r['z'], inh_z*100, color=pal[k], lw=2, label=lbl)

ax_feq.set_xlabel('z [m]');  ax_feq.set_ylabel('f_eq = 1 − β  [−]')
ax_feq.set_title('Thermodynamic driving force\n(1=irreversible, 0=at equilibrium)')
ax_feq.set_ylim(0, 1.05);  ax_feq.legend(fontsize=7);  ax_feq.grid(True, alpha=0.3)

ax_inh.set_xlabel('z [m]');  ax_inh.set_ylabel('K_OH·p_H2O/p_H2^0.5 / DEN  [%]')
ax_inh.set_title('H2O inhibition fraction of DEN\n(0%=no inhibition)')
ax_inh.legend(fontsize=7);  ax_inh.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'ghsv_plot6_rate_decomposition_T{T_C_PROF}C.png');  plt.show()

# ── Plot 7: CH4 and H2 mole-fraction profiles ────────────────────────────────
T_K_p7 = T_C_PROF + 273.15
fig, (ax_ch4, ax_h2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'CH4 and H2 profiles  |  T = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  isothermal', fontsize=11)
for k, i_g in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_g))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    lbl = f"GHSV = {e['GHSV']:.1f}"
    C_tot = r['C_CO2'] + r['C_H2'] + r['C_CH4'] + r['C_H2O']
    y_CH4 = r['C_CH4'] / np.maximum(C_tot, 1e-30) * 100
    y_H2  = r['C_H2']  / np.maximum(C_tot, 1e-30) * 100
    ax_ch4.plot(r['z'], y_CH4, color=pal[k], lw=2, label=lbl)
    ax_h2.plot( r['z'], y_H2,  color=pal[k], lw=2, label=lbl)
ax_ch4.set_xlabel('z [m]');  ax_ch4.set_ylabel('y_CH4 [mol%]')
ax_ch4.set_title('CH4 mole fraction along bed');  ax_ch4.legend(fontsize=7);  ax_ch4.grid(True, alpha=0.3)
ax_h2.set_xlabel( 'z [m]');  ax_h2.set_ylabel( 'y_H2 [mol%]')
ax_h2.set_title( 'H2 mole fraction along bed');   ax_h2.legend(fontsize=7);  ax_h2.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'ghsv_plot7_CH4_H2_profiles_T{T_C_PROF}C.png');  plt.show()

# ── Plot 8: Heat generation — axial profiles + bed-averaged totals ───────────
# ΔH_rxn: Sabatier CO2 + 4H2 → CH4 + 2H2O at ~280°C ≈ −165 kJ/mol
# ΔH_ads: isosteric heat H2O on 13X ≈ −55 kJ/mol (literature estimate)
dH_rxn = 165e3   # J/mol CO2  (positive = released)
dH_ads =  55e3   # J/mol H2O  (positive = released; negative when solid desorbs)

T_K_p8    = T_C_PROF + 273.15
_sel_ig   = [0, len(GHSV_LIST)//2, len(GHSV_LIST)-1]   # low / mid / high GHSV
_sel_cols = ['tab:blue', 'tab:orange', 'tab:red']

fig, (ax_prof, ax_int) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f'Heat generation (estimate)  |  T = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s  |  '
    f'ΔH_rxn = −{dH_rxn/1e3:.0f} kJ/mol,  ΔH_ads = −{dH_ads/1e3:.0f} kJ/mol',
    fontsize=10)

def _heat_terms(res):
    p_H2O = res['C_H2O'] * R_gas * T_K_p8 / 1e5
    ads   = K_LDF(T_K_p8, p_H2O) * (q_star(T_K_p8, p_H2O) - res['q'])
    Q_r   = rho_bed_cat * res['r'] * dH_rxn / 1e3             # kW/m³  (r already in mol/kg/s)
    Q_a   = rho_bed_ads * ads * dH_ads / 1e3                  # kW/m³
    return Q_r, Q_a, Q_r + Q_a

# left: axial profiles for 3 selected GHSV cases
for ci, i_g in zip(_sel_cols, _sel_ig):
    e = all_results.get((T_C_PROF, i_g))
    if e is None or e['res'] is None:
        continue
    Q_r, Q_a, Q_t = _heat_terms(e['res'])
    lbl = f"GHSV={e['GHSV']:.1f}"
    ax_prof.plot(e['res']['z'], Q_r, color=ci, lw=2, ls='-',  label=f'{lbl} rxn')
    ax_prof.plot(e['res']['z'], Q_a, color=ci, lw=2, ls='--', label=f'{lbl} ads')
    ax_prof.plot(e['res']['z'], Q_t, color=ci, lw=2, ls=':',  label=f'{lbl} total')
ax_prof.axhline(0, color='k', lw=0.8)
ax_prof.set_xlabel('z [m]');  ax_prof.set_ylabel('Heat generation [kW/m³_bed]')
ax_prof.set_title('Axial profiles  (— rxn  -- ads  ··· total)', fontsize=9)
ax_prof.legend(fontsize=7);  ax_prof.grid(True, alpha=0.3)

# right: bed-averaged heat (∫Q dz / L_b) vs GHSV
ghsv_v, Qr_v, Qa_v, Qt_v = [], [], [], []
for i_g, gv in enumerate(GHSV_LIST):
    e = all_results.get((T_C_PROF, i_g))
    if e is None or e['res'] is None:
        continue
    Q_r, Q_a, Q_t = _heat_terms(e['res'])
    z = e['res']['z']
    ghsv_v.append(gv)
    Qr_v.append(np.trapz(Q_r, z) / L_b)
    Qa_v.append(np.trapz(Q_a, z) / L_b)
    Qt_v.append(np.trapz(Q_t, z) / L_b)
ax_int.plot(ghsv_v, Qr_v, 'o-',  color='tab:blue',   lw=2, ms=6, label='reaction')
ax_int.plot(ghsv_v, Qa_v, 's--', color='tab:orange',  lw=2, ms=6, label='adsorption')
ax_int.plot(ghsv_v, Qt_v, 'D:',  color='tab:red',     lw=2, ms=6, label='total')
ax_int.axhline(0, color='k', lw=0.8)
ax_int.set_xlabel('GHSV [m³/(kg_ads·h)]');  ax_int.set_ylabel('Bed-avg heat generation [kW/m³_bed]')
ax_int.set_title('Bed-averaged  (∫Q dz / L_b)', fontsize=9)
ax_int.legend(fontsize=8);  ax_int.grid(True, alpha=0.3)

plt.tight_layout()
_savefig(f'ghsv_plot8_heat_generation_T{T_C_PROF}C.png');  plt.show()

# ── Plot 9: Heat released vs z at GHSV = 0.5  (Bareschino Fig.5 style) ───────
_ig05 = int(np.argmin(np.abs(GHSV_LIST - 0.5)))
_e05  = all_results.get((T_C_PROF, _ig05))
if _e05 is not None and _e05['res'] is not None:
    Q_r05, Q_a05, Q_t05 = _heat_terms(_e05['res'])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(_e05['res']['z'], Q_r05, color='tab:red',    lw=2.5, label='Reaction')
    ax.plot(_e05['res']['z'], Q_a05, color='tab:blue',   lw=2.5, label='Adsorption')
    ax.plot(_e05['res']['z'], Q_t05, color='tab:green',  lw=2.5, ls='--', label='Total')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('z [m]', fontsize=12);  ax.set_ylabel('Heat generation [kW/m³_bed]', fontsize=12)
    ax.set_title(
        f'Heat released vs axial position  |  GHSV = {GHSV_LIST[_ig05]:.1f} m³/(kg·h)  |  '
        f'T = {T_C_PROF} C  |  u_s = {u_s_fixed*1e3:.1f} mm/s  (steady state)',
        fontsize=10)
    ax.legend(fontsize=11);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(f'ghsv_plot9_heat_bareschino_style_GHSV05_T{T_C_PROF}C.png');  plt.show()
# endregion
