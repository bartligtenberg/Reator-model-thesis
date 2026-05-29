"""
Moving Packed Bed (MPB) Reactor Model — Steady-State, Isothermal, CO2 Methanation
==================================================================================

Counter-current flow:
    gas  : z = 0 (inlet, bottom)  ->  z = L (outlet, top)    u_g > 0
    solid: z = L (inlet, top)     ->  z = 0 (outlet, bottom)  u_s > 0 (magnitude)

Isothermal (T = T_in throughout).  Wall cooling is strong (U_a = 8000 W/(m3.K)),
so the isothermal assumption is a well-justified first approximation.

Solved by decoupled Gauss-Seidel iteration:
  1. Gas IVP   (z = 0 -> L, solve_ivp BDF) with fixed q(z)
  2. Solid IVP (zeta = 0 -> L, zeta = L-z, solve_ivp BDF) with fixed C_H2O(z)
  3. Repeat until max|delta_q| < tol

State variables at steady state:
    C_CO2(z), C_H2(z), C_CH4(z), C_H2O(z)  [mol/m3]
    q(z)                                      [mol/kg]
"""

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
# Rate: r [mol/(kg_cat·s)] = k*(p_CO2*p_H2)^0.5 * f_eq / DEN² * 1000
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
Cp_CO2 = 45.4;  Cp_H2 = 29.3;  Cp_CH4 = 46.9;  Cp_H2O = 34.2;  Cp_N2 = 29.5  # [J/(mol·K)] — NIST at ~550 K
U_a    = 8000.0        # [W/(m³_bed·K)]  — volumetric wall heat transfer coeff., own estimate, h = 100 W/(M2*K) (justifies temporary isothermal assumption)

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

print(f"MPB: d={d_b*100:.0f} cm, L={L_b:.1f} m, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3, u_g_STP={u_g_STP*1e3:.1f} mm/s")
# endregion


# region 2. INHERITED FUNCTIONS
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
    D_M  = 2.5e-5*(T_K/300.0)**1.75
    p    = np.asarray(p_arr, dtype=float)
    dp   = 1.0/1e5
    dqsp = (q_star_vec(T_K, p+dp, W0, E, n) - q_star_vec(T_K, np.maximum(p-dp,1e-15), W0, E, n))/2.0
    dqsp = np.maximum(dqsp, 1e-30)
    return (15.0*D_M*MW_H2O*eps_p
            / (0.5*d_p**2*tau_p*rho_water(T_K)*R_gas*T_K*dqsp))

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
    DEN   = (1.0 + K_OH*np.maximum(p_H2O,0)/p_H2_s**0.5
             + K_H2*p_H2_s**0.5 + K_mix*p_CO2_s)
    return k*(p_CO2_s*p_H2_s)**0.5*f_eq/DEN**2*1000.0

def q_star(T_K, p_H2O):
    return q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

def K_LDF(T_K, p_H2O):
    return K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

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
# endregion


# region 3. DECOUPLED SOLVER
# =============================================================================
def solve_mpb(u_s, u_g, T_K, C_in_CO2, C_in_H2, C_in_CH4,
              max_iter=200, tol=1e-4, N=100, q_init=None):
    """
    Counter-current MPB at steady state, isothermal (T = T_K throughout).

    Decoupled Gauss-Seidel iteration:
      - Gas IVP  : z = 0 -> L (BDF), uses q(z) from previous iteration
      - Solid IVP: zeta = L-z, zeta=0->L (BDF), uses C_H2O(z) from current gas step

    q_init: optional initial q(z) profile [mol/kg] on N points — warm-start from a
            nearby case to reduce iterations needed.
    """
    z_grid = np.linspace(0.0, L_b, N)

    if q_init is not None:
        q_prof = np.interp(z_grid, np.linspace(0, L_b, len(q_init)), q_init)
    else:
        q_prof = np.zeros(N)

    converged = False
    for it in range(max_iter):
        q_fn = interp1d(z_grid, q_prof, kind='linear',
                        bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))

        # ── Step 1: gas IVP from z=0 to z=L ─────────────────────────────────
        def gas_rhs(z, y):
            C_CO2_l = max(y[0], 0.0)
            C_H2_l  = max(y[1], 0.0)
            C_CH4_l = max(y[2], 0.0)
            C_H2O_l = max(y[3], 0.0)
            q_l     = max(float(q_fn(z)), 0.0)

            p_CO2 = C_CO2_l*R_gas*T_K/1e5
            p_H2  = C_H2_l *R_gas*T_K/1e5
            p_CH4 = C_CH4_l*R_gas*T_K/1e5
            p_H2O = C_H2O_l*R_gas*T_K/1e5

            r   = float(reaction_rate_SI(T_K,
                        np.array([p_CO2]), np.array([p_H2]),
                        np.array([p_CH4]), np.array([p_H2O]))[0])
            qs  = float(q_star(T_K, np.array([p_H2O]))[0])
            Kl  = float(K_LDF(T_K,  np.array([p_H2O]))[0])
            ads = Kl*(qs - q_l)

            return [
                -rho_bed_cat*r / (eps_b*u_g),
                -4.0*rho_bed_cat*r / (eps_b*u_g),
                +rho_bed_cat*r / (eps_b*u_g),
                (2.0*rho_bed_cat*r - rho_bed_ads*ads) / (eps_b*u_g),
            ]

        gs = solve_ivp(gas_rhs, [0.0, L_b], [C_in_CO2, C_in_H2, C_in_CH4, 0.0],
                       method='BDF', rtol=1e-4, atol=1e-8,
                       t_eval=z_grid, dense_output=False)
        if not gs.success:
            return None

        C_H2O_prof = np.maximum(gs.y[3], 0.0)
        C_H2O_fn   = interp1d(z_grid, C_H2O_prof, kind='linear',
                               bounds_error=False,
                               fill_value=(C_H2O_prof[0], C_H2O_prof[-1]))

        # ── Step 2: solid IVP in zeta = L-z from zeta=0 (z=L) to zeta=L (z=0)
        # dq/d_zeta = K_LDF*(q* - q)/u_s   [stable: q starts at 0 and grows]
        def solid_rhs(zeta, q_arr):
            z_pos = L_b - float(zeta)
            p_H2O = max(float(C_H2O_fn(z_pos)), 0.0)*R_gas*T_K/1e5
            qs = float(q_star(T_K, np.array([p_H2O]))[0])
            Kl = float(K_LDF(T_K,  np.array([p_H2O]))[0])
            q_val = max(float(q_arr[0]), 0.0)
            return [Kl*(qs - q_val)/u_s]

        zeta_eval = np.linspace(0.0, L_b, N)
        ss = solve_ivp(solid_rhs, [0.0, L_b], [0.0],
                       method='BDF', rtol=1e-4, atol=1e-8,
                       t_eval=zeta_eval, dense_output=False)
        if not ss.success:
            return None

        # Map zeta -> z:  z = L - zeta  (zeta=0 -> z=L, zeta=L -> z=0)
        z_from_zeta = L_b - ss.t          # z values in DECREASING order
        q_from_zeta = np.maximum(ss.y[0], 0.0)

        # Interpolate onto z_grid (which is increasing)
        sort_idx = np.argsort(z_from_zeta)
        q_new = np.interp(z_grid, z_from_zeta[sort_idx], q_from_zeta[sort_idx])

        # Damped update (avoids oscillatory non-convergence at slow u_s)
        q_prof_new = 0.5 * q_prof + 0.5 * q_new

        # Check convergence
        scale = max(np.max(q_prof_new), 1e-8)
        err   = np.max(np.abs(q_prof_new - q_prof)) / scale
        q_prof = q_prof_new
        if err < tol:
            converged = True
            break

    if not converged:
        pass   # return best estimate anyway

    # ── Recompute final gas profiles with converged q ────────────────────────
    q_fn_final = interp1d(z_grid, q_prof, kind='linear',
                          bounds_error=False, fill_value=(q_prof[0], q_prof[-1]))

    def gas_rhs_final(z, y):
        C_CO2_l = max(y[0], 0.0);  C_H2_l  = max(y[1], 0.0)
        C_CH4_l = max(y[2], 0.0);  C_H2O_l = max(y[3], 0.0)
        q_l     = max(float(q_fn_final(z)), 0.0)
        p_CO2 = C_CO2_l*R_gas*T_K/1e5;  p_H2  = C_H2_l *R_gas*T_K/1e5
        p_CH4 = C_CH4_l*R_gas*T_K/1e5;  p_H2O = C_H2O_l*R_gas*T_K/1e5
        r   = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                      np.array([p_CH4]), np.array([p_H2O]))[0])
        qs  = float(q_star(T_K, np.array([p_H2O]))[0])
        Kl  = float(K_LDF(T_K,  np.array([p_H2O]))[0])
        ads = Kl*(qs - q_l)
        return [
            -rho_bed_cat*r/(eps_b*u_g),
            -4.0*rho_bed_cat*r/(eps_b*u_g),
            +rho_bed_cat*r/(eps_b*u_g),
            (2.0*rho_bed_cat*r - rho_bed_ads*ads)/(eps_b*u_g),
        ]

    z_fine = np.linspace(0.0, L_b, 300)
    gf = solve_ivp(gas_rhs_final, [0.0, L_b], [C_in_CO2, C_in_H2, C_in_CH4, 0.0],
                   method='BDF', rtol=1e-6, atol=1e-10,
                   t_eval=z_fine, dense_output=False)

    C_CO2f = np.maximum(gf.y[0], 0.0);  C_H2f  = np.maximum(gf.y[1], 0.0)
    C_CH4f = np.maximum(gf.y[2], 0.0);  C_H2Of = np.maximum(gf.y[3], 0.0)
    q_fine = np.interp(z_fine, z_grid, q_prof)
    T_fine = T_K*np.ones(len(z_fine))

    p_CO2f = C_CO2f*R_gas*T_fine/1e5;  p_H2f  = C_H2f *R_gas*T_fine/1e5
    p_CH4f = C_CH4f*R_gas*T_fine/1e5;  p_H2Of = C_H2Of*R_gas*T_fine/1e5
    r_fine = reaction_rate_SI(T_fine, p_CO2f, p_H2f, p_CH4f, p_H2Of)
    X_CO2  = np.clip(1.0 - C_CO2f/C_in_CO2, 0.0, 1.0)

    return dict(z=z_fine, C_CO2=C_CO2f, C_H2=C_H2f, C_CH4=C_CH4f,
                C_H2O=C_H2Of, q=q_fine, T=T_fine, r=r_fine, X_CO2=X_CO2,
                converged=converged, n_iter=it+1)
# endregion


# region 4. SOLVE LOOP
# =============================================================================
def _q_physics_init(T_K, u_g, C_in_CO2, C_in_H2, C_in_CH4, N=150):
    """
    Physics-based initial guess for q(z).

    Solves the gas ODE with no adsorption (pure reaction, like a saturated fixed bed),
    then returns q*(T, p_H2O(z)). This is the maximum possible solid loading given
    the H2O profile from reaction alone — a much better start than q=0 for slow u_s.
    """
    def rhs_noads(_z, y):
        C_CO2_l = max(y[0], 0.0); C_H2_l  = max(y[1], 0.0)
        C_CH4_l = max(y[2], 0.0); C_H2O_l = max(y[3], 0.0)
        p_CO2 = C_CO2_l*R_gas*T_K/1e5; p_H2  = C_H2_l *R_gas*T_K/1e5
        p_CH4 = C_CH4_l*R_gas*T_K/1e5; p_H2O = C_H2O_l*R_gas*T_K/1e5
        r = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                   np.array([p_CH4]), np.array([p_H2O]))[0])
        return [
            -rho_bed_cat*r/(eps_b*u_g),
            -4.0*rho_bed_cat*r/(eps_b*u_g),
            +rho_bed_cat*r/(eps_b*u_g),
             2.0*rho_bed_cat*r/(eps_b*u_g),   # no adsorption sink
        ]
    z_grid = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs_noads, [0, L_b], [C_in_CO2, C_in_H2, C_in_CH4, 0.0],
                    method='BDF', rtol=1e-4, atol=1e-8, t_eval=z_grid)
    p_H2O_prof  = np.maximum(sol.y[3], 0.0) * R_gas * T_K / 1e5
    X_CO2_noSE  = float(np.clip(1.0 - sol.y[0, -1] / C_in_CO2, 0.0, 1.0))
    return dict(q=q_star(T_K, p_H2O_prof), X_CO2_noSE=X_CO2_noSE)

def _fmt_seconds(s):
    """Format seconds as e.g. '1m 23s' or '45s'."""
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"

all_results  = {}   # key: (T_C, i_us)
noSE_results = {}   # key: T_C  →  non-SE fixed-bed CO2 conversion [-]

n_total     = len(T_IN_LIST) * len(U_S_LIST)
n_done      = 0
t_run_start = time.perf_counter()

for T_C in T_IN_LIST:
    T_K      = T_C + 273.15
    u_g      = u_g_STP*(T_K/T_STP)
    C_in_CO2 = y_CO2_in*P_Pa/(R_gas*T_K)
    C_in_H2  = y_H2_in *P_Pa/(R_gas*T_K)
    C_in_CH4 = y_CH4_in*P_Pa/(R_gas*T_K)
    Cp_mix_est = y_CH4_in*Cp_CH4 + y_CO2_in*Cp_CO2 + y_H2_in*Cp_H2
    rho_g_mol  = P_Pa/(R_gas*T_K)
    u_s_star   = u_g*rho_g_mol*Cp_mix_est/(rho_bed_tot*Cp_cat)

    print(f"\n{'='*60}")
    print(f"  T_in = {T_C} C  |  u_g = {u_g*1e3:.1f} mm/s  "
          f"|  u_s* = {u_s_star*1e3:.3f} mm/s")
    print(f"{'='*60}")

    _phys      = _q_physics_init(T_K, u_g, C_in_CO2, C_in_H2, C_in_CH4)
    q_init_raw = _phys['q']
    q_init     = q_init_raw[::-1]          # flip (z=L→0 shape); fully saturated = correct steady-state guess
    noSE_results[T_C] = _phys['X_CO2_noSE']
    print(f"  non-SE fixed-bed conversion: {noSE_results[T_C]*100:.1f}%")
    z_init     = np.linspace(0, L_b, len(q_init))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(z_init, q_init_raw, lw=2, color='grey',      ls='--', label='raw q* (wrong direction)')
    ax.plot(z_init, q_init,     lw=2, color='steelblue',           label='flipped, fully saturated (used as q_init)')
    ax.axvline(L_b, color='tab:orange', lw=1, ls=':', alpha=0.7, label=f'z=L (solid inlet, q=0 BC)')
    ax.set_xlabel('z [m]');  ax.set_ylabel('q [mol/kg]')
    ax.set_title(f'Physics-based initial guess  |  T_in = {T_C} C')
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
    plt.tight_layout();  plt.show()

    for i_us, u_s in enumerate(U_S_LIST):
        t0  = time.perf_counter()
        res = solve_mpb(u_s, u_g, T_K, C_in_CO2, C_in_H2, C_in_CH4, q_init=q_init)
        dt  = time.perf_counter() - t0
        n_done += 1
        elapsed = time.perf_counter() - t_run_start
        eta     = elapsed / n_done * (n_total - n_done)

        if res is not None:
            X_out = float(res['X_CO2'][-1])*100
            q_out = float(res['q'][0])
            tag   = "ok" if res['converged'] else "not-conv"
            print(f"  u_s={u_s*1e3:.4f} mm/s  X={X_out:.1f}%  "
                  f"q(0)={q_out:.3f} mol/kg  [{tag}, {res['n_iter']} iter]"
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
    X_out   = float(res['X_CO2'][-1])
    q_out   = float(res['q'][0])            # solid loading at z=0 (solid outlet)
    r_max   = float(np.max(res['r']))*1e3   # mmol/(kg_cat.s)
    p_H2O_max = float(np.max(res['C_H2O']))*R_gas*T_K/1e5*1000  # mbar
    # sorbent utilisation: q at solid outlet vs q* at the peak H2O in the bed
    # (NOT at z=L where fresh solid instantly adsorbs all H2O → p_H2O≈0 → q*≈0 → util→∞)
    p_H2O_peak = float(np.max(res['C_H2O']))*R_gas*T_K/1e5
    q_star_out = float(q_star(T_K, np.array([max(p_H2O_peak, 1e-8)]))[0])
    return dict(X_CO2=X_out, q_out=q_out, r_max=r_max,
                p_H2O_max=p_H2O_max, q_star_out=q_star_out,
                sorbent_util=q_out/max(q_star_out,1e-10))
# endregion


# region 6. PLOTS
# =============================================================================
markers = ['o', 's', '^', 'D']
cmap    = plt.cm.viridis

# ── Plot 1: Axial profiles at T_in = 300 C for 5 u_s values ─────────────────
T_C_PROF = T_IN_LIST[0]
i_plot   = np.arange(len(U_S_LIST))
pal      = plt.cm.plasma(np.linspace(0.1, 0.85, len(i_plot)))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f'MPB axial profiles  |  T_in = {T_C_PROF} C  |  isothermal, counter-current',
             fontsize=11)

plotted = 0
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
    plotted += 1

labels_units = [('C_CO2 [mmol/m3]',    'CO2 concentration'),
                ('q [mol/kg]',          'Solid H2O loading'),
                ('CO2 conversion [%]',  'CO2 conversion along bed'),
                ('r [mmol/(kg_cat.s)]', 'Reaction rate')]
for ax, (ylabel, title) in zip(axes.flat, labels_units):
    ax.set_xlabel('z [m]', fontsize=10)
    ax.set_ylabel(ylabel,  fontsize=10)
    ax.set_title(title,    fontsize=10)
    ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)
    ax.axvline(0,   color='tab:blue',   lw=1, ls=':', alpha=0.5)
    ax.axvline(L_b, color='tab:orange', lw=1, ls=':', alpha=0.5)

plt.tight_layout()
plt.show()

# ── Plot 2: CO2 conversion at z=L vs u_s for each T_in ─────────────────────
pal2 = cmap(np.linspace(0.1, 0.85, len(T_IN_LIST)))
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    us_ok, X_ok = [], []
    for i_us in range(len(U_S_LIST)):
        e = all_results.get((T_C, i_us))
        if e and e['res']:
            m = get_metrics(e)
            if m:
                us_ok.append(e['u_s']*1e3)
                X_ok.append(m['X_CO2']*100)
    if us_ok:
        ax.semilogx(us_ok, X_ok, marker=markers[j], color=pal2[j],
                    lw=2, ms=6, label=f'{T_C} C (MPB)')
        ax.axhline(equilibrium_conversion(T_C+273.15), color=pal2[j],
                   lw=1, ls=':', alpha=0.5, label=f'{T_C} C thermo. eq.')
        if T_C in noSE_results:
            ax.axhline(noSE_results[T_C]*100, color=pal2[j],
                       lw=1.5, ls='--', alpha=0.8, label=f'{T_C} C non-SE (fixed bed)')

ax.set_xlabel('u_s [mm/s]', fontsize=11)
ax.set_ylabel('CO2 conversion at z=L [%]', fontsize=11)
ax.set_title('MPB  --  CO2 conversion vs solid velocity\n'
             '(dotted = thermo. eq., dashed = non-SE fixed bed)', fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3);  ax.set_ylim(0, 105)
plt.tight_layout();  plt.show()

# ── Plot 3: Sorbent utilisation vs u_s ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    us_ok, util_ok = [], []
    for i_us in range(len(U_S_LIST)):
        e = all_results.get((T_C, i_us))
        if e and e['res']:
            m = get_metrics(e)
            if m:
                us_ok.append(e['u_s']*1e3)
                util_ok.append(m['sorbent_util'] * 100)
    if us_ok:
        ax.semilogx(us_ok, util_ok, marker=markers[j], color=pal2[j],
                    lw=2, ms=6, label=f'{T_C} C')

ax.axhline(100, color='grey', lw=1.5, ls='--', label='q = q* (fully saturated)')
ax.set_xlabel('u_s [mm/s]', fontsize=11)
ax.set_ylabel('Sorbent utilisation  q(z=0) / q*(p_H2O_max)  [%]', fontsize=11)
ax.set_title('MPB  --  Sorbent utilisation vs solid velocity', fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
plt.tight_layout();  plt.show()

# ── Plot 4: q profile and C_H2O profile for several u_s at T=300 C ─────────
fig, (ax_q, ax_h) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'H2O profiles along bed  |  T_in = {T_C_PROF} C', fontsize=11)
for k, i_us in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_us))
    if e is None or e['res'] is None:
        continue
    r   = e['res']
    lbl = f"u_s = {e['u_s']*1e3:.3f} mm/s"
    ax_q.plot(r['z'], r['q'],                              color=pal[k], lw=2, label=lbl)
    ax_h.plot(r['z'], r['C_H2O']*R_gas*T_C_PROF/1e2,    color=pal[k], lw=2, label=lbl)

ax_q.set_xlabel('z [m]');  ax_q.set_ylabel('q [mol/kg]')
ax_q.set_title('Solid H2O loading q(z)')
ax_q.legend(fontsize=7);   ax_q.grid(True, alpha=0.3)
ax_h.set_xlabel('z [m]');  ax_h.set_ylabel('p_H2O [mbar]')
ax_h.set_title('Gas-phase H2O partial pressure')
ax_h.legend(fontsize=7);   ax_h.grid(True, alpha=0.3)
plt.tight_layout();  plt.show()
# endregion
