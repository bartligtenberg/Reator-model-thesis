"""
Validation Test Suite — Full-Scale SEM Column Model
====================================================
Five test groups cover every axis a thesis examiner is likely to probe:

  A. Internal consistency   — mass/carbon balance, stoichiometry, Le Chatelier
  B. Numerical convergence  — grid refinement, solver-tolerance sensitivity
  C. Thermodynamic limits   — equilibrium at low GHSV, isothermal wall, adiabatic bound
  D. Sub-model unit tests   — K_eq direction, DA isotherm limits, kinetic rate
  E. Literature comparison  — SE conversion enhancement, H2O breakthrough timing

Run:
    python validation_tests.py

Each test prints PASS / FAIL with the key numbers. A summary is printed at the end.
Plots are generated for B1, C1, D1, D2, and E2.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# =============================================================================
# SHARED PARAMETERS  (identical to full scale SEM LHHW nonisothermal.py)
# =============================================================================

d_b   = 0.050
L_b   = 2.000
A_b   = np.pi / 4 * d_b**2
V_bed = A_b * L_b
eps_b = 0.4

M_cat = 0.064
M_ads = 1.22
rho_bed_cat = M_cat / V_bed
rho_bed_ads = M_ads / V_bed
rho_bed_tot = (M_cat + M_ads) / V_bed

d_p   = 0.75e-3
eps_p = 0.615
tau_p = 3.0

W0_DA = 341.00e-6
E_DA  = 1192.25e3
n_DA  = 1.55

P_bar = 1.0
P_Pa  = P_bar * 1e5

y_CO2_in = 0.04
y_H2_in  = 0.16
y_CH4_in = 0.80

GHSV_NOM = 0.4
T_STP    = 273.15
R_gas    = 8.314
MW_H2O   = 0.018015

T_ref_K = 555.0
k_ref   = 3.46e-4
Ea_k    = 77.5e3
A_OH    = 0.50;  dH_OH  =  22.4e3
A_H2    = 0.44;  dH_H2  =  -6.2e3
A_mix   = 0.88;  dH_mix = -10.0e3
P_FLOOR = 1e-4

dH_r   = -165.0e3
dH_ads = -45.0e3
Cp_cat = 1100.0
Cp_CO2 = 45.4;  Cp_H2 = 29.3;  Cp_CH4 = 46.9;  Cp_H2O = 34.2;  Cp_N2 = 29.5

U_a_NOM = 8000.0
N_REF   = 100

T_LIST = [260, 280, 300, 320]   # temperatures used in E-group tests


# =============================================================================
# FUNCTIONS  (identical to main model)
# =============================================================================

def P_sat_bar(T_K):
    log10_p = (29.8605 - 3.1522e3 / T_K
               - 7.3037 * np.log10(T_K)
               + 2.4247e-9 * T_K
               + 1.8090e-6 * T_K**2)
    return 10.0**log10_p * 133.322e-5


def rho_water(T_K):
    return 996.0 / (1.0 + 2.0e-3 * (T_K - 298.15))


def q_star_vec(T_K, p_arr, W0=W0_DA, E=E_DA, n=n_DA):
    p      = np.asarray(p_arr, dtype=float)
    Psat   = P_sat_bar(T_K)
    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_raw  = (R_gas / MW_H2O) * T_K * np.log(Psat / p_safe)
    A      = np.where((p <= 0.0) | (p >= Psat), 0.0, A_raw)
    W      = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    return np.where(p <= 0.0, 0.0, rho_water(T_K) / MW_H2O * W)


def K_LDF_vec(T_K, p_arr):
    D_M       = 2.5e-5 * (T_K / 300.0) ** 1.75
    p         = np.asarray(p_arr, dtype=float)
    dp        = 1.0 / 1e5
    dqstar_dp = (q_star_vec(T_K, np.maximum(p + dp, 1e-15))
                 - q_star_vec(T_K, np.maximum(p - dp, 1e-15))) / 2.0
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)
    return (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_water(T_K) * R_gas * T_K * dqstar_dp))


def K_eq_sabatier(T_K):
    return 137.0 * T_K**(-3.994) * np.exp(158700.0 / (R_gas * T_K))


def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    vH    = lambda dH: np.exp(-dH / R_gas * (1.0 / T_K - 1.0 / T_ref_K))
    k     = k_ref * np.exp(-Ea_k / R_gas * (1.0 / T_K - 1.0 / T_ref_K))
    K_OH  = A_OH  * vH(dH_OH)
    K_H2  = A_H2  * vH(dH_H2)
    K_mix = A_mix * vH(dH_mix)
    K_eq  = K_eq_sabatier(T_K)
    p_CO2_s = np.maximum(p_CO2, P_FLOOR)
    p_H2_s  = np.maximum(p_H2,  P_FLOOR)
    beta = (p_CH4 * p_H2O**2) / (K_eq * p_CO2_s * p_H2_s**4)
    f_eq = np.maximum(1.0 - beta, 0.0)
    DEN  = (1.0
            + K_OH  * np.maximum(p_H2O, 0.0) / p_H2_s**0.5
            + K_H2  * p_H2_s**0.5
            + K_mix * p_CO2_s)
    return k * (p_CO2_s * p_H2_s)**0.5 * f_eq / DEN**2 * 1000.0


def equilibrium_conversion_pct(T_K_val):
    """Equilibrium X_CO2 [%] for the Bareschino feed at 1 bar."""
    K = K_eq_sabatier(T_K_val)
    def f(X):
        d     = 1.0 - 0.08 * X
        p_CO2 = 0.04 * (1 - X)       / d
        p_H2  = 0.16 * (1 - X)       / d
        p_CH4 = (0.80 + 0.04 * X)    / d
        p_H2O = 0.08 * X             / d
        return p_CH4 * p_H2O**2 / (p_CO2 * p_H2**4 + 1e-100) - K
    try:
        return brentq(f, 1e-9, 1 - 1e-9) * 100.0
    except Exception:
        return 100.0


# =============================================================================
# SOLVER HELPER
# =============================================================================

def make_rhs(N_val, U_a_val):
    """Return a closure over (N_val, U_a_val) with the standard solve_ivp signature."""
    dz_val = L_b / (N_val - 1)

    def rhs(t, y, se_on, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O, T_in, T_wall):
        C_CO2 = np.maximum(y[0*N_val : 1*N_val], 0.0)
        C_H2  = np.maximum(y[1*N_val : 2*N_val], 0.0)
        C_CH4 = np.maximum(y[2*N_val : 3*N_val], 0.0)
        C_H2O = np.maximum(y[3*N_val : 4*N_val], 0.0)
        q     = np.maximum(y[4*N_val : 5*N_val], 0.0)
        T     = np.maximum(y[5*N_val : 6*N_val], 200.0)

        p_CO2 = C_CO2 * R_gas * T / 1e5
        p_H2  = C_H2  * R_gas * T / 1e5
        p_CH4 = C_CH4 * R_gas * T / 1e5
        p_H2O = C_H2O * R_gas * T / 1e5

        r    = reaction_rate_SI(T, p_CO2, p_H2, p_CH4, p_H2O)
        qs   = q_star_vec(T, p_H2O)
        Kl   = K_LDF_vec(T, p_H2O)
        dqdt = Kl * (qs - q) if se_on else np.zeros(N_val)

        C_CO2_up = np.concatenate([[C_in_CO2], C_CO2[:-1]])
        C_H2_up  = np.concatenate([[C_in_H2],  C_H2[:-1]])
        C_CH4_up = np.concatenate([[C_in_CH4], C_CH4[:-1]])
        C_H2O_up = np.concatenate([[C_in_H2O], C_H2O[:-1]])

        dCdt_CO2 = (-u*(C_CO2 - C_CO2_up)/dz_val + rho_bed_cat*(-1)*r) / eps_b
        dCdt_H2  = (-u*(C_H2  - C_H2_up) /dz_val + rho_bed_cat*(-4)*r) / eps_b
        dCdt_CH4 = (-u*(C_CH4 - C_CH4_up)/dz_val + rho_bed_cat*(+1)*r) / eps_b
        dCdt_H2O = (-u*(C_H2O - C_H2O_up)/dz_val + rho_bed_cat*(+2)*r
                                                   - rho_bed_ads*dqdt) / eps_b

        y_CO2l = p_CO2 / P_bar
        y_H2l  = p_H2  / P_bar
        y_CH4l = p_CH4 / P_bar
        y_H2Ol = p_H2O / P_bar
        y_N2l  = np.maximum(1.0 - y_CO2l - y_H2l - y_CH4l - y_H2Ol, 0.0)

        Cp_mix    = (y_CO2l*Cp_CO2 + y_H2l*Cp_H2 + y_CH4l*Cp_CH4
                     + y_H2Ol*Cp_H2O + y_N2l*Cp_N2)
        rho_g_mol = P_Pa / (R_gas * T)
        Cp_eff    = rho_bed_tot * Cp_cat + eps_b * rho_g_mol * Cp_mix

        T_up   = np.concatenate([[T_in], T[:-1]])
        Q_rxn  = (-dH_r)   * rho_bed_cat * r
        Q_ads  = (-dH_ads) * rho_bed_ads * dqdt
        Q_wall = -U_a_val  * (T - T_wall)

        dTdt = (-u * rho_g_mol * Cp_mix * (T - T_up) / dz_val
                + Q_rxn + Q_ads + Q_wall) / Cp_eff

        return np.concatenate([dCdt_CO2, dCdt_H2, dCdt_CH4, dCdt_H2O, dqdt, dTdt])

    return rhs


def solve_one(T_C, se_on, N_val=N_REF, U_a_val=U_a_NOM, ghsv_val=GHSV_NOM,
              rtol_val=1e-4, t_end_override=None, label=""):
    """
    Run one SEM column simulation.  Returns (sol, meta).
    meta contains T_K, u, inlet concentrations, q_at_max, t_sat_est.
    """
    T_K      = T_C + 273.15
    Q_STP    = ghsv_val * M_ads / 3600.0
    u        = (Q_STP / A_b) * (T_K / T_STP)
    C_in_CO2 = y_CO2_in * P_Pa / (R_gas * T_K)
    C_in_H2  = y_H2_in  * P_Pa / (R_gas * T_K)
    C_in_CH4 = y_CH4_in * P_Pa / (R_gas * T_K)
    C_in_H2O = 0.0

    p_H2O_max = 2 * y_CO2_in * P_bar / (1 - 2 * y_CO2_in)
    q_at_max  = float(q_star_vec(T_K, np.array([p_H2O_max]))[0])
    F_CO2_in  = C_in_CO2 * u * A_b
    t_sat_est = q_at_max * M_ads / (2.0 * F_CO2_in)

    if t_end_override is not None:
        t_end = t_end_override
    elif se_on:
        t_end = min(2.5 * t_sat_est, 7200.0)
    else:
        t_end = 600.0   # SE-off reaches steady state well within 600 s

    rhs  = make_rhs(N_val, U_a_val)
    y0   = np.zeros(6 * N_val)
    y0[0*N_val : 1*N_val] = C_in_CO2
    y0[1*N_val : 2*N_val] = C_in_H2
    y0[2*N_val : 3*N_val] = C_in_CH4
    y0[5*N_val : 6*N_val] = T_K
    if not se_on:
        y0[4*N_val : 5*N_val] = q_at_max   # pre-saturated sorbent → no adsorption

    atol = np.concatenate([1e-8*np.ones(4*N_val), 1e-8*np.ones(N_val), 1e-2*np.ones(N_val)])

    tag = label or f"T={T_C}°C se_on={se_on} N={N_val} U_a={U_a_val:.0f} GHSV={ghsv_val:.2f}"
    print(f"  [{tag}] ...", end="", flush=True)

    sol = solve_ivp(
        rhs,
        t_span=[0.0, t_end],
        y0=y0,
        args=(se_on, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O, T_K, T_K),
        method='BDF', rtol=rtol_val, atol=atol, dense_output=True,
    )
    status = "OK" if sol.success else f"FAILED — {sol.message}"
    print(f" {status}")
    if not sol.success:
        raise RuntimeError(f"ODE solver failed ({tag}): {sol.message}")

    meta = dict(T_K=T_K, u=u, C_in_CO2=C_in_CO2, C_in_H2=C_in_H2,
                C_in_CH4=C_in_CH4, q_at_max=q_at_max, t_sat_est=t_sat_est,
                N_val=N_val, ghsv_val=ghsv_val)
    return sol, meta


def outlet_last(sol, N_val):
    """Extract outlet concentrations and T at the final (steady-state) time step."""
    y = sol.y[:, -1]
    return dict(
        C_CO2 = max(float(y[  N_val - 1]), 0.0),
        C_H2  = max(float(y[2*N_val - 1]), 0.0),
        C_CH4 = max(float(y[3*N_val - 1]), 0.0),
        C_H2O = max(float(y[4*N_val - 1]), 0.0),
        T_out = max(float(y[6*N_val - 1]), 200.0),
        T_max = float(np.max(y[5*N_val : 6*N_val])),
    )


def X_CO2_timeseries(sol, C_in_CO2, N_val):
    """Return (t_arr, X_CO2_arr) for the full time history."""
    t   = sol.t
    y   = sol.sol(t)
    Cout = np.maximum(y[N_val - 1, :], 0.0)
    return t, np.clip((C_in_CO2 - Cout) / C_in_CO2, 0.0, 1.0)


# =============================================================================
# GROUP A — INTERNAL CONSISTENCY
# =============================================================================

def test_A1_carbon_balance(results_300):
    """
    A1 — Carbon and hydrogen stoichiometry at outlet steady state (SE off).

    In this constant-velocity model the species equations are exactly coupled
    by stoichiometry, so the ratios below should hold to within ODE tolerance
    (~0.1 %).  A deviation >2 % indicates a sign error or mismatched coefficient.

    Expected:
        ΔCH4 / ΔCO2 = 1.00   (one methane produced per CO2 consumed)
        ΔH2  / ΔCO2 = 4.00   (four hydrogen consumed per CO2 consumed)
        C_H2O_out / ΔCO2 = 2.00  (two water produced, no adsorption in SE-off)
    """
    print("\n── A1  Carbon / hydrogen balance at steady state (SE off) ──")
    sol, meta = results_300['off']
    N = meta['N_val']
    out = outlet_last(sol, N)

    dCO2 = meta['C_in_CO2'] - out['C_CO2']   # consumed  [mol/m³]
    dH2  = meta['C_in_H2']  - out['C_H2']    # consumed
    dCH4 = out['C_CH4']     - meta['C_in_CH4']  # produced
    dH2O = out['C_H2O']                       # produced (inlet = 0)

    r_CH4 = dCH4 / max(dCO2, 1e-12)
    r_H2  = dH2  / max(dCO2, 1e-12)
    r_H2O = dH2O / max(dCO2, 1e-12)

    err_CH4 = abs(r_CH4 - 1.0) * 100   # %
    err_H2  = abs(r_H2  - 4.0) / 4.0 * 100
    err_H2O = abs(r_H2O - 2.0) / 2.0 * 100

    tol = 2.0   # % — limited by ODE rtol=1e-4
    passed = err_CH4 < tol and err_H2 < tol and err_H2O < tol

    print(f"    ΔCH4/ΔCO2 = {r_CH4:.4f}  (expect 1.000)  error = {err_CH4:.3f}%")
    print(f"    ΔH2 /ΔCO2 = {r_H2 :.4f}  (expect 4.000)  error = {err_H2 :.3f}%")
    print(f"    H2O /ΔCO2 = {r_H2O:.4f}  (expect 2.000)  error = {err_H2O:.3f}%")
    print(f"    → {'PASS' if passed else 'FAIL'}  (tolerance {tol} %)")
    return passed


def test_A2_stoichiometry_profile(results_300):
    """
    A2 — Axial stoichiometry: H2 : CO2 consumption ratio equals 4 at every node.

    Checks that the species source terms in the RHS are consistent across the
    entire spatial domain, not only at the outlet.
    """
    print("\n── A2  Axial stoichiometry H2 : CO2 = 4 (SE off, final snapshot) ──")
    sol, meta = results_300['off']
    N = meta['N_val']
    y  = sol.y[:, -1]
    C_CO2 = np.maximum(y[0*N : 1*N], 0.0)
    C_H2  = np.maximum(y[1*N : 2*N], 0.0)

    dCO2_z = meta['C_in_CO2'] - C_CO2   # consumed up to each node
    dH2_z  = meta['C_in_H2']  - C_H2

    # Avoid division by near-zero at inlet (no reaction yet)
    mask = dCO2_z > 0.01 * meta['C_in_CO2']
    ratio = dH2_z[mask] / dCO2_z[mask]
    max_dev = float(np.max(np.abs(ratio - 4.0)) / 4.0 * 100)

    passed = max_dev < 2.0
    print(f"    Max deviation of H2:CO2 ratio from 4.0 across bed: {max_dev:.3f}%")
    print(f"    → {'PASS' if passed else 'FAIL'}  (tolerance 2 %)")
    return passed


def test_A3_le_chatelier(results_300):
    """
    A3 — Le Chatelier: SE-on conversion must never be lower than SE-off.

    Water removal by adsorption shifts the Sabatier equilibrium forward, so
    X_SE_on ≥ X_SE_off must hold at every time point.  Any inversion is a
    sign-convention or coupling error.
    """
    print("\n── A3  Le Chatelier: X_SE_on ≥ X_SE_off at all times ──")
    sol_on,  meta_on  = results_300['on']
    sol_off, meta_off = results_300['off']
    N = meta_on['N_val']

    t_on,  X_on  = X_CO2_timeseries(sol_on,  meta_on['C_in_CO2'],  N)
    t_off, X_off = X_CO2_timeseries(sol_off, meta_off['C_in_CO2'], N)

    # Interpolate SE-off onto SE-on time grid
    X_off_interp = np.interp(t_on, t_off, X_off)
    violations   = np.sum(X_on < X_off_interp - 1e-4)   # small numerical slack
    max_inv      = float(np.max(np.maximum(X_off_interp - X_on, 0.0)) * 100)

    passed = violations == 0
    print(f"    Time points checked: {len(t_on)}")
    print(f"    Violations (X_on < X_off - 1e-4): {violations}")
    print(f"    Max inversion: {max_inv:.4f} percentage points")
    print(f"    → {'PASS' if passed else 'FAIL'}")
    return passed


# =============================================================================
# GROUP B — NUMERICAL CONVERGENCE
# =============================================================================

def test_B1_grid_refinement(T_C=300):
    """
    B1 — Grid-refinement study: N = 25, 50, 100, 200.

    The steady-state outlet X_CO2 (SE off) should converge monotonically.
    The solution is considered grid-independent if N=100 and N=200 agree
    within 1 % of each other.
    """
    print("\n── B1  Grid refinement  (SE off, T = 300 °C) ──")
    N_vals = [25, 50, 100, 200]
    X_ss   = {}
    for N_val in N_vals:
        sol, meta = solve_one(T_C, se_on=False, N_val=N_val, label=f"N={N_val}")
        out = outlet_last(sol, N_val)
        X_ss[N_val] = (meta['C_in_CO2'] - out['C_CO2']) / meta['C_in_CO2'] * 100

    X_ref = X_ss[200]
    print(f"\n    {'N':>6}  {'X_CO2 [%]':>12}  {'Δ vs N=200 [%]':>16}")
    print(f"    {'─'*40}")
    for N_val in N_vals:
        dev = abs(X_ss[N_val] - X_ref)
        print(f"    {N_val:>6}  {X_ss[N_val]:>12.4f}  {dev:>16.4f}")

    dev_100 = abs(X_ss[100] - X_ss[200])
    passed  = dev_100 < 1.0
    print(f"\n    N=100 vs N=200: {dev_100:.4f} pp")
    print(f"    → {'PASS' if passed else 'FAIL'}  (tolerance 1 pp)")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(N_vals, [X_ss[n] for n in N_vals], 'ko-', lw=2, ms=7)
    ax.axhline(X_ref, color='grey', ls='--', lw=1, label=f'N=200 reference ({X_ref:.2f} %)')
    ax.set_xlabel('Number of spatial nodes  N  [–]', fontsize=11)
    ax.set_ylabel('Steady-state outlet X_CO₂  [%]', fontsize=11)
    ax.set_title('B1 — Grid-refinement study  (T = 300 °C, SE off)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_B1_grid_refinement.png', dpi=150, bbox_inches='tight')
    plt.show()

    return passed, X_ss


def test_B2_solver_tolerance(T_C=300, X_ss_N100=None):
    """
    B2 — Solver-tolerance sensitivity: rtol = 1e-3, 1e-4, 1e-5.

    The nominal rtol=1e-4 should agree with the tightest rtol=1e-5 to within 0.5 %.
    """
    print("\n── B2  Solver-tolerance sensitivity  (SE off, T = 300 °C) ──")
    rtols = [1e-3, 1e-4, 1e-5]
    X_tol = {}
    for rtol in rtols:
        sol, meta = solve_one(T_C, se_on=False, rtol_val=rtol,
                              label=f"rtol={rtol:.0e}")
        out = outlet_last(sol, meta['N_val'])
        X_tol[rtol] = (meta['C_in_CO2'] - out['C_CO2']) / meta['C_in_CO2'] * 100

    X_ref = X_tol[1e-5]
    print(f"\n    {'rtol':>8}  {'X_CO2 [%]':>12}  {'Δ vs rtol=1e-5 [pp]':>22}")
    print(f"    {'─'*46}")
    for rtol in rtols:
        dev = abs(X_tol[rtol] - X_ref)
        print(f"    {rtol:>8.0e}  {X_tol[rtol]:>12.4f}  {dev:>22.4f}")

    dev_nom = abs(X_tol[1e-4] - X_tol[1e-5])
    passed  = dev_nom < 0.5
    print(f"\n    rtol=1e-4 vs rtol=1e-5: {dev_nom:.4f} pp")
    print(f"    → {'PASS' if passed else 'FAIL'}  (tolerance 0.5 pp)")
    return passed


# =============================================================================
# GROUP C — THERMODYNAMIC LIMITS
# =============================================================================

def test_C1_equilibrium_limit(T_C=300):
    """
    C1 — At very low GHSV (0.04 m³/kg/h, 10× below nominal), SE-off steady-state
    X_CO2 should approach the thermodynamic equilibrium conversion.

    A gap >5 percentage points indicates a kinetic or thermodynamic inconsistency.
    """
    print("\n── C1  Equilibrium limit at low GHSV  (SE off, T = 300 °C) ──")
    ghsv_low = 0.04
    sol_nom, meta_nom = solve_one(T_C, se_on=False, label="GHSV=0.40 (nominal)")
    sol_low, meta_low = solve_one(T_C, se_on=False, ghsv_val=ghsv_low,
                                  t_end_override=3000.0, label=f"GHSV={ghsv_low}")

    out_nom = outlet_last(sol_nom, meta_nom['N_val'])
    out_low = outlet_last(sol_low, meta_low['N_val'])
    X_nom = (meta_nom['C_in_CO2'] - out_nom['C_CO2']) / meta_nom['C_in_CO2'] * 100
    X_low = (meta_low['C_in_CO2'] - out_low['C_CO2']) / meta_low['C_in_CO2'] * 100
    X_eq  = equilibrium_conversion_pct(T_C + 273.15)

    gap = X_eq - X_low
    passed = gap < 5.0 and X_low > X_nom   # must increase and must be close to X_eq

    print(f"    GHSV = {GHSV_NOM:.2f}  →  X = {X_nom:.2f} %")
    print(f"    GHSV = {ghsv_low:.2f}  →  X = {X_low:.2f} %")
    print(f"    Equilibrium   →  X = {X_eq:.2f} %")
    print(f"    Gap (X_eq − X_low) = {gap:.2f} pp")
    print(f"    → {'PASS' if passed else 'FAIL'}  (tolerance 5 pp, must increase with lower GHSV)")

    # Plot X vs time for both GHSV values
    t_nom, X_t_nom = X_CO2_timeseries(sol_nom, meta_nom['C_in_CO2'], meta_nom['N_val'])
    t_low, X_t_low = X_CO2_timeseries(sol_low, meta_low['C_in_CO2'], meta_low['N_val'])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_nom, X_t_nom * 100, 'tab:blue',  lw=2, label=f'GHSV = {GHSV_NOM:.2f} m³/(kg·h)')
    ax.plot(t_low, X_t_low * 100, 'tab:orange', lw=2, label=f'GHSV = {ghsv_low:.2f} m³/(kg·h)')
    ax.axhline(X_eq, color='k', ls='--', lw=1.5, label=f'Equilibrium  {X_eq:.1f} %')
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Outlet X_CO₂  [%]', fontsize=11)
    ax.set_title('C1 — Equilibrium approach at low GHSV  (T = 300 °C, SE off)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_C1_equilibrium_limit.png', dpi=150, bbox_inches='tight')
    plt.show()

    return passed


def test_C2_isothermal_limit(T_C=300):
    """
    C2 — With U_a → ∞ (= 1e6 W/m³/K), the bed temperature must stay flat.

    A maximum |ΔT| < 2 K across the entire bed and simulation confirms that
    the wall-cooling term has the correct sign and magnitude.
    """
    print("\n── C2  Isothermal wall limit  (U_a = 1e6, SE off, T = 300 °C) ──")
    U_a_iso = 1e6
    sol, meta = solve_one(T_C, se_on=False, U_a_val=U_a_iso, label="U_a=1e6")
    N  = meta['N_val']
    T0 = meta['T_K']

    # Sample temperature field at several time points
    t_sample = sol.t[::max(1, len(sol.t)//20)]
    dT_max   = 0.0
    for t_s in t_sample:
        T_prof = sol.sol(t_s)[5*N : 6*N]
        dT_max = max(dT_max, float(np.max(np.abs(T_prof - T0))))

    passed = dT_max < 2.0
    print(f"    Max |T(z,t) − T_in| with U_a=1e6:  {dT_max:.3f} K")
    print(f"    → {'PASS' if passed else 'FAIL'}  (tolerance 2 K)")
    return passed


def test_C3_adiabatic_vs_cooled(T_C=300):
    """
    C3 — Adiabatic (U_a=0) hot-spot must exceed the cooled (U_a=8000) hot-spot.

    The wall-cooling term removes heat, so removing it must increase ΔT_max.
    """
    print("\n── C3  Adiabatic (U_a=0) hot-spot > cooled (U_a=8000) ──")
    sol_cool, meta_cool = solve_one(T_C, se_on=False, U_a_val=8000.0, label="cooled U_a=8000")
    sol_adb,  meta_adb  = solve_one(T_C, se_on=False, U_a_val=0.0,    label="adiabatic U_a=0")

    N  = N_REF
    T0 = meta_cool['T_K']

    dT_cool = outlet_last(sol_cool, N)['T_max'] - T0
    dT_adb  = outlet_last(sol_adb,  N)['T_max'] - T0

    passed = dT_adb > dT_cool
    print(f"    ΔT_max  cooled    (U_a=8000): {dT_cool:.2f} K")
    print(f"    ΔT_max  adiabatic (U_a=0  ): {dT_adb :.2f} K")
    print(f"    → {'PASS' if passed else 'FAIL'}  (adiabatic must be hotter)")
    return passed


# =============================================================================
# GROUP D — SUB-MODEL UNIT TESTS
# =============================================================================

def test_D1_keq_direction():
    """
    D1 — K_eq must decrease monotonically with temperature (exothermic reaction).

    The Sabatier reaction is strongly exothermic (ΔH = −165 kJ/mol), so K_eq
    must fall with rising T.  A non-monotonic K_eq signals a wrong sign in the
    exponent.

    As a magnitude check: at 300 °C (573 K), published data give
    K_eq ≈ 10^4–10^5 bar⁻² (Koschany 2016, Table 4), so a value below 10
    would indicate an error.
    """
    print("\n── D1  K_eq direction and magnitude ──")
    T_arr   = np.array([200, 260, 300, 320, 400, 500, 600]) + 273.15
    K_arr   = np.array([K_eq_sabatier(T) for T in T_arr])
    monotone = bool(np.all(np.diff(K_arr) < 0))
    K_at_300 = float(K_eq_sabatier(573.15))

    print(f"\n    {'T [°C]':>8}  {'K_eq [bar⁻²]':>16}")
    print(f"    {'─'*28}")
    for T_C, K in zip(T_arr - 273.15, K_arr):
        print(f"    {T_C:>8.0f}  {K:>16.2e}")
    print(f"\n    Monotonically decreasing: {monotone}")
    print(f"    K_eq(300 °C) = {K_at_300:.2e}  (expect > 10)")

    passed = monotone and K_at_300 > 10.0
    print(f"    → {'PASS' if passed else 'FAIL'}")

    # Plot
    T_fine = np.linspace(150, 600, 200) + 273.15
    K_fine = np.array([K_eq_sabatier(T) for T in T_fine])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(T_fine - 273.15, K_fine, 'tab:blue', lw=2)
    ax.scatter(T_arr - 273.15, K_arr, color='tab:red', zorder=5, s=50)
    ax.set_xlabel('Temperature [°C]', fontsize=11)
    ax.set_ylabel('K_eq  [bar⁻²]', fontsize=11)
    ax.set_title('D1 — Equilibrium constant K_eq vs temperature', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_D1_Keq.png', dpi=150, bbox_inches='tight')
    plt.show()

    return passed


def test_D2_isotherm_limits():
    """
    D2 — DA isotherm physical limits.

    Four assertions must all hold:
      (a) q* → 0 as p → 0
      (b) q* → q_max = W0·ρ_water/M_w  as p → P_sat
      (c) q* increases with partial pressure at fixed T
      (d) q* decreases with temperature at fixed relative humidity (p/P_sat)
    """
    print("\n── D2  DA isotherm physical limits ──")
    T_test  = 573.15   # 300 °C
    Psat    = float(P_sat_bar(T_test))
    q_max_theory = W0_DA * rho_water(T_test) / MW_H2O

    q_near_zero = float(q_star_vec(T_test, np.array([1e-8])))
    q_near_sat  = float(q_star_vec(T_test, np.array([0.9999 * Psat])))

    p_range   = np.linspace(1e-5, 0.99 * Psat, 50)
    q_profile = q_star_vec(T_test, p_range)
    monotone_p = bool(np.all(np.diff(q_profile) >= 0))

    T_list = [260, 300, 340]
    p_rel  = 0.5    # fixed relative humidity
    q_at_Tlist = [float(q_star_vec(T + 273.15, np.array([p_rel * float(P_sat_bar(T + 273.15))])))
                  for T in T_list]
    monotone_T = bool(q_at_Tlist[0] > q_at_Tlist[1] > q_at_Tlist[2])

    print(f"    q*(T=300°C, p→0)        = {q_near_zero:.4e} mol/kg  (expect ≈ 0)")
    print(f"    q*(T=300°C, p→P_sat)    = {q_near_sat:.4f} mol/kg")
    print(f"    q_max theory (W0·ρ/M_w) = {q_max_theory:.4f} mol/kg")
    print(f"    Increasing with p: {monotone_p}")
    print(f"    q* at p/Psat=0.5 across T:  {' < '.join(f'{T}°C: {q:.3f}' for T, q in zip(T_list, q_at_Tlist))}")
    print(f"    Decreasing with T (at fixed rel. humidity): {monotone_T}")

    rel_err_sat = abs(q_near_sat - q_max_theory) / q_max_theory * 100
    passed = (q_near_zero < 1e-3 and rel_err_sat < 1.0
              and monotone_p and monotone_T)
    print(f"    q*(p→P_sat) / q_max_theory error = {rel_err_sat:.4f}%")
    print(f"    → {'PASS' if passed else 'FAIL'}")

    # Plot: q* vs p at three temperatures
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['tab:blue', 'tab:orange', 'tab:red']
    for T_C, col in zip([260, 300, 340], colors):
        T_K_i  = T_C + 273.15
        Psat_i = float(P_sat_bar(T_K_i))
        p_i    = np.linspace(1e-5, 0.999 * Psat_i, 200)
        q_i    = q_star_vec(T_K_i, p_i)
        ax.plot(p_i * 1000, q_i, color=col, lw=2, label=f'{T_C} °C  (P_sat={Psat_i*1000:.1f} mbar)')
    ax.set_xlabel('p_H₂O  [mbar]', fontsize=11)
    ax.set_ylabel('q*  [mol/kg]', fontsize=11)
    ax.set_title('D2 — DA isotherm: equilibrium H₂O loading vs partial pressure', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_D2_isotherm.png', dpi=150, bbox_inches='tight')
    plt.show()

    return passed


def test_D3_reaction_rate():
    """
    D3 — Kinetic rate sanity checks.

    (a) At the reference state (T=T_ref, stoichiometric inlet, β=0) the rate
        must be positive and finite.  It should equal approximately
        k_ref · √(p_CO2·p_H2) / DEN² · 1000.
    (b) Rate must equal zero when β = 1 (thermodynamic equilibrium).
    (c) Rate must increase with temperature in the kinetically limited regime
        (low T, β≈0).
    """
    print("\n── D3  Reaction rate sanity checks ──")

    # (a) Reference state
    p_CO2_ref, p_H2_ref = 0.04, 0.16   # inlet partial pressures at P=1 bar
    r_ref = float(reaction_rate_SI(
        np.array([T_ref_K]), p_CO2_ref, p_H2_ref, 0.0, 0.0))
    vH    = lambda dH: np.exp(-dH / R_gas * (1.0 / T_ref_K - 1.0 / T_ref_K))
    K_OH_ref  = A_OH  * vH(dH_OH)
    K_H2_ref  = A_H2  * vH(dH_H2)
    K_mix_ref = A_mix * vH(dH_mix)
    DEN_ref   = 1.0 + K_H2_ref * p_H2_ref**0.5 + K_mix_ref * p_CO2_ref
    r_analytic = k_ref * (p_CO2_ref * p_H2_ref)**0.5 / DEN_ref**2 * 1000.0
    print(f"    r at T_ref, inlet feed:  {r_ref:.6f} mol/(kg·s)")
    print(f"    Analytic approx (β=0):   {r_analytic:.6f} mol/(kg·s)")
    err_ref = abs(r_ref - r_analytic) / r_analytic * 100
    print(f"    Relative error: {err_ref:.2f}%")

    # (b) Rate = 0 at equilibrium
    T_eq = 573.15
    K    = K_eq_sabatier(T_eq)
    # Choose partial pressures that exactly satisfy equilibrium:
    # p_CH4 * p_H2O^2 = K * p_CO2 * p_H2^4
    # Simple: p_CO2=0.01, p_H2=0.01, then p_CH4*p_H2O^2 = K*0.01*0.01^4
    p_CO2_e, p_H2_e = 0.01, 0.01
    p_CH4_p_H2O2 = K * p_CO2_e * p_H2_e**4
    # pick p_CH4 = 0.80, then p_H2O = sqrt(p_CH4_p_H2O2/p_CH4)
    p_CH4_e = 0.80
    p_H2O_e = np.sqrt(p_CH4_p_H2O2 / p_CH4_e)
    r_eq = float(reaction_rate_SI(
        np.array([T_eq]), p_CO2_e, p_H2_e, p_CH4_e, p_H2O_e))
    print(f"\n    r at equilibrium conditions: {r_eq:.2e} mol/(kg·s)  (expect ≈ 0)")

    # (c) Rate increases with T in low-T regime
    T_low  = [220, 260, 300] # °C
    r_low  = [float(reaction_rate_SI(np.array([T+273.15]), p_CO2_ref, p_H2_ref, 0.0, 0.0))
              for T in T_low]
    print(f"\n    Rate vs T (β=0, inlet feed):")
    for T_C, r in zip(T_low, r_low):
        print(f"      T = {T_C}°C  →  r = {r:.6f} mol/(kg·s)")
    monotone_T = all(r_low[i] < r_low[i+1] for i in range(len(r_low)-1))
    print(f"    Rate increases with T: {monotone_T}")

    passed = (r_ref > 0 and np.isfinite(r_ref)
              and r_eq < 1e-6
              and monotone_T
              and err_ref < 5.0)
    print(f"    → {'PASS' if passed else 'FAIL'}")
    return passed


# =============================================================================
# GROUP E — LITERATURE COMPARISON
# =============================================================================

def test_E1_breakthrough_times(results_all):
    """
    E1 — H2O breakthrough times (SE on), compared to Bareschino (2023) Table 4.

    Threshold definition: y_H2O_outlet ≥ 10 % of the equilibrium y_H2O, which
    matches Bareschino's definition of breakthrough.

    Literature source: Bareschino et al. (2023), Table 4,
    non-isothermal cooled reactor at GHSV = 0.5 m³/(kg_ads·h), P = 1 bar.
      T=280 °C → t_BT,H2O = 126.6 min
      T=300 °C → t_BT,H2O =  88.3 min
      T=320 °C → t_BT,H2O =  71.2 min

    GHSV scaling: our model uses GHSV = 0.4 m³/(kg_ads·h).  Since breakthrough
    time scales as t_BT ∝ q_max / (GHSV × X_CO2), lower GHSV gives longer t_BT.
    The GHSV-scaled reference is: t_ref_scaled = t_BT,Bareschino × (0.5 / 0.4).
    We accept a ±20 % deviation from this scaled reference.

    T=260 °C is not reported by Bareschino; only the physical-plausibility and
    monotonicity checks apply there.
    """
    print("\n── E1  H2O breakthrough times (SE on) ──")
    print("    Reference: Bareschino et al. (2023) Table 4, non-isothermal cooled,")
    print("    GHSV=0.5 m³/(kg_ads·h).  Our model: GHSV=0.4 → scaled ref = ref × 1.25")

    # H2O breakthrough times from Bareschino Table 4 (non-isothermal cooled, GHSV=0.5)
    bareschino_ref = {260: None, 280: 126.6, 300: 88.3, 320: 71.2}   # [min]
    ghsv_scale     = GHSV_NOM / 0.5   # 0.4/0.5 = 0.8 → our GHSV is lower → longer t_BT
    # scaled reference: t_BT_expected = bareschino / ghsv_scale = bareschino * 1.25

    t_bt = {}
    for T_C in T_LIST:
        T_K       = T_C + 273.15
        X_eq      = equilibrium_conversion_pct(T_K) / 100.0
        y_H2O_eq  = 0.08 * X_eq / (1.0 - 0.08 * X_eq)
        threshold = 0.10 * y_H2O_eq

        sol, meta = results_all[T_C]['on']
        N         = meta['N_val']
        t_arr     = sol.t
        y_arr     = sol.sol(t_arr)
        C_H2O_out = np.maximum(y_arr[4*N - 1, :], 0.0)
        T_out     = np.maximum(y_arr[6*N - 1, :], 200.0)
        y_H2O_out = C_H2O_out * R_gas * T_out / P_Pa

        idx = np.where(y_H2O_out >= threshold)[0]
        t_bt[T_C] = t_arr[idx[0]] / 60.0 if len(idx) > 0 else None

    print(f"\n    {'T [°C]':>8}  {'Model [min]':>12}  {'Bareschino':>12}  "
          f"{'Scaled ref':>12}  {'Error vs scaled':>16}")
    print(f"    {'─'*66}")
    within_tol = []
    for T_C in T_LIST:
        ref      = bareschino_ref[T_C]
        t_model  = t_bt[T_C]
        t_str    = f"{t_model:.1f}" if t_model is not None else "no BT"
        ref_str  = f"{ref:.1f}"     if ref      is not None else "—"
        if ref is not None and t_model is not None:
            t_scaled = ref / ghsv_scale
            err_pct  = (t_model - t_scaled) / t_scaled * 100
            sc_str   = f"{t_scaled:.1f}"
            er_str   = f"{err_pct:+.1f}%"
            within_tol.append(abs(err_pct) <= 20.0)
        else:
            sc_str, er_str = "—", "—"
        print(f"    {T_C:>8}  {t_str:>12}  {ref_str:>12}  {sc_str:>12}  {er_str:>16}")

    # Qualitative checks (apply to all temperatures including 260 °C)
    t_vals     = [t_bt[T] for T in T_LIST if t_bt[T] is not None]
    in_range   = all(10.0 <= t <= 200.0 for t in t_vals) if t_vals else False
    decreasing = all(t_vals[i] >= t_vals[i+1] for i in range(len(t_vals)-1))
    quant_ok   = all(within_tol) if within_tol else True

    passed = in_range and decreasing and quant_ok
    print(f"\n    All times in 10–200 min:              {in_range}")
    print(f"    Decreasing with T:                    {decreasing}")
    print(f"    Within ±20% of GHSV-scaled reference: {quant_ok}")
    print(f"    → {'PASS' if passed else 'FAIL'}")
    return passed, t_bt


def test_E2_conversion_enhancement(results_all):
    """
    E2 — SE enhancement factor X_SE_on / X_SE_off at each temperature.

    Enhancement must be >1 at all temperatures (Le Chatelier, global check over
    the full run range) and must decrease with increasing temperature (higher T
    reduces adsorption capacity, so the SE benefit shrinks).

    Literature source (Bareschino 2023):
    - SE X_CO2: the paper states explicitly that "a complete CO2 conversion
      (X_CO2 = 1) was observed at the reactor outlet until H2 breakthrough was
      attained" (section 4.1). This holds for all T ≤ 350 °C investigated.
      → bareschino_X_on = 100 % at T = 280, 300, 320 °C.
    - Non-SE X_CO2 at steady state: not directly tabulated by Bareschino for
      their cooled reactor. Fig. 7 shows only the SE case. The non-SE reference
      can be added once Fig. 7 is digitised (set bareschino_X_off entries).
    """
    print("\n── E2  SE conversion enhancement vs temperature ──")

    # ── Literature reference values (Bareschino 2023) ──────────────────────────
    # SE: complete conversion until H2 breakthrough (paper section 4.1, all T ≤ 350 °C)
    bareschino_X_on = {260: None, 280: 100.0, 300: 100.0, 320: 100.0}  # [%]
    # Non-SE reference: not explicitly tabulated by Bareschino for their cooled reactor.
    # To add it: digitise the steady-state X_CO2 dashed line from Fig. 7 of the paper
    # and insert values into a bareschino_X_off dict, then add a comparison column below.

    X_off_ss  = {}
    X_on_ini  = {}
    enh       = {}

    for T_C in T_LIST:
        sol_off, meta_off = results_all[T_C]['off']
        sol_on,  meta_on  = results_all[T_C]['on']
        N       = meta_on['N_val']
        t_sat   = meta_on['t_sat_est']

        # SE-off steady state: average over last half of simulation
        t_off, X_t_off = X_CO2_timeseries(sol_off, meta_off['C_in_CO2'], N)
        mid = max(1, len(X_t_off) // 2)
        X_off_ss[T_C] = float(np.mean(X_t_off[mid:])) * 100

        # SE-on early period: 10–40 % of estimated saturation time (fresh sorbent)
        t_on, X_t_on = X_CO2_timeseries(sol_on, meta_on['C_in_CO2'], N)
        mask = (t_on >= 0.10 * t_sat) & (t_on <= 0.40 * t_sat)
        if mask.sum() == 0:
            mask = np.ones(len(t_on), dtype=bool)
        X_on_ini[T_C] = float(np.mean(X_t_on[mask])) * 100

        enh[T_C] = X_on_ini[T_C] / max(X_off_ss[T_C], 0.1)

    print(f"\n    {'T [°C]':>8}  {'X_off SS [%]':>14}  {'X_on ini [%]':>14}  "
          f"{'Enh. [−]':>10}  {'Ref X_on [%]':>14}  {'Err X_on [pp]':>14}")
    print(f"    {'─'*80}")
    for T_C in T_LIST:
        ref_on   = bareschino_X_on[T_C]
        r_on_str = f"{ref_on:.1f}"                    if ref_on  is not None else "—"
        err_str  = f"{X_on_ini[T_C] - ref_on:+.2f}"  if ref_on  is not None else "—"
        print(f"    {T_C:>8}  {X_off_ss[T_C]:>14.2f}  {X_on_ini[T_C]:>14.2f}  "
              f"{enh[T_C]:>10.3f}  {r_on_str:>14}  {err_str:>14}")

    # All enhancements > 1, and enhancement should decrease with T
    all_enh_gt1  = all(enh[T] > 1.0 for T in T_LIST)
    enh_vals     = [enh[T] for T in T_LIST]
    enh_decrease = all(enh_vals[i] >= enh_vals[i+1] for i in range(len(enh_vals)-1))

    passed = all_enh_gt1 and enh_decrease
    print(f"\n    All enhancement factors > 1: {all_enh_gt1}")
    print(f"    Enhancement decreasing with T: {enh_decrease}")
    print(f"    → {'PASS' if passed else 'FAIL'}")

    # Bar chart
    x    = np.arange(len(T_LIST))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, [X_off_ss[T] for T in T_LIST], w, label='Non-SE (steady state)',
           color='tab:orange', alpha=0.85)
    ax.bar(x + w/2, [X_on_ini[T] for T in T_LIST], w, label='SE (fresh sorbent)',
           color='tab:blue',   alpha=0.85)
    T_fine = np.linspace(200, 370, 80)
    X_eq_l = [equilibrium_conversion_pct(T + 273.15) for T in T_fine]
    ax.plot(np.interp(T_fine, T_LIST, x), X_eq_l, 'k--', lw=1.5, label='Equilibrium')
    ax.set_xticks(x); ax.set_xticklabels([f'{T} °C' for T in T_LIST])
    ax.set_ylabel('CO₂ conversion [%]', fontsize=11)
    ax.set_title('E2 — SE enhancement at each operating temperature', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('validation_E2_enhancement.png', dpi=150, bbox_inches='tight')
    plt.show()

    return passed


# =============================================================================
# MAIN — RUN ALL TESTS
# =============================================================================

if __name__ == '__main__':

    print("=" * 70)
    print("  VALIDATION TEST SUITE — Full-Scale SEM LHHW Non-isothermal Model")
    print("=" * 70)

    # ── Pre-compute the main run batch (4 T × 2 modes) used by groups A and E ──
    print("\n[Pre-computing main run batch: 4 temperatures × SE on/off]")
    results_all = {}
    for T_C in T_LIST:
        results_all[T_C] = {}
        results_all[T_C]['off'] = solve_one(T_C, se_on=False, label=f"T={T_C}°C SE-off")
        results_all[T_C]['on']  = solve_one(T_C, se_on=True,  label=f"T={T_C}°C SE-on ")

    results_300 = {'on': results_all[300]['on'], 'off': results_all[300]['off']}

    # ── Run all tests ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GROUP A — Internal consistency")
    print("=" * 70)
    pA1 = test_A1_carbon_balance(results_300)
    pA2 = test_A2_stoichiometry_profile(results_300)
    pA3 = test_A3_le_chatelier(results_300)

    print("\n" + "=" * 70)
    print("  GROUP B — Numerical convergence")
    print("=" * 70)
    pB1, _ = test_B1_grid_refinement(T_C=300)
    pB2    = test_B2_solver_tolerance(T_C=300)

    print("\n" + "=" * 70)
    print("  GROUP C — Thermodynamic limits")
    print("=" * 70)
    pC1 = test_C1_equilibrium_limit(T_C=300)
    pC2 = test_C2_isothermal_limit(T_C=300)
    pC3 = test_C3_adiabatic_vs_cooled(T_C=300)

    print("\n" + "=" * 70)
    print("  GROUP D — Sub-model unit tests")
    print("=" * 70)
    pD1 = test_D1_keq_direction()
    pD2 = test_D2_isotherm_limits()
    pD3 = test_D3_reaction_rate()

    print("\n" + "=" * 70)
    print("  GROUP E — Literature comparison")
    print("=" * 70)
    pE1, _ = test_E1_breakthrough_times(results_all)
    pE2    = test_E2_conversion_enhancement(results_all)

    # ── Summary ──────────────────────────────────────────────────────────────
    results_summary = {
        'A1 Carbon balance':          pA1,
        'A2 Axial stoichiometry':     pA2,
        'A3 Le Chatelier':            pA3,
        'B1 Grid refinement':         pB1,
        'B2 Solver tolerance':        pB2,
        'C1 Equilibrium limit':       pC1,
        'C2 Isothermal wall':         pC2,
        'C3 Adiabatic > cooled':      pC3,
        'D1 K_eq direction':          pD1,
        'D2 DA isotherm limits':      pD2,
        'D3 Reaction rate':           pD3,
        'E1 Breakthrough times':      pE1,
        'E2 Conversion enhancement':  pE2,
    }

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    n_pass = sum(results_summary.values())
    n_total = len(results_summary)
    for name, passed in results_summary.items():
        status = "PASS" if passed else "FAIL"
        marker = "✓" if passed else "✗"
        print(f"  {marker}  {name:<35}  {status}")
    print(f"\n  {n_pass}/{n_total} tests passed")
    print("=" * 70)
