"""
Fit Ligtenberg (2026) DA isotherm parameters (W0, E, n) to Wei et al.
breakthrough capacity data using differential evolution on the analytical
DA isotherm.  Works because the system is equilibrium-controlled (MTZ << L_bed).
After finding the optimum, the full column simulation is run to verify.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# ── geometry / feed (must match adsorption_simulation copy.py) ──────────────
d_b      = 0.010
L_b      = 0.100
A_b      = np.pi / 4 * d_b**2
V_bed    = A_b * L_b
m_cat    = 6.5e-3
rho_bed  = m_cat / V_bed
eps_b    = 0.40
d_p      = 0.75e-3
y_H2O_in = 5.0 / 95.0
P_bar    = 1.0
P_Pa     = P_bar * 1e5
Q_STP    = 100e-6 / 60
T_STP    = 273.15
u_STP    = Q_STP / A_b
R_gas    = 8.314
MW_H2O   = 0.018015
rho_ads  = 998.2
eps_p    = 0.6
tau_p    = 3.0
BT_FRACTION = 0.10
N  = 30
dz = L_b / (N - 1)

# ── experimental data ────────────────────────────────────────────────────────
wei_T   = np.array([260, 280, 300, 320], dtype=float)   # °C
wei_cap = np.array([1.56, 1.27, 1.0, 0.80])             # mmol/g

p_in = y_H2O_in * P_bar   # H2O partial pressure at inlet [bar]

# ── thermodynamic helpers ────────────────────────────────────────────────────
def P_sat_bar(T_K):
    return 10.0 ** (5.40221 - 1838.675 / (T_K - 31.737))

def q_star_scalar(T_K, p_bar, W0, E, n):
    Psat  = P_sat_bar(T_K)
    p_s   = np.clip(p_bar, 1e-15, Psat * (1 - 1e-10))
    A     = (R_gas / MW_H2O) * T_K * np.log(Psat / p_s)
    W     = W0 * np.exp(-min((A / E) ** n, 500.0))
    return rho_ads / MW_H2O * W   # mol/kg = mmol/g

def q_star_vec(T_K, p_arr, W0, E, n):
    p    = np.asarray(p_arr, dtype=float)
    Psat = P_sat_bar(T_K)
    p_s  = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_r  = (R_gas / MW_H2O) * T_K * np.log(Psat / p_s)
    A    = np.where((p <= 0) | (p >= Psat), 0.0, A_r)
    W    = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    return np.where(p <= 0, 0.0, rho_ads / MW_H2O * W)

def K_LDF_vec(T_K, p_arr, W0, E, n):
    D_M = 2.5e-5 * (T_K / 300.0) ** 1.75
    p   = np.asarray(p_arr, dtype=float)
    dp  = 1.0 / 1e5
    dqs = (q_star_vec(T_K, p + dp, W0, E, n)
           - q_star_vec(T_K, np.maximum(p - dp, 1e-15), W0, E, n)) / 2.0
    dqs = np.maximum(dqs, 1e-30)
    return 15.0 * D_M * MW_H2O * eps_p / (0.5 * d_p**2 * tau_p * rho_ads * R_gas * T_K * dqs)

# ── step 1 : fast analytical fit ─────────────────────────────────────────────
def obj_analytical(params):
    W0, E, n = params
    sse = sum(
        (q_star_scalar(T + 273.15, p_in, W0, E, n) - cap) ** 2
        for T, cap in zip(wei_T, wei_cap)
    )
    return sse

print("Step 1 — fitting DA isotherm analytically (instant) …")
res = differential_evolution(
    obj_analytical,
    bounds=[(50e-6, 500e-6), (700e3, 1600e3), (1.0, 3.5)],
    seed=42, maxiter=2000, tol=1e-10, popsize=20,
    mutation=(0.5, 1.0), recombination=0.7, disp=False,
)
W0_opt, E_opt, n_opt = res.x

print(f"\n  Best-fit Ligtenberg (2026) parameters:")
print(f"    W0 = {W0_opt*1e6:.2f}e-6  m³/kg")
print(f"    E  = {E_opt/1e3:.2f}e3   J/kg")
print(f"    n  = {n_opt:.4f}")
print(f"    Analytical SSE = {res.fun:.6f}")

print("\n  Analytical check:")
for T_C, cap_exp in zip(wei_T, wei_cap):
    cap_sim = q_star_scalar(T_C + 273.15, p_in, W0_opt, E_opt, n_opt)
    print(f"    {T_C:.0f}°C  sim={cap_sim:.3f}  exp={cap_exp:.3f}  err={cap_sim-cap_exp:+.3f}")

# ── step 2 : verify with full column simulation ───────────────────────────────
def rhs_column(t, y, T_K, u, C_in, W0, E, n):
    C = np.maximum(y[:N], 0.0)
    q = np.maximum(y[N:], 0.0)
    p = C * R_gas * T_K / 1e5
    dqdt = K_LDF_vec(T_K, p, W0, E, n) * (q_star_vec(T_K, p, W0, E, n) - q)
    C_up = np.concatenate([[C_in], C[:-1]])
    dCdt = (-u * (C - C_up) / dz - rho_bed * dqdt) / eps_b
    return np.concatenate([dCdt, dqdt])

def simulate_capacity(W0, E, n, T_C):
    T_K  = T_C + 273.15
    u    = u_STP * (T_K / T_STP)
    C_in = y_H2O_in * P_Pa / (R_gas * T_K)
    qs0  = q_star_scalar(T_K, p_in, W0, E, n)
    t_end = min(5.0 * qs0 * m_cat / (C_in * u * A_b), 3e4) if C_in > 0 else 3e4

    def bt_event(t, y, T_K, u, C_in, W0, E, n):
        return y[N - 1] - BT_FRACTION * C_in
    bt_event.terminal  = True
    bt_event.direction = 1

    sol = solve_ivp(rhs_column, [0.0, t_end], np.zeros(2 * N),
                    args=(T_K, u, C_in, W0, E, n),
                    method='BDF', events=bt_event, rtol=1e-4, atol=1e-8)
    if not sol.success or len(sol.t_events[0]) == 0:
        return np.nan
    return float(np.mean(sol.y[N:, -1]))

print("\nStep 2 — full column verification …")
print(f"  {'T [°C]':>6}  {'sim [mmol/g]':>12}  {'exp [mmol/g]':>12}  {'err':>8}")
for T_C, cap_exp in zip(wei_T, wei_cap):
    cap_sim = simulate_capacity(W0_opt, E_opt, n_opt, T_C)
    print(f"  {T_C:>6.0f}  {cap_sim:>12.3f}  {cap_exp:>12.3f}  {cap_sim-cap_exp:>+8.3f}")

# ── step 3 : refine using the column simulation ───────────────────────────────
# Analytical fit minimizes against q*(T,p_in); column gives ~8-10% less at
# 10% BT (front isn't fully through the bed).  Nelder-Mead from the
# analytical solution converges in ~50-80 evaluations.

print("\nStep 3 -- refining with full column simulation (Nelder-Mead) ...")

from scipy.optimize import minimize

eval_count = [0]

def obj_column(params):
    W0, E, n = params
    if W0 <= 0 or E <= 0 or n <= 0:
        return 1e10
    sse = 0.0
    for T_C, cap_exp in zip(wei_T, wei_cap):
        cap_sim = simulate_capacity(W0, E, n, T_C)
        if np.isnan(cap_sim):
            return 1e10
        sse += (cap_sim - cap_exp) ** 2
    eval_count[0] += 1
    print(f"  eval {eval_count[0]:3d}  W0={W0*1e6:.1f}e-6  E={E/1e3:.1f}k  n={n:.3f}  SSE={sse:.5f}")
    return sse

res2 = minimize(
    obj_column,
    x0=[W0_opt, E_opt, n_opt],
    method='Nelder-Mead',
    options={'xatol': 1e-8, 'fatol': 1e-6, 'maxiter': 500, 'adaptive': True},
)

W0_col, E_col, n_col = res2.x
print(f"\n  Best-fit Ligtenberg (2026) parameters (column-fitted):")
print(f"    W0 = {W0_col*1e6:.2f}e-6  m3/kg")
print(f"    E  = {E_col/1e3:.2f}e3   J/kg")
print(f"    n  = {n_col:.4f}")
print(f"    Column SSE = {res2.fun:.6f}")

print(f"\n  Column verification:")
print(f"  {'T [C]':>6}  {'sim [mmol/g]':>12}  {'exp [mmol/g]':>12}  {'err':>8}")
for T_C, cap_exp in zip(wei_T, wei_cap):
    cap_sim = simulate_capacity(W0_col, E_col, n_col, T_C)
    print(f"  {T_C:>6.0f}  {cap_sim:>12.3f}  {cap_exp:>12.3f}  {cap_sim-cap_exp:>+8.3f}")

print(f"\nCopy into adsorption_simulation copy.py:")
print(f"    'Ligtenberg (2026)': {{'W0': {W0_col:.4e}, 'E': {E_col:.4e}, 'n': {n_col:.4f}}},")

