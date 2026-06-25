"""
Moving Packed Bed (MPB) Reactor Model — Steady-State, Isothermal, Pseudo-Homogeneous
MOLAR FLUX FORM — BACKWARD SHOOTING BVP SOLVER
==========================================================================================

Counter-current flow:
    gas  : z = 0 (inlet, bottom)  ->  z = L (outlet, top)    u_g > 0
    solid: z = L (inlet, top)     ->  z = 0 (outlet, bottom)  u_s > 0 (magnitude)

State vector: y = [F_CO2, F_H2, F_CH4, F_H2O, q]

ODE system (in z, from 0 to L):
    dF_CO2/dz = -rho_bed_cat * r
    dF_H2 /dz = -4*rho_bed_cat * r
    dF_CH4/dz = +rho_bed_cat * r
    dF_H2O/dz =  2*rho_bed_cat*r - rho_bed_ads*K_LDF*(q*-q)
    dq/dz     = -K_LDF*(q*-q)/u_s          [solid moves in -z direction]

Boundary conditions:
    z=0: F_CO2=F_in, F_H2=F_in, F_CH4=F_in, F_H2O=0    [4 gas inlet BCs]
    z=L: q=0                                              [solid inlet: fresh]

WHY BACKWARD SHOOTING (not solve_bvp):
    K_LDF/u_s ~ 40,000 /m (for u_s=0.5 mm/s), so the q boundary layer at z=L
    has a width of ~25 µm. solve_bvp (collocation) cannot handle this — it
    exceeds max nodes and produces garbage. Instead:
    - Integrate BACKWARD in xi = L-z using BDF (stiff solver).
    - At xi=0 (z=L): q=0 (solid inlet BC known); F_i(L) are the 4 free parameters.
    - At xi=L (z=0): must recover the gas inlet BCs.
    - The backward q-ODE  dq/dxi = +K_LDF*(q*-q)/u_s  is STABLE (eigenvalue
      -K_LDF/u_s < 0 when linearised around q=q*), so BDF handles the thin
      boundary layer in the first few steps and coasts thereafter.
    - Use scipy.optimize.fsolve to find the 4 unknowns F_i(L).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq


# region 1. PARAMETERS
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
rho_p = 1400

W0_DA = 190.00e-6
E_DA  = 1190e3
n_DA  = 1.55

T_ref_K = 555.0
k_ref   = 3.46e-4
Ea_k    = 77.5e3
A_OH    = 0.50;  dH_OH  =  22.4e3
A_H2    = 0.44;  dH_H2  =  -6.2e3
A_mix   = 0.88;  dH_mix = -10.0e3
P_FLOOR = 1e-4

R_gas  = 8.314
MW_H2O = 0.018015

P_bar = 1.0
P_Pa  = P_bar * 1e5
y_CO2_in = 0.04
y_H2_in  = 0.16
y_CH4_in = 0.80

T_STP   = 273.15
GHSV    = 0.5
Q_STP   = GHSV * M_ads / 3600.0
u_g_STP = Q_STP / A_b
F_total_in = u_g_STP * P_Pa / (R_gas * T_STP)
F_in_CO2   = y_CO2_in * F_total_in
F_in_H2    = y_H2_in  * F_total_in
F_in_CH4   = y_CH4_in * F_total_in

U_S_LIST  = np.array([0.5, 1.0, 2.0, 3.0, 4.0]) * 1e-3
T_IN_LIST = [280]

print(f"MPB backward-shooting (isothermal): d={d_b*100:.0f} cm, L={L_b:.1f} m, "
      f"rho_bed_tot={rho_bed_tot:.0f} kg/m3, u_g_STP={u_g_STP*1e3:.1f} mm/s")
print(f"  F_in_total={F_total_in:.4f} mol/(m2·s)  "
      f"F_in_CO2={F_in_CO2:.4f}  F_in_H2={F_in_H2:.4f}")
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
    p_s  = np.clip(p, 1e-15, Psat*(1-1e-10))
    A    = R_gas/MW_H2O * T_K * np.log(Psat/p_s)
    A    = np.where((p <= 0) | (p >= Psat), 0.0, A)
    W    = W0 * np.exp(-np.minimum((A/E)**n, 500.0))
    qs   = rho_water(T_K) / MW_H2O * W
    return np.where(p <= 0, 0.0, qs)

def K_LDF_vec(T_K, p_arr, W0, E, n):
    D_M  = 2.5e-5 * (T_K / 300.0)**1.75
    p    = np.asarray(p_arr, dtype=float)
    dp   = 1.0 / 1e5
    dqsp = (q_star_vec(T_K, p + dp, W0, E, n)
            - q_star_vec(T_K, np.maximum(p - dp, 1e-15), W0, E, n)) / 2.0
    dqsp = np.maximum(dqsp, 1e-30)
    r_p  = 0.5 * d_p
    return 15.0 * eps_p * D_M / (r_p**2 * tau_p * rho_p * R_gas * T_K * dqsp)

def K_eq_sabatier(T_K):
    return 137.0 * T_K**(-3.994) * np.exp(158700.0 / (R_gas * T_K))

def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    vH   = lambda dH: np.exp(-dH/R_gas * (1.0/T_K - 1.0/T_ref_K))
    k    = k_ref * np.exp(-Ea_k/R_gas * (1.0/T_K - 1.0/T_ref_K))
    K_OH  = A_OH*vH(dH_OH);  K_H2 = A_H2*vH(dH_H2);  K_mix = A_mix*vH(dH_mix)
    K_eq  = K_eq_sabatier(T_K)
    p_CO2_s = np.maximum(p_CO2, P_FLOOR);  p_H2_s = np.maximum(p_H2, P_FLOOR)
    beta  = (p_CH4 * p_H2O**2) / (K_eq * p_CO2_s * p_H2_s**4)
    f_eq  = np.maximum(1.0 - np.where(np.isfinite(beta), beta, 1e10), 0.0)
    DEN   = (1.0 + K_OH * np.maximum(p_H2O, 0) / p_H2_s**0.5
             + K_H2 * p_H2_s**0.5 + K_mix * p_CO2_s**0.5)
    return k * (p_CO2_s * p_H2_s)**0.5 * f_eq / DEN**2 * 1000.0

def q_star(T_K, p_H2O):
    return q_star_vec(T_K, p_H2O, W0_DA, E_DA, n_DA)

_K_LDF_MAX = 20.0

def K_LDF(T_K, p_H2O):
    return np.minimum(K_LDF_vec(T_K, p_H2O, W0_DA, E_DA, n_DA), _K_LDF_MAX)

def equilibrium_conversion(T_K_val):
    K = K_eq_sabatier(T_K_val)
    def f(X):
        d = 1.0 - 0.08 * X
        return ((0.80 + 0.04*X)/d * (0.08*X/d)**2
                / ((0.04*(1-X)/d) * (0.16*(1-X)/d)**4 + 1e-100) - K)
    try:
        return brentq(f, 1e-9, 1-1e-9) * 100.0
    except Exception:
        return 100.0

def _partial_pressures(F_CO2, F_H2, F_CH4, F_H2O):
    F_tot = F_CO2 + F_H2 + F_CH4 + F_H2O
    if F_tot < 1e-30:
        return 0.0, 0.0, 0.0, 0.0
    return (F_CO2/F_tot*P_bar, F_H2/F_tot*P_bar,
            F_CH4/F_tot*P_bar, F_H2O/F_tot*P_bar)
# endregion


# region 3. BACKWARD SHOOTING SOLVER
# =============================================================================
def _noads_outlet(T_K):
    """No-adsorption forward IVP; returns F_i at z=L as initial guess for F_out."""
    def rhs(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r_s = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                     np.array([p_CH4]), np.array([p_H2O]))[0])
        return [-rho_bed_cat*r_s, -4.0*rho_bed_cat*r_s,
                +rho_bed_cat*r_s,  2.0*rho_bed_cat*r_s]
    sol = solve_ivp(rhs, [0.0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                    method='BDF', rtol=1e-6, atol=1e-10)
    return sol.y[:4, -1]


def solve_mpb_shoot(u_s, T_K, F_out_guess=None):
    """
    Solve counter-current MPB by backward shooting (BDF IVP + fsolve).

    Integrates in xi = L-z from xi=0 (z=L, solid inlet) to xi=L (z=0, gas inlet).
    Backward q-ODE: dq/dxi = +K_LDF*(q*-q)/u_s  — stable because eigenvalue
    around q=q* is -K_LDF/u_s < 0. BDF resolves the thin q-layer near xi=0
    automatically.

    Free parameters: F_i(z=L) = [F_CO2_L, F_H2_L, F_CH4_L, F_H2O_L] (4 unknowns).
    Target residuals: F_i(z=0) - [F_in_CO2, F_in_H2, F_in_CH4, 0] = 0.
    """
    def _backward_ode(xi, state):
        F_CO2_v = max(state[0], 0.0)
        F_H2_v  = max(state[1], 0.0)
        F_CH4_v = max(state[2], 0.0)
        F_H2O_v = max(state[3], 0.0)
        q_v     = max(state[4], 0.0)

        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_v, F_H2_v, F_CH4_v, F_H2O_v)
        r_s = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                     np.array([p_CH4]), np.array([p_H2O]))[0])
        qs  = float(q_star(T_K, np.array([p_H2O]))[0])
        Kl  = float(K_LDF(T_K, np.array([p_H2O]))[0])
        drv = Kl * (qs - q_v)

        # Signs are negated relative to forward direction (d/dxi = -d/dz):
        return [+rho_bed_cat * r_s,
                +4.0 * rho_bed_cat * r_s,
                -rho_bed_cat * r_s,
                -2.0 * rho_bed_cat * r_s + rho_bed_ads * drv,
                +drv / u_s]

    def _integrate(F_out, t_eval=None):
        ic = [max(F_out[0], 1e-20), max(F_out[1], 1e-20),
              max(F_out[2], 1e-20), max(F_out[3], 0.0), 0.0]
        kwargs = dict(method='BDF', rtol=1e-8, atol=1e-12)
        if t_eval is not None:
            kwargs['t_eval'] = t_eval
        return solve_ivp(_backward_ode, [0.0, L_b], ic, **kwargs)

    target = np.array([F_in_CO2, F_in_H2, F_in_CH4, 0.0])
    scale  = np.array([F_in_CO2, F_in_H2, F_in_CH4, F_in_CO2])

    def _residuals(F_out):
        sol = _integrate(F_out)
        if not sol.success:
            return np.full(4, 1.0)
        return (sol.y[:4, -1] - target) / scale

    if F_out_guess is None:
        F_out_guess = _noads_outlet(T_K)

    F_sol, info, ier, msg = fsolve(_residuals, F_out_guess,
                                   full_output=True, xtol=1e-9, ftol=1e-9)
    converged = (ier == 1)
    n_fev     = int(info['nfev'])
    res_norm  = float(np.max(np.abs(_residuals(F_sol))))

    if not converged:
        print(f"    [SHOOT] {msg}  (res={res_norm:.1e})")

    # Reconstruct fine grid (backward xi, then flip to z direction)
    xi_eval  = np.linspace(0.0, L_b, 400)
    sol_fine = _integrate(F_sol, t_eval=xi_eval)

    # Reverse: xi=0→z=L becomes index 0; xi=L→z=0 becomes index -1 → flip
    z_fine = L_b - sol_fine.t[::-1]
    y_rev  = sol_fine.y[:, ::-1]

    F_CO2f = np.maximum(y_rev[0], 0.0)
    F_H2f  = np.maximum(y_rev[1], 0.0)
    F_CH4f = np.maximum(y_rev[2], 0.0)
    F_H2Of = np.maximum(y_rev[3], 0.0)
    q_fine = np.maximum(y_rev[4], 0.0)

    F_totf  = np.maximum(F_CO2f + F_H2f + F_CH4f + F_H2Of, 1e-30)
    p_CO2f  = F_CO2f/F_totf*P_bar;  p_H2f  = F_H2f /F_totf*P_bar
    p_CH4f  = F_CH4f/F_totf*P_bar;  p_H2Of = F_H2Of/F_totf*P_bar
    r_fine  = reaction_rate_SI(T_K, p_CO2f, p_H2f, p_CH4f, p_H2Of)
    X_CO2   = np.clip(1.0 - F_CO2f / F_in_CO2, 0.0, 1.0)
    u_g_f   = F_totf * R_gas * T_K / P_Pa

    return dict(
        z=z_fine,
        C_CO2=F_CO2f/u_g_f, C_H2=F_H2f/u_g_f,
        C_CH4=F_CH4f/u_g_f, C_H2O=F_H2Of/u_g_f,
        F_H2O=F_H2Of,     # molar flux — used directly in H2O balance
        q=q_fine, r=r_fine, X_CO2=X_CO2,
        converged=converged, n_fev=n_fev, res_norm=res_norm,
        F_out=F_sol,      # warmstart for next u_s
    )
# endregion


# region 4. REFERENCE AND SOLVE LOOP
# =============================================================================
def _compute_noSE(T_K, N=300):
    """Fixed-bed isothermal reference (u_s=0, no sorption enhancement)."""
    def rhs(_z, y):
        F_CO2_l = max(y[0], 0.0);  F_H2_l  = max(y[1], 0.0)
        F_CH4_l = max(y[2], 0.0);  F_H2O_l = max(y[3], 0.0)
        p_CO2, p_H2, p_CH4, p_H2O = _partial_pressures(
            F_CO2_l, F_H2_l, F_CH4_l, F_H2O_l)
        r_s = float(reaction_rate_SI(T_K, np.array([p_CO2]), np.array([p_H2]),
                                     np.array([p_CH4]), np.array([p_H2O]))[0])
        return [-rho_bed_cat*r_s, -4.0*rho_bed_cat*r_s,
                +rho_bed_cat*r_s,  2.0*rho_bed_cat*r_s]
    z_grid = np.linspace(0, L_b, N)
    sol = solve_ivp(rhs, [0, L_b], [F_in_CO2, F_in_H2, F_in_CH4, 0.0],
                    method='BDF', rtol=1e-5, atol=1e-9, t_eval=z_grid)
    F_CO2_f = np.maximum(sol.y[0], 0.0)
    F_tot_f = np.maximum(sol.y[0]+sol.y[1]+sol.y[2]+sol.y[3], 1e-30)
    p_H2O_f = np.maximum(sol.y[3], 0.0) / F_tot_f * P_bar
    p_CO2_f = F_CO2_f / F_tot_f * P_bar
    p_H2_f  = np.maximum(sol.y[1], 0.0) / F_tot_f * P_bar
    p_CH4_f = np.maximum(sol.y[2], 0.0) / F_tot_f * P_bar
    r_f     = reaction_rate_SI(T_K, p_CO2_f, p_H2_f, p_CH4_f, p_H2O_f)
    X_f     = np.clip(1.0 - F_CO2_f / F_in_CO2, 0.0, 1.0)
    u_g_f   = F_tot_f * R_gas * T_K / P_Pa
    return dict(X_CO2_noSE=float(X_f[-1]),
                profile=dict(z=z_grid,
                             C_CO2=F_CO2_f/u_g_f,
                             C_H2O=np.maximum(sol.y[3], 0.0)/u_g_f,
                             q=q_star(T_K, p_H2O_f), r=r_f, X_CO2=X_f))

def _fmt(s):
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"


all_results  = {}
noSE_results = {}
n_total      = len(T_IN_LIST) * len(U_S_LIST)
n_done       = 0
t0_run       = time.perf_counter()

for T_C in T_IN_LIST:
    T_K = T_C + 273.15

    print(f"\n{'='*60}")
    print(f"  T_in = {T_C} C  |  u_g_STP = {u_g_STP*1e3:.1f} mm/s  |  backward shooting")
    print(f"{'='*60}")

    _noSE = _compute_noSE(T_K)
    noSE_results[T_C] = _noSE
    print(f"  non-SE fixed-bed conversion: {_noSE['X_CO2_noSE']*100:.1f}%")

    F_out_prev = None

    for i_us, u_s in enumerate(U_S_LIST):
        t0 = time.perf_counter()
        res = solve_mpb_shoot(u_s, T_K, F_out_guess=F_out_prev)
        dt = time.perf_counter() - t0
        n_done += 1
        elapsed = time.perf_counter() - t0_run
        eta = elapsed / n_done * (n_total - n_done)

        X_out = float(res['X_CO2'][-1]) * 100
        q_out = float(res['q'][0])
        tag   = "ok" if res['converged'] else "not-conv"

        print(f"  u_s={u_s*1e3:.4f} mm/s  X={X_out:.1f}%  q(0)={q_out:.3f}  "
              f"[{tag}, {res['n_fev']} fev, res={res['res_norm']:.1e}]"
              f"  ({dt:.1f}s, ETA {_fmt(eta)})")

        F_H2O_gas_out   = float(res['F_H2O'][-1])
        F_H2O_solid_out = u_s * rho_bed_ads * q_out
        F_H2O_prod      = 2.0 * F_in_CO2 * float(res['X_CO2'][-1])
        bal_err = ((F_H2O_gas_out + F_H2O_solid_out - F_H2O_prod)
                   / max(F_H2O_prod, 1e-30) * 100)
        print(f"    H2O balance [mmol/(m²·s)]:  produced={F_H2O_prod*1e3:.3f}  "
              f"gas_out={F_H2O_gas_out*1e3:.3f}  solid_out={F_H2O_solid_out*1e3:.3f}  "
              f"err={bal_err:+.1f}%")

        all_results[(T_C, i_us)] = {'res': res, 'u_s': u_s, 'T_K': T_K}

        if res['converged']:
            F_out_prev = res['F_out']

print(f"\nAll done.  Total: {_fmt(time.perf_counter() - t0_run)}")
# endregion


# region 5. POST-PROCESSING
# =============================================================================
def get_metrics(entry):
    res = entry['res']
    if res is None:
        return None
    T_K = entry['T_K']
    X_out       = float(res['X_CO2'][-1])
    q_out       = float(res['q'][0])
    p_H2O_peak  = float(np.max(res['C_H2O'])) * R_gas * T_K / 1e5
    q_star_peak = float(q_star(T_K, np.array([max(p_H2O_peak, 1e-8)]))[0])
    return dict(X_CO2=X_out, q_out=q_out,
                sorbent_util=q_out / max(q_star_peak, 1e-10))
# endregion


# region 6. PLOTS
# =============================================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def _savefig(fn):
    plt.savefig(os.path.join(SAVE_DIR, fn), dpi=150, bbox_inches='tight')

T_C_PROF = T_IN_LIST[0]
i_plot   = np.arange(len(U_S_LIST))
pal      = plt.cm.plasma(np.linspace(0.1, 0.85, len(i_plot)))
pal2     = plt.cm.viridis(np.linspace(0.1, 0.85, len(T_IN_LIST)))
markers  = ['o', 's', '^', 'D']
_p0      = noSE_results.get(T_C_PROF, {}).get('profile')

# Plot 1: Axial profiles
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f'MPB backward-shooting (isothermal)  T={T_C_PROF} C  counter-current', fontsize=11)
for k, i_us in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_us))
    if not e:
        continue
    r   = e['res']
    lbl = f"u_s = {e['u_s']*1e3:.3f} mm/s"
    axes[0,0].plot(r['z'], r['C_CO2']*1e3, color=pal[k], lw=2, label=lbl)
    axes[0,1].plot(r['z'], r['q'],          color=pal[k], lw=2, label=lbl)
    axes[1,0].plot(r['z'], r['X_CO2']*100,  color=pal[k], lw=2, label=lbl)
    axes[1,1].plot(r['z'], r['r']*1e3,      color=pal[k], lw=2, label=lbl)
if _p0 is not None:
    axes[0,0].plot(_p0['z'], _p0['C_CO2']*1e3, color='k', lw=2, ls='--', label='u_s=0')
    axes[0,1].plot(_p0['z'], _p0['q'],          color='k', lw=2, ls='--', label='u_s=0')
    axes[1,0].plot(_p0['z'], _p0['X_CO2']*100,  color='k', lw=2, ls='--', label='u_s=0')
    axes[1,1].plot(_p0['z'], _p0['r']*1e3,      color='k', lw=2, ls='--', label='u_s=0')
for ax, (yl, ti) in zip(axes.flat,
        [('C_CO2 [mmol/m3]','CO2 concentration'), ('q [mol/kg]','Solid H2O loading'),
         ('CO2 conversion [%]','CO2 conversion'), ('r [mmol/kg_cat/s]','Reaction rate')]):
    ax.set_xlabel('z [m]'); ax.set_ylabel(yl); ax.set_title(ti)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'shoot_plot1_axial_T{T_C_PROF}C.png'); plt.show()

# Plot 2: Conversion vs u_s
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    us_ok, X_ok = [], []
    for i_us in range(len(U_S_LIST)):
        e = all_results.get((T_C, i_us))
        if e:
            m = get_metrics(e)
            if m:
                us_ok.append(e['u_s']*1e3); X_ok.append(m['X_CO2']*100)
    if us_ok:
        ax.semilogx(us_ok, X_ok, marker=markers[j % 4], color=pal2[j],
                    lw=2, ms=6, label=f'{T_C} C (MPB)')
        ax.axhline(equilibrium_conversion(T_C+273.15), color=pal2[j],
                   lw=1, ls=':', label=f'{T_C} C eq.')
        if T_C in noSE_results:
            ax.axhline(noSE_results[T_C]['X_CO2_noSE']*100, color=pal2[j],
                       lw=1.5, ls='--', label=f'{T_C} C u_s=0')
ax.set_xlabel('u_s [mm/s]'); ax.set_ylabel('CO2 conversion [%]')
ax.set_title('MPB backward-shooting (isothermal) — conversion vs solid velocity')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
plt.tight_layout()
_savefig('shoot_plot2_conversion_vs_us.png'); plt.show()

# Plot 3: Sorbent utilisation
fig, ax = plt.subplots(figsize=(9, 5))
for j, T_C in enumerate(T_IN_LIST):
    us_ok, util_ok = [], []
    for i_us in range(len(U_S_LIST)):
        e = all_results.get((T_C, i_us))
        if e:
            m = get_metrics(e)
            if m:
                us_ok.append(e['u_s']*1e3); util_ok.append(m['sorbent_util']*100)
    if us_ok:
        ax.semilogx(us_ok, util_ok, marker=markers[j % 4], color=pal2[j],
                    lw=2, ms=6, label=f'{T_C} C')
ax.axhline(100, color='grey', lw=1.5, ls='--', label='q=q* (saturated)')
ax.set_xlabel('u_s [mm/s]'); ax.set_ylabel('q(z=0)/q*(p_H2O_max) [%]')
ax.set_title('MPB backward-shooting (isothermal) — sorbent utilisation')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
_savefig('shoot_plot3_sorbent_util.png'); plt.show()

# Plot 4: H2O profiles
T_K_prof = T_C_PROF + 273.15
fig, (ax_q, ax_h) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'H2O profiles  T={T_C_PROF} C  backward-shooting isothermal', fontsize=11)
for k, i_us in enumerate(i_plot):
    e = all_results.get((T_C_PROF, i_us))
    if not e:
        continue
    r   = e['res']
    lbl = f"u_s = {e['u_s']*1e3:.3f} mm/s"
    ax_q.plot(r['z'], r['q'],                          color=pal[k], lw=2, label=lbl)
    ax_h.plot(r['z'], r['C_H2O']*R_gas*T_K_prof/1e2,  color=pal[k], lw=2, label=lbl)
if _p0 is not None:
    ax_q.plot(_p0['z'], _p0['q'],                         color='k', lw=2, ls='--', label='u_s=0')
    ax_h.plot(_p0['z'], _p0['C_H2O']*R_gas*T_K_prof/1e2, color='k', lw=2, ls='--', label='u_s=0')
ax_q.set_xlabel('z [m]'); ax_q.set_ylabel('q [mol/kg]')
ax_q.set_title('Solid H2O loading'); ax_q.legend(fontsize=7); ax_q.grid(True, alpha=0.3)
ax_h.set_xlabel('z [m]'); ax_h.set_ylabel('p_H2O [mbar]')
ax_h.set_title('Gas H2O partial pressure'); ax_h.legend(fontsize=7); ax_h.grid(True, alpha=0.3)
plt.tight_layout()
_savefig(f'shoot_plot4_H2O_T{T_C_PROF}C.png'); plt.show()
# endregion
