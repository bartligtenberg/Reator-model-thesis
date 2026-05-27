"""
Full-Scale SEM Column Model  —  1D Non-Isothermal Transient
============================================================

Scaled up from Wei's lab tube to Bareschino's pilot geometry:
    d = 50 mm,  L = 2 m,  d_p = 2.5 mm

Flow is set via GHSV = 0.5 m³_STP / (kg_cat · h), matching Bareschino.
All kinetics, isotherm, and feed composition are identical to the Wei model.

State vector  (6 × N values — one value per axial node)
---------------------------------------------------------
    y[0 : N]     C_CO2  [mol/m³]   gas-phase CO2 concentration
    y[N : 2N]    C_H2   [mol/m³]   gas-phase H2 concentration
    y[2N : 3N]   C_CH4  [mol/m³]   gas-phase CH4 concentration
    y[3N : 4N]   C_H2O  [mol/m³]   gas-phase H2O concentration
    y[4N : 5N]   q      [mol/kg]   solid-phase H2O loading
    y[5N : 6N]   T      [K]        local bed temperature
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# region 1. PARAMETERS
# =============================================================================
# 1. PARAMETERS
# =============================================================================

# --------------- Bed geometry (Bareschino pilot scale) -----------------------
d_b   = 0.050               # bed diameter                        [m]
L_b   = 2.000               # bed length                          [m]
A_b   = np.pi / 4 * d_b**2  # cross-sectional area                [m²]
V_bed = A_b * L_b            # total bed volume                    [m³]
eps_b = 0.4                # void fraction between particles      [-]

# Separate catalyst and adsorbent masses (Bareschino Table 3)
M_cat = 0.064               # catalyst mass (Ni/Ce component)     [kg]
M_ads = 1.22                # adsorbent mass (zeolite 13X)        [kg]
rho_bed_cat = M_cat / V_bed # catalyst bulk density               [kg_cat / m³_bed]
rho_bed_ads = M_ads / V_bed # adsorbent bulk density              [kg_ads / m³_bed]
rho_bed_tot = (M_cat + M_ads) / V_bed  # total (for energy balance solid Cp)

# --------------- Particle properties (Bareschino 2.5 mm pellets) -------------
d_p   = 0.75e-3              # particle diameter                   [m]
eps_p = 0.615               # intraparticle void fraction          [-]  (Bareschino Table S1)
tau_p = 3.0                 # tortuosity factor                    [-]

# rho_ads is now temperature-dependent — see rho_water(T_K) in region 2

# --------------- DA isotherm parameters: Mette (2014) ------------------------
W0_DA = 341.00e-6           # micropore volume                     [m³/kg_sorbent]
E_DA  = 1192.25e3           # characteristic adsorption energy     [J/kg]
n_DA  = 1.55                # DA heterogeneity parameter           [-]

# --------------- Operating conditions ----------------------------------------
T_LIST = [280, 300, 320]                                  # inlet temperatures [°C]
P_bar = 1.0                 # total pressure                       [bar]
P_Pa  = P_bar * 1e5         # total pressure                       [Pa]

# --------------- Feed composition (Bareschino 2022: CO2=4%, H2=16%, CH4=80%) --
y_CO2_in = 0.04
y_H2_in  = 0.16
y_CH4_in = 0.80
y_N2_in  = 0.00

# --------------- Gas flow: GHSV = 0.5 m³_STP/(kg_ads·h)  (Bareschino) -------
GHSV  = 0.5                  # m³_STP / (kg_ads · h)              [m³/(kg·h)]
T_STP = 273.15               # STP temperature                     [K]
Q_STP = GHSV * M_ads / 3600  # volumetric flow at STP             [m³/s]
u_STP = Q_STP / A_b          # superficial velocity at STP         [m/s]

# --------------- Physical constants ------------------------------------------
R_gas  = 8.314
MW_H2O = 0.018015

# --------------- LHHW kinetic parameters (Koschany et al. 2016, Table 6) -----
T_ref_K = 555.0
k_ref   = 3.46e-4
Ea_k    = 77.5e3
A_OH    = 0.50;  dH_OH  =  22.4e3
A_H2    = 0.44;  dH_H2  =  -6.2e3
A_mix   = 0.88;  dH_mix = -10.0e3
P_FLOOR = 1e-4

# --------------- Thermal parameters ------------------------------------------
dH_r   = -165.0e3           # heat of Sabatier reaction            [J / mol_CO2]
dH_ads = -45.0e3            # isosteric heat of H2O adsorption     [J / mol_H2O]  (Bareschino Table 3)
Cp_cat = 1100.0             # solid heat capacity                  [J / (kg · K)]  (Bareschino Table 3)

# Molar heat capacities at 573 K (300 °C) — NIST Shomate equations  [J / (mol · K)]
Cp_CO2 = 45.4
Cp_H2  = 29.3
Cp_CH4 = 46.9
Cp_H2O = 34.2
Cp_N2  = 29.5

# Wall heat-transfer coefficient × specific area [W/(m³·K)]
# U_a = h_wall × (4/d_b).  Keeping h_wall = 500 W/(m²·K):
#   Wei  (d=10 mm): 500 × 400 = 200 000
#   Here (d=50 mm): 500 ×  80 =  40 000  — larger tube is less well cooled
U_a = 8000              # [W / (m³ · K)]

# --------------- Spatial discretisation --------------------------------------
N  = 100
dz = L_b / (N - 1)
z_m = np.linspace(0, L_b, N)   # [m] for plots

print(f"Bed: d = {d_b*100:.0f} cm,  L = {L_b:.1f} m,  V_bed = {V_bed*1e3:.2f} L")
print(f"M_cat = {M_cat:.3f} kg,  M_ads = {M_ads:.3f} kg,  total = {M_cat+M_ads:.3f} kg")
print(f"rho_bed_cat = {rho_bed_cat:.1f}  rho_bed_ads = {rho_bed_ads:.1f}  rho_bed_tot = {rho_bed_tot:.1f} kg/m³")
print(f"GHSV  = {GHSV:.1f} m³/(kg_ads·h)  (Q = {Q_STP*3600*1e3:.1f} L/h)")
print(f"Q_STP = {Q_STP*1e3:.3f} L/s  =  {Q_STP*60*1e3:.2f} L/min")
print(f"u_STP = {u_STP:.4f} m/s")


# endregion

# region 2. THERMODYNAMIC AND KINETIC FUNCTIONS
# =============================================================================
# 2. THERMODYNAMIC AND KINETIC FUNCTIONS
# =============================================================================

def P_sat_bar(T_K):
    """Saturation vapour pressure of water [bar] — Kowalska & Ambrozek (2017), Bareschino Eq. S.17."""
    log10_p_mmHg = (29.8605 - 3.1522e3 / T_K
                    - 7.3037 * np.log10(T_K)
                    + 2.4247e-9 * T_K
                    + 1.8090e-6 * T_K**2)
    return 10.0**log10_p_mmHg * 133.322e-5   # mmHg → bar


def rho_water(T_K):
    """Temperature-dependent liquid water density [kg/m³] — Schaefer & Thess (2018), Bareschino Eq. S.16."""
    return 996.0 / (1.0 + 2.0e-3 * (T_K - 298.15))


def q_star_vec(T_K, p_arr, W0, E, n):
    """Equilibrium H2O loading [mol/kg] — Dubinin-Astakhov isotherm."""
    p    = np.asarray(p_arr, dtype=float)
    Psat = P_sat_bar(T_K)
    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_raw  = (R_gas / MW_H2O) * T_K * np.log(Psat / p_safe)
    A  = np.where((p <= 0.0) | (p >= Psat), 0.0, A_raw)
    W  = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    qs = rho_water(T_K) / MW_H2O * W
    return np.where(p <= 0.0, 0.0, qs)


def K_LDF_vec(T_K, p_arr, W0, E, n):
    """LDF mass-transfer coefficient [1/s]."""
    D_M    = 2.5e-5 * (T_K / 300.0) ** 1.75
    p      = np.asarray(p_arr, dtype=float)
    dp_bar = 1.0 / 1e5
    p_lo = np.maximum(p - dp_bar, 1e-15)
    p_hi = p + dp_bar
    dqstar_dp = (q_star_vec(T_K, p_hi, W0, E, n)
                 - q_star_vec(T_K, p_lo, W0, E, n)) / 2.0
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)
    return (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_water(T_K) * R_gas * T_K * dqstar_dp))


def K_eq_sabatier(T_K):
    """
    Equilibrium constant for CO2 + 4H2 → CH4 + 2H2O [dimensionless, p in bar].
    Koschany et al. (2016):  K_eq = 137 · T^(−3.994) · exp(158.7 kJ/mol / RT)
    """
    return 137.0 * T_K**(-3.994) * np.exp(158700.0 / (R_gas * T_K))


def reaction_rate_SI(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    """
    LHHW CO2 methanation rate [mol / (kg_cat · s)].
    Koschany et al. (2016).  T_K may be a scalar or an array of length N.
    """
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

    DEN = (1.0
           + K_OH  * np.maximum(p_H2O, 0.0) / p_H2_s**0.5
           + K_H2  * p_H2_s**0.5
           + K_mix * p_CO2_s)

    r_g_s = k * (p_CO2_s * p_H2_s)**0.5 * f_eq / DEN**2
    return r_g_s * 1000.0   # mol/(g·s) → mol/(kg·s)


# endregion

# region 4. RIGHT-HAND SIDE (NON-ISOTHERMAL)
# =============================================================================
# 4. RIGHT-HAND SIDE OF THE ODE SYSTEM  (non-isothermal, 6 × N equations)
# =============================================================================

def rhs_sem_noniso(t, y, se_on, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O,
                   T_in, T_wall):
    """
    Time derivatives for the non-isothermal SEM column (6N equations).

    Parameters
    ----------
    se_on    : bool   — True = SE mode (adsorption active)
    u        : float  — superficial gas velocity at inlet T [m/s]
    C_in_*   : float  — inlet concentrations [mol/m³] at inlet T
    T_in     : float  — inlet / feed temperature [K]
    T_wall   : float  — wall / furnace temperature [K]
    """
    # --- Unpack and clip state vector ---
    C_CO2 = np.maximum(y[0*N : 1*N], 0.0)
    C_H2  = np.maximum(y[1*N : 2*N], 0.0)
    C_CH4 = np.maximum(y[2*N : 3*N], 0.0)
    C_H2O = np.maximum(y[3*N : 4*N], 0.0)
    q     = np.maximum(y[4*N : 5*N], 0.0)
    T     = np.maximum(y[5*N : 6*N], 200.0)

    # --- Partial pressures using LOCAL temperature ---
    p_CO2 = C_CO2 * R_gas * T / 1e5
    p_H2  = C_H2  * R_gas * T / 1e5
    p_CH4 = C_CH4 * R_gas * T / 1e5
    p_H2O = C_H2O * R_gas * T / 1e5

    # --- Rates with LOCAL T ---
    r    = reaction_rate_SI(T, p_CO2, p_H2, p_CH4, p_H2O)
    qs   = q_star_vec(T, p_H2O, W0_DA, E_DA, n_DA)
    Kl   = K_LDF_vec( T, p_H2O, W0_DA, E_DA, n_DA)
    dqdt = Kl * (qs - q) if se_on else np.zeros(N)

    # --- Upwind mass balances ---
    C_CO2_up = np.concatenate([[C_in_CO2], C_CO2[:-1]])
    C_H2_up  = np.concatenate([[C_in_H2],  C_H2[:-1]])
    C_CH4_up = np.concatenate([[C_in_CH4], C_CH4[:-1]])
    C_H2O_up = np.concatenate([[C_in_H2O], C_H2O[:-1]])

    dCdt_CO2 = (-u * (C_CO2 - C_CO2_up) / dz  +  rho_bed_cat * (-1) * r) / eps_b
    dCdt_H2  = (-u * (C_H2  - C_H2_up)  / dz  +  rho_bed_cat * (-4) * r) / eps_b
    dCdt_CH4 = (-u * (C_CH4 - C_CH4_up) / dz  +  rho_bed_cat * (+1) * r) / eps_b
    dCdt_H2O = (-u * (C_H2O - C_H2O_up) / dz  +  rho_bed_cat * (+2) * r
                                                 -  rho_bed_ads * dqdt) / eps_b

    # --- Energy balance ---
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
    Q_wall = -U_a * (T - T_wall)

    dTdt = (-u * rho_g_mol * Cp_mix * (T - T_up) / dz
            + Q_rxn + Q_ads + Q_wall) / Cp_eff

    return np.concatenate([dCdt_CO2, dCdt_H2, dCdt_CH4, dCdt_H2O, dqdt, dTdt])


# endregion

# region 5. SOLVE
# =============================================================================
# 5. SOLVE
# =============================================================================

all_results = {}

p_H2O_max = 2 * y_CO2_in * P_bar / (1 - 2 * y_CO2_in)

atol_vec = np.concatenate([
    1e-8 * np.ones(4 * N),
    1e-8 * np.ones(N),
    1e-2 * np.ones(N),
])

for T_C in T_LIST:
    T_K      = T_C + 273.15
    u        = u_STP * (T_K / T_STP)
    C_in_CO2 = y_CO2_in * P_Pa / (R_gas * T_K)
    C_in_H2  = y_H2_in  * P_Pa / (R_gas * T_K)
    C_in_CH4 = y_CH4_in * P_Pa / (R_gas * T_K)
    C_in_H2O = 0.0

    q_at_max  = float(q_star_vec(T_K, np.array([p_H2O_max]), W0_DA, E_DA, n_DA)[0])
    F_CO2_in  = C_in_CO2 * u * A_b
    t_sat_est = q_at_max * M_ads / (2.0 * F_CO2_in)
    t_end     = min(2.5 * t_sat_est, 7200.0)

    print("=" * 60)
    print(f"  Full-scale SEM — T_in = {T_C} °C,  P = {P_bar} bar,  U_a = {U_a:.0f} W/(m³·K)")
    print(f"  Feed:  CO2 = {y_CO2_in:.1%},  H2 = {y_H2_in:.0%},  "
          f"CH4 = {y_CH4_in:.1%},  N2 = {y_N2_in:.0%}")
    print(f"  GHSV:  {Q_STP*3600/M_ads:.2f} m³/(kg_ads·h)  =  {Q_STP*3600*1e6/(M_ads*1e3):.0f} mL/(g_ads·h)")
    print(f"  t_sat_est ≈ {t_sat_est/60:.0f} min  →  t_end = {t_end/60:.0f} min")

    y0 = np.zeros(6 * N)
    y0[0*N : 1*N] = C_in_CO2
    y0[1*N : 2*N] = C_in_H2
    y0[2*N : 3*N] = C_in_CH4
    y0[5*N : 6*N] = T_K

    results = {}
    for se_on in [True, False]:
        tag = "SE on  (reaction + adsorption)" if se_on else "SE off (reaction only)    "
        print(f"  Solving {tag} ...", end="", flush=True)
        y0_run = y0.copy()
        if not se_on:
            y0_run[4*N : 5*N] = q_at_max
        sol = solve_ivp(
            rhs_sem_noniso,
            t_span=[0.0, t_end],
            y0=y0_run,
            args=(se_on, u, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O, T_K, T_K),
            method='BDF',
            rtol=1e-4,
            atol=atol_vec,
            dense_output=True,
        )
        print(f"  {'OK' if sol.success else 'FAILED — ' + sol.message}")
        if not sol.success:
            raise RuntimeError(f"ODE solver did not converge ({tag}): {sol.message}")
        results[se_on] = sol

    all_results[T_C] = {
        'results':   results,
        'T_K':       T_K,
        'C_in_CO2':  C_in_CO2,
        'q_at_max':  q_at_max,
        't_sat_est': t_sat_est,
        't_end':     t_end,
    }

print("=" * 60)


# endregion

# region 6. POST-PROCESSING
# =============================================================================
# 6. POST-PROCESSING
# =============================================================================

def extract_outlet(sol, C_in_CO2_loc):
    """Extract outlet CO2 conversion, H2O pressure, and peak bed temperature."""
    t_arr     = sol.t
    y_arr     = sol.sol(t_arr)
    C_CO2_out = np.maximum(y_arr[N - 1,     :], 0.0)
    C_H2O_out = np.maximum(y_arr[4*N - 1,   :], 0.0)
    T_profile = y_arr[5*N : 6*N,            :]
    T_out     = T_profile[-1, :]
    T_max     = T_profile.max(axis=0)

    X_CO2      = np.clip((C_in_CO2_loc - C_CO2_out) / C_in_CO2_loc, 0.0, 1.0)
    p_H2O_mbar = C_H2O_out * R_gas * T_out / 1e5 * 1000

    return t_arr, X_CO2, p_H2O_mbar, T_max


# endregion

# region 7. PLOT — time-series (4 panels per temperature)
# =============================================================================
# 7. PLOT  — time-series
# =============================================================================
if True:
    n_rows = len(T_LIST)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5 * n_rows), squeeze=False)
    fig.suptitle(
        f'Full-scale SEM  —  d = {d_b*100:.0f} cm, L = {L_b:.0f} m  |  '
        f'P = {P_bar} bar,  U_a = {U_a:.0f} W/(m³·K)\n'
        f'Feed: {y_CO2_in:.1%} CO₂ / {y_H2_in:.0%} H₂ / {y_CH4_in:.1%} CH₄ / '
        f'{y_N2_in:.0%} N₂  —  GHSV = {GHSV:.1f} m³/(kg·h)',
        fontsize=11
    )

    for row, T_C in enumerate(T_LIST):
        data         = all_results[T_C]
        results      = data['results']
        C_in_CO2_row = data['C_in_CO2']
        q_max_row    = data['q_at_max']

        t_on,  X_on,  pH2O_on,  Tmax_on  = extract_outlet(results[True],  C_in_CO2_row)
        t_off, X_off, pH2O_off, Tmax_off = extract_outlet(results[False], C_in_CO2_row)

        sol_on   = results[True]
        sol_off  = results[False]
        t_snaps  = np.linspace(sol_on.t[1], sol_on.t[-1], 5)
        snap_col = plt.cm.plasma(np.linspace(0.15, 0.85, len(t_snaps)))

        ax1, ax2, ax3, ax4 = axes[row]

        ax1.plot(t_on  / 60, X_on  * 100, color='tab:blue',   lw=2.5, label='SE on')
        ax1.plot(t_off / 60, X_off * 100, color='tab:orange', lw=2.5, ls='--', label='SE off')
        ax1.set_xlabel('Time [min]'); ax1.set_ylabel('CO₂ conversion [%]')
        ax1.set_title(f'Outlet CO₂ conversion — T_in = {T_C} °C')
        ax1.set_ylim(0, 105); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2.plot(t_on  / 60, Tmax_on  - (T_C + 273.15), color='tab:blue',   lw=2.5, label='SE on')
        ax2.plot(t_off / 60, Tmax_off - (T_C + 273.15), color='tab:orange', lw=2.5, ls='--', label='SE off')
        ax2.set_xlabel('Time [min]'); ax2.set_ylabel('ΔT_max [K]')
        ax2.set_title(f'Peak hot-spot above T_in — {T_C} °C')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        for i, t_s in enumerate(t_snaps):
            y_s = sol_on.sol(t_s)
            q_s = np.maximum(y_s[4*N : 5*N], 0.0)
            ax3.plot(z_m, q_s, color=snap_col[i], lw=2.0, label=f't = {t_s/60:.0f} min')
        ax3.axhline(q_max_row, color='grey', ls=':', lw=1.0, label=f'q* = {q_max_row:.2f} mol/kg')
        ax3.set_xlabel('Bed position z [m]'); ax3.set_ylabel('q  [mol/kg]')
        ax3.set_title(f'Solid-phase H₂O loading (SE on) — {T_C} °C')
        ax3.legend(fontsize=8, loc='upper left'); ax3.grid(True, alpha=0.3)

        y_on_arr  = sol_on.sol(sol_on.t)
        y_off_arr = sol_off.sol(sol_off.t)
        T_out_on  = y_on_arr[6*N - 1, :]
        T_out_off = y_off_arr[6*N - 1, :]
        to_mbar = lambda C, T_loc: np.maximum(C, 0.0) * R_gas * T_loc / 1e5 * 1000

        for blk, label, col in [(0,'CO₂','tab:blue'), (1,'H₂','tab:green'),
                                 (3,'H₂O','tab:purple')]:
            C_on  = y_on_arr[ (blk+1)*N - 1, :]
            C_off = y_off_arr[(blk+1)*N - 1, :]
            ax4.plot(sol_on.t  / 60, to_mbar(C_on,  T_out_on),  color=col, lw=2.0,
                     label=f'{label} (SE on)')
            ax4.plot(sol_off.t / 60, to_mbar(C_off, T_out_off), color=col, lw=2.0,
                     ls='--', label=f'{label} (SE off)')
        ax4.set_xlabel('Time [min]')
        ax4.set_ylabel('Outlet partial pressure [mbar]')
        ax4.set_title(f'Outlet species — T_in = {T_C} °C')
        ax4.legend(fontsize=7, ncol=2)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    plt.show()
# endregion

# region 7b. PLOT — CO2 conversion + hot-spot vs temperature
# =============================================================================
# 7b. PLOT — CO2 conversion vs temperature  +  peak ΔT vs temperature
# =============================================================================
from scipy.optimize import brentq


def equilibrium_conversion(T_K_val):
    """Equilibrium CO2 conversion for Bareschino's feed (4% CO2, 16% H2, 80% CH4) at 1 bar."""
    K = K_eq_sabatier(T_K_val)
    def f(X):
        d     = 1.0 - 0.08 * X
        p_CO2 = 0.04 * (1 - X) / d
        p_H2  = 0.16 * (1 - X) / d
        p_CH4 = (0.80 + 0.04 * X) / d
        p_H2O = 0.08 * X / d
        return p_CH4 * p_H2O**2 / (p_CO2 * p_H2**4 + 1e-100) - K
    try:
        return brentq(f, 1e-9, 1 - 1e-9) * 100
    except Exception:
        return 100.0


T_arr    = np.array(T_LIST, dtype=float)
X_off_ss = []
X_on_ini = []
dT_off   = []
dT_on    = []

for T_C in T_LIST:
    data         = all_results[T_C]
    results      = data['results']
    T_K_row      = data['T_K']
    C_in_CO2_row = data['C_in_CO2']
    t_sat_row    = data['t_sat_est']

    t_off, X_off, _, Tmax_off = extract_outlet(results[False], C_in_CO2_row)
    t_on,  X_on,  _, Tmax_on  = extract_outlet(results[True],  C_in_CO2_row)

    mid = max(1, len(X_off) // 2)
    X_off_ss.append(float(np.mean(X_off[mid:])) * 100)
    dT_off.append(float(np.mean(Tmax_off[mid:])) - T_K_row)

    mask = (t_on >= 0.10 * t_sat_row) & (t_on <= 0.40 * t_sat_row)
    if mask.sum() == 0:
        mask = np.ones(len(t_on), dtype=bool)
    X_on_ini.append(float(np.mean(X_on[mask])) * 100)
    dT_on.append(float(np.mean(Tmax_on[mask])) - T_K_row)

T_fine = np.linspace(170, 370, 120)
X_eq   = [equilibrium_conversion(T + 273.15) for T in T_fine]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
fig.suptitle(
    f'5%Ni2.5%Ce13X  —  Full-scale SEM  (d = {d_b*100:.0f} cm, L = {L_b:.0f} m)\n'
    f'GHSV = {GHSV:.1f} m³/(kg·h),  P = {P_bar} bar,  '
    f'U_a = {U_a:.0f} W/(m³·K)  ({"adiabatic" if U_a == 0 else "cooled"})\n'
    f'Feed: {y_CO2_in:.1%} CO₂ / {y_H2_in:.0%} H₂ / {y_CH4_in:.1%} CH₄',
    fontsize=10
)

ax_top.plot(T_fine, X_eq,     'k--',  lw=1.5, label='Equilibrium (isothermal ref.)')
ax_top.plot(T_arr,  X_off_ss, 'ko--', lw=2.0, ms=7, label='Non-SE (steady state)')
ax_top.plot(T_arr,  X_on_ini, 'r^-',  lw=2.0, ms=7, label='SE (fresh sorbent)')
ax_top.set_ylabel('CO₂ conversion [%]', fontsize=12)
ax_top.set_ylim(0, 105)
ax_top.legend(fontsize=10)
ax_top.grid(True, alpha=0.3)

ax_bot.plot(T_arr, dT_off, 'ko--', lw=2.0, ms=7, label='Non-SE')
ax_bot.plot(T_arr, dT_on,  'r^-',  lw=2.0, ms=7, label='SE (fresh sorbent)')
ax_bot.axhline(0, color='grey', lw=0.8, ls=':')
ax_bot.set_xlabel('Inlet temperature [°C]', fontsize=12)
ax_bot.set_ylabel('Peak ΔT  (T_max − T_in)  [K]', fontsize=12)
ax_bot.set_title('Hot-spot magnitude')
ax_bot.legend(fontsize=10)
ax_bot.grid(True, alpha=0.3)

ax_bot.set_xlim(170, 370)
plt.tight_layout()
plt.show()
# endregion

# region 7c. PLOT — Bareschino Fig. 7 style: species mole fractions vs time
# =============================================================================
# 7c. PLOT — outlet mole fractions over time for T = 280, 300, 320 °C (SE on)
# =============================================================================
T_PLOT3 = [280, 300, 320]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex='col')
fig.suptitle(
    f'Full-scale SEM (SE on)  —  d = {d_b*100:.0f} cm, L = {L_b:.0f} m\n'
    f'Feed: {y_CO2_in:.1%} CO₂ / {y_H2_in:.0%} H₂ / {y_CH4_in:.1%} CH₄  —  '
    f'GHSV = {GHSV:.1f} m³/(kg·h)',
    fontsize=11
)

for col, T_C in enumerate(T_PLOT3):
    data         = all_results[T_C]
    sol_on       = data['results'][True]
    C_in_CO2_row = data['C_in_CO2']

    t_arr = sol_on.t
    y_arr = sol_on.sol(t_arr)

    C_CO2_out = np.maximum(y_arr[  N - 1, :], 0.0)
    C_H2_out  = np.maximum(y_arr[2*N - 1, :], 0.0)
    C_CH4_out = np.maximum(y_arr[3*N - 1, :], 0.0)
    C_H2O_out = np.maximum(y_arr[4*N - 1, :], 0.0)
    T_out     = np.maximum(y_arr[6*N - 1, :], 200.0)

    C_tot = np.maximum(C_CO2_out + C_H2_out + C_CH4_out + C_H2O_out, 1e-15)
    y_CO2 = C_CO2_out / C_tot
    y_H2  = C_H2_out  / C_tot
    y_CH4 = C_CH4_out / C_tot
    y_H2O = C_H2O_out / C_tot
    X_CO2 = np.clip((C_in_CO2_row - C_CO2_out) / C_in_CO2_row, 0.0, 1.0)
    t_min = t_arr / 60

    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    ax_top.plot(t_min, y_CH4, 'k-',  lw=2, label='$y_{CH_4}$')
    ax_top.plot(t_min, X_CO2, 'k--', lw=2, label='$X_{CO_2}$')
    ax_top.set_ylim(0, 1.1)
    ax_top.set_title(f'T = {T_C} °C', fontsize=11)
    ax_top.text(0.05, 0.08, chr(ord('a') + col), transform=ax_top.transAxes,
                fontsize=13, fontweight='bold')
    if col == 0:
        ax_top.set_ylabel('$y_{CH_4}$,  $X_{CO_2}$  [–]', fontsize=10)
    ax_top.legend(fontsize=9, loc='lower left')
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(t_min, y_CO2, color='red',   lw=2, label='$y_{CO_2}$')
    ax_bot.plot(t_min, y_H2,  color='green', lw=2, label='$y_{H_2}$')
    ax_bot.plot(t_min, y_H2O, color='blue',  lw=2, label='$y_{H_2O}$')
    ax_bot.set_ylim(0, 0.12)
    ax_bot.set_xlabel('time [min]', fontsize=10)
    if col == 0:
        ax_bot.set_ylabel('$y_{CO_2}$, $y_{H_2}$, $y_{H_2O}$  [–]', fontsize=10)
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# endregion

# region 7d. PLOT — Axial temperature profile at T_in = 300 °C (SE on)
# =============================================================================
# 7d. PLOT — T(z) snapshots for SE on at T_in = 300 °C
# =============================================================================
T_C_prof     = 300
sol_prof     = all_results[T_C_prof]['results'][True]
t_max_avail  = sol_prof.t[-1]
t_snaps_min  = [5, 15, 30, 50, 90, 120]
snap_colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(t_snaps_min)))

fig, ax = plt.subplots(figsize=(9, 5))
for i, t_min in enumerate(t_snaps_min):
    t_s = t_min * 60.0
    if t_s > t_max_avail:
        print(f"  t = {t_min} min exceeds solver range ({t_max_avail/60:.0f} min) — skipped")
        continue
    T_prof = sol_prof.sol(t_s)[5*N : 6*N]
    ax.plot(z_m, T_prof - 273.15, color=snap_colors[i], lw=2.0, label=f't = {t_min} min')

ax.axhline(T_C_prof, color='grey', lw=1.0, ls=':', label=f'T_in = {T_C_prof} °C')
ax.set_xlabel('Bed position  z  [m]', fontsize=12)
ax.set_ylabel('Temperature  [°C]', fontsize=12)
ax.set_title(f'Axial temperature profile — T_in = {T_C_prof} °C, SE on', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# endregion

# region 8. BREAKTHROUGH TIMES
# =============================================================================
# 8. H2O BREAKTHROUGH TIMES (SE on)
#    Defined as first time outlet y_H2O exceeds 0.5 % of total gas flow.
# =============================================================================
# Threshold = 10 % of equilibrium y_H2O at each process temperature (Bareschino definition)
print("\n" + "=" * 60)
print("  H2O breakthrough times  (y_H2O_outlet ≥ 10% of y_H2O_eq)")
print("=" * 60)
for T_C in T_LIST:
    T_K_bt    = T_C + 273.15
    X_eq_bt   = equilibrium_conversion(T_K_bt) / 100.0          # 0–1
    y_H2O_eq  = 0.08 * X_eq_bt / (1.0 - 0.08 * X_eq_bt)        # equilibrium y_H2O
    threshold = 0.10 * y_H2O_eq

    data      = all_results[T_C]
    sol_on    = data['results'][True]
    t_arr     = sol_on.t
    y_arr     = sol_on.sol(t_arr)
    C_H2O_out = np.maximum(y_arr[4*N - 1, :], 0.0)
    T_out_bt  = np.maximum(y_arr[6*N - 1, :], 200.0)
    y_H2O_out = C_H2O_out * R_gas * T_out_bt / P_Pa

    idx = np.where(y_H2O_out >= threshold)[0]
    if len(idx) > 0:
        t_bt = t_arr[idx[0]] / 60
        print(f"  T_in = {T_C:3d} °C  |  y_H2O_eq = {y_H2O_eq:.4f},  "
              f"threshold = {threshold:.4f}  →  t_BT ≈ {t_bt:.1f} min")
    else:
        print(f"  T_in = {T_C:3d} °C  |  threshold = {threshold:.4f}  →  "
              f"no breakthrough within {t_arr[-1]/60:.0f} min")
print("=" * 60)
# endregion
