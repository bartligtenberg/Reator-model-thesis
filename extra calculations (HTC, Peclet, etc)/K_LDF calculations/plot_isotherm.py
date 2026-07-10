"""
DA adsorption isotherm for H2O on 13X zeolite — Mette (2014) parameters
========================================================================
Panel 1 : q* [mol/kg]         vs p_H2O [mbar], linear x-axis 0-300 mbar
Panel 2 : dq*/dp [mol/kg/bar] vs p_H2O [mbar], linear x-axis 0-300 mbar
Panel 3 : X  [g/kg]           vs p_H2O [mbar], log x-axis — matches Mette Fig 2 left
Panel 4 : K_LDF [1/s]         vs p_H2O [mbar], log x-axis (MPB model parameters)
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
R_gas   = 8.314       # J/(mol·K)
MW_H2O  = 0.018015    # kg/mol
rho_ads = 998.2       # kg/m³  (liquid water)

# ── DA isotherm parameters — Mette (2014) ───────────────────────────────────
W0 = 341.00e-6    # m³/kg
E  = 1192.25e3    # J/kg
n  = 1.55

# ── Antoine equation for p_sat [bar] (NIST, valid ~274–441 K) ───────────────
def p_sat_bar(T_K):
    return 10.0 ** (5.40221 - 1838.675 / (T_K - 31.737))

# ── DA equilibrium loading [mol/kg] ─────────────────────────────────────────
def q_star(T_K, p_bar):
    p = np.asarray(p_bar, dtype=float)
    Psat = p_sat_bar(T_K)
    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A = (R_gas / MW_H2O) * T_K * np.log(Psat / p_safe)
    W = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    qs = rho_ads / MW_H2O * W
    return np.where(p <= 0.0, 0.0, qs)

# ── Pellet parameters (MPB model, Bareschino setup) ─────────────────────────
eps_p     = 0.615
tau_p     = 3.0
rho_p     = 1400.0    # kg/m³
K_LDF_MAX = 20.0      # 1/s  (cap to prevent singularity at flat isotherm)

def D_mol(T_K):
    return 2.5e-5 * (T_K / 300.0) ** 1.75   # m²/s, Chapman-Enskog scaling

def K_LDF(T_K, p_bar, d_p):
    r_p    = 0.5 * d_p
    D_M    = D_mol(T_K)
    dqs    = np.maximum(dqstar_dp(T_K, p_bar), 1e-30)  # mol/(kg·bar)
    dqs_Pa = dqs / 1e5                                   # convert to mol/(kg·Pa)
    k      = 15.0 * eps_p * D_M / (r_p**2 * tau_p * rho_p * R_gas * T_K * dqs_Pa)
    return np.minimum(k, K_LDF_MAX)

particle_cases = [
    (0.75e-3, "-",  "0.75 mm"),
    (3.00e-3, "--", "3.00 mm"),
]

# ── dq*/dp via central finite difference [mol/kg/bar] ───────────────────────
def dqstar_dp(T_K, p_bar):
    dp = 1e-6   # bar  (1e-4 mbar, small enough for accurate FD)
    p = np.asarray(p_bar, dtype=float)
    p_lo = np.maximum(p - dp, 1e-15)
    return (q_star(T_K, p + dp) - q_star(T_K, p_lo)) / (2 * dp)

# ── Temperatures ─────────────────────────────────────────────────────────────
temps = [(280 + 273.15, "280 °C"), (300 + 273.15, "300 °C")]
colors = ["#1f77b4", "#d62728"]

# x-axis arrays
p_mbar_lin = np.linspace(0.01, 300, 800)          # linear, for panels 1 & 2
p_bar_lin  = p_mbar_lin / 1e3
p_mbar_log = np.logspace(np.log10(0.1), np.log10(300), 800)  # log, for panel 3
p_bar_log  = p_mbar_log / 1e3

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 14))

for (T_K, label), color in zip(temps, colors):
    qs_lin = q_star(T_K, p_bar_lin)
    qs_log = q_star(T_K, p_bar_log)
    X_g_kg = qs_log * MW_H2O * 1e3          # mol/kg → g_H2O / kg_sorbent

    ax1.plot(p_mbar_lin, qs_lin,  color=color, lw=2, label=label)
    ax2.plot(p_mbar_log, dqstar_dp(T_K, p_bar_log), color=color, lw=2, label=label)
    ax3.plot(p_mbar_log, X_g_kg,  color=color, lw=2, label=label)

    for d_p, ls, dp_label in particle_cases:
        kldf = K_LDF(T_K, p_bar_log, d_p)
        ax4.plot(p_mbar_log, kldf, color=color, lw=2, ls=ls,
                 label=f"{label}, $d_p$ = {dp_label}")

# ── Panel 1: q* [mol/kg], linear x ───────────────────────────────────────────
ax1.set_ylabel("$q^*$ [mol/kg]", fontsize=12)
ax1.set_xlim(0, 300)
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=9)
ax1.set_title("DA isotherm — H$_2$O on 13X zeolite (Mette 2014)", fontsize=12)
ax1.set_xlabel("$p_{H_2O}$ [mbar]", fontsize=11)
ax1.grid(True, alpha=0.3)

# ── Panel 2: dq*/dp, linear x ────────────────────────────────────────────────
ax2.set_xlabel("$p_{H_2O}$ [mbar]", fontsize=11)
ax2.set_ylabel("$dq^*/dp$ [mol kg$^{-1}$ bar$^{-1}$]", fontsize=12)
ax2.set_xscale("log")
ax2.set_xlim(0.1, 300)
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=9)
ax2.set_title("Isotherm slope (thermodynamic factor in $K_{LDF}$)", fontsize=12)
ax2.grid(True, alpha=0.3, which="both")

# ── Panel 3: X [g/kg], log x — cf. Mette Fig 2 left ─────────────────────────
ax3.set_xlabel("$p_{H_2O}$ [mbar]", fontsize=11)
ax3.set_ylabel("$X$ [g$_{H_2O}$ kg$_{sorbent}^{-1}$]", fontsize=12)
ax3.set_xscale("log")
ax3.set_xlim(0.1, 300)
ax3.set_ylim(bottom=0)
ax3.legend(fontsize=9)
ax3.set_title("Same isotherm in g/kg on log scale — cf. Mette (2014) Fig. 2 left", fontsize=12)
ax3.grid(True, alpha=0.3, which="both")

# secondary y-axis in kg/kg
ax3b = ax3.twinx()
ax3b.set_ylabel("$X$ [kg$_{H_2O}$ kg$_{sorbent}^{-1}$]", fontsize=11, color="grey")
ax3b.tick_params(axis="y", labelcolor="grey")
# scale the right axis to match: kg/kg = g/kg / 1000
y_top = ax3.get_ylim()[1]
ax3b.set_ylim(0, y_top / 1e3)

# ── Panel 4: K_LDF [1/s], log x ──────────────────────────────────────────────
ax4.set_xlabel("$p_{H_2O}$ [mbar]", fontsize=11)
ax4.set_ylabel("$K_{LDF}$ [s$^{-1}$]", fontsize=12)
ax4.set_xscale("log")
ax4.set_xlim(0.1, 300)
ax4.set_ylim(bottom=0)
ax4.legend(fontsize=9)
ax4.set_title(f"LDF coefficient ($\\rho_p$ = {rho_p:.0f} kg/m³, solid=0.75 mm, dashed=3.00 mm)", fontsize=12)
ax4.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("isotherm_280_300C.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: isotherm_280_300C.png")
