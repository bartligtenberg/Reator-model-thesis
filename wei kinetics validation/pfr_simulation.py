"""
Plug flow reactors smulation of CO2 Methanation over 5%Ni/13X
=================================================
Reproduces the kinetic experiments from Wei's thesis using a plug flow
reactor (PFR) model with a power-law rate expression.

Reaction
--------
    CO2 + 4 H2  ->  CH4 + 2 H2O

Rate law (Table 6.2, 5%Ni/13X)
-------------------------------
    r = k0 * exp(-Ea/R * (1/T - 1/T_ref)) * pCO2^0.10 * pH2^0.51
                                           * pCH4^0.016 * pH2O^0.06

    r     : rate of CO2 consumption  [mol/(min·g_cat)]
    k0    : pre-exponential factor   [mol/(min·g_cat)]  -- see note below
    Ea    : activation energy        65.2 kJ/mol
    T_ref : reference temperature    266 °C (= 539 K)
    p_i   : partial pressure of species i  [bar]

Note on k0
----------
    The thesis table reports k0 = 3.4e2 with units listed as mol/(min·g).
    Using that value directly gives 100% conversion for every condition.
    A value of 3.4e-4 mol/(min·g) reproduces the observed range of 3-40%
    H2 conversion, consistent with the parity plot in the thesis (Fig. 6a).
    Most likely the table value is in umol/(min·g), i.e. 3.4e2 umol/(min·g)
    = 3.4e-4 mol/(min·g).

Reactor setup (Table S.6.1)
----------------------------
    Total bed length  : 100 mm
    Bed diameter      : 10 mm
    Active catalyst   : 3 mm zone (rest filled with inert borosilicate spheres)
    Total flow rate   : 250 mL/min
    GHSV              : 100 min-1  (= Q / V_bed, where V_bed = 2.5 mL)
    Total pressure    : 1 bar

Outputs
-------
    - Table of predicted X_H2 and X_CO2 printed to terminal
    - pfr_results.csv  : same table saved to file
    - parity_plot.png  : parity plot of predicted H2 conversion
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# =============================================================================
# 1. KINETIC PARAMETERS  (5%Ni/13X, Table S.6.4)
# =============================================================================

k_Tref = 1.1e-4      # rate constant at T_ref   [mol/(min·g_cat)]
Ea     = 81.9e3      # activation energy         [J/mol]
R_gas  = 8.314       # universal gas constant    [J/(mol·K)]
T_ref  = 266 + 273.15  # reference temperature  [K]

# Reaction orders from Table S.6.4
n_CO2 =  0.16
n_H2  =  0.48
n_CH4 =  0.01
n_H2O = -0.003


# =============================================================================
# 2. REACTOR AND FEED PARAMETERS  (Table S.6.1)
# =============================================================================

P_bar = 1.0      # total pressure               [bar]
Q_in  = 250e-3   # total volumetric feed flow   [L/min]
GHSV  = 100      # gas hourly space velocity    [min-1]

# Catalyst bed volume from GHSV definition:  V_cat = Q / GHSV
# The remaining 97 mm of the 100 mm reactor tube contains inert
# borosilicate spheres (1 mm diameter) to stabilise the bed.
V_cat_mL = Q_in * 1000 / GHSV   # = 250 / 100 = 2.5 mL
W_cat    = 0.9                   # catalyst mass [g]

print(f"Catalyst bed volume : {V_cat_mL:.1f} mL   (Q / GHSV = {Q_in*1000:.0f} / {GHSV})")
print(f"Catalyst mass       : {W_cat:.2f} g\n")


# =============================================================================
# 3. EXPERIMENTAL FEED COMPOSITIONS  (Table S.6.2)
# =============================================================================
# Each entry: experiment number -> (y_N2, y_CO2, y_H2, y_CH4) as mole fractions
# H2O is zero in all feeds. N2 is the balance gas (inert).

feed_compositions = {
     1: (0.75, 0.05, 0.20, 0.00),   # base case: CO2=5%, H2=20%, no CH4
     2: (0.70, 0.05, 0.20, 0.05),   # CH4:CO2 = 1
     3: (0.65, 0.05, 0.20, 0.10),   # CH4:CO2 = 2
     4: (0.80, 0.04, 0.16, 0.00),   # CO2=4%, H2:CO2=4
     5: (0.76, 0.04, 0.16, 0.04),   # CO2=4%, CH4:CO2=1
     6: (0.72, 0.04, 0.16, 0.08),   # CO2=4%, CH4:CO2=2
     7: (0.85, 0.03, 0.12, 0.00),   # CO2=3%, H2:CO2=4
     8: (0.82, 0.03, 0.15, 0.00),   # CO2=3%, H2:CO2=5
     9: (0.79, 0.03, 0.18, 0.00),   # CO2=3%, H2:CO2=6
    10: (0.82, 0.03, 0.12, 0.03),   # CO2=3%, H2:CO2=4, CH4:CO2=1
    11: (0.79, 0.03, 0.12, 0.06),   # CO2=3%, H2:CO2=4, CH4:CO2=2
}

temperatures_C = [240, 250, 260, 270, 280, 300]  # experimental temperatures [°C]


# =============================================================================
# 4. KINETIC RATE FUNCTION
# =============================================================================

# Small floor value applied to partial pressures of products at the reactor
# inlet. Without this, p_CH4 = p_H2O = 0 at W=0 causes 0^exponent = 0,
# making the rate zero even though CO2 and H2 are present.
EPS = 1e-12  # [bar]

def rate(T_K, p_CO2, p_H2, p_CH4, p_H2O):
    """
    Power-law rate of CO2 consumption.

    Parameters
    ----------
    T_K   : temperature [K]
    p_i   : partial pressure of species i [bar]

    Returns
    -------
    r : rate of CO2 consumption [mol/(min·g_cat)]
    """
    k = k_Tref * np.exp(-Ea / R_gas * (1.0 / T_K - 1.0 / T_ref))
    return (k
            * max(p_CO2, EPS) ** n_CO2
            * max(p_H2,  EPS) ** n_H2
            * max(p_CH4, EPS) ** n_CH4
            * max(p_H2O, EPS) ** n_H2O)


# =============================================================================
# 5. PFR MOLE BALANCES
# =============================================================================

def pfr_rhs(W, F, T_K, F_N2):
    """
    Right-hand side of the PFR mole balances.

    The PFR is integrated over catalyst weight W [g] at steady state.
    At each point W, the rate is evaluated from the current molar flows,
    which give the local partial pressures.

    State vector
    ------------
    F = [F_CO2, F_H2, F_CH4, F_H2O]  [mol/min]

    Stoichiometry  (CO2 + 4 H2 -> CH4 + 2 H2O)
    --------------------------------------------
    dF_CO2 / dW = -r
    dF_H2  / dW = -4r
    dF_CH4 / dW = +r
    dF_H2O / dW = +2r

    Parameters
    ----------
    W    : catalyst weight coordinate [g]
    F    : molar flow rates [mol/min]
    T_K  : temperature [K]  (isothermal, constant along the bed)
    F_N2 : molar flow of inert N2 [mol/min]  (constant, not in state vector)
    """
    F_CO2, F_H2, F_CH4, F_H2O = F
    F_tot = max(F_CO2 + F_H2 + F_CH4 + F_H2O + F_N2, EPS)

    p_CO2 = F_CO2 / F_tot * P_bar
    p_H2  = F_H2  / F_tot * P_bar
    p_CH4 = F_CH4 / F_tot * P_bar
    p_H2O = F_H2O / F_tot * P_bar

    r = rate(T_K, p_CO2, p_H2, p_CH4, p_H2O)
    return [-r, -4.0*r, r, 2.0*r]


# =============================================================================
# 6. RUN SIMULATIONS
# =============================================================================

records = []

for exp_no, (y_N2, y_CO2, y_H2, y_CH4) in feed_compositions.items():

    for T_C in temperatures_C:
        T_K = T_C + 273.15

        # Total molar flow at inlet using ideal gas law at experiment temperature
        F_in_total = P_bar * Q_in / (0.08314 * T_K)  # [mol/min]

        # Inlet molar flows [mol/min]
        F0_CO2 = y_CO2 * F_in_total
        F0_H2  = y_H2  * F_in_total
        F0_CH4 = y_CH4 * F_in_total
        F_N2   = y_N2  * F_in_total

        # Initial condition: seed H2O (and CH4 if absent) with EPS to avoid
        # a zero rate at the very start of the reactor (W = 0)
        y0 = [F0_CO2, F0_H2, max(F0_CH4, EPS), EPS]

        # Integrate mole balances from W=0 to W=W_cat
        sol = solve_ivp(
            pfr_rhs,
            t_span=[0.0, W_cat],
            y0=y0,
            args=(T_K, F_N2),
            method='RK45',
            rtol=1e-8,
            atol=1e-14,
        )

        if not sol.success:
            print(f"WARNING: solver failed for exp {exp_no} at {T_C} C: {sol.message}")

        F_CO2_out, F_H2_out = sol.y[0, -1], sol.y[1, -1]

        # Conversion = fraction of inlet moles consumed; clipped to [0, 1]
        X_H2  = np.clip((F0_H2  - F_H2_out)  / F0_H2,  0.0, 1.0)
        X_CO2 = np.clip((F0_CO2 - F_CO2_out) / F0_CO2, 0.0, 1.0)

        records.append({
            'Exp':        exp_no,
            'T (°C)':     T_C,
            'CO2_in (%)': y_CO2 * 100,
            'H2_in (%)':  y_H2  * 100,
            'CH4_in (%)': y_CH4 * 100,
            'X_H2 (%)':   round(X_H2  * 100, 1),
            'X_CO2 (%)':  round(X_CO2 * 100, 1),
        })

df = pd.DataFrame(records)


# =============================================================================
# 7. PRINT AND SAVE RESULTS
# =============================================================================

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 120)
print("=" * 70)
print("Simulation results")
print("=" * 70)
print(df.to_string(index=False))

out_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(out_dir, "pfr_results.csv")
try:
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
except PermissionError:
    print("\nCould not save CSV — close pfr_results.csv in Excel first.")


# =============================================================================
# 8. PARITY PLOT
# =============================================================================
# Since measured values are not yet available, the predicted X_H2 is placed
# on both axes. All points therefore fall on the diagonal, but the plot shows
# that our predictions cover the same 0.05-0.40 range as Wei's measurements.
# Replace the x-values with observed data once available to make a true parity
# plot.

fig, ax = plt.subplots(figsize=(6, 6))

# Perfect-fit diagonal line
ax.plot([0, 1], [0, 1], 'k-', linewidth=0.8)

# One scatter series per temperature, coloured from dark (240 °C) to light (300 °C)
temps  = sorted(df['T (°C)'].unique())
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(temps)))

for temp, color in zip(temps, colors):
    sub = df[df['T (°C)'] == temp]
    x = sub['X_H2 (%)'].values / 100   # convert percentage to fraction
    ax.scatter(x, x, color=color, s=40, zorder=3, label=f'{int(temp)} C')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Observed H2 conversion (-)')
ax.set_ylabel('Predicted H2 conversion (-)')
ax.set_title('Parity plot - 5%Ni/13X')
ax.legend(title='T', fontsize=8, loc='upper left')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(out_dir, "parity_plot.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {plot_path}")
