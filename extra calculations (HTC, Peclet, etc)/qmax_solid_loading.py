"""
Required solid loading at 100% CO2 conversion — MPB steady state
=================================================================

At steady state, the overall H2O balance across the whole bed is:

    F_H2O_produced = F_H2O_gas_out + u_s * rho_bed_ads * q(z=0)

where:
    F_H2O_produced  = 2 * F_in_CO2        [mol/(m²·s)]  — 2 mol H2O per mol CO2 (Sabatier)
    F_H2O_gas_out   = H2O leaving in exit gas  [mol/(m²·s)]
    u_s             = solid velocity       [m/s]
    rho_bed_ads     = M_ads / V_bed        [kg/m³]
    q(z=0)          = solid loading at solid outlet  [mol/kg]

If ALL produced H2O is captured by the solid (F_H2O_gas_out = 0), the solid exit
loading is at its theoretical maximum:

    q_max = 2 * F_in_CO2 / (u_s * rho_bed_ads)

This is an upper bound. If the model gives q(z=0) < q_max, the difference is
carried out in the gas phase. q(z=0) > q_max would violate the mass balance.

Bed geometry and feed composition match the Bareschino lab-scale setup used in
the MPB reactor model (d=5 cm, L=2 m, M_ads=1.22 kg, 4% CO2 / 16% H2 / 80% CH4).
"""

import numpy as np

# ── Fixed bed / feed parameters (Bareschino lab setup) ──────────────────────
d_b      = 0.050          # bed diameter [m]
L_b      = 2.000          # bed length [m]
A_b      = np.pi / 4 * d_b**2
V_bed    = A_b * L_b

M_ads    = 1.22           # sorbent mass [kg]
rho_bed_ads = M_ads / V_bed  # [kg/m³]

y_CO2_in = 0.04           # CO2 mole fraction at inlet
T_STP    = 273.15         # [K]
P_Pa     = 1.0e5          # operating pressure [Pa]
R_gas    = 8.314          # [J/(mol·K)]


def q_max_100pct(GHSV, u_s_mms, verbose=True):
    """
    Calculate the maximum solid exit loading for 100% CO2 conversion.

    Parameters
    ----------
    GHSV     : float  — gas hourly space velocity [NL / (g_ads · h)]
    u_s_mms  : float  — solid velocity [mm/s]
    verbose  : bool   — print step-by-step breakdown

    Returns
    -------
    q_max : float  [mol / kg_ads]
    """
    u_s = u_s_mms * 1e-3  # mm/s → m/s

    # Superficial gas velocity at STP
    Q_STP    = GHSV * (M_ads * 1e3) / 3600.0   # NL/s  (M_ads in g)
    u_g_STP  = Q_STP * 1e-3 / A_b              # m/s   (NL → m³)

    # Total molar flux at inlet (temperature-independent at STP)
    F_total  = u_g_STP * P_Pa / (R_gas * T_STP)  # mol/(m²·s)
    F_in_CO2 = y_CO2_in * F_total                 # mol/(m²·s)

    # H2O production at 100% conversion: CO2 + 4H2 → CH4 + 2H2O
    F_H2O_produced = 2.0 * F_in_CO2              # mol/(m²·s)

    # Mass balance: all H2O adsorbed onto solid
    q_max = F_H2O_produced / (u_s * rho_bed_ads)

    if verbose:
        print("=" * 55)
        print(f"  GHSV      = {GHSV} NL/(g_ads·h)")
        print(f"  u_s       = {u_s_mms} mm/s")
        print("-" * 55)
        print(f"  A_b         = {A_b*1e4:.4f} cm²")
        print(f"  V_bed       = {V_bed*1e6:.1f} cm³")
        print(f"  rho_bed_ads = {rho_bed_ads:.1f} kg/m³")
        print(f"  u_g_STP     = {u_g_STP*1e3:.2f} mm/s")
        print(f"  F_total_in  = {F_total:.4f} mol/(m2*s)")
        print(f"  F_in_CO2    = {F_in_CO2:.4f} mol/(m2*s)")
        print(f"  F_H2O_prod  = 2 x {F_in_CO2:.4f} = {F_H2O_produced:.4f} mol/(m2*s)")
        print(f"  u_s x rho_ads = {u_s:.4f} x {rho_bed_ads:.1f} = {u_s*rho_bed_ads:.4f} mol/(m2*s) per mol/kg")
        print("-" * 55)
        print(f"  q_max(z=0)  = {F_H2O_produced:.4f} / {u_s*rho_bed_ads:.4f}")
        print(f"              = {q_max:.4f} mol/kg_ads")
        print("=" * 55)

    return q_max


if __name__ == "__main__":
    # Default case from MPB model
    print("\nDefault case (GHSV = 0.5 NL/g/h):")
    for u_s in [0.5, 1.0, 2.0, 3.0, 4.0]:
        q = q_max_100pct(GHSV=0.5, u_s_mms=u_s, verbose=False)
        print(f"  u_s = {u_s} mm/s  ->  q_max = {q:.4f} mol/kg")

    print()
    # Detailed breakdown for u_s = 3 mm/s
    q_max_100pct(GHSV=0.5, u_s_mms=3.0)
