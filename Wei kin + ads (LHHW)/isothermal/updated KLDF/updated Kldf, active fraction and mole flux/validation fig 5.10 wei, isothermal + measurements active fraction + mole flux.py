"""
Validation — Wei (2022) Figure 5.10  (model + measurements overlay, molar-flux form)
======================================================================================

Reproduces the sorption-enhanced CO2 methanation breakthrough curve from
Wei's thesis Fig. 5.10:
  "Typical water breakthrough capacity and duration of bifunctional
   catalyst-sorbent 5%Ni2.5%Ce13X. Experiment at 240 °C, P = 1 bar,
   GHSV = 923 mL/(g_cat·h). Feed: 10 mL/min H2, 2.5 mL/min CO2,
   81.5 mL/min CH4, 6 mL/min N2."

Same coupled SEM column model as "validation fig 5.10 wei, isothermal +
measurements active fraction.py", but the gas velocity is no longer held
constant along the bed. CO2 + 4H2 -> CH4 + 2H2O is mole-reducing
(5 -> 3 mol), and H2O is further removed from the gas phase by adsorption,
so total gas moles shrink as the front moves down the column. Since the
column is isothermal and isobaric, ideal-gas law pins the total gas
concentration C_total = P/(RT) constant in z and t — so local velocity
must drop to compensate, via a quasi-steady total-molar-flux closure
(see _compute_fields). The previous constant-u model silently violated
this conservation law at high conversion/adsorption; this version
restores it as an (approximately, up to solver tolerance) exact
invariant of the ODE system.

Measured data digitised from measurements Wei.png (Wei Fig. 5.10) are
overlaid as scatter markers on the model lines.

Plot (dual y-axis, matching Fig. 5.10):
    Left y-axis  :  H2, CO2, H2O outlet mol%          (0 – 10 %)
    Right y-axis :  CH4 outlet mol% & CO2 conversion  (−10 – 110 %)
                    CO2 equilibrium conversion (flat dashed line)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# 1. PARAMETERS  (identical to SEM LHHW.py)
# =============================================================================

# Bed geometry — Wei (2022) Fig. 5.10 lab setup: 5%Ni2.5%Ce13X, 6.5 g packed bed
d_b     = 0.010                    # bed diameter                [m]
L_b     = 0.100                    # bed length                  [m]
A_b     = np.pi / 4 * d_b**2      # cross-sectional area        [m²]
V_bed   = A_b * L_b                # total bed volume            [m³]
m_cat_total     = 6.5e-3           # total bed material mass     [kg]  (Wei Fig. 5.10, 6.5 g — GHSV basis)
active_fraction = 0.10              # fraction of m_cat_total that is catalytically active (Ni) [-]
rho_bed = m_cat_total / V_bed      # full bulk density — all material adsorbs (sorbent) [kg/m³]
rho_cat = active_fraction * rho_bed  # catalytically active density, used for reaction rate only [kg/m³]
eps_b   = 0.40                     # void fraction               [-]

# Particle transport properties — Bareschino et al. (2023) Table 1, 13X zeolite pellets
d_p     = 0.75e-3                  # particle diameter           [m]
eps_p   = 0.6                    # intraparticle void fraction [-]
tau_p   = 4.0                      # tortuosity                  [-]

rho_ads = 998.2                    # liquid water density        [kg/m³]
MW_H2O  = 0.018015                 # molar mass water            [kg/mol]
R_gas   = 8.314                    # gas constant                [J/(mol·K)]

# DA isotherm parameters — fitted to Wei (2022) 300 °C H2O breakthrough; Mette (2014) W0=340 overestimates
W0_DA   = 150.00e-6                # micropore volume            [m³/kg]
E_DA    = 1190e3                # characteristic energy       [J/kg]
n_DA    = 1.55                     # heterogeneity exponent      [-]

# Operating conditions
T_C     = 260
T_K     = T_C + 273.15
P_bar   = 1.0
P_Pa    = P_bar * 1e5

# Feed composition (Wei Fig. 5.10: 10/2.5/81.5/6 mL/min, total 100 mL/min)
y_CO2_in = 0.025
y_H2_in  = 0.100
y_CH4_in = 0.815
y_N2_in  = 0.060

Q_STP   = 100e-6 / 60             # volumetric flow STP         [m³/s]
T_STP   = 273.15                   # STP temperature             [K]
u_in    = (Q_STP / A_b) * (T_K / T_STP)  # inlet superficial velocity (boundary condition) [m/s]

# LHHW kinetic parameters (Koschany et al. 2016)
T_ref_K = 555.0
k_ref   = 3.46e-4
Ea_k    = 77.5e3
A_OH    = 0.50;  dH_OH   = 22.4e3
A_H2    = 0.44;  dH_H2   = -6.2e3
A_mix   = 0.88;  dH_mix  = -10.0e3

P_FLOOR = 1e-4                     # partial-pressure floor      [bar]

# Total-molar-flux closure safeguards (node velocity is derived, not solved for
# directly — clip the implied total flux to keep it positive and bounded so a
# transient BDF/Newton trial iterate can't produce backflow or a blow-up)
U_FLOOR_FRAC = 1e-3                 # floor on F_tot_rightface, as a fraction of F_total_in [-]
U_CEIL_FRAC  = 10.0                 # defensive ceiling, same units                          [-]

# Spatial discretisation
N  = 50
dz = L_b / (N - 1)


# =============================================================================
# 2. THERMODYNAMIC AND KINETIC FUNCTIONS  (copied unchanged from SEM LHHW.py)
# =============================================================================

def P_sat_bar(T):
    # Antoine equation for water (T in K, returns bar)
    return 10.0 ** (5.40221 - 1838.675 / (T - 31.737))


def q_star_vec(T, p_arr, W0, E, n):
    # Dubinin-Astakhov isotherm: q* = rho_liq/MW * W0*exp(-(A/E)^n)
    # A = (R/MW)*T*ln(Psat/p) is the adsorption potential [J/kg]
    p      = np.asarray(p_arr, dtype=float)
    Psat   = P_sat_bar(T)
    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_raw  = (R_gas / MW_H2O) * T * np.log(Psat / p_safe)
    A      = np.where((p <= 0.0) | (p >= Psat), 0.0, A_raw)
    W      = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    return np.where(p <= 0.0, 0.0, rho_ads / MW_H2O * W)


def K_LDF_vec(T, p_arr, W0, E, n):
    # Glueckauf LDF coefficient: K_LDF = 15*D_eff / (r_p^2 * dq*/dC)
    # dq*/dp obtained by numerical central difference to handle the nonlinear DA isotherm
    # D_M: molecular diffusivity of H2O vapour, Chapman-Enskog T^1.75 scaling (Bareschino 2023)
    D_M       = 3.36e-9 * T_K**1.75
    p         = np.asarray(p_arr, dtype=float)
    dp_bar    = 1.0 / 1e5
    p_lo      = np.maximum(p - dp_bar, 1e-15)
    p_hi      = p + dp_bar
    dqstar_dp = (q_star_vec(T, p_hi, W0, E, n)
                 - q_star_vec(T, p_lo, W0, E, n)) / 2.0
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)
    return (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_ads * R_gas * T * dqstar_dp))


def K_eq_sabatier(T):
    # Equilibrium constant for CO2 + 4H2 ⇌ CH4 + 2H2O — fit from Koschany et al. (2016)
    return 137.0 * T**(-3.994) * np.exp(158700.0 / (R_gas * T))


def reaction_rate_SI(T, p_CO2, p_H2, p_CH4, p_H2O):
    vH      = lambda dH: np.exp(-dH / R_gas * (1.0 / T - 1.0 / T_ref_K))
    k       = k_ref * np.exp(-Ea_k / R_gas * (1.0 / T - 1.0 / T_ref_K))
    K_OH    = A_OH  * vH(dH_OH)
    K_H2    = A_H2  * vH(dH_H2)
    K_mix   = A_mix * vH(dH_mix)
    K_eq    = K_eq_sabatier(T)
    p_CO2_s = np.maximum(p_CO2, P_FLOOR)
    p_H2_s  = np.maximum(p_H2,  P_FLOOR)
    beta    = np.minimum((p_CH4 * p_H2O**2) / (K_eq * p_CO2_s * p_H2_s**4), 1.0)  # clamped: beta>1 → f_eq=0 anyway; prevents overflow when H2 depleted
    f_eq    = np.maximum(1.0 - beta, 0.0)
    DEN     = (1.0
               + K_OH  * np.maximum(p_H2O, 0.0) / p_H2_s**0.5
               + K_H2  * p_H2_s**0.5
               + K_mix * p_CO2_s**0.5)
    return k * (p_CO2_s * p_H2_s)**0.5 * f_eq / DEN**2 * 1000.0


# =============================================================================
# 3. TOTAL-MOLAR-FLUX CLOSURE  (replaces the old constant superficial velocity)
# =============================================================================

def _compute_fields(T, C_CO2, C_H2, C_CH4, C_H2O, q, F_total_in, C_total):
    """
    Local reaction/adsorption rates, plus the node-local gas velocity implied
    by them. CO2 + 4H2 -> CH4 + 2H2O consumes gas moles (stoichiometric sum
    -1-4+1+2 = -2 per unit reaction extent) and adsorption removes H2O from
    the gas phase (-1 per unit adsorption extent). Since T and P are both
    fixed here (isothermal, isobaric), C_total = P/(RT) cannot change, so the
    total molar flux F_tot must fall along z to satisfy dF_tot/dz = S_tot at
    every instant (quasi-steady closure). Integrated with a plain rectangle-
    rule cumsum to exactly match the existing first-order-upwind finite-
    volume scheme (a trapezoidal rule would blend neighbouring source values
    and break exact mass conservation between this closure and the species
    transport terms).
    """
    p_CO2 = C_CO2 * R_gas * T / 1e5
    p_H2  = C_H2  * R_gas * T / 1e5
    p_CH4 = C_CH4 * R_gas * T / 1e5
    p_H2O = C_H2O * R_gas * T / 1e5

    r    = reaction_rate_SI(T, p_CO2, p_H2, p_CH4, p_H2O)  # mol/(kg_cat·s)
    qs   = q_star_vec(T, p_H2O, W0_DA, E_DA, n_DA)         # equilibrium loading [mol/kg]
    Kl   = K_LDF_vec(T,  p_H2O, W0_DA, E_DA, n_DA)         # LDF rate constant   [1/s]
    dqdt = Kl * (qs - q)                                    # adsorption rate     [mol/(kg·s)]

    S_tot = rho_cat * (-2.0) * r - rho_bed * dqdt          # net gas-mole source [mol/(m3_bed·s)]

    F_tot_rightface = F_total_in + np.cumsum(S_tot) * dz
    F_tot_rightface = np.clip(F_tot_rightface,
                               U_FLOOR_FRAC * F_total_in,
                               U_CEIL_FRAC  * F_total_in)

    u_node    = F_tot_rightface / C_total
    u_node_up = np.concatenate([[F_total_in / C_total], u_node[:-1]])

    return r, dqdt, S_tot, u_node, u_node_up


# =============================================================================
# 4. RHS  (sorption-enhanced only — se_on always True here)
# =============================================================================

def rhs_sem_flux(t, y, T, F_total_in, C_total, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O):
    C_CO2 = np.maximum(y[0*N : 1*N], 0.0)
    C_H2  = np.maximum(y[1*N : 2*N], 0.0)
    C_CH4 = np.maximum(y[2*N : 3*N], 0.0)
    C_H2O = np.maximum(y[3*N : 4*N], 0.0)
    q     = np.maximum(y[4*N : 5*N], 0.0)

    r, dqdt, S_tot, u_node, u_node_up = _compute_fields(
        T, C_CO2, C_H2, C_CH4, C_H2O, q, F_total_in, C_total)

    # Upwind (first-order) advection: node 0 sees inlet, node i sees node i-1
    C_CO2_up = np.concatenate([[C_in_CO2], C_CO2[:-1]])
    C_H2_up  = np.concatenate([[C_in_H2],  C_H2[:-1]])
    C_CH4_up = np.concatenate([[C_in_CH4], C_CH4[:-1]])
    C_H2O_up = np.concatenate([[C_in_H2O], C_H2O[:-1]])

    # 1D transient PFR balance: eps_b * dC/dt = -d(u*C)/dz + rho * stoich * r
    # u is now node-local (u_node/u_node_up), not a single constant — see _compute_fields.
    # Reaction scales with rho_cat (active_fraction of bed); adsorption scales with the full rho_bed (sorbent = all material).
    dCdt_CO2 = (-(u_node * C_CO2 - u_node_up * C_CO2_up) / dz + rho_cat * (-1) * r) / eps_b
    dCdt_H2  = (-(u_node * C_H2  - u_node_up * C_H2_up ) / dz + rho_cat * (-4) * r) / eps_b
    dCdt_CH4 = (-(u_node * C_CH4 - u_node_up * C_CH4_up) / dz + rho_cat * (+1) * r) / eps_b
    dCdt_H2O = (-(u_node * C_H2O - u_node_up * C_H2O_up) / dz + rho_cat * (+2) * r
                                                                - rho_bed * dqdt) / eps_b  # adsorption removes H2O from gas

    return np.concatenate([dCdt_CO2, dCdt_H2, dCdt_CH4, dCdt_H2O, dqdt])


# =============================================================================
# 5. SOLVE
# =============================================================================

C_total    = P_Pa / (R_gas * T_K)   # total molar concentration at T, P [mol/m³] — fixed (isothermal, isobaric)
F_total_in = u_in * C_total         # inlet total molar flux [mol/(m²·s)] — boundary condition
C_in_CO2   = y_CO2_in * C_total
C_in_H2    = y_H2_in  * C_total
C_in_CH4   = y_CH4_in * C_total
C_in_H2O   = 0.0

t_end = 7200.0                     # 120 min

y0 = np.zeros(5 * N)
y0[0*N : 1*N] = C_in_CO2
y0[1*N : 2*N] = C_in_H2
y0[2*N : 3*N] = C_in_CH4

GHSV = Q_STP * 3600 * 1e6 / (m_cat_total * 1e3)
print(f"Validating Wei Fig. 5.10 (molar-flux form):  T = {T_C} °C,  P = {P_bar} bar,  "
      f"GHSV = {GHSV:.0f} mL/(g·h)")
print(f"Bed (sorbent) density: {rho_bed:.1f} kg/m³  |  Catalyst density: {rho_cat:.1f} kg/m³  "
      f"(active_fraction = {active_fraction:.0%} of {m_cat_total*1000:.1f} g)")
print(f"Inlet velocity: {u_in*1e3:.3f} mm/s  |  Inlet total molar flux F_total_in = {F_total_in:.4f} mol/(m²·s)")
print(f"Solving sorption-enhanced case for {t_end/60:.0f} min ...")

sol = solve_ivp(
    rhs_sem_flux,
    t_span=[0.0, t_end],
    y0=y0,
    args=(T_K, F_total_in, C_total, C_in_CO2, C_in_H2, C_in_CH4, C_in_H2O),
    method='BDF',
    rtol=1e-4,
    atol=1e-8,
    dense_output=True,
)
if not sol.success:
    raise RuntimeError(f"ODE solver failed: {sol.message}")
print("Done.")


# =============================================================================
# 6. POST-PROCESS
# =============================================================================

t_plot = np.linspace(0.0, t_end, 400)
y_out  = sol.sol(t_plot)

C_CO2_prof = np.maximum(y_out[0*N : 1*N, :], 0.0)   # shape (N, 400)
C_H2_prof  = np.maximum(y_out[1*N : 2*N, :], 0.0)
C_CH4_prof = np.maximum(y_out[2*N : 3*N, :], 0.0)
C_H2O_prof = np.maximum(y_out[3*N : 4*N, :], 0.0)
q_prof     = np.maximum(y_out[4*N : 5*N, :], 0.0)

# Recompute node velocities (and the floor/ceiling engagement diagnostic) on the
# accepted solution, using the same _compute_fields helper the RHS used — keeps
# the reported diagnostics guaranteed consistent with what was actually integrated.
n_t       = len(t_plot)
u_out     = np.zeros(n_t)
floor_hit = np.zeros(n_t, dtype=bool)
ceil_hit  = np.zeros(n_t, dtype=bool)
for i in range(n_t):
    r_i, dqdt_i, S_tot_i, u_node_i, u_node_up_i = _compute_fields(
        T_K, C_CO2_prof[:, i], C_H2_prof[:, i], C_CH4_prof[:, i],
        C_H2O_prof[:, i], q_prof[:, i], F_total_in, C_total)
    u_out[i] = u_node_i[-1]
    F_tot_rf_i = F_total_in + np.cumsum(S_tot_i) * dz
    floor_hit[i] = np.any(F_tot_rf_i <= U_FLOOR_FRAC * F_total_in * (1 + 1e-9))
    ceil_hit[i]  = np.any(F_tot_rf_i >= U_CEIL_FRAC  * F_total_in * (1 - 1e-9))

# Outlet concentrations: last spatial node of each species block
C_CO2_out = C_CO2_prof[-1, :]
C_H2_out  = C_H2_prof[-1, :]
C_CH4_out = C_CH4_prof[-1, :]
C_H2O_out = C_H2O_prof[-1, :]

# Mol% — total gas concentration is pinned by the ideal gas law (isothermal,
# isobaric), so C_total is the correct fixed denominator (replaces the old ad
# hoc C_gas_out = sum of species + fixed C_N2_in).
pct_CO2 = C_CO2_out / C_total * 100
pct_H2  = C_H2_out  / C_total * 100
pct_CH4 = C_CH4_out / C_total * 100
pct_H2O = C_H2O_out / C_total * 100

# N2 diagnostic: N2 is inert and untracked, so whatever isn't the four tracked
# species must be N2. N2's molar FLUX (not concentration) is what's conserved
# for a passive species advected by a spatially-varying velocity -- as u falls
# along the bed (reaction + adsorption shrink the carrier gas), N2 necessarily
# gets locally *concentrated* (C_N2 = F_N2_in/u_node rises) to keep its flux
# constant. So C_N2_implied rising above C_N2_in is expected physics, not an
# error; it should never drop below C_N2_in (that would mean gas moles were
# created from nothing) nor exceed C_total (unphysical).
C_N2_in      = y_N2_in * C_total
C_N2_implied = np.maximum(C_total - (C_CO2_out + C_H2_out + C_CH4_out + C_H2O_out), 0.0)
n2_max_rise_pct = (np.max(C_N2_implied) / C_N2_in - 1.0) * 100
n2_below_inlet  = np.any(C_N2_implied < C_N2_in * (1 - 1e-6))
print(f"\nN2 concentration diagnostic: max rise above inlet = {n2_max_rise_pct:.2f}% "
      f"(expected: carrier gas shrinks as reaction+adsorption remove moles, "
      f"so N2 gets concentrated -- this is NOT an error)")
if n2_below_inlet:
    print("WARNING: C_N2_implied dropped below C_N2_in somewhere -- this would indicate "
          "spurious gas-mole creation and should be investigated.")
if np.any(floor_hit):
    print(f"WARNING: velocity floor engaged at {np.sum(floor_hit)}/{n_t} output times "
          f"(exact mass conservation not guaranteed there)")
else:
    print("Velocity floor never engaged on the accepted solution (exact mass conservation holds).")
if np.any(ceil_hit):
    print(f"WARNING: velocity ceiling engaged at {np.sum(ceil_hit)}/{n_t} output times")
print(f"Outlet/inlet velocity ratio at t_end: {u_out[-1] / u_in:.4f}")

# CO2 conversion — flux-based, since outlet velocity now differs from inlet
# velocity once total gas moles change (the old concentration-based formula
# implicitly assumed u_out = u_in, which is exactly the assumption removed here).
F_CO2_in  = C_in_CO2 * u_in
F_CO2_out = u_out * C_CO2_out
X_CO2 = np.clip(1.0 - F_CO2_out / F_CO2_in, 0.0, 1.0) * 100

def equilibrium_conversion_pct(T_K_val):
    K  = K_eq_sabatier(T_K_val)
    def f(X):
        return ((0.815 + 0.025*X) * 0.0025 * X**2 * (1 - 0.05*X)**2
                / (2.5e-6 * (1 - X)**5) - K)
    try:
        return brentq(f, 1e-9, 1 - 1e-9) * 100
    except Exception:
        return 100.0

X_eq  = equilibrium_conversion_pct(T_K)
t_min = t_plot / 60

print(f"CO2 equilibrium conversion at {T_C} °C: {X_eq:.1f} %")
print(f"CH4 mol% at full CO2 conversion (N2 dilution): "
      f"{(y_CH4_in + y_CO2_in) / (1 - y_CO2_in) * 100:.1f} %  "
      f"(Wei caption: 94 %)")


# =============================================================================
# 7. MEASURED DATA  (digitised from measurements Wei.png — Wei Fig. 5.10)
# =============================================================================

# Left axis (mol%)
t_H2_meas  = np.array([ 0.090,  3.927,  7.734, 11.550, 15.458, 19.185, 22.911,
                        26.729, 30.638, 34.277, 38.283, 41.923, 45.742, 49.564])
H2_meas    = np.array([ 9.899,  1.956,  0.726,  0.504,  0.383,  0.323,  0.242,
                         0.222,  0.222,  0.444,  1.169,  1.593,  1.673,  2.056])

t_CO2_meas = np.array([ 0.114,  4.000,  8.250, 12.500, 16.750, 21.000, 25.250,
                        29.500, 33.750, 38.000, 42.250, 46.500, 50.750, 55.000])
CO2_meas   = np.array([ 2.560,  0.060,  0.000,  0.000,  0.000,  0.000,  0.000,
                         0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000])

t_H2O_meas = np.array([ 0.000, 42.182, 43.000, 43.371, 43.832, 44.202, 44.571,
                        45.484, 46.942, 48.308, 49.764, 52.220, 52.497, 53.862,
                        54.863, 55.047])
H2O_meas   = np.array([ 0.000,  0.040,  0.060,  0.867,  1.573,  2.218,  2.802,
                         3.306,  3.690,  3.911,  4.052,  4.274,  4.718,  4.819,
                         4.940,  5.202])

# Right axis
t_CH4_meas = np.array([ 0.182,  4.067,  7.810, 11.629, 15.357, 19.357, 22.994,
                        26.903, 30.721, 34.448, 38.266, 42.084, 45.992, 49.709])
CH4_meas   = np.array([ 0.403, 73.387, 90.726, 92.339, 92.742, 92.944, 93.145,
                        93.347, 93.347, 93.145, 92.540, 92.137, 91.935, 80.645])

t_X_meas   = np.array([0,   4.7,  10,  15,  20,  25,  30,  35,  40,  43,  47,  50,  52])
X_meas     = np.array([0,100, 100, 100, 100, 100, 100, 100, 100,  100,  100,  100,  100])


# =============================================================================
# 7b. H₂O SLOPE COMPARISON  (K_LDF validation via breakthrough steepness)
# =============================================================================

def fit_breakthrough_slope(t_arr, y_arr, frac_lo=0.10, frac_hi=0.90, t_max=None):
    """
    Linear fit to the 10–90 % rise of a breakthrough curve.
    Optional t_max caps the upper end of the fit window (overrides frac_hi).
    Returns (slope [mol%/min], (t_start, t_end), poly_coefficients).
    """
    y_max = np.nanmax(y_arr)
    mask  = (y_arr >= frac_lo * y_max) & (y_arr <= frac_hi * y_max)
    if t_max is not None:
        mask &= (t_arr <= t_max)
    if mask.sum() < 2:
        return np.nan, None, None
    t_sel = t_arr[mask]
    y_sel = y_arr[mask]
    poly  = np.polyfit(t_sel, y_sel, 1)   # [slope, intercept]
    return poly[0], (float(t_sel[0]), float(t_sel[-1])), poly


# Measured slope: direct two-point slope between the digitised anchor points
# (43.0 min, 0.06 mol%) and (48.308 min, 3.911 mol%) — matches the dashed line drawn on the plot.
t_anchor_meas = (43.0, 48.308)
y_anchor_meas = (0.06, 3.911)
slope_meas = (y_anchor_meas[1] - y_anchor_meas[0]) / (t_anchor_meas[1] - t_anchor_meas[0])
twin_meas  = t_anchor_meas

slope_model, twin_model, poly_model = fit_breakthrough_slope(t_min, pct_H2O)

print(f"\n--- H₂O breakthrough slope (K_LDF validation) ---")
print(f"  Measured : {slope_meas:.3f} mol%/min  "
      f"(two-point slope over t = {twin_meas[0]:.1f}–{twin_meas[1]:.1f} min)")
print(f"  Model    : {slope_model:.3f} mol%/min  "
      f"(fitted over t = {twin_model[0]:.1f}–{twin_model[1]:.1f} min)")
print(f"  Ratio (model / meas): {slope_model / slope_meas:.2f}")


# =============================================================================
# 8. PLOT — model lines + measured scatter, dual-axis
# =============================================================================

fig, ax1 = plt.subplots(figsize=(10, 6))

# --- Left y-axis: model lines ---
h_H2,  = ax1.plot(t_min, pct_H2,  'k-',  lw=1.8, label='H₂ (model)')
h_CO2, = ax1.plot(t_min, pct_CO2, color='tab:red', lw=1.8, label='CO₂ (model)')
h_H2O, = ax1.plot(t_min, pct_H2O, color='olive',   lw=2.2, label='H₂O (model)')

# --- Left y-axis: measured scatter ---
h_H2_m  = ax1.scatter(t_H2_meas,  H2_meas,  color='k',
                       marker='D', s=55, zorder=5,
                       label='H₂ (meas.)')
h_CO2_m = ax1.scatter(t_CO2_meas, CO2_meas, color='tab:red',
                       marker='o', s=55, zorder=5,
                       label='CO₂ (meas.)')
h_H2O_m = ax1.scatter(t_H2O_meas, H2O_meas, color='olive',
                       marker='^', s=55, zorder=5,
                       label='H₂O (meas.)')

ax1.set_xlabel('Time (min)', fontsize=12)
ax1.set_ylabel('H₂, CO₂, H₂O concentration (mol%)', fontsize=11)
ax1.set_xlim(0, 60)
ax1.set_ylim(-0.5, 10.5)
ax1.set_yticks([0, 2, 4, 6, 8, 10])

# --- Right y-axis: model lines ---
ax2 = ax1.twinx()

h_CH4, = ax2.plot(t_min, pct_CH4,
                   color='tab:blue', lw=1.8, label='CH₄ (model)')
h_Xeq, = ax2.plot([0, 60], [X_eq, X_eq],
                   color='m', lw=1.5, ls='--',
                   label=f'CO₂ eq. conversion ({X_eq:.0f} %)')
h_X,   = ax2.plot(t_min, X_CO2,
                   color='m', lw=1.8, label='CO₂ conversion (model)')

# --- Right y-axis: measured scatter ---
h_CH4_m = ax2.scatter(t_CH4_meas, CH4_meas,
                       color='tab:blue', marker='o', s=65, zorder=5,
                       label='CH₄ (meas.)')
h_X_m   = ax2.scatter(t_X_meas,   X_meas,
                       color='m', marker='s', s=65, zorder=5,
                       label='CO₂ conversion (meas.)')

ax2.set_ylabel('CH₄ concentration (mol%)  /  CO₂ conversion (%)', fontsize=11)
ax2.set_ylim(-10, 110)
ax2.set_yticks([0, 20, 40, 60, 80, 100])

# --- H₂O slope lines (K_LDF validation) ---
t_fit_meas  = np.array([43.0, 48.308])
y_fit_meas  = np.array([0.06, 3.911])
t_fit_model = np.linspace(twin_model[0], twin_model[1], 80)
h_slope_m,   = ax1.plot(t_fit_meas,  y_fit_meas,
                         color='olive', ls='--', lw=2.2, alpha=0.75,
                         label=f'H₂O slope meas. ({slope_meas:.2f} mol%/min)')
h_slope_mod, = ax1.plot(t_fit_model, np.polyval(poly_model, t_fit_model),
                         color='darkgreen', ls='--', lw=2.2, alpha=0.75,
                         label=f'H₂O slope model ({slope_model:.2f} mol%/min)')

# --- Combined legend ---
all_handles = [h_CH4, h_CH4_m, h_Xeq, h_X, h_X_m,
               h_H2, h_H2_m, h_CO2, h_CO2_m, h_H2O, h_H2O_m,
               h_slope_m, h_slope_mod]
ax2.legend(handles=all_handles, loc='center left', bbox_to_anchor=(0.02, 0.5),
           fontsize=8.5, framealpha=0.9, ncol=2)

ax1.grid(True, alpha=0.3)

plt.tight_layout()
fname = f'validation_fig5_10_wei_isothermal_with_measurements_mole_flux_T{T_C:.0f}C_af{active_fraction:.0%}.png'
plt.savefig(os.path.join(_DIR, fname), dpi=300, bbox_inches='tight')
plt.show()
