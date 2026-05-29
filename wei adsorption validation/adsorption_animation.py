"""
Animated adsorption front — H₂O moving through the packed bed over time.

Shows two panels updating frame by frame:
  Left  — gas-phase H₂O concentration C(z) normalised by inlet value (0 → 1)
  Right — solid-phase loading q(z) normalised by inlet equilibrium q* (0 → 1)

At t=0 the bed is clean.  As time progresses the adsorption front sweeps from
left (inlet) to right (outlet).  Breakthrough is reached when the outlet
concentration hits 10% of the inlet value.

Run this script on its own — it does not depend on adsorption_simulation.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# =============================================================================
# 1. PARAMETERS  (identical to adsorption_simulation.py)
# =============================================================================

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
N           = 50          # finer grid for smoother animation
dz          = L_b / (N - 1)
z_nodes     = np.linspace(0, L_b, N) * 100   # [cm] for plot axis

# Case to animate — change these two lines to explore other conditions
T_C    = 300
PARAMS = {'W0': 341.00e-6, 'E': 1192.25e3, 'n': 1.55}   # Mette (2014)
LABEL  = 'Mette (2014)'

N_FRAMES = 250   # number of animation frames


# =============================================================================
# 2. ISOTHERM AND LDF FUNCTIONS  (same physics as adsorption_simulation.py)
# =============================================================================

def P_sat_bar(T_K):
    return 10.0 ** (5.40221 - 1838.675 / (T_K - 31.737))


def q_star_vec(T_K, p_arr, W0, E, n):
    p     = np.asarray(p_arr, dtype=float)
    Psat  = P_sat_bar(T_K)
    p_safe = np.clip(p, 1e-15, Psat * (1 - 1e-10))
    A_raw  = (R_gas / MW_H2O) * T_K * np.log(Psat / p_safe)
    A      = np.where((p <= 0.0) | (p >= Psat), 0.0, A_raw)
    W      = W0 * np.exp(-np.minimum((A / E) ** n, 500.0))
    qs     = rho_ads / MW_H2O * W
    return np.where(p <= 0.0, 0.0, qs)


def K_LDF_vec(T_K, p_arr, W0, E, n):
    D_M    = 2.5e-5 * (T_K / 300.0) ** 1.75
    p      = np.asarray(p_arr, dtype=float)
    dp_bar = 1.0 / 1e5
    p_lo   = np.maximum(p - dp_bar, 1e-15)
    p_hi   = p + dp_bar
    dqstar_dp = (q_star_vec(T_K, p_hi, W0, E, n)
                 - q_star_vec(T_K, p_lo, W0, E, n)) / 2.0
    dqstar_dp = np.maximum(dqstar_dp, 1e-30)
    return (15.0 * D_M * MW_H2O * eps_p
            / (0.5 * d_p**2 * tau_p * rho_ads * R_gas * T_K * dqstar_dp))


def rhs_column(t, y, T_K, u, C_in, W0, E, n):
    C    = np.maximum(y[:N], 0.0)
    q    = np.maximum(y[N:], 0.0)
    p    = C * R_gas * T_K / 1e5
    qs   = q_star_vec(T_K, p, W0, E, n)
    Kl   = K_LDF_vec(T_K, p, W0, E, n)
    dqdt = Kl * (qs - q)
    C_up = np.concatenate([[C_in], C[:-1]])
    dCdt = (-u * (C - C_up) / dz - rho_bed * dqdt) / eps_b
    return np.concatenate([dCdt, dqdt])


# =============================================================================
# 3. RUN SIMULATION WITH DENSE OUTPUT
# =============================================================================

W0, E, n = PARAMS['W0'], PARAMS['E'], PARAMS['n']
T_K  = T_C + 273.15
u    = u_STP * (T_K / T_STP)
C_in = y_H2O_in * P_Pa / (R_gas * T_K)
p_in = y_H2O_in * P_bar

qs_inlet = float(q_star_vec(T_K, np.array([p_in]), W0, E, n)[0])

# Estimate end time from equilibrium front velocity, then add 20% margin.
t_bt_est = qs_inlet * m_cat / (C_in * u * A_b)
t_end    = min(1.5 * t_bt_est, 3e4)

def bt_event(t, y, T_K, u, C_in, W0, E, n):
    return y[N - 1] - BT_FRACTION * C_in
bt_event.terminal  = True
bt_event.direction = 1

print(f"Running simulation: T = {T_C} °C, {LABEL}")
print(f"  Inlet q* = {qs_inlet:.3f} mol/kg")
print(f"  Estimated breakthrough ≈ {t_bt_est/60:.1f} min")
print("  Solving ODE ... ", end="", flush=True)

sol = solve_ivp(
    rhs_column,
    t_span=[0.0, t_end],
    y0=np.zeros(2 * N),
    args=(T_K, u, C_in, W0, E, n),
    method='BDF',
    events=bt_event,
    dense_output=True,   # enables sol.sol(t) interpolation at any time
    rtol=1e-4,
    atol=1e-8,
)

# Actual breakthrough time from the event trigger
t_bt = sol.t_events[0][0] if sol.t_events[0].size > 0 else sol.t[-1]
print(f"done.  Breakthrough at t = {t_bt/60:.1f} min")

# Evaluate solution at N_FRAMES evenly spaced times from 0 to t_bt
t_frames = np.linspace(0, t_bt, N_FRAMES)
y_frames = sol.sol(t_frames)                  # shape (2N, N_FRAMES)
C_frames = np.maximum(y_frames[:N, :], 0.0)  # gas concentration (N, N_FRAMES)
q_frames = np.maximum(y_frames[N:, :], 0.0)  # solid loading     (N, N_FRAMES)

# Normalise for plotting
C_norm = C_frames / C_in          # 0 = clean gas, 1 = saturated with inlet H₂O
q_norm = q_frames / qs_inlet      # 0 = empty solid, 1 = solid at inlet equilibrium


# =============================================================================
# 4. BUILD ANIMATION
# =============================================================================

fig = plt.figure(figsize=(12, 5))
fig.suptitle(
    f'H₂O adsorption front  —  {LABEL},  T = {T_C} °C\n'
    f'Inlet y_{{H₂O}} = {y_H2O_in*100:.2f}%,  P = {P_bar} bar',
    fontsize=12
)

gs  = gridspec.GridSpec(1, 2, wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# --- Left panel: gas-phase concentration ---
ax1.set_xlim(0, L_b * 100)
ax1.set_ylim(-0.02, 1.08)
ax1.set_xlabel('Bed position z [cm]')
ax1.set_ylabel('C / C_in  [-]')
ax1.set_title('Gas-phase H₂O concentration')
ax1.axhline(BT_FRACTION, color='red', ls='--', lw=1, label=f'Breakthrough ({int(BT_FRACTION*100)}%)')
ax1.axhline(1.0, color='grey', ls=':', lw=0.8)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

line1, = ax1.plot([], [], 'tab:blue', lw=2.5)
fill1  = ax1.fill_between([], [], alpha=0.15, color='tab:blue')

# --- Right panel: solid-phase loading ---
ax2.set_xlim(0, L_b * 100)
ax2.set_ylim(-0.02, 1.08)
ax2.set_xlabel('Bed position z [cm]')
ax2.set_ylabel('q / q*_inlet  [-]')
ax2.set_title('Solid-phase H₂O loading')
ax2.axhline(1.0, color='grey', ls=':', lw=0.8, label='Inlet equilibrium q*')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)

line2, = ax2.plot([], [], 'tab:orange', lw=2.5)
fill2  = ax2.fill_between([], [], alpha=0.15, color='tab:orange')

# Time label shown in both panels
time_text = fig.text(0.5, 0.01,  '', ha='center', fontsize=10)

# Progress bar drawn as a thin rectangle at the top of each axis
prog1 = ax1.axvspan(0, 0, ymin=0.97, ymax=1.0, color='tab:blue',  alpha=0.5)
prog2 = ax2.axvspan(0, 0, ymin=0.97, ymax=1.0, color='tab:orange', alpha=0.5)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line1, line2, time_text


def update(frame):
    global fill1, fill2, prog1, prog2

    t   = t_frames[frame]
    c   = C_norm[:, frame]
    q   = q_norm[:, frame]
    frac = t / t_bt   # fraction of breakthrough time elapsed

    # Update concentration curve and shaded area below it
    line1.set_data(z_nodes, c)
    fill1.remove()
    fill1 = ax1.fill_between(z_nodes, c, alpha=0.15, color='tab:blue')

    # Update loading curve
    line2.set_data(z_nodes, q)
    fill2.remove()
    fill2 = ax2.fill_between(z_nodes, q, alpha=0.15, color='tab:orange')

    # Progress bars (span from 0 to frac * L_b)
    prog1.remove()
    prog1 = ax1.axvspan(0, frac * L_b * 100, ymin=0.965, ymax=1.0,
                        color='tab:blue', alpha=0.6)
    prog2.remove()
    prog2 = ax2.axvspan(0, frac * L_b * 100, ymin=0.965, ymax=1.0,
                        color='tab:orange', alpha=0.6)

    time_text.set_text(
        f't = {t/60:.1f} min  ({frac*100:.0f}% of breakthrough time  |  '
        f't_bt = {t_bt/60:.1f} min)'
    )
    return line1, line2, time_text


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    init_func=init,
    interval=40,      # ms between frames → ~25 fps
    blit=False,       # blit=False needed because fill_between patches change
    repeat=True,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
