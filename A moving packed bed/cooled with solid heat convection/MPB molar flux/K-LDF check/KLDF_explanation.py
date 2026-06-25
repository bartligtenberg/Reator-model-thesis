import numpy as np
import matplotlib.pyplot as plt
import os

# DA isotherm parameters (from model)
R_gas  = 8.314
MW_H2O = 0.018015
W0_DA  = 190.00e-6
E_DA   = 1190e3
n_DA   = 1.55
T_K    = 553.15  # 280 C

# Particle / bed parameters
d_p         = 0.75e-3
eps_p       = 0.615
tau_p       = 3.0
rho_bed_ads = 1.22 / (np.pi/4 * 0.05**2 * 2.0)
rho_p       = rho_bed_ads / (1 - 0.4)

def P_sat_bar(T):
    log10_p = (29.8605 - 3.1522e3/T - 7.3037*np.log10(T)
               + 2.4247e-9*T + 1.8090e-6*T**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5

def rho_water(T):
    return 996.0 / (1.0 + 2.0e-3*(T - 298.15))

def q_star_val(T, p):
    Psat = P_sat_bar(T)
    if p <= 0 or p >= Psat:
        return 0.0
    A = (R_gas/MW_H2O)*T*np.log(Psat/p)
    return rho_water(T)/MW_H2O * W0_DA * np.exp(-min((A/E_DA)**n_DA, 500.0))  # mol/kg

def K_LDF_val(T, p):
    D_M  = 2.5e-5*(T/300.0)**1.75
    dp   = 1.0/1e5
    dqsp = (q_star_val(T, p+dp) - q_star_val(T, max(p-dp, 1e-15))) / 2.0
    dqsp = max(dqsp, 1e-30)
    return min(15.0*eps_p*D_M / (0.5 * d_p**2 * tau_p * rho_p * R_gas * T * dqsp), 0.5)

# Relevant reactor range: p_H2O from near 0 to ~80 mbar (= max ~8% H2O at 1 bar)
# Use log spacing so the steep low-p region is well resolved
p_arr = np.logspace(np.log10(5e-5), np.log10(0.08), 400)   # bar, log-spaced
q_arr = np.array([q_star_val(T_K, p) for p in p_arr])
K_arr = np.array([K_LDF_val(T_K, p) for p in p_arr])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Why desorption is slow at low $p_{H_2O}$ — DA isotherm at 280 °C', fontsize=12)

# ── Panel 1: isotherm with slope illustration ────────────────────────────────
ax1.plot(p_arr*1e3, q_arr, 'k-', lw=2.5)

dq = 0.15  # same Δq shown at both points

# Point A: steep region (low p)
pA     = 0.002
qA     = q_star_val(T_K, pA)
dp_num = 1e-7
sA     = (q_star_val(T_K, pA+dp_num) - q_star_val(T_K, pA-dp_num)) / (2*dp_num)
dpA    = dq / sA

# Point B: flat region (high p)
pB     = 0.040
qB     = q_star_val(T_K, pB)
sB     = (q_star_val(T_K, pB+dp_num) - q_star_val(T_K, pB-dp_num)) / (2*dp_num)
dpB    = dq / sB

# Annotate point A
ax1.annotate('', xy=(pA*1e3, qA+dq), xytext=(pA*1e3, qA),
             arrowprops=dict(arrowstyle='<->', color='tab:blue', lw=2))
ax1.annotate('', xy=((pA+dpA)*1e3, qA-0.02), xytext=(pA*1e3, qA-0.02),
             arrowprops=dict(arrowstyle='<->', color='tab:red', lw=2))
ax1.text(pA*1e3*0.85, qA + dq/2, 'Δq', color='tab:blue', fontsize=11, ha='right', va='center')
ax1.text((pA + dpA/2)*1e3, qA - 0.09,
         f'Δp = {dpA*1e3:.1f} mbar\n→ tiny gas gradient\n→ slow diffusion',
         color='tab:red', fontsize=8, ha='center')

# Annotate point B
ax1.annotate('', xy=(pB*1e3, qB+dq), xytext=(pB*1e3, qB),
             arrowprops=dict(arrowstyle='<->', color='tab:blue', lw=2))
ax1.annotate('', xy=((pB+dpB)*1e3, qB-0.02), xytext=(pB*1e3, qB-0.02),
             arrowprops=dict(arrowstyle='<->', color='tab:green', lw=2))
ax1.text(pB*1e3*0.97, qB + dq/2, 'Δq', color='tab:blue', fontsize=11, ha='right', va='center')
ax1.text((pB + dpB/2)*1e3, qB - 0.09,
         f'Δp = {dpB*1e3:.1f} mbar\n→ large gas gradient\n→ fast diffusion',
         color='tab:green', fontsize=8, ha='center')

ax1.axvspan(0.05, 0.8,  alpha=0.10, color='tab:red',   label='steep — small $K_{LDF}$')
ax1.axvspan(2,   80,   alpha=0.07, color='tab:green', label='flat — large $K_{LDF}$')
ax1.set_xlabel('$p_{H_2O}$ [mbar]', fontsize=11)
ax1.set_ylabel('$q^*$ [mol/kg]', fontsize=11)
ax1.set_title('Same Δq in solid → very different Δp in pore gas', fontsize=10)
ax1.set_xscale('log')
ax1.set_xlim(0.05, 80)
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, which='both')

# ── Panel 2: K_LDF vs p ──────────────────────────────────────────────────────
ax2.plot(p_arr*1e3, K_arr, 'k-', lw=2.5)
ax2.axhline(0.5, color='grey', lw=1.5, ls='--', label='$K_{LDF,max}$ = 0.5 s$^{-1}$')
ax2.fill_between(p_arr*1e3, K_arr, 0,
                 where=(p_arr < 1e-3), alpha=0.15, color='tab:red',
                 label='slow: pore diffusion limited')
ax2.fill_between(p_arr*1e3, K_arr, 0,
                 where=(p_arr > 2e-3), alpha=0.15, color='tab:green',
                 label='fast: hits cap → near equilibrium')
ax2.set_xlabel('$p_{H_2O}$ [mbar]', fontsize=11)
ax2.set_ylabel('$K_{LDF}$ [s$^{-1}$]', fontsize=11)
ax2.set_title('LDF mass transfer coefficient vs $p_{H_2O}$  (log x-axis)', fontsize=10)
ax2.set_xscale('log')
ax2.set_xlim(0.05, 80)
ax2.set_ylim(0, 0.55)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'KLDF_explanation.png'),
            dpi=150, bbox_inches='tight')
plt.show()
