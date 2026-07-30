"""
Load a saved (u_s, GHSV) feasibility/sizing sweep result and print it -- reads the .pkl
produced by Step 5 of 'MPB_flux_form active frac heatmap.py'. No plotting; console output only.

Edit PKL_PATH below if you want to load a different saved sweep (e.g. a different
active_fraction).
"""

import os
import pickle
import numpy as np

HERE     = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(HERE, 'sweep_results_active_frac_20%.pkl')

with open(PKL_PATH, 'rb') as f:
    sweep = pickle.load(f)

US_GRID       = sweep['US_GRID']          # [m/s]
GHSV_GRID     = sweep['GHSV_GRID']        # [m3_STP/(kg_ads·h)]
T_SWEEP_C     = sweep['T_SWEEP_C']
L_b           = sweep['L_b']
LAMBDA_MIN    = sweep['LAMBDA_MIN']
Lambda_thermo = sweep['Lambda_thermo']
mask_survive  = sweep['mask_survive']
q_star_max    = sweep['q_star_max']
p_H2O_full    = sweep.get('p_H2O_full')   # not present in sweeps saved before this field was added
cell_results  = sweep['cell_results']
Z995, Z974    = sweep['Z995'], sweep['Z974']
TMAX, NITER, SOLVET = sweep['TMAX'], sweep['NITER'], sweep['SOLVET']

N_US, N_GHSV = len(US_GRID), len(GHSV_GRID)
n_attempted  = sum(1 for v in cell_results.values() if v is not None)
n_converged  = sum(1 for v in cell_results.values() if v is not None and v.get('converged'))
n_reach_995  = int(np.sum(~np.isnan(Z995)))

# region SUMMARY
# =============================================================================
print(f"Loaded: {PKL_PATH}")
print(f"Sweep grid: {N_US} x {N_GHSV} = {N_US*N_GHSV} cells")
print(f"  u_s  in [{US_GRID.min()*1e3:.2f}, {US_GRID.max()*1e3:.2f}] mm/s")
print(f"  GHSV in [{GHSV_GRID.min():.2f}, {GHSV_GRID.max():.2f}] m3_STP/(kg_ads.h)")
print(f"  T = {T_SWEEP_C:.0f} C, L_trial = {L_b:.1f} m")
q_star_line = f"  q_star_max = {q_star_max:.3f} mol/kg"
if p_H2O_full is not None:
    q_star_line += f"  @ p_H2O_full = {p_H2O_full*1e3:.1f} mbar"
print(q_star_line)
print(f"  survive Lambda_thermo >= {LAMBDA_MIN}: {int(mask_survive.sum())}/{N_US*N_GHSV}")
print(f"  attempted: {n_attempted}, converged: {n_converged}, reach X_CO2=99.5%: {n_reach_995}")
# endregion


# region PER-CELL TABLE
# =============================================================================
print(f"\n{'u_s[mm/s]':>10} {'GHSV':>6} {'Lambda':>7} {'conv':>5} {'iter':>5} "
      f"{'z995/L':>7} {'z974/L':>7} {'Tmax[C]':>8} {'H2O err%':>9}")
for i in range(N_US):
    for j in range(N_GHSV):
        e = cell_results.get((i, j))
        if e is None or e.get('z') is None:
            continue
        bal     = e.get('h2o_balance')     # None for sweeps saved before this field was added
        bal_str = f"{bal['bal_err_pct']:+.1f}" if bal is not None else "n/a"
        print(f"{US_GRID[i]*1e3:10.2f} {GHSV_GRID[j]:6.2f} {Lambda_thermo[i, j]:7.2f} "
              f"{str(e['converged']):>5} {e['n_iter']:5d} "
              f"{Z995[i, j]/L_b:7.3f} {Z974[i, j]/L_b:7.3f} {TMAX[i, j]:8.1f} {bal_str:>9}")
# endregion
