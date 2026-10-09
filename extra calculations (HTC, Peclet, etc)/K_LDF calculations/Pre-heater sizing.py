import numpy as np

# ============================================================
# Parameters (same physical situation as the once-through case)
# ============================================================
m_solid    = 91.4     # kg - full solid inventory in the reservoir
cp_solid   = 1100.0   # J/(kg*K) - solid heat capacity (Table 6.1)

T_amb      = 20.0     # deg C - ambient / initial solid temperature
T_target   = 280.0    # deg C - target solid preheat temperature
T_N2_in    = 400.0    # deg C - hot N2 supply temperature entering the reservoir

cp_N2      = 1040.0   # J/(kg*K) - N2 specific heat
M_N2       = 0.028    # kg/mol - N2 molar mass
R_gas      = 8.314     # J/(mol*K) - universal gas constant

t_preheat_h = 3.0
t_preheat   = t_preheat_h * 3600

# --- Reservoir geometry (for loop inventory estimate) ---
V_bed_total = 0.1088   # m3 - total reservoir volume (solid + voids), from sizing calc
eps_bed     = 0.4      # - bed voidage
V_pipework  = 0.010    # m3 - nominal extra free volume in ducting/blower/heater housing

# ============================================================
# Energy and average power required by the solid (unchanged)
# ============================================================
delta_T_solid = T_target - T_amb
E_solid = m_solid * cp_solid * delta_T_solid   # J
P_solid_avg = E_solid / t_preheat              # W

# ============================================================
# Mass flow of N2 needed (unchanged - set by how much heat each
# kg of gas gives up crossing the bed, not by how it's supplied)
# ============================================================
delta_T_N2 = T_N2_in - T_target                # K - temperature drop across the bed per pass
m_dot_N2   = P_solid_avg / (cp_N2 * delta_T_N2)  # kg/s
m_dot_N2_kgh = m_dot_N2 * 3600

# ============================================================
# ONCE-THROUGH case (for comparison)
# Fresh ambient N2 heated to T_N2_in every pass, then vented.
# ============================================================
P_heater_once_through = m_dot_N2 * cp_N2 * (T_N2_in - T_amb)   # W
m_N2_total_once_through = m_dot_N2_kgh * t_preheat_h            # kg, total consumed

# ============================================================
# RECYCLE case
# Gas exits the bed at ~T_target, is reheated back to T_N2_in,
# and returned - so the heater only has to make up the heat that
# actually went into the solid, not reheat fresh gas from ambient
# every pass.
# ============================================================
P_heater_recycle = m_dot_N2 * cp_N2 * (T_N2_in - T_target)   # W
# (this should come out equal to P_solid_avg, by construction)

# --- Loop inventory: how much N2 is actually "in the pipes" at once ---
V_free_gas = eps_bed * V_bed_total + V_pipework   # m3 - free gas volume in the loop
P_atm = 101325.0        # Pa
T_N2_in_K = T_N2_in + 273.15
m_loop_inventory = (P_atm * V_free_gas * M_N2) / (R_gas * T_N2_in_K)   # kg

# ============================================================
# Results
# ============================================================
print(f"Avg. power required by solid:      {P_solid_avg:.0f} W")
print(f"N2 mass flow rate:                 {m_dot_N2_kgh:.1f} kg/h")
print()
print("--- Once-through ---")
print(f"Heater power:                      {P_heater_once_through/1000:.2f} kW")
print(f"Total N2 consumed over {t_preheat_h:.0f} h:      {m_N2_total_once_through:.1f} kg")
print()
print("--- Recycle ---")
print(f"Heater power:                      {P_heater_recycle/1000:.2f} kW")
print(f"Free gas volume in loop:           {V_free_gas*1000:.1f} L")
print(f"N2 mass held in loop at any time:  {m_loop_inventory*1000:.1f} g")