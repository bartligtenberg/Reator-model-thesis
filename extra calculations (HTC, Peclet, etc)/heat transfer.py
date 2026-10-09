import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================
T_bed  = 292.0   # deg C - local peak bed temperature (hot zone near gas inlet)
T_wall = 280.0   # deg C - controlled wall temperature (heater setpoint)
T_amb  = 20.0    # deg C - ambient room temperature

r1     = 25.0    # mm - reactor inner radius (bed / wall interface)
d_p    = 2.5     # mm - particle diameter (13X pellets, matches main reactor model)
                 # bed-to-wall film resistance is concentrated within ~1 particle
                 # diameter of the wall, not spread across the full bed radius
r_wall = 30.0    # mm - outer radius of the metal wall (5 mm wall)
r2     = 90.0    # mm - outer radius of insulation (r_wall + 60 mm mineral wool)
r_far  = 100.0   # mm - nominal radius where ambient conditions are reached
                 # (kept close to r2 since the outer film is a thin boundary
                 #  layer, not a real spatial distance - see note below)

L        = 2.0     # m - reactor length
k_ins    = 0.04    # W/(m*K) - thermal conductivity of the mineral wool insulation
k_steel  = 45.0     # W/(m*K) - carbon-steel wall thermal conductivity (nominal)
h_out    = 9.0     # W/(m^2*K) - combined convection + radiation coefficient, outer surface

# ============================================================
# Compute Q for this insulation thickness
# Q = 2*pi*L*deltaT / [ ln(r_wall/r1)/k_steel + ln(r2/r_wall)/k_ins + 1/(h_out*r2) ]
# (r2 in meters; wall conduction included for completeness, though it is
#  negligible next to the insulation resistance - see printed values below)
# ============================================================
r1_m     = r1 / 1000.0
r_wall_m = r_wall / 1000.0
r2_m     = r2 / 1000.0
delta_T  = T_wall - T_amb

R_cond_wall = np.log(r_wall_m / r1_m) / (2 * np.pi * L * k_steel)   # K/W, steel wall
R_cond      = np.log(r2_m / r_wall_m) / (2 * np.pi * L * k_ins)     # K/W, insulation
R_conv      = 1 / (h_out * 2 * np.pi * r2_m * L)                    # K/W, outer film
Q = delta_T / (R_cond_wall + R_cond + R_conv)                       # W, total steady-state heat loss

print(f"R_cond_wall = {R_cond_wall:.5f} K/W  (negligible vs. insulation)")
print(f"R_cond = {R_cond:.3f} K/W")
print(f"R_conv = {R_conv:.3f} K/W")
print(f"Q      = {Q:.1f} W  (steady-state ambient loss)")

# Precompute the coefficient for the log-conduction formula:
#   T(r) = T_wall - log_coeff * ln(r / r_wall)
log_coeff = Q / (2 * np.pi * L * k_ins)



















# ============================================================
# Electric heater sizing
# The heater must supply two things:
#   1) The steady-state ambient loss Q (computed above) - continuous duty.
#   2) Enough extra power to heat the wall + insulation thermal mass from
#      ambient to setpoint within a chosen ramp time - transient duty.
# ============================================================

# --- Material properties (nominal, illustrative) ---
rho_steel = 7900.0   # kg/m^3 - wall material density
cp_steel  = 500.0    # J/(kg*K) - wall material specific heat
rho_ins   = 100.0    # kg/m^3 - mineral wool bulk density
cp_ins    = 840.0    # J/(kg*K) - mineral wool specific heat

t_ramp_min = 120.0   # min - target startup ramp time to reach 280 C
t_ramp     = t_ramp_min * 60.0  # s

# --- Wall thermal mass ---
# Annular volume between the bed radius r1 and the wall outer radius r_wall.
V_wall = np.pi * (r_wall_m ** 2 - r1_m ** 2) * L   # m^3
m_wall = rho_steel * V_wall                                    # kg
C_wall = m_wall * cp_steel                                     # J/K

# The wall is treated as isothermal (negligible internal resistance), so it
# heats through the full delta_T from ambient to the wall setpoint.
E_wall = C_wall * delta_T   # J

# --- Insulation thermal mass ---
# Annular volume between the wall outer radius r_wall and insulation outer
# radius r2.
V_ins = np.pi * (r2_m ** 2 - r_wall_m ** 2) * L   # m^3
m_ins = rho_ins * V_ins                            # kg
C_ins = m_ins * cp_ins                             # J/K

# The insulation has an internal gradient from T_wall (inner face) down to
# the outer surface temperature T_surface, so it does NOT heat through the
# full delta_T on average - use the average of the two end temperatures.
T_surface = T_wall - log_coeff * np.log(r2_m / r_wall_m)   # deg C, outer surface temp
delta_T_ins_avg = ((T_wall + T_surface) / 2) - T_amb        # K, average rise
E_ins = C_ins * delta_T_ins_avg   # J

print(f"T_surface = {T_surface:.1f} degC  (insulation outer/touch temperature)")

# --- Total transient energy and power ---
E_total = E_wall + E_ins        # J - total energy to bring hardware to setpoint
P_ramp  = E_total / t_ramp      # W - power needed to supply that energy in t_ramp

# --- Total heater power ---
P_heater = P_ramp + Q   # W - transient ramp power + steady-state loss

print()
print(f"m_wall = {m_wall:.2f} kg,  E_wall = {E_wall/1e6:.3f} MJ")
print(f"m_ins  = {m_ins:.2f} kg,  E_ins  = {E_ins/1e6:.3f} MJ  (avg dT = {delta_T_ins_avg:.1f} K)")
print(f"E_total = {E_total/1e6:.3f} MJ")
print(f"t_ramp  = {t_ramp_min:.0f} min")
print(f"P_ramp  = {P_ramp:.1f} W")
print()
print(f"Steady-state power (Q)  = {Q:.1f} W")
print(f"Ramp power (P_ramp)     = {P_ramp:.1f} W")
print(f"P_heater (ramp + steady loss) = {P_heater:.1f} W")

# ============================================================
# Build the radial grid
# ============================================================
r = np.linspace(0, r_far, 400)  # radial positions, mm
T = np.zeros_like(r)            # temperature at each radial position, deg C

# ============================================================
# Region 1a: bulk bed (0 <= r <= r1 - d_p)
# Flat temperature - the pseudo-homogeneous model assumes no radial
# gradient within the bulk bed itself.
# ============================================================
r_film_start = r1 - d_p
mask_bed_bulk = r <= r_film_start
T[mask_bed_bulk] = T_bed

# ============================================================
# Region 1b: bed-to-wall film (r1 - d_p < r <= r1)
# The real resistance between bed and wall is concentrated in a thin
# zone about one particle diameter thick, right against the wall
# (disrupted packing, higher local voidage) - not spread across the
# whole bed radius. Drawn as an exponential decay - steepest right at
# the wall (r1), flattening asymptotically toward T_bed moving back
# into the bed - the mirror image of the ambient-film curve below.
# Same caveat as before: h_w is a lumped boundary condition, so this
# exact shape isn't derived, only positioned/scaled to sit in the
# right place and decay the right way.
# ============================================================
mask_bed_film = (r > r_film_start) & (r <= r1)
delta_bed_film = d_p / 4.0
T[mask_bed_film] = T_bed - (T_bed - T_wall) * np.exp(-(r1 - r[mask_bed_film]) / delta_bed_film)

# ============================================================
# Region 2: wall (r1 < r <= r_wall)
# Essentially flat at T_wall - the steel's own conduction resistance
# is negligible (R_cond_wall computed above is ~3-4 orders of
# magnitude smaller than the insulation's), so almost none of the
# bed-to-ambient temperature drop happens across the metal itself.
# ============================================================
mask_wall = (r > r1) & (r <= r_wall)
T[mask_wall] = T_wall

# ============================================================
# Region 3: insulation (r_wall < r <= r2)
# True radial conduction through a cylindrical shell gives a
# logarithmic profile, not a straight line, because the area the
# heat flows through grows with r while Q stays constant.
# ============================================================
mask_ins = (r > r_wall) & (r <= r2)
T[mask_ins] = T_wall - log_coeff * np.log(r[mask_ins] / r_wall)

# ============================================================
# Region 4: outer air film down to ambient (r2 < r <= r_far)
# NOT a real conduction distance - convection is a lumped surface
# relationship (Newton's law of cooling), not a function of r. Drawn as
# an exponential decay (steepest gradient at the wall, flattening
# asymptotically toward T_amb) to match the classic thermal
# boundary-layer profile, rather than a straight line. The decay
# length is chosen only so the curve looks ~95% relaxed by r_far -
# it is not derived from h_out (a real boundary-layer solution would
# require solving the momentum/energy equations, well beyond this
# lumped-parameter model).
# ============================================================
mask_film = r > r2
delta_film = (r_far - r2) / 3.0
T[mask_film] = T_amb + (T_surface - T_amb) * np.exp(-(r[mask_film] - r2) / delta_film)

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(r, T, color="#D85A30", lw=2)

ax.axvspan(0, r_film_start, color="#F0997B", alpha=0.15, label="Bed (bulk)")
ax.axvspan(r_film_start, r1, color="#C9455C", alpha=0.25, label="Bed-wall film")
ax.axvspan(r1, r_wall, color="gray", alpha=0.15, label="Wall")
ax.axvspan(r_wall, r2, color="#5DCAA5", alpha=0.15, label="Insulation (60 mm)")
ax.axvspan(r2, r_far, color="#85B7EB", alpha=0.15, label="Ambient film")

ax.set_xlabel("Radial position r (mm)")
ax.set_ylabel("Temperature T (\u00b0C)")
ax.set_title("Radial temperature profile: bed \u2192 film \u2192 wall \u2192 insulation \u2192 ambient")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("radial_temperature_profile.png", dpi=200)
plt.show()