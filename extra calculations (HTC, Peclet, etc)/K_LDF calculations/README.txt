K_LDF CALCULATIONS — SUMMARY
=============================
Script:  plot_isotherm.py
Output:  isotherm_280_300C.png
Material: binderless zeolite 13X (Köstrolith 13XBFK)
Isotherm: Dubinin-Astakhov (DA), parameters from Mette et al. (2014)
  W0 = 341.00e-6 m³/kg,  E = 1192.25 kJ/kg,  n = 1.55


WHAT WAS COMPUTED
-----------------
The DA isotherm gives the equilibrium H2O loading q* [mol/kg] on the sorbent
as a function of temperature T and H2O partial pressure p_H2O:

  A  = (R/MW_H2O) * T * ln(p_sat / p)        [adsorption potential, J/kg]
  W  = W0 * exp(-(A/E)^n)                     [adsorbed volume, m³/kg]
  q* = (rho_ads / MW_H2O) * W                 [mol/kg]

Saturation pressure p_sat from the Antoine equation (NIST, valid 274-441 K):
  log10(p_sat [bar]) = 5.40221 - 1838.675 / (T[K] - 31.737)

The K_LDF (Linear Driving Force coefficient) governs how fast the actual
solid loading q approaches equilibrium q*:

  dq/dt = K_LDF * (q* - q)

with K_LDF derived from pore diffusion in a spherical pellet:

  K_LDF = 15 * eps_p * D_M / (r_p^2 * tau_p * rho_p * R * T * dq*/dp_Pa)

where dq*/dp_Pa is the isotherm slope in mol/(kg·Pa).

Pellet parameters (Bareschino lab setup, MPB model):
  d_p   = 0.75 mm    particle diameter
  eps_p = 0.615      intraparticle porosity
  tau_p = 3.0        pore tortuosity
  rho_p = 1400 kg/m³ particle density
  D_M   = 2.5e-5 * (T/300)^1.75  m²/s   (Chapman-Enskog scaling)
  K_LDF_MAX = 20 s⁻¹  (numerical cap for flat-isotherm singularity)


EQUILIBRIUM LOADINGS AT KEY CONDITIONS (280 °C / 300 °C)
---------------------------------------------------------
  p_H2O [mbar]   q* 280C [mol/kg]   q* 300C [mol/kg]
     0.1              0.10               0.06
     1.0              0.39               0.26
     5.0              0.88               0.63
    10.0              1.23               0.91
    80.0              3.10               2.45   ← reaction zone (~4% CO2, 100% conv.)
   100.0              3.39               2.70

The reaction zone (methanation products) has p_H2O ≈ 80 mbar at 1 bar total,
4% CO2 feed, 100% conversion. The feed zone has p_H2O < 10 mbar.


MAIN CONCLUSIONS
----------------

1. ISOTHERM SHAPE (panel 1 & 3)
   The DA isotherm is concave on a linear pressure axis — steepest near p=0,
   continuously flattening with increasing p. This is Type I micropore filling,
   NOT an S-shaped isotherm. On a log axis (panel 3) the shape matches
   Mette et al. (2014) Fig. 2 left. At 280°C the loading is higher than at
   300°C at every pressure, reflecting lower equilibrium capacity at higher T.

2. ISOTHERM SLOPE dq*/dp (panel 2)
   The slope dq*/dp is largest (600+ mol/kg/bar) at very low p_H2O and decays
   to near-zero at high p_H2O. This slope is the thermodynamic factor in K_LDF:
   large slope → small K_LDF → slow adsorption/desorption kinetics.

3. K_LDF BEHAVIOUR (panel 4)
   K_LDF starts near zero at low p_H2O (< 1 mbar) and rises to ~5-7 s⁻¹ by
   300 mbar. Within the reactor operating range (0-100 mbar), the cap of
   20 s⁻¹ is never reached — adsorption kinetics are genuinely rate-limited
   throughout, not just in the desorption zone.

4. KINETIC ASYMMETRY — WHY THE MPB WORKS
   Adsorption zone (reaction zone, ~80 mbar):
     - dq*/dp ≈ 10-20 mol/kg/bar  →  K_LDF ≈ 1.5-3 s⁻¹  (moderate, adequate)
   Desorption zone (feed zone, < 10 mbar):
     - dq*/dp ≈ 100-600 mol/kg/bar  →  K_LDF < 1 s⁻¹  (slow)

   The DA isotherm creates a natural kinetic asymmetry: adsorption in the
   reaction zone is faster than desorption in the feed zone. This means the
   solid retains most of its loading as it passes through the feed zone and
   exits the reactor still carrying water — which is exactly the desired
   mechanism for water removal from the gas phase.

5. DESORPTION IS DELIBERATELY INCOMPLETE
   The thermodynamic driving force to desorb in the feed zone is large:
   q* drops from ~3 mol/kg (reaction zone) to ~0.4-1.2 mol/kg (feed zone).
   But K_LDF < 1 s⁻¹ means the solid only partially equilibrates during its
   residence time. It exits still carrying ~2-3 mol/kg — removing water from
   the reactor. If the solid fully equilibrated, it would dump water back into
   the feed gas. The slow desorption kinetics are therefore a feature, not a bug.

6. WORKING CAPACITY
   The working capacity per cycle is approximately q_exit_rxn - q_entry_rxn.
   If the solid spends too long in the feed zone, it over-desorbs (q → 0.4 mol/kg)
   and arrives at the reaction zone with low loading → max working capacity next
   cycle but no water exported. If it spends too little time, it enters the
   reaction zone already nearly saturated → small additional adsorption.
   Solid velocity u_s is therefore a key design variable.


REFERENCE
---------
Mette, B., Kerskes, H., Drück, H., Müller-Steinhagen, H. (2014).
"Experimental and numerical investigations on the water vapor adsorption
isotherms and kinetics of binderless zeolite 13X."
Int. J. Heat Mass Transfer, 71, 555-561.
