# Thesis Overview

**Title:** Design of a Lab-Scale Reactor for Sorption-Enhanced CO2 Methanation with Bi-functional Catalyst-Sorbent Material

**Author:** Bart Ligtenberg
**Supervisor:** Prof.dr.ir. W. de Jong (TU Delft)
**Faculty:** Faculty of Process and Energy Engineering, Delft
**Duration:** March 2026 – December 2026

---

## Central Research Question

How should a lab-scale sorption-enhanced CO2 hydrogenation process with a bi-functional catalyst-sorbent material, continuous solid circulation between a reactor and a dedicated regeneration vessel, and electrically driven regeneration be designed and modelled?

Five sub-questions, addressed across the chapters below:
1. Most suitable target product (Ch. 2)
2. Most suitable reactor type (Ch. 4)
3. Predicted reactor performance under varying operating conditions (Ch. 6–7)
4. Configuration and energy balance of the surrounding process (future work — Aspen Plus)
5. Mechanical design of the full lab-scale process (future work — construction drawings)

---

## Target Product: Methane

Selected via weighted multi-criteria assessment across 5 criteria:

| Criterion | Weight | Methane | DME | Methanol | CO |
|-----------|--------|---------|-----|----------|----|
| Downstream process complexity | 35% | 5 | 3 | 1 | 1 |
| Infrastructure readiness | 20% | 5 | 3 | 3 | 1 |
| Bi-functional material maturity | 20% | 4 | 1 | 2 | 3 |
| Market value | 15% | 3 | 5 | 5 | 3 |
| Operating conditions | 10% | 5 | 3 | 2 | 1 |
| **Weighted total** | | **4.50** | **2.90** | **2.30** | **1.70** |

Decisive advantage: sorption enhancement on methanation already delivers a stream close to Dutch gas grid injection specification — only water removal needed downstream.

---

## Reactor Type: Moving Packed Bed (MPB)

Selected over fixed bed, fluidised bed (circulating BFB, parallel BFB, riser), and trickle-flow configurations via a 6-way weighted multi-criteria comparison (MPB scores 4.30, next best circulating BFB at 3.70) because it:
- Decouples gas velocity from solid throughput → widest operational bandwidth, no minimum-fluidisation floor
- Enables continuous solid circulation and regeneration → continuous production
- Only generates compressive contact forces at the base of the column (minimises attrition)
- Counter-current contact moderates peak temperature and maximises sorbent utilisation

**Process layout:** Two-stage (see block flow diagram, Ch. 4.5)
1. Conventional fixed-bed first stage, ~300–400°C — bulk conversion (~80% CO2 conversion, no sorbent)
2. Condenser removes bulk liquid water
3. MPB second stage, ~250–300°C — drives conversion to completion via sorption enhancement
4. Spent catalyst-sorbent withdrawn continuously from the reactor bottom, regenerated electrically (TSA, 300–400°C, N2 purge) in a separate vessel, returned to reactor top

---

## Bi-functional Material

**5 wt% Ni + 2.5 wt% Ce on zeolite 13X** (developed by Wei, 2022)

- Demonstrated 100% CO2 conversion and near-complete CH4 selectivity over 100 adsorption-regeneration cycles
- Catalytic and water-adsorbing functions integrated in a single particle
- Validated up to TRL 5 in fixed-bed reactors
- Particle diameter 0.5–1 mm, water uptake capacity ≈1.3 mmol/g_cat; bed diluted with 75.7 wt% inert filler to moderate volumetric heat release

---

## Reaction

**Sabatier reaction:**
CO2 + 4H2 ⇌ CH4 + 2H2O     ΔH°298 = −165 kJ/mol

Competing side reactions (neglected in the reactor model — see below):
- CO2 + H2 ⇌ CO + H2O     ΔH° = +41 kJ/mol (rWGS)
- CO + 3H2 ⇌ CH4 + H2O    ΔH° = −206 kJ/mol

**Operating window:** 200–300°C, atmospheric to moderate pressure

---

## Kinetics Model

Langmuir-Hinshelwood-Hougen-Watson (LHHW) model from Koschany, Schlereth, and Hinrichsen. Validated over Ni-based catalyst at 180–360°C and 1–15 bar.

Rate expression:

r = k · (pH2 · pCO2)^0.5 · (1 − pCH4·pH2O² / (pH2²·pCO2·Keq)) / (1 + KOH·pH2O/pH2^0.5 + KH2·pH2^0.5 + Kmix·pCO2)²

Where r is in mol kg_cat⁻¹ s⁻¹ and all partial pressures are in bar.

Combined with a Dubinin–Astakhov equilibrium isotherm for water uptake on the zeolite and a linear-driving-force (LDF) mass-transfer coefficient for adsorption kinetics.

---

## Reactor Model (Chapter 6)

1D, steady-state, pseudo-homogeneous model of the MPB reactor along the axial coordinate. Gas and solid flow counter-currently, so the model is a split boundary-value problem (gas BCs at the inlet, solid loading/temperature BC at the solid inlet); solved by Gauss–Seidel iteration alternating between a gas-phase initial-value problem and a solid-phase initial-value problem, each integrated with a stiff BDF solver (SciPy `solve_ivp`).

Key assumptions (each checked against a dimensionless criterion, see Table 6.2 in the PDF):
- Plug flow, no axial dispersion (gas or solid)
- 1D (radial gradients neglected; bounded conservatively at ≈17 K center-to-wall)
- Pseudo-homogeneous (gas and solid at the same local temperature; NTU ≈ 7×10³)
- Negligible pressure drop (Ergun equation, ΔP/P ≈ 1%)
- Intrinsic kinetics — no internal mass-transfer limitation (Weisz–Prater)
- Only the Sabatier reaction modelled (CH4 selectivity = 1); only water adsorbs (no CO2 co-adsorption)
- Feed to the MPB stage fixed at 80% CH4 / 16% H2 / 4% CO2 (dry basis — see Section 7.5 note below)

**Validation:** no moving-bed SEM experiment exists in the literature, so the assembled reactor model cannot be validated end-to-end. Instead, each constitutive sub-model is validated separately against Wei's fixed-bed breakthrough data (Fig. 5.10): the catalytically active mass fraction is fitted to the non-sorption-enhanced conversion (20 wt% active fraction reproduces it well), and the isotherm capacity W₀ is fitted to the measured breakthrough capacity, using two independent, non-overlapping datasets. The LHHW kinetic parameters and the LDF coefficient K_LDF are carried over unfitted from literature (Koschany et al.; Mette et al.). The coupled breakthrough experiment then reproduces the plateau conversion almost exactly (99.5% modelled vs 100.0% measured) and the breakthrough timing to within batch-to-batch variation, with the modelled front slightly steeper than measured (attributed to K_LDF not transferring exactly from pure zeolite to the bi-functional composite).

---

## Results and Design Point (Chapter 7)

**Capacity margin Λ** (Eq. 7.1) — ratio of the solid's maximum water-removal rate to the reaction's water production rate — is the central operability criterion. Λ ≥ 2.5 is adopted as the operating floor (below this the sorbent partially desorbs near the solid outlet and the solver loses convergence); a solid-velocity floor u_s ≥ 3 mm/s is added for a separate numerical/physical reason (residence time too long → poor solver coupling, and physically difficult to maintain uniform slow solids flow).

**Operating envelope:** reactor length L = 2 m, internal diameter d_t = 50 mm, sized against the worst case in the target window (GHSV = 2.0 m³/(kg_ads·h) reaches 99% conversion at z = 1.72 m, giving a ~16% length margin). Operating window: GHSV ∈ [0, 2] m³_STP/(kg_ads·h), corresponding u_s ∈ [3, 7.7] mm/s, with a feedforward control law (Eq. 7.2) holding Λ = 2.5 across the range.

**Design point** (GHSV = 1.5, 75% of the envelope): u_s = 5.8 mm/s, CO2 conversion 99.77% (vs. 84.5% unenhanced thermodynamic ceiling), temperature rise only +12 K (peak at z ≈ 0.02 m, decaying back to inlet by z ≈ 1.25 m) because the counter-current solid stream removes 88.5% of generated heat (wall removes 11.5%, gas net sensible heat ≈ 0). Mass and energy balances close to <0.3%.

**Sensitivity analysis (7.4):** of the three sub-model parameters carried over from literature, the LHHW rate constant *k* is the dominant sensitivity (halving *k* drops conversion 99.77% → 96.72% and roughly doubles the bed length needed for 95% conversion); K_LDF and the isotherm capacity W₀ have only marginal effect even under 5–25× and 2× uncertainty respectively. The design has built-in flexibility (bed dilution currently 75.7 wt%, and u_s/GHSV as free operating variables) to absorb this uncertainty once experimental data on the real material is available.

---

## Open / In Progress: Section 7.5 "Other Operating Scenarios"

This section of the v19 draft is currently just a stub — six candidate scenarios listed but not yet analysed or written up:

1. **Feed water content** — currently in progress (relaxing the assumption that the condenser removes all water before the MPB stage; see `Sensitivity analysis and scenarios/water in feed/`)
2. Different first-stage conversion (currently fixed at ~80%)
3. Temperature analysis / colder zeolite entering (relaxing T(solid inlet) = T_in)
4. Adiabatic operation (no wall cooling)
5. Little-to-no bed dilution (currently 75.7 wt% inert filler)
6. Not-fully-regenerated zeolite at inlet (relaxing q(solid inlet) = 0)



## Thesis Chapter Structure (v19)

| Chapter | Topic | Status |
|---------|-------|--------|
| 1 | Introduction, motivation & research questions | Done |
| 2 | Target product selection | Done |
| 3 | SEM literature review | Done |
| 4 | Basis of design (reactor selection, block flow diagram) | Done |
| 5 | Conclusions & research plan | Done |
| 6 | Reactor model (equations, assumptions, sub-model validation) | Done |
| 7 | Results & discussion (capacity margin, operating envelope, design point, sensitivity) | Done through 7.4; **7.5 "Other operating scenarios" in progress** |
| A | GANTT chart | Done |
| B | Riser calculation (supports Ch. 4 riser elimination) | Done |
| — | Process energy balance (Aspen Plus) | To do |
| — | Mechanical design drawings | To do |

---

## Key Reference

Wei, L. (2022). *Sorption enhanced CO2 methanation for large scale energy storage: Catalyst and Process development.* PhD thesis, Delft University of Technology.
