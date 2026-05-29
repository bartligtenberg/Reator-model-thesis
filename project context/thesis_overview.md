# Thesis Overview

**Title:** Design of a Lab-Scale Reactor for Sorption-Enhanced CO2 Methanation with Bi-functional Catalyst-Sorbent Material

**Author:** Bart Ligtenberg
**Supervisor:** Prof.dr.ir. W. de Jong (TU Delft)
**Faculty:** Faculty of Process and Energy Engineering, Delft
**Duration:** March 2026 – December 2026

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

Selected over fixed bed, fluidised bed, and other configurations because it:
- Decouples gas velocity from solid throughput
- Enables continuous solid circulation and regeneration
- Only generates compressive contact forces (minimises attrition)
- Counter-current contact moderates peak temperature and maximises sorbent utilisation

**Process layout:** Two-stage
1. Conventional fixed-bed first stage — bulk conversion
2. MPB second stage — drives conversion to completion
3. Spent solid continuously transferred to a separate regenerator and returned to reactor top

---

## Bi-functional Material

**5 wt% Ni + 2.5 wt% Ce on zeolite 13X** (developed by Wei, 2022)

- Demonstrated 100% CO2 conversion and near-complete CH4 selectivity over 100 adsorption-regeneration cycles
- Catalytic and water-adsorbing functions integrated in a single particle
- Validated up to TRL 5 in fixed-bed reactors

---

## Reaction

**Sabatier reaction:**
CO2 + 4H2 ⇌ CH4 + 2H2O     ΔH°298 = −165 kJ/mol

Competing side reactions:
- CO2 + H2 ⇌ CO + H2O     ΔH° = +41 kJ/mol (rWGS)
- CO + 3H2 ⇌ CH4 + H2O    ΔH° = −206 kJ/mol

**Operating window:** 200–300°C, atmospheric to moderate pressure

---

## Kinetics Model

Langmuir-Hinshelwood-Hougen-Watson (LHHW) model from Koschany, Schlereth, and Hinrichsen. Validated over Ni-based catalyst at 180–360°C and 1–15 bar.

Rate expression:

r = k · (pH2 · pCO2)^0.5 · (1 − pCH4·pH2O² / (pH2²·pCO2·Keq)) / (1 + KOH·pH2O/pH2^0.5 + KH2·pH2^0.5 + Kmix·pCO2)²

Where r is in mol kg_cat⁻¹ s⁻¹ and all partial pressures are in bar.

---

## Thesis Chapter Structure

| Chapter | Topic | Status |
|---------|-------|--------|
| 1 | Introduction & motivation | Done (literature review) |
| 2 | Target product selection | Done |
| 3 | SEM literature review | Done |
| 4 | Basis of design (reactor selection, block flow diagram) | Done |
| 5 | Conclusions & research plan | Done |
| — | Reactor performance model | To do |
| — | Process energy balance | To do |
| — | Mechanical design drawings | To do |

---

## Key Reference

Wei, L. (2022). *Sorption enhanced CO2 methanation for large scale energy storage: Catalyst and Process development.* PhD thesis, Delft University of Technology.
