"""
Cascade methanation feed calculation
Two conventional reactors with flash separators (25C, 25 bar).

Reaction (Sabatier):  CO2 + 4H2 -> CH4 + 2H2O   (dH = -165 kJ/mol, strongly exothermic)

The purpose of cascading two reactors with an intermediate flash is to overcome
the equilibrium limitation of methanation: removing the product water shifts the
equilibrium back towards methane, allowing higher overall CO2 conversion.

Assumptions:
- 80% CO2 conversion per reactor pass (kinetically limited; equilibrium is >99% below ~400C)
- Flash separators operate at 25C and 25 bar
- Only water condenses in the flash; H2, CO2, CH4 are supercritical at 25C and stay in gas phase
- Basis: 1 mol total feed
"""

# ── Feed ─────────────────────────────────────────────────────────────────────
X_per_pass = 0.80   # CO2 conversion per reactor pass

# Stoichiometric H2/CO2 ratio for CO2 methanation is 4:1, so 80% H2 + 20% CO2
feed = {"H2": 0.80, "CO2": 0.20, "CH4": 0.00, "H2O": 0.00}  # moles (basis: 1 mol total)


def react(stream, conversion):
    """
    Apply one reactor pass: CO2 + 4H2 -> CH4 + 2H2O.

    The stoichiometry fixes the ratios: every mole of CO2 converted consumes
    4 mol H2 and produces 1 mol CH4 and 2 mol H2O. Total moles decrease because
    5 moles of reactants become 3 moles of products.
    """
    n_CO2_reacted = conversion * stream["CO2"]
    return {
        "H2":  stream["H2"]  - 4 * n_CO2_reacted,   # 4 mol H2 consumed per mol CO2
        "CO2": stream["CO2"] -     n_CO2_reacted,
        "CH4": stream["CH4"] +     n_CO2_reacted,    # 1 mol CH4 produced per mol CO2
        "H2O": stream["H2O"] + 2 * n_CO2_reacted,   # 2 mol H2O produced per mol CO2
    }


def flash(stream, P_bar=25.0, T_C=25.0):
    """
    Model a flash drum: cool and hold at pressure, drain condensed water.

    Physics: at temperature T, water has a saturation pressure P_sat below which
    it exists as liquid. In a gas mixture at total pressure P_total, the maximum
    mole fraction of water that can remain as vapour is (Raoult's law, dilute limit):

        y_H2O = P_sat / P_total

    Any water above this fraction condenses and is removed as liquid.
    High operating pressure is therefore beneficial: it forces more water out
    (lower y_H2O_sat), which also shifts the methanation equilibrium towards CH4.

    H2, CO2, and CH4 are well above their critical temperatures at 25C, so they
    have no meaningful vapour pressure and pass through the flash unchanged.

    P_sat is calculated from the Antoine equation (valid 1-100C):
        log10(P_sat / mmHg) = A - B / (C + T)
    with A=8.07131, B=1730.63, C=233.426 for water.
    """
    P_sat = 10 ** (8.07131 - 1730.63 / (233.426 + T_C)) * 0.00133322  # mmHg -> bar

    # Maximum water vapour fraction at equilibrium (Raoult's law)
    y_H2O_sat = P_sat / P_bar

    # Non-condensable species pass through unchanged
    dry = {k: v for k, v in stream.items() if k != "H2O"}
    n_dry = sum(dry.values())

    # Solve for water remaining in vapour: y = n_H2O / (n_dry + n_H2O)
    n_H2O_vapour = y_H2O_sat * n_dry / (1 - y_H2O_sat)
    water_removed = stream["H2O"] - n_H2O_vapour   # liquid drained from drum

    out = dict(dry)
    out["H2O"] = n_H2O_vapour
    return out, water_removed, P_sat


def normalize(stream):
    """Divide each species by total moles to get mole fractions."""
    total = sum(stream.values())
    return {k: v / total for k, v in stream.items()}, total


def print_stream(label, stream):
    normed, total = normalize(stream)
    print(f"\n  {label}")
    print(f"  {'Species':<8} {'Moles':>10}  {'Mol%':>8}")
    print(f"  {'-'*30}")
    for sp, n in stream.items():
        print(f"  {sp:<8} {n:>10.4f}  {normed[sp]*100:>7.2f}%")
    print(f"  {'Total':<8} {total:>10.4f}")


# ── Stage 1 ───────────────────────────────────────────────────────────────────
print("=" * 50)
print("  CASCADE METHANATION — FEED CALCULATION")
print("=" * 50)

print_stream("Feed", feed)

after_r1 = react(feed, X_per_pass)
print_stream("After Reactor 1 (80% conv.)", after_r1)

# Flash 1: removes most of the water produced in Reactor 1.
# This allows Reactor 2 to start with a lower water content, shifting
# the equilibrium and enabling further CO2 conversion.
after_f1, w1, Psat1 = flash(after_r1)
print(f"\n  Flash 1 — P_sat(25C) = {Psat1*1000:.2f} mbar | water removed: {w1:.4f} mol")
print_stream("After Flash 1", after_f1)

# ── Stage 2 ───────────────────────────────────────────────────────────────────
after_r2 = react(after_f1, X_per_pass)
print_stream("After Reactor 2 (80% conv.)", after_r2)

# Flash 2: final water removal before the gas enters the MPB reactor.
# Residual water vapour is set by VLE at 25C/25bar (~0.13 mol%).
# Even this small amount matters for the MPB: water adsorbs preferentially
# on 13X zeolite over CO2, so it competes for sorbent active sites.
after_f2, w2, Psat2 = flash(after_r2)
print(f"\n  Flash 2 — P_sat(25C) = {Psat2*1000:.2f} mbar | water removed: {w2:.4f} mol")
print_stream("After Flash 2 (MPB feed)", after_f2)

# ── Overall summary ───────────────────────────────────────────────────────────
CO2_in  = feed["CO2"]
CO2_out = after_f2["CO2"]
overall_X = (CO2_in - CO2_out) / CO2_in

final_norm, _ = normalize(after_f2)

print("\n" + "=" * 50)
print("  SUMMARY")
print("=" * 50)
print(f"  Overall CO2 conversion : {overall_X*100:.1f}%")
print(f"  Final CH4 purity       : {final_norm['CH4']*100:.2f} mol%")
print(f"  Residual H2            : {final_norm['H2']*100:.2f} mol%")
print(f"  Residual CO2           : {final_norm['CO2']*100:.2f} mol%")
print(f"  Residual H2O (vapour)  : {final_norm['H2O']*100:.4f} mol%  <- MPB feed water content")
