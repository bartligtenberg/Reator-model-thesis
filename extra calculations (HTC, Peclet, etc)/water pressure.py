import numpy as np


def P_sat_bar(T_K):
    """
    Saturation vapour pressure of water [bar]. Extended Antoine equation (Wexler-Hyland form); coefficients give log10(P/mmHg),
    converted to bar via 1 mmHg = 133.322e-5 bar. Valid over a wide T range (unlike a simple 2-constant Antoine fit, which only holds ~273-303 K).
    np.clip prevents overflow at extreme temperatures.
    """
    log10_p = (29.8605 - 3.1522e3/T_K - 7.3037*np.log10(T_K)
               + 2.4247e-9*T_K + 1.8090e-6*T_K**2)
    return 10.0**np.clip(log10_p, -10, 10) * 133.322e-5   # [mmHg] -> [bar]


def P_sat_bar2(T_K):
    """
    Saturation vapour pressure of water [bar]  —  Antoine equation.
    Copied unchanged from adsorption_simulation.py.
    """
    return 10.0 ** (5.40221 - 1838.675 / (T_K - 31.737))


print(P_sat_bar(500))
print(P_sat_bar2(500))
