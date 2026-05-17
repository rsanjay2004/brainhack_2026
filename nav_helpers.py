# nav_helpers.py
#
# Pure-function navigation helpers. No drone state, no async — safe to call
# from anywhere. Split out so qualifier_main.py is not cluttered with
# math utilities.

import math


def yaw_to_cardinal(yaw_deg: float) -> str:
    """Convert NED yaw (0=North, 90=East) to 8-point cardinal label."""
    d = yaw_deg % 360
    idx = int((d + 22.5) / 45) % 8
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][idx]


def sector_cardinals(yaw_deg: float):
    """Return (fwd_card, left_card, right_card) labels for L/C/R camera sectors."""
    fwd  = yaw_deg % 360
    left = (yaw_deg - 90) % 360
    rgt  = (yaw_deg + 90) % 360
    return yaw_to_cardinal(fwd), yaw_to_cardinal(left), yaw_to_cardinal(rgt)


def nearest_on_seg(n, e, n1, e1, n2, e2):
    """Return point on segment (n1,e1)-(n2,e2) closest to (n,e)."""
    dn, de = n2 - n1, e2 - e1
    seg_sq = dn * dn + de * de
    if seg_sq < 1e-9:
        return n1, e1
    t = max(0.0, min(1.0, ((n - n1) * dn + (e - e1) * de) / seg_sq))
    return n1 + t * dn, e1 + t * de
