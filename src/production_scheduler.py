"""Production scheduling — assign stopes to periods to maximize discounted NPV."""

import numpy as np


def schedule_production(
    selected_stopes: list[dict],
    dev_network: dict,
    mining_rate_tpa: float = 500_000.0,
    discount_rate: float = 0.08,
    gold_price: float = 1900.0,
    mining_cost: float = 45.0,
    processing_cost: float = 25.0,
    ga_cost: float = 10.0,
    recovery: float = 0.92,
) -> dict:
    if not selected_stopes:
        return {"periods": [], "total_dcf_npv": 0, "undiscounted_npv": 0, "mine_life_years": 0}

    stopes = sorted(selected_stopes, key=lambda s: -s["avg_grade"])

    periods = []
    period = 1
    tonnage_remaining = mining_rate_tpa
    period_stopes = []
    period_tonnage = 0
    period_metal = 0
    period_value = 0

    for s in stopes:
        if s["total_tonnage"] <= tonnage_remaining:
            period_stopes.append(s["stope_id"])
            period_tonnage += s["total_tonnage"]
            period_metal += s["total_metal_oz"]
            period_value += s["total_value"]
            tonnage_remaining -= s["total_tonnage"]
        else:
            if period_stopes:
                revenue = period_metal * recovery * gold_price
                opex = period_tonnage * (mining_cost + processing_cost + ga_cost)
                periods.append({
                    "period": period,
                    "stope_ids": period_stopes,
                    "n_stopes": len(period_stopes),
                    "tonnage": round(period_tonnage, 1),
                    "metal_oz": round(period_metal, 2),
                    "avg_grade": round(period_metal * 31.1035 / max(period_tonnage, 1), 3),
                    "revenue": round(revenue, 2),
                    "opex": round(opex, 2),
                    "cash_flow": round(revenue - opex, 2),
                })
            period += 1
            period_stopes = [s["stope_id"]]
            period_tonnage = s["total_tonnage"]
            period_metal = s["total_metal_oz"]
            period_value = s["total_value"]
            tonnage_remaining = mining_rate_tpa - s["total_tonnage"]

    if period_stopes:
        revenue = period_metal * recovery * gold_price
        opex = period_tonnage * (mining_cost + processing_cost + ga_cost)
        periods.append({
            "period": period,
            "stope_ids": period_stopes,
            "n_stopes": len(period_stopes),
            "tonnage": round(period_tonnage, 1),
            "metal_oz": round(period_metal, 2),
            "avg_grade": round(period_metal * 31.1035 / max(period_tonnage, 1), 3),
            "revenue": round(revenue, 2),
            "opex": round(opex, 2),
            "cash_flow": round(revenue - opex, 2),
        })

    dev_cost_per_period = dev_network.get("total_cost", 0) / max(len(periods), 1)

    total_dcf = 0
    undiscounted = 0
    cumulative_cf = 0
    for p in periods:
        cf = p["cash_flow"] - dev_cost_per_period
        p["dev_cost"] = round(dev_cost_per_period, 2)
        p["net_cash_flow"] = round(cf, 2)
        discount_factor = 1 / (1 + discount_rate) ** p["period"]
        p["discount_factor"] = round(discount_factor, 4)
        p["discounted_cf"] = round(cf * discount_factor, 2)
        total_dcf += cf * discount_factor
        undiscounted += cf
        cumulative_cf += cf * discount_factor
        p["cumulative_dcf"] = round(cumulative_cf, 2)

    return {
        "periods": periods,
        "total_dcf_npv": round(total_dcf, 2),
        "undiscounted_npv": round(undiscounted, 2),
        "mine_life_years": len(periods),
        "discount_rate": discount_rate,
        "mining_rate_tpa": mining_rate_tpa,
    }
