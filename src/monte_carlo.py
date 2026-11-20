"""Monte Carlo simulation for NPV sensitivity under price, grade, and cost uncertainty."""

import numpy as np
import pandas as pd


def run_monte_carlo(
    selected_stopes: list[dict],
    dev_cost: float,
    base_gold_price: float = 1900.0,
    base_mining_cost: float = 45.0,
    base_processing_cost: float = 25.0,
    base_ga_cost: float = 10.0,
    recovery: float = 0.92,
    n_simulations: int = 1000,
    price_std_pct: float = 0.15,
    grade_std_pct: float = 0.10,
    cost_std_pct: float = 0.08,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    stope_tonnages = np.array([s["total_tonnage"] for s in selected_stopes])
    stope_metal_oz = np.array([s["total_metal_oz"] for s in selected_stopes])
    stope_avg_grades = np.array([s["avg_grade"] for s in selected_stopes])

    total_tonnage = stope_tonnages.sum()
    total_metal = stope_metal_oz.sum()

    prices = rng.normal(base_gold_price, base_gold_price * price_std_pct, n_simulations)
    prices = np.maximum(prices, 500)

    grade_factors = rng.normal(1.0, grade_std_pct, n_simulations)
    grade_factors = np.maximum(grade_factors, 0.5)

    mining_costs = rng.normal(base_mining_cost, base_mining_cost * cost_std_pct, n_simulations)
    proc_costs = rng.normal(base_processing_cost, base_processing_cost * cost_std_pct, n_simulations)
    ga_costs = rng.normal(base_ga_cost, base_ga_cost * cost_std_pct, n_simulations)

    npvs = np.zeros(n_simulations)
    revenues = np.zeros(n_simulations)
    total_costs_arr = np.zeros(n_simulations)

    for i in range(n_simulations):
        adj_metal = total_metal * grade_factors[i]
        rev = adj_metal * prices[i]
        cost = total_tonnage * (mining_costs[i] + proc_costs[i] + ga_costs[i]) + dev_cost
        npvs[i] = rev - cost
        revenues[i] = rev
        total_costs_arr[i] = cost

    percentiles = {
        "P10": float(np.percentile(npvs, 10)),
        "P25": float(np.percentile(npvs, 25)),
        "P50": float(np.percentile(npvs, 50)),
        "P75": float(np.percentile(npvs, 75)),
        "P90": float(np.percentile(npvs, 90)),
    }

    prob_positive = float(np.mean(npvs > 0) * 100)

    # Tornado chart data: sensitivity of NPV to each variable
    base_rev = total_metal * base_gold_price
    base_cost = total_tonnage * (base_mining_cost + base_processing_cost + base_ga_cost) + dev_cost
    base_npv = base_rev - base_cost

    tornado = []
    for name, low_mult, high_mult in [
        ("Gold Price", 0.8, 1.2),
        ("Grade", 0.85, 1.15),
        ("Mining Cost", 1.15, 0.85),
        ("Processing Cost", 1.15, 0.85),
    ]:
        if name == "Gold Price":
            npv_low = total_metal * base_gold_price * low_mult - base_cost
            npv_high = total_metal * base_gold_price * high_mult - base_cost
        elif name == "Grade":
            npv_low = total_metal * low_mult * base_gold_price - base_cost
            npv_high = total_metal * high_mult * base_gold_price - base_cost
        elif name == "Mining Cost":
            npv_low = base_rev - (total_tonnage * (base_mining_cost * low_mult + base_processing_cost + base_ga_cost) + dev_cost)
            npv_high = base_rev - (total_tonnage * (base_mining_cost * high_mult + base_processing_cost + base_ga_cost) + dev_cost)
        elif name == "Processing Cost":
            npv_low = base_rev - (total_tonnage * (base_mining_cost + base_processing_cost * low_mult + base_ga_cost) + dev_cost)
            npv_high = base_rev - (total_tonnage * (base_mining_cost + base_processing_cost * high_mult + base_ga_cost) + dev_cost)

        tornado.append({
            "variable": name,
            "npv_low": round(npv_low, 2),
            "npv_high": round(npv_high, 2),
            "range": round(abs(npv_high - npv_low), 2),
        })

    tornado.sort(key=lambda t: t["range"], reverse=True)

    return {
        "npvs": npvs.tolist(),
        "revenues": revenues.tolist(),
        "total_costs": total_costs_arr.tolist(),
        "prices_used": prices.tolist(),
        "grade_factors_used": grade_factors.tolist(),
        "mean_npv": float(np.mean(npvs)),
        "std_npv": float(np.std(npvs)),
        "percentiles": percentiles,
        "prob_positive_pct": prob_positive,
        "base_npv": base_npv,
        "tornado": tornado,
        "n_simulations": n_simulations,
    }
