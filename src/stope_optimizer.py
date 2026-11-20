"""Optimize stope layout using Binary Integer Linear Programming (PuLP 4.x API)."""

import pulp


def optimize_stope_layout(
    candidates: list[dict],
    block_to_stopes: dict,
    min_pillar_blocks: int = 1,
    time_limit: int = 120,
) -> dict:
    if not candidates:
        return {"status": "No candidates", "selected": [], "total_value": 0,
                "n_selected": 0, "n_candidates": 0, "total_tonnage": 0, "total_metal_oz": 0}

    prob = pulp.LpProblem(name="StopeLO", sense=pulp.const.LpMaximize)

    x = {}
    for s in candidates:
        x[s["stope_id"]] = prob.add_variable(name=f"x_{s['stope_id']}", cat="Binary")

    prob.setObjective(
        sum(s["total_value"] * x[s["stope_id"]] for s in candidates)
    )

    for bid, sids in block_to_stopes.items():
        if len(sids) > 1:
            prob.addConstraint(
                sum(x[sid] for sid in sids) <= 1,
                name=f"overlap_b{bid}"
            )

    # Pillar constraints via shared blocks
    checked = set()
    for s in candidates:
        for other_s in candidates:
            if s["stope_id"] >= other_s["stope_id"]:
                continue
            pair = (s["stope_id"], other_s["stope_id"])
            if pair in checked:
                continue
            checked.add(pair)
            if set(s["block_ids"]) & set(other_s["block_ids"]):
                prob.addConstraint(
                    x[s["stope_id"]] + x[other_s["stope_id"]] <= 1,
                    name=f"pillar_{pair[0]}_{pair[1]}"
                )

    solver = pulp.HiGHS(msg=False, timeLimit=time_limit)
    result = prob.solve(solver)

    selected = []
    total_value = 0.0
    total_tonnage = 0.0
    total_metal = 0.0

    for s in candidates:
        v = x[s["stope_id"]].value()
        if v is not None and v > 0.5:
            selected.append(s)
            total_value += s["total_value"]
            total_tonnage += s["total_tonnage"]
            total_metal += s["total_metal_oz"]

    status_str = str(result).split("status: ")[1].split(" (")[0] if "status:" in str(result) else str(result)

    return {
        "status": status_str,
        "selected": selected,
        "n_selected": len(selected),
        "n_candidates": len(candidates),
        "total_value": round(total_value, 2),
        "total_tonnage": round(total_tonnage, 2),
        "total_metal_oz": round(total_metal, 4),
    }


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    from stope_generator import generate_candidate_stopes

    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / "economic_block_model.csv")

    candidates, b2s = generate_candidate_stopes(
        df, step=3, min_width=2, max_width=3,
        min_height=3, max_height=4, min_length=2, max_length=3
    )
    print(f"Candidates: {len(candidates)}")

    result = optimize_stope_layout(candidates, b2s)
    print(f"Status: {result['status']}")
    print(f"Selected: {result['n_selected']} / {result['n_candidates']}")
    print(f"Total value: ${result['total_value']:,.0f}")
    print(f"Total tonnage: {result['total_tonnage']:,.0f} t")
    print(f"Total metal: {result['total_metal_oz']:,.1f} oz")
