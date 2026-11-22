"""Flask backend for Automated Underground Mine Design web application."""

import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request

sys.path.insert(0, str(Path(__file__).parent / "src"))

from block_model_generator import generate_block_model
from economic_evaluator import evaluate_economics
from stope_generator import generate_candidate_stopes
from stope_optimizer import optimize_stope_layout
from mine_layout import generate_development_network
from monte_carlo import run_monte_carlo
from grade_prediction import train_grade_model
from production_scheduler import schedule_production
from geotechnical import assess_stability
from kriging import run_kriging_analysis, run_simple_kriging, run_indicator_kriging, run_sgs
from bolani_loader import load_drillhole_data, build_bolani_block_model, get_bolani_stats
from reserve_reporting import classify_resource, generate_reserve_table
from stochastic_optimizer import run_stochastic_optimization
from multi_objective import run_multi_objective_optimization
from implicit_geology import build_implicit_model
from ventilation import design_ventilation_network
from backfill import optimize_backfill

import numpy as np
import pandas as pd
import json

BASE_DIR = Path(__file__).parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))

# In-memory cache
cache = {}


def np_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: np_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [np_safe(i) for i in obj]
    return obj


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "service": "automated-underground-mine-design"})


@app.route("/api/generate-block-model", methods=["POST"])
def api_generate_block_model():
    p = request.json
    nx = p.get("nx", 30)
    ny = p.get("ny", 25)
    nz = p.get("nz", 15)
    block_size = p.get("block_size", 5)
    seed = p.get("seed", 42)

    df = generate_block_model(nx=nx, ny=ny, nz=nz, block_size=block_size, seed=seed)
    cache["block_model"] = df
    cache["block_size"] = block_size

    gold_price = p.get("gold_price", 1900)
    mining_cost = p.get("mining_cost", 45)
    processing_cost = p.get("processing_cost", 25)
    ga_cost = p.get("ga_cost", 10)
    recovery = p.get("recovery", 92) / 100

    df_econ, cutoff = evaluate_economics(
        df, gold_price_per_oz=gold_price, mining_cost_per_t=mining_cost,
        processing_cost_per_t=processing_cost, ga_cost_per_t=ga_cost,
        recovery=recovery, block_size=block_size,
    )
    cache["df_econ"] = df_econ
    cache["cutoff"] = cutoff
    cache["econ_params"] = {
        "gold_price": gold_price, "mining_cost": mining_cost,
        "processing_cost": processing_cost, "ga_cost": ga_cost,
        "recovery": recovery,
    }

    sample = df_econ.sample(min(5000, len(df_econ)), random_state=42)
    ore_blocks = df_econ[df_econ["above_cutoff"]]
    high_grade = df_econ[df_econ["grade_au_gt"] > 2.0]

    z_levels = sorted(df_econ["z"].unique().tolist())

    return jsonify(np_safe({
        "total_blocks": len(df_econ),
        "ore_blocks": int(df_econ["above_cutoff"].sum()),
        "mean_grade": round(float(df_econ["grade_au_gt"].mean()), 3),
        "max_grade": round(float(df_econ["grade_au_gt"].max()), 3),
        "cutoff_grade": round(float(cutoff), 3),
        "z_levels": z_levels,
        "all_blocks": {
            "x": sample["x"].tolist(), "y": sample["y"].tolist(),
            "z": sample["z"].tolist(), "grade": sample["grade_au_gt"].tolist(),
            "value": sample["economic_value"].tolist(),
        },
        "ore_blocks_data": {
            "x": ore_blocks["x"].tolist(), "y": ore_blocks["y"].tolist(),
            "z": ore_blocks["z"].tolist(), "grade": ore_blocks["grade_au_gt"].tolist(),
            "value": ore_blocks["economic_value"].tolist(),
        },
        "high_grade_data": {
            "x": high_grade["x"].tolist(), "y": high_grade["y"].tolist(),
            "z": high_grade["z"].tolist(), "grade": high_grade["grade_au_gt"].tolist(),
            "value": high_grade["economic_value"].tolist(),
        },
        "grade_histogram": {
            "ore_grades": ore_blocks["grade_au_gt"].tolist(),
            "waste_grades": df_econ[~df_econ["above_cutoff"]]["grade_au_gt"].tolist(),
        },
        "grade_tonnage": {
            "cum_tonnes": ore_blocks.sort_values("grade_au_gt", ascending=False)["tonnage"].cumsum().tolist(),
            "grades": ore_blocks.sort_values("grade_au_gt", ascending=False)["grade_au_gt"].tolist(),
        },
    }))


@app.route("/api/cross-section", methods=["POST"])
def api_cross_section():
    p = request.json
    z_val = p.get("z")
    if "df_econ" not in cache:
        return jsonify({"error": "Generate block model first"}), 400
    section = cache["df_econ"][cache["df_econ"]["z"] == z_val]
    return jsonify(np_safe({
        "x": section["x"].tolist(),
        "y": section["y"].tolist(),
        "grade": section["grade_au_gt"].tolist(),
    }))


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    if "df_econ" not in cache:
        return jsonify({"error": "Generate block model first"}), 400

    p = request.json
    block_size = cache["block_size"]

    candidates, b2s = generate_candidate_stopes(
        cache["df_econ"], block_size=block_size,
        min_width=p.get("min_width", 2), max_width=p.get("max_width", 3),
        min_height=p.get("min_height", 3), max_height=p.get("max_height", 5),
        min_length=p.get("min_length", 2), max_length=p.get("max_length", 4),
        step=p.get("step", 2),
    )

    if not candidates:
        return jsonify({"error": "No viable candidate stopes found"}), 400

    result = optimize_stope_layout(candidates, b2s)
    cache["opt_result"] = result

    selected_data = []
    for s in result["selected"]:
        selected_data.append({
            "stope_id": s["stope_id"],
            "x_min": s["x_min"], "x_max": s["x_max"],
            "y_min": s["y_min"], "y_max": s["y_max"],
            "z_min": s["z_min"], "z_max": s["z_max"],
            "total_value": s["total_value"],
            "total_tonnage": s["total_tonnage"],
            "total_metal_oz": s["total_metal_oz"],
            "avg_grade": s["avg_grade"],
            "width_m": s["width_m"], "height_m": s["height_m"],
            "length_m": s["length_m"], "ore_fraction": s["ore_fraction"],
        })

    ore = cache["df_econ"][cache["df_econ"]["above_cutoff"]]
    ore_sample = ore.sample(min(2000, len(ore)), random_state=42)

    return jsonify(np_safe({
        "status": result["status"],
        "n_candidates": result["n_candidates"],
        "n_selected": result["n_selected"],
        "total_value": result["total_value"],
        "total_tonnage": result["total_tonnage"],
        "total_metal_oz": result["total_metal_oz"],
        "selected": selected_data,
        "ore_background": {
            "x": ore_sample["x"].tolist(), "y": ore_sample["y"].tolist(),
            "z": ore_sample["z"].tolist(),
        },
    }))


@app.route("/api/mine-layout", methods=["POST"])
def api_mine_layout():
    if "opt_result" not in cache or not cache["opt_result"]["selected"]:
        return jsonify({"error": "Run optimization first"}), 400

    selected = cache["opt_result"]["selected"]
    block_size = cache["block_size"]
    dev = generate_development_network(selected, block_size=block_size)
    cache["dev"] = dev

    stope_boxes = []
    for s in selected:
        stope_boxes.append({
            "x_min": s["x_min"], "x_max": s["x_max"],
            "y_min": s["y_min"], "y_max": s["y_max"],
            "z_min": s["z_min"], "z_max": s["z_max"],
            "stope_id": s["stope_id"],
        })

    segments = []
    for seg in dev["segments"]:
        segments.append({
            "type": seg["type"],
            "x1": seg["x1"], "y1": seg["y1"], "z1": seg["z1"],
            "x2": seg["x2"], "y2": seg["y2"], "z2": seg["z2"],
            "length_m": seg["length_m"],
        })

    seg_df = pd.DataFrame(dev["segments"])
    summary = []
    if len(seg_df) > 0:
        for t, group in seg_df.groupby("type"):
            summary.append({
                "type": t, "count": len(group),
                "total_length": round(group["length_m"].sum(), 1),
                "cost": round(group["length_m"].sum() * 5000, 0),
            })

    return jsonify(np_safe({
        "total_length_m": dev["total_length_m"],
        "total_cost": dev["total_cost"],
        "n_levels": dev["n_levels"],
        "stope_boxes": stope_boxes,
        "segments": segments,
        "summary": summary,
    }))


@app.route("/api/economics", methods=["POST"])
def api_economics():
    if "opt_result" not in cache or not cache["opt_result"]["selected"]:
        return jsonify({"error": "Run optimization first"}), 400

    result = cache["opt_result"]
    selected = result["selected"]
    ep = cache["econ_params"]
    block_size = cache["block_size"]
    dev = generate_development_network(selected, block_size=block_size)

    total_revenue = sum(s["total_metal_oz"] for s in selected) * ep["gold_price"]
    total_mining = result["total_tonnage"] * ep["mining_cost"]
    total_proc = result["total_tonnage"] * ep["processing_cost"]
    total_ga = result["total_tonnage"] * ep["ga_cost"]
    dev_cost = dev["total_cost"]
    npv = total_revenue - total_mining - total_proc - total_ga - dev_cost

    prices = list(range(1200, 2800, 100))
    npvs = []
    total_metal = sum(s["total_metal_oz"] for s in selected)
    opex = result["total_tonnage"] * (ep["mining_cost"] + ep["processing_cost"] + ep["ga_cost"]) + dev_cost
    for p in prices:
        npvs.append(round(total_metal * p - opex, 2))

    stope_values = [s["total_value"] for s in selected]

    return jsonify(np_safe({
        "npv": round(npv, 2),
        "total_revenue": round(total_revenue, 2),
        "total_cost": round(total_mining + total_proc + total_ga + dev_cost, 2),
        "cost_breakdown": {
            "categories": ["Mining", "Processing", "G&A", "Development"],
            "values": [round(total_mining, 2), round(total_proc, 2), round(total_ga, 2), round(dev_cost, 2)],
        },
        "sensitivity": {"prices": prices, "npvs": npvs},
        "current_price": ep["gold_price"],
        "stope_values": stope_values,
    }))


@app.route("/api/monte-carlo", methods=["POST"])
def api_monte_carlo():
    if "opt_result" not in cache or not cache["opt_result"]["selected"]:
        return jsonify({"error": "Run optimization first"}), 400

    p = request.json
    selected = cache["opt_result"]["selected"]
    block_size = cache["block_size"]
    dev = generate_development_network(selected, block_size=block_size)
    ep = cache["econ_params"]

    mc = run_monte_carlo(
        selected, dev["total_cost"],
        base_gold_price=ep["gold_price"],
        base_mining_cost=ep["mining_cost"],
        base_processing_cost=ep["processing_cost"],
        base_ga_cost=ep["ga_cost"],
        recovery=ep["recovery"],
        n_simulations=p.get("n_simulations", 1000),
        price_std_pct=p.get("price_std", 15) / 100,
        grade_std_pct=p.get("grade_std", 10) / 100,
    )
    cache["mc_result"] = mc

    return jsonify(np_safe({
        "mean_npv": mc["mean_npv"],
        "std_npv": mc["std_npv"],
        "prob_positive_pct": mc["prob_positive_pct"],
        "base_npv": mc["base_npv"],
        "percentiles": mc["percentiles"],
        "npvs": mc["npvs"],
        "prices_used": mc["prices_used"],
        "tornado": mc["tornado"],
    }))


@app.route("/api/ml-predict", methods=["POST"])
def api_ml_predict():
    if "block_model" not in cache:
        return jsonify({"error": "Generate block model first"}), 400

    p = request.json
    model_type = p.get("model_type", "random_forest")
    test_size = p.get("test_pct", 20) / 100

    ml = train_grade_model(cache["block_model"], model_type=model_type, test_size=test_size)
    cache["ml_result"] = ml

    sample_n = min(2000, len(ml["y_test"]))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ml["y_test"]), sample_n, replace=False)
    y_t = np.array(ml["y_test"])[idx]
    y_p = np.array(ml["y_pred_test"])[idx]

    imp = ml["feature_importances"]
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)

    z_levels = sorted(cache["block_model"]["z"].unique().tolist())

    return jsonify(np_safe({
        "model_type": ml["model_type"],
        "train_metrics": ml["train_metrics"],
        "test_metrics": ml["test_metrics"],
        "actual_vs_predicted": {"actual": y_t.tolist(), "predicted": y_p.tolist()},
        "feature_importances": {"names": [k for k, _ in imp_sorted], "values": [v for _, v in imp_sorted]},
        "residuals": ml["residuals"],
        "has_uncertainty": ml["uncertainty"] is not None,
        "z_levels": z_levels,
    }))


@app.route("/api/ml-uncertainty", methods=["POST"])
def api_ml_uncertainty():
    if "ml_result" not in cache or cache["ml_result"]["uncertainty"] is None:
        return jsonify({"error": "Train Random Forest model first"}), 400

    p = request.json
    z_val = p.get("z")
    df = cache["block_model"].copy()
    df["uncertainty"] = cache["ml_result"]["uncertainty"]
    df["predicted_grade"] = cache["ml_result"]["y_pred_all"]

    section = df[df["z"] == z_val]
    return jsonify(np_safe({
        "x": section["x"].tolist(),
        "y": section["y"].tolist(),
        "predicted_grade": section["predicted_grade"].tolist(),
        "uncertainty": section["uncertainty"].tolist(),
    }))


@app.route("/api/scenarios", methods=["GET"])
def api_scenarios_list():
    scenarios = cache.get("scenarios", {})
    return jsonify(np_safe([
        {
            "id": sid,
            "name": s["name"],
            "gold_price": s["econ_params"]["gold_price"],
            "total_value": s["opt_result"]["total_value"],
            "total_tonnage": s["opt_result"]["total_tonnage"],
            "n_selected": s["opt_result"]["n_selected"],
            "total_metal_oz": s["opt_result"]["total_metal_oz"],
        }
        for sid, s in scenarios.items()
    ]))


@app.route("/api/scenarios/save", methods=["POST"])
def api_scenarios_save():
    if "opt_result" not in cache or not cache["opt_result"]["selected"]:
        return jsonify({"error": "Run optimization first"}), 400

    p = request.json
    name = p.get("name", f"Scenario {len(cache.get('scenarios', {})) + 1}")
    sid = f"s{len(cache.get('scenarios', {})) + 1}"

    if "scenarios" not in cache:
        cache["scenarios"] = {}

    cache["scenarios"][sid] = {
        "name": name,
        "econ_params": cache["econ_params"].copy(),
        "block_size": cache["block_size"],
        "opt_result": {
            "status": cache["opt_result"]["status"],
            "n_candidates": cache["opt_result"]["n_candidates"],
            "n_selected": cache["opt_result"]["n_selected"],
            "total_value": cache["opt_result"]["total_value"],
            "total_tonnage": cache["opt_result"]["total_tonnage"],
            "total_metal_oz": cache["opt_result"]["total_metal_oz"],
        },
        "selected_summary": [
            {k: s[k] for k in ["stope_id", "avg_grade", "total_value", "total_tonnage", "total_metal_oz"]}
            for s in cache["opt_result"]["selected"]
        ],
    }

    return jsonify({"saved": sid, "name": name, "total": len(cache["scenarios"])})


@app.route("/api/scenarios/compare", methods=["POST"])
def api_scenarios_compare():
    scenarios = cache.get("scenarios", {})
    if len(scenarios) < 2:
        return jsonify({"error": "Save at least 2 scenarios first"}), 400

    comparison = []
    for sid, s in scenarios.items():
        ep = s["econ_params"]
        opt = s["opt_result"]
        total_revenue = opt["total_metal_oz"] * ep["gold_price"]
        total_cost = opt["total_tonnage"] * (ep["mining_cost"] + ep["processing_cost"] + ep["ga_cost"])
        npv_est = total_revenue - total_cost
        comparison.append({
            "id": sid,
            "name": s["name"],
            "gold_price": ep["gold_price"],
            "mining_cost": ep["mining_cost"],
            "recovery_pct": round(ep["recovery"] * 100, 1),
            "n_selected": opt["n_selected"],
            "total_value": opt["total_value"],
            "total_tonnage": opt["total_tonnage"],
            "total_metal_oz": opt["total_metal_oz"],
            "npv_estimate": round(npv_est, 2),
        })

    return jsonify(np_safe(comparison))


@app.route("/api/scenarios/delete/<sid>", methods=["DELETE"])
def api_scenarios_delete(sid):
    scenarios = cache.get("scenarios", {})
    if sid not in scenarios:
        return jsonify({"error": "Scenario not found"}), 404
    del scenarios[sid]
    return jsonify({"deleted": sid})


@app.route("/api/kriging", methods=["POST"])
def api_kriging():
    if "block_model" not in cache:
        return jsonify({"error": "Generate block model first"}), 400

    p = request.json
    variogram_model = p.get("variogram_model", "spherical")
    result = run_kriging_analysis(
        cache["block_model"],
        variogram_model=variogram_model,
        test_fraction=p.get("test_fraction", 0.2),
    )
    cache["kriging_result"] = result
    return jsonify(np_safe(result))


@app.route("/api/kriging/simple", methods=["POST"])
def api_kriging_simple():
    if "block_model" not in cache:
        return jsonify({"error": "Generate block model first"}), 400
    p = request.json
    result = run_simple_kriging(cache["block_model"],
                                variogram_model=p.get("variogram_model", "spherical"),
                                test_fraction=p.get("test_fraction", 0.2))
    return jsonify(np_safe(result))


@app.route("/api/kriging/indicator", methods=["POST"])
def api_kriging_indicator():
    if "block_model" not in cache:
        return jsonify({"error": "Generate block model first"}), 400
    p = request.json
    result = run_indicator_kriging(cache["block_model"],
                                   variogram_model=p.get("variogram_model", "spherical"))
    return jsonify(np_safe(result))


@app.route("/api/kriging/sgs", methods=["POST"])
def api_kriging_sgs():
    if "block_model" not in cache:
        return jsonify({"error": "Generate block model first"}), 400
    p = request.json
    result = run_sgs(cache["block_model"],
                     n_simulations=p.get("n_simulations", 10),
                     variogram_model=p.get("variogram_model", "spherical"))
    return jsonify(np_safe(result))


@app.route("/api/schedule", methods=["POST"])
def api_schedule():
    if "opt_result" not in cache or not cache["opt_result"]["selected"]:
        return jsonify({"error": "Run optimization first"}), 400

    p = request.json
    selected = cache["opt_result"]["selected"]
    block_size = cache["block_size"]
    dev = generate_development_network(selected, block_size=block_size)
    ep = cache["econ_params"]

    result = schedule_production(
        selected, dev,
        mining_rate_tpa=p.get("mining_rate", 500000),
        discount_rate=p.get("discount_rate", 8) / 100,
        gold_price=ep["gold_price"],
        mining_cost=ep["mining_cost"],
        processing_cost=ep["processing_cost"],
        ga_cost=ep["ga_cost"],
        recovery=ep["recovery"],
    )
    cache["schedule"] = result

    return jsonify(np_safe(result))


@app.route("/api/geotechnical", methods=["POST"])
def api_geotechnical():
    if "opt_result" not in cache or not cache["opt_result"]["selected"]:
        return jsonify({"error": "Run optimization first"}), 400

    selected = cache["opt_result"]["selected"]
    result = assess_stability(selected)
    cache["geotech"] = result

    return jsonify(np_safe(result))


# ── Bolani Drillhole Data ──────────────────────────────────────────────────────

@app.route("/api/bolani/load", methods=["POST"])
def api_bolani_load():
    """Load and parse Bolani drillhole data, return composites + stats."""
    try:
        df_comp, collar = load_drillhole_data()
        cache["bolani_composites"] = df_comp
        cache["bolani_collar"] = collar
        stats = get_bolani_stats(df_comp)
        drillholes = []
        for _, row in collar.iterrows():
            drillholes.append({
                "bh_no": row["bh_no"],
                "x": float(row.get("e", 0)), "y": float(row.get("n", 0)),
                "z": float(row.get("z", 0)), "depth": float(row.get("depth", 0)),
            })
        # Return composite samples for 3D drillhole display
        samples = df_comp[["x", "y", "z", "fe", "lithocode", "bh_no"]].dropna(subset=["fe"])
        return jsonify(np_safe({
            "n_drillholes": len(collar),
            "n_composites": len(df_comp),
            "drillholes": drillholes,
            "composites": {
                "x": samples["x"].tolist(), "y": samples["y"].tolist(),
                "z": samples["z"].tolist(), "fe": samples["fe"].tolist(),
                "litho": samples["lithocode"].tolist(),
                "bh": samples["bh_no"].tolist(),
            },
            "stats": stats,
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bolani/block-model", methods=["POST"])
def api_bolani_block_model():
    """Build IDW block model from Bolani composites."""
    if "bolani_composites" not in cache:
        return jsonify({"error": "Load Bolani data first"}), 400
    p = request.json
    block_size = p.get("block_size", 10)
    df_bm = build_bolani_block_model(cache["bolani_composites"], block_size=block_size)
    if df_bm.empty:
        return jsonify({"error": "Block model generation failed"}), 500
    cache["bolani_block_model"] = df_bm

    # Grade range filtering data
    fe_min = float(df_bm["fe"].min())
    fe_max = float(df_bm["fe"].max())
    sample = df_bm.sample(min(3000, len(df_bm)), random_state=42)

    return jsonify(np_safe({
        "n_blocks": len(df_bm),
        "fe_min": round(fe_min, 2), "fe_max": round(fe_max, 2),
        "fe_mean": round(float(df_bm["fe"].mean()), 2),
        "block_size": block_size,
        "blocks": {
            "x": sample["x"].tolist(), "y": sample["y"].tolist(),
            "z": sample["z"].tolist(), "fe": sample["fe"].tolist(),
            "sio2": sample["sio2"].tolist(), "al2o3": sample["al2o3"].tolist(),
        },
    }))


# ── Reserve Reporting ─────────────────────────────────────────────────────────

@app.route("/api/reserve-report", methods=["POST"])
def api_reserve_report():
    if "df_econ" not in cache:
        return jsonify({"error": "Generate block model first"}), 400
    p = request.json
    commodity = p.get("commodity", "Au")
    grade_col = "grade_au_gt"
    cutoff_base = cache.get("cutoff", 0.5)
    cutoff_grades = [cutoff_base * f for f in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]

    df_class = classify_resource(
        cache["df_econ"], grade_col=grade_col,
        block_size=cache.get("block_size", 5)
    )
    ep = cache.get("econ_params", {})
    report = generate_reserve_table(
        df_class, cutoff_grades, grade_col=grade_col, commodity=commodity,
        price_per_unit=ep.get("gold_price", 1900),
        recovery=ep.get("recovery", 0.92),
        mining_cost=ep.get("mining_cost", 45),
        processing_cost=ep.get("processing_cost", 25),
        ga_cost=ep.get("ga_cost", 10),
        block_size=cache.get("block_size", 5),
    )
    return jsonify(np_safe(report))


# ── Stochastic Optimization ───────────────────────────────────────────────────

@app.route("/api/stochastic-optimize", methods=["POST"])
def api_stochastic_optimize():
    if "df_econ" not in cache:
        return jsonify({"error": "Generate block model first"}), 400
    p = request.json
    block_size = cache["block_size"]
    from stope_generator import generate_candidate_stopes
    candidates, b2s = generate_candidate_stopes(
        cache["df_econ"], block_size=block_size,
        min_width=p.get("min_width", 2), max_width=p.get("max_width", 3),
        min_height=p.get("min_height", 3), max_height=p.get("max_height", 5),
        min_length=p.get("min_length", 2), max_length=p.get("max_length", 4),
        step=p.get("step", 3),
    )
    if not candidates:
        return jsonify({"error": "No candidate stopes"}), 400

    result = run_stochastic_optimization(
        cache["df_econ"], candidates, b2s,
        n_simulations=p.get("n_simulations", 15),
        p_min=p.get("p_min", 0.7),
    )
    return jsonify(np_safe(result))


# ── Multi-Objective Optimization ──────────────────────────────────────────────

@app.route("/api/multi-objective", methods=["POST"])
def api_multi_objective():
    if "df_econ" not in cache:
        return jsonify({"error": "Generate block model first"}), 400
    p = request.json
    block_size = cache["block_size"]
    from stope_generator import generate_candidate_stopes
    candidates, b2s = generate_candidate_stopes(
        cache["df_econ"], block_size=block_size,
        step=p.get("step", 3),
    )
    if not candidates:
        return jsonify({"error": "No candidate stopes"}), 400

    result = run_multi_objective_optimization(
        candidates, b2s,
        alpha1=p.get("alpha1", 1.0),
        alpha2=p.get("alpha2", 0.5),
        alpha3=p.get("alpha3", 0.3),
        n_epsilon_steps=p.get("n_steps", 7),
    )
    return jsonify(np_safe(result))


# ── Implicit Geological Modeling ──────────────────────────────────────────────

@app.route("/api/implicit-geology", methods=["POST"])
def api_implicit_geology():
    if "block_model" not in cache and "bolani_composites" not in cache:
        return jsonify({"error": "Generate block model first"}), 400
    p = request.json
    if "bolani_composites" in cache:
        df = cache["bolani_composites"].rename(columns={"fe": "grade_au_gt"})
        grade_col = "grade_au_gt"
    else:
        df = cache["block_model"]
        grade_col = "grade_au_gt"
    block_df = cache.get("bolani_block_model") or cache.get("df_econ")
    result = build_implicit_model(
        df, block_df=block_df,
        grade_col=grade_col,
        ore_threshold=p.get("ore_threshold"),
        block_size=cache.get("block_size", 5),
    )
    return jsonify(np_safe(result))


# ── Ventilation Design ────────────────────────────────────────────────────────

@app.route("/api/ventilation", methods=["POST"])
def api_ventilation():
    if "opt_result" not in cache or not cache["opt_result"]["selected"]:
        return jsonify({"error": "Run optimization first"}), 400
    p = request.json
    selected = cache["opt_result"]["selected"]
    dev_segs = cache.get("dev", {}).get("segments", [])
    result = design_ventilation_network(
        dev_segs, selected,
        diesel_power_kw=p.get("diesel_kw", 500),
        ambient_temp_c=p.get("ambient_temp", 20),
    )
    cache["ventilation"] = result
    return jsonify(np_safe(result))


# ── Backfill Optimization ─────────────────────────────────────────────────────

@app.route("/api/backfill", methods=["POST"])
def api_backfill():
    if "opt_result" not in cache or not cache["opt_result"]["selected"]:
        return jsonify({"error": "Run optimization first"}), 400
    p = request.json
    selected = cache["opt_result"]["selected"]
    result = optimize_backfill(
        selected,
        block_size=cache.get("block_size", 5),
        fill_type=p.get("fill_type", "paste"),
        batch_capacity_m3_day=p.get("batch_capacity", 500),
    )
    cache["backfill"] = result
    return jsonify(np_safe(result))


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, port=port, host="0.0.0.0")
