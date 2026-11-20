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


if __name__ == "__main__":
    app.run(debug=False, port=5050, host="0.0.0.0")
