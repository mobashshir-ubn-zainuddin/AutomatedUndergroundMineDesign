"""Flask backend for Automated Underground Mine Design web application."""

import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request, make_response
from werkzeug.utils import secure_filename

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
    # Stamp static assets with their mtime so browsers pick up edits instead of
    # serving a stale app.js from cache.
    stamp = 0
    for rel in ("js/app.js", "css/style.css"):
        f = BASE_DIR / "static" / rel
        if f.exists():
            stamp = max(stamp, int(f.stat().st_mtime))
    resp = make_response(render_template("index.html", asset_v=stamp))
    # The page carries the asset stamps, so it must never be served from cache
    # or the browser keeps an old layout with new scripts.
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


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


@app.route("/api/bolani/drillhole-profile", methods=["POST"])
def api_bolani_dh_profile():
    """Return per-drillhole depth profiles for grade/element variation."""
    if "bolani_composites" not in cache:
        return jsonify({"error": "Load Bolani data first"}), 400
    p = request.json
    bh_no = p.get("bh_no", "ALL")
    element = p.get("element", "fe")

    df = cache["bolani_composites"].copy()
    if bh_no != "ALL":
        df = df[df["bh_no"] == bh_no]

    df = df.dropna(subset=[element]).sort_values(["bh_no", "from"])

    # Build per-drillhole series
    drillholes = df["bh_no"].unique().tolist()
    series = {}
    for bh in drillholes:
        sub = df[df["bh_no"] == bh].sort_values("from")
        series[bh] = {
            "depth": ((sub["from"] + sub["to"]) / 2).tolist(),
            "value": sub[element].tolist(),
            "litho": sub["lithocode"].tolist(),
            "from": sub["from"].tolist(),
            "to": sub["to"].tolist(),
        }

    # Statistics by lithology
    litho_stats = {}
    for litho, grp in df.groupby("lithocode"):
        vals = grp[element].values
        litho_stats[litho] = {
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std()), 2),
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
            "n": int(len(vals)),
            "values": vals.tolist()[:200],
        }

    # All values pooled for the element histogram
    all_vals = df[element].values
    hist_counts, hist_bins = np.histogram(all_vals, bins=25)

    return jsonify(np_safe({
        "element": element,
        "drillholes": drillholes,
        "series": series,
        "litho_stats": litho_stats,
        "histogram": {"counts": hist_counts.tolist(), "bins": hist_bins.tolist()},
        "overall_mean": round(float(all_vals.mean()), 2),
        "overall_std": round(float(all_vals.std()), 2),
    }))


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
    cache["bolani_block_size"]  = block_size

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


@app.route("/api/bolani/combined-csv", methods=["GET"])
def api_bolani_combined_csv():
    """Download the Bolani dataset as one CSV, also usable as an upload template."""
    f = BASE_DIR / "bolani_data" / "bolani_combined.csv"
    if not f.exists():
        return jsonify({"error": "Combined CSV not found"}), 404
    resp = make_response(f.read_text(encoding="utf-8"))
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=bolani_combined.csv"
    return resp


@app.route("/api/csv/upload-combined", methods=["POST"])
def api_csv_upload_combined():
    """Load a single CSV that carries collar, assay and geology in one table."""
    try:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(f)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        bh_col = next((c for c in ("bh_no", "bhno", "hole_id", "holeid", "drillhole", "bh")
                       if c in df.columns), None)
        if bh_col is None:
            return jsonify({"error": "No drillhole ID column found (expected bh_no or hole_id)"}), 400
        df = df.rename(columns={bh_col: "bh_no"})

        # Composite coordinates, derived from the collar if not supplied directly
        if not {"x", "y", "z"}.issubset(df.columns):
            east = next((c for c in ("collar_east", "easting", "e") if c in df.columns), None)
            north = next((c for c in ("collar_north", "northing", "n") if c in df.columns), None)
            rl = next((c for c in ("collar_rl", "elevation", "rl") if c in df.columns), None)
            if not (east and north and rl):
                return jsonify({"error": "Need x/y/z or collar_east/collar_north/collar_rl columns"}), 400
            if "from" not in df.columns or "to" not in df.columns:
                return jsonify({"error": "Need from/to depth columns to place samples"}), 400
            mid = (df["from"] + df["to"]) / 2
            df["x"], df["y"], df["z"] = df[east], df[north], df[rl] - mid

        for col in ("from", "to"):
            if col not in df.columns:
                df[col] = 0.0
        if "lithocode" not in df.columns:
            df["lithocode"] = "Unknown"
        if "interval_len" not in df.columns:
            df["interval_len"] = df["to"] - df["from"]

        grade_cols = [c for c in ("fe", "sio2", "al2o3", "p", "s", "au", "cu")
                      if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not grade_cols:
            return jsonify({"error": "No numeric grade columns found"}), 400
        for c in grade_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        primary = "fe" if "fe" in grade_cols else grade_cols[0]
        if primary != "fe":
            df["fe"] = df[primary]
        for c in ("sio2", "al2o3"):
            if c not in df.columns:
                df[c] = np.nan

        collar = (df.groupby("bh_no")
                    .agg(e=("x", "first"), n=("y", "first"), z=("z", "max"))
                    .reset_index())
        collar["depth"] = df.groupby("bh_no")["to"].max().values

        cache["bolani_composites"] = df
        cache["bolani_collar"] = collar

        stats = get_bolani_stats(df)
        sample = df.dropna(subset=["x", "y", "z", "fe"])
        sample = sample.sample(min(3000, len(sample)), random_state=42)

        return jsonify(np_safe({
            "n_drillholes": int(df["bh_no"].nunique()),
            "n_composites": len(df),
            "drillholes": [{"bh_no": b} for b in df["bh_no"].astype(str).unique().tolist()],
            "columns": grade_cols,
            "grade_col": primary,
            "stats": stats,
            "composites": {
                "x": sample["x"].tolist(), "y": sample["y"].tolist(),
                "z": sample["z"].tolist(), "fe": sample["fe"].tolist(),
                "litho": sample["lithocode"].astype(str).tolist(),
                "bh": sample["bh_no"].astype(str).tolist(),
            },
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bolani/scatter", methods=["POST"])
def api_bolani_scatter():
    """Cross-plot any two assayed compounds, split by lithology."""
    if "bolani_composites" not in cache:
        return jsonify({"error": "Load drillhole data first"}), 400

    p = request.json or {}
    df = cache["bolani_composites"]
    numeric = [c for c in ("fe", "sio2", "al2o3", "p", "s")
               if c in df.columns and df[c].notna().any()]
    if not numeric:
        return jsonify({"error": "No assayed compounds available"}), 400

    x_col = p.get("x", "fe")
    y_col = p.get("y", "sio2" if "sio2" in numeric else numeric[-1])
    if x_col not in numeric or y_col not in numeric:
        return jsonify({"error": f"Choose compounds from: {', '.join(numeric)}"}), 400

    sub = df.dropna(subset=[x_col, y_col])
    if sub.empty:
        return jsonify({"error": f"No samples assayed for both {x_col.upper()} and {y_col.upper()}"}), 400

    # One series per lithology so the plot carries geological meaning
    series = []
    for litho, grp in sub.groupby("lithocode"):
        series.append({
            "litho": str(litho),
            "x": grp[x_col].tolist(),
            "y": grp[y_col].tolist(),
            "bh": grp["bh_no"].astype(str).tolist(),
            "depth": ((grp["from"] + grp["to"]) / 2).round(1).tolist(),
        })
    series.sort(key=lambda s: -len(s["x"]))

    corr = float(sub[x_col].corr(sub[y_col])) if len(sub) > 2 else 0.0
    # Least-squares trend for the cross-plot
    if len(sub) > 2:
        slope, intercept = np.polyfit(sub[x_col], sub[y_col], 1)
        xs = [float(sub[x_col].min()), float(sub[x_col].max())]
        trend = {"x": xs, "y": [float(slope * v + intercept) for v in xs]}
    else:
        trend = {"x": [], "y": []}

    return jsonify(np_safe({
        "x_col": x_col, "y_col": y_col,
        "available": numeric,
        "n_samples": len(sub),
        "correlation": round(corr, 3),
        "trend": trend,
        "series": series,
    }))


# ── Bolani → Active Model ─────────────────────────────────────────────────────

@app.route("/api/bolani/use-as-model", methods=["POST"])
def api_bolani_use_as_model():
    """Convert the Bolani IDW block model into df_econ format so all downstream
    tabs (Stope Optimization, Mine Layout, Economics, Ventilation, Backfill,
    Kriging, ML, Geotechnical, Monte Carlo, Reserve Report) work on Bolani data."""
    if "bolani_block_model" not in cache:
        return jsonify({"error": "Build the Bolani block model first (Bolani tab → Build Block Model)"}), 400

    p = request.json or {}
    bm = cache["bolani_block_model"].copy()

    # Iron ore economics: revenue per tonne scales with Fe grade.
    # Defaults are set so the breakeven cutoff lands near 55% Fe, the usual
    # marketable-lump threshold for Indian haematite ore.
    fe_price_per_pct_t = p.get("fe_price_per_pct_t", 0.45)   # USD per 1% Fe per tonne
    mining_cost      = p.get("mining_cost", 12.0)            # USD/t
    processing_cost  = p.get("processing_cost", 8.0)         # USD/t
    ga_cost          = p.get("ga_cost", 3.5)                 # USD/t
    recovery         = p.get("recovery", 0.95)
    density          = p.get("density", 3.2)                 # t/m³ for iron ore
    block_size       = cache.get("bolani_block_size", 10)

    # Map Fe% to grade_au_gt column name so all existing code works unchanged
    bm["grade_au_gt"]   = bm["fe"].fillna(bm["fe"].median())
    bm["density_tpm3"]  = density
    bm["tonnage"]       = block_size ** 3 * density

    revenue_per_t       = bm["grade_au_gt"] * fe_price_per_pct_t * recovery
    cost_per_t          = mining_cost + processing_cost + ga_cost
    bm["revenue_per_t"] = revenue_per_t.round(2)
    bm["cost_per_t"]    = cost_per_t
    bm["profit_per_t"]  = (revenue_per_t - cost_per_t).round(2)
    bm["economic_value"] = (bm["tonnage"] * bm["profit_per_t"]).round(2)
    # metal_content_oz repurposed as "contained Fe tonnes"
    bm["metal_content_oz"] = (bm["tonnage"] * bm["grade_au_gt"] / 100.0).round(2)

    breakeven = cost_per_t / (fe_price_per_pct_t * recovery)   # % Fe cutoff
    bm["above_cutoff"] = bm["grade_au_gt"] >= breakeven
    bm["ore_class"]    = np.where(bm["above_cutoff"], "Ore", "Waste")

    # Stope generation indexes blocks by block_id, which the IDW grid lacks
    bm = bm.reset_index(drop=True)
    bm["block_id"] = bm.index

    cache["df_econ"]      = bm
    cache["block_model"]  = bm
    cache["block_size"]   = block_size
    cache["cutoff"]       = breakeven
    cache["econ_params"]  = {
        "gold_price": fe_price_per_pct_t * 100,  # proxy so Economics tab math works
        "mining_cost": mining_cost,
        "processing_cost": processing_cost,
        "ga_cost": ga_cost,
        "recovery": recovery,
        "dataset": "bolani",
    }

    ore  = bm[bm["above_cutoff"]]
    sample = bm.sample(min(5000, len(bm)), random_state=42)
    z_levels = sorted(bm["z"].unique().tolist())

    # High grade = top quartile of Fe
    hg_threshold = float(bm["grade_au_gt"].quantile(0.75))
    high = bm[bm["grade_au_gt"] >= hg_threshold]

    # Drillhole composites, overlaid on the block model for context
    comp = cache.get("bolani_composites")
    dh_payload = {"x": [], "y": [], "z": [], "fe": [], "bh": [], "litho": []}
    if comp is not None:
        c = comp.dropna(subset=["x", "y", "z", "fe"])
        c = c.sample(min(3000, len(c)), random_state=42)
        dh_payload = {
            "x": c["x"].tolist(), "y": c["y"].tolist(), "z": c["z"].tolist(),
            "fe": c["fe"].tolist(),
            "bh": c["bh_no"].astype(str).tolist(),
            "litho": c.get("lithocode", pd.Series(["—"] * len(c))).astype(str).tolist(),
        }

    return jsonify(np_safe({
        "ok": True,
        "dataset": "Bolani Iron Ore",
        "total_blocks": len(bm),
        "ore_blocks": int(bm["above_cutoff"].sum()),
        "mean_fe": round(float(bm["grade_au_gt"].mean()), 2),
        "cutoff_fe": round(float(breakeven), 2),
        "block_size": block_size,
        "z_levels": z_levels,
        "all_blocks": {
            "x": sample["x"].tolist(), "y": sample["y"].tolist(),
            "z": sample["z"].tolist(),
            "grade": sample["grade_au_gt"].tolist(),
            "value": sample["economic_value"].tolist(),
        },
        "ore_blocks_data": {
            "x": ore["x"].tolist(), "y": ore["y"].tolist(),
            "z": ore["z"].tolist(),
            "grade": ore["grade_au_gt"].tolist(),
            "value": ore["economic_value"].tolist(),
        },
        "high_grade_data": {
            "x": high["x"].tolist(), "y": high["y"].tolist(),
            "z": high["z"].tolist(),
            "grade": high["grade_au_gt"].tolist(),
            "value": high["economic_value"].tolist(),
        },
        "high_grade_threshold": round(hg_threshold, 2),
        "drillholes": dh_payload,
        "grade_histogram": {
            "ore_grades": ore["grade_au_gt"].tolist(),
            "waste_grades": bm[~bm["above_cutoff"]]["grade_au_gt"].tolist(),
        },
        "grade_tonnage": {
            "cum_tonnes": ore.sort_values("grade_au_gt", ascending=False)["tonnage"].cumsum().tolist(),
            "grades": ore.sort_values("grade_au_gt", ascending=False)["grade_au_gt"].tolist(),
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


# ── CSV Upload — Custom drillhole data ──────────────────────────────────────

@app.route("/api/csv/upload", methods=["POST"])
def api_csv_upload():
    """Accept user-uploaded collar + assay (+ optional geology/survey) CSVs,
    parse them into the same format as the Bolani loader, cache the result."""
    try:
        collar_file = request.files.get("collar")
        assay_file  = request.files.get("assay")
        geo_file    = request.files.get("geology")
        survey_file = request.files.get("survey")

        if not collar_file or not assay_file:
            return jsonify({"error": "collar and assay CSV files are required"}), 400

        collar = pd.read_csv(collar_file)
        assay  = pd.read_csv(assay_file)

        # Normalise column names to lower-case, strip spaces
        collar.columns = [c.strip().lower().replace(" ", "_") for c in collar.columns]
        assay.columns  = [c.strip().lower().replace(" ", "_") for c in assay.columns]

        # Try to identify the drillhole ID column
        bh_candidates = ["bh_no", "bhno", "hole_id", "holeid", "dhno", "drillhole", "bh", "id"]
        bh_col_collar = next((c for c in bh_candidates if c in collar.columns), collar.columns[0])
        bh_col_assay  = next((c for c in bh_candidates if c in assay.columns), assay.columns[0])
        collar = collar.rename(columns={bh_col_collar: "bh_no"})
        assay  = assay.rename(columns={bh_col_assay:  "bh_no"})

        # Coordinate columns
        for old, new in [("easting","e"),("northing","n"),("elevation","z"),("rl","z"),("x","e"),("y","n")]:
            if old in collar.columns and new not in collar.columns:
                collar = collar.rename(columns={old: new})
        if "e" not in collar.columns: collar["e"] = 0.0
        if "n" not in collar.columns: collar["n"] = 0.0
        if "z" not in collar.columns: collar["z"] = 0.0
        if "depth" not in collar.columns:
            depth_cand = [c for c in collar.columns if "depth" in c or "length" in c]
            collar["depth"] = collar[depth_cand[0]] if depth_cand else 100.0

        # From/To columns in assay
        for old, new in [("from_m","from"),("to_m","to"),("depth_from","from"),("depth_to","to")]:
            if old in assay.columns and new not in assay.columns:
                assay = assay.rename(columns={old: new})
        if "from" not in assay.columns: assay["from"] = 0.0
        if "to"   not in assay.columns: assay["to"]   = assay["from"] + 1.0

        # Detect grade columns
        grade_candidates = ["fe","au","cu","pb","zn","ni","grade","value","assay"]
        grade_cols = [c for c in assay.columns if any(g in c for g in grade_candidates)]
        if not grade_cols:
            grade_cols = [c for c in assay.columns if c not in ["bh_no","from","to","lithocode","rocktype"]]

        # Merge collar + assay → composites
        merged = assay.merge(collar[["bh_no","e","n","z","depth"]], on="bh_no", how="left")
        mid = (merged["from"] + merged["to"]) / 2
        merged["x"] = merged["e"]
        merged["y"] = merged["n"]
        merged["z_comp"] = merged["z"] - mid   # vertical hole desurvey
        merged = merged.rename(columns={"z_comp": "z_dh"})
        merged["z"] = merged["z_dh"]

        # Geology
        if geo_file:
            geo = pd.read_csv(geo_file)
            geo.columns = [c.strip().lower().replace(" ", "_") for c in geo.columns]
            bh_col_geo = next((c for c in bh_candidates if c in geo.columns), geo.columns[0])
            geo = geo.rename(columns={bh_col_geo: "bh_no"})
            if "lithocode" not in geo.columns:
                lit_cand = [c for c in geo.columns if "lith" in c or "rock" in c or "code" in c]
                if lit_cand: geo = geo.rename(columns={lit_cand[0]: "lithocode"})
            from_cand = next((c for c in geo.columns if "from" in c), None)
            to_cand   = next((c for c in geo.columns if "to" in c and c != from_cand), None)
            if from_cand and to_cand:
                geo = geo.rename(columns={from_cand: "geo_from", to_cand: "geo_to"})
                # Interval join: assign lithocode to composites
                def assign_litho(row):
                    sub = geo[(geo["bh_no"] == row["bh_no"]) &
                              (geo["geo_from"] <= row["from"]) &
                              (geo["geo_to"]   >= row["to"])]
                    return sub["lithocode"].iloc[0] if len(sub) else "Unknown"
                merged["lithocode"] = merged.apply(assign_litho, axis=1)
            elif "lithocode" in geo.columns:
                merged = merged.merge(geo[["bh_no","lithocode"]].drop_duplicates("bh_no"), on="bh_no", how="left")
        if "lithocode" not in merged.columns:
            merged["lithocode"] = "Unknown"

        # Identify primary grade column (prefer fe, then first detected)
        primary_grade = "fe" if "fe" in grade_cols else (grade_cols[0] if grade_cols else None)
        if primary_grade and primary_grade in merged.columns:
            merged[primary_grade] = pd.to_numeric(merged[primary_grade], errors="coerce")
            fe_mean = float(merged[primary_grade].dropna().mean())
            fe_max  = float(merged[primary_grade].dropna().max())
        else:
            fe_mean = fe_max = None

        # Cache as bolani_composites (compatible format)
        if primary_grade and primary_grade != "fe":
            merged["fe"] = merged[primary_grade]
        for col in ["sio2","al2o3"]:
            if col not in merged.columns:
                merged[col] = np.nan
        cache["bolani_composites"] = merged
        cache["bolani_collar"]     = collar

        # Sample for 3D display
        sample = merged.dropna(subset=["x","y","z"]).sample(min(2000, len(merged)), random_state=42)
        composites = {
            "x":  sample["x"].tolist(),
            "y":  sample["y"].tolist(),
            "z":  sample["z"].tolist(),
            "fe": sample["fe"].tolist() if "fe" in sample.columns else [],
            "litho": sample["lithocode"].tolist(),
            "bh": sample["bh_no"].tolist(),
        }

        return jsonify(np_safe({
            "n_drillholes": int(merged["bh_no"].nunique()),
            "n_composites": len(merged),
            "drillholes":   merged["bh_no"].unique().tolist(),
            "columns":      grade_cols,
            "grade_col":    primary_grade,
            "fe_mean":      fe_mean,
            "fe_max":       fe_max,
            "composites":   composites,
            "preview":      merged[["bh_no","from","to"] + grade_cols[:3]].head(8).fillna("").to_dict(orient="records"),
        }))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, port=port, host="0.0.0.0")
