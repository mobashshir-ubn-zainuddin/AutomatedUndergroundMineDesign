"""Streamlit web interface for Automated Underground Mine Design."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from block_model_generator import generate_block_model
from economic_evaluator import evaluate_economics
from stope_generator import generate_candidate_stopes
from stope_optimizer import optimize_stope_layout
from mine_layout import generate_development_network
from monte_carlo import run_monte_carlo
from grade_prediction import train_grade_model

st.set_page_config(
    page_title="Underground Mine Design",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Automated Underground Mine Design")
st.markdown("**Stope Layout Optimization with BILP & 3D Visualization**")


# --- Sidebar Controls ---
st.sidebar.header("Parameters")

with st.sidebar.expander("Block Model", expanded=False):
    nx = st.number_input("Grid X blocks", 10, 60, 30, 5)
    ny = st.number_input("Grid Y blocks", 10, 60, 25, 5)
    nz = st.number_input("Grid Z blocks", 5, 30, 15, 5)
    block_size = st.number_input("Block size (m)", 3, 10, 5)
    seed = st.number_input("Random seed", 1, 999, 42)

with st.sidebar.expander("Economic Parameters", expanded=True):
    gold_price = st.slider("Gold Price ($/oz)", 1000, 3000, 1900, 50)
    mining_cost = st.slider("Mining Cost ($/t)", 20, 100, 45, 5)
    processing_cost = st.slider("Processing Cost ($/t)", 10, 60, 25, 5)
    ga_cost = st.slider("G&A Cost ($/t)", 5, 30, 10, 5)
    recovery = st.slider("Recovery (%)", 70, 99, 92) / 100

with st.sidebar.expander("Stope Constraints", expanded=True):
    min_width = st.number_input("Min Width (blocks)", 1, 5, 2)
    max_width = st.number_input("Max Width (blocks)", 2, 6, 3)
    min_height = st.number_input("Min Height (blocks)", 2, 8, 3)
    max_height = st.number_input("Max Height (blocks)", 3, 10, 5)
    min_length = st.number_input("Min Length (blocks)", 1, 6, 2)
    max_length = st.number_input("Max Length (blocks)", 2, 8, 4)
    step_size = st.number_input("Search step", 1, 4, 2)


# --- Generate Block Model ---
@st.cache_data
def cached_block_model(nx, ny, nz, block_size, seed):
    return generate_block_model(nx=nx, ny=ny, nz=nz, block_size=block_size, seed=seed)


@st.cache_data
def cached_economics(df_hash, gold_price, mining_cost, processing_cost, ga_cost, recovery, block_size):
    df = st.session_state["block_model"]
    return evaluate_economics(
        df, gold_price_per_oz=gold_price, mining_cost_per_t=mining_cost,
        processing_cost_per_t=processing_cost, ga_cost_per_t=ga_cost,
        recovery=recovery, block_size=block_size,
    )


df_blocks = cached_block_model(nx, ny, nz, block_size, seed)
st.session_state["block_model"] = df_blocks
df_hash = hash(df_blocks.to_csv())

df_econ, cutoff_grade = cached_economics(
    df_hash, gold_price, mining_cost, processing_cost, ga_cost, recovery, block_size
)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Block Model", "🔍 Ore Classification", "⛏️ Stope Optimization",
    "🏗️ Mine Layout", "📈 Economics", "🎲 Monte Carlo", "🤖 ML Grade Prediction"
])

# ============ TAB 1: Block Model ============
with tab1:
    st.subheader("3D Block Model Viewer")

    view_mode = st.radio(
        "Display", ["All Blocks", "Ore Blocks Only (above cutoff)", "High Grade (>2 g/t)"],
        horizontal=True
    )

    if view_mode == "All Blocks":
        plot_df = df_econ.sample(min(5000, len(df_econ)), random_state=42)
    elif "Ore" in view_mode:
        plot_df = df_econ[df_econ["above_cutoff"]]
    else:
        plot_df = df_econ[df_econ["grade_au_gt"] > 2.0]

    if len(plot_df) > 0:
        fig = go.Figure(data=go.Scatter3d(
            x=plot_df["x"], y=plot_df["y"], z=plot_df["z"],
            mode="markers",
            marker=dict(
                size=3,
                color=plot_df["grade_au_gt"],
                colorscale="YlOrRd",
                colorbar=dict(title="Grade (g/t)"),
                opacity=0.6,
            ),
            text=[f"Grade: {g:.2f} g/t<br>Value: ${v:,.0f}"
                  for g, v in zip(plot_df["grade_au_gt"], plot_df["economic_value"])],
            hoverinfo="text",
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                zaxis_title="Elevation (m)", aspectmode="data"
            ),
            height=600, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Blocks", f"{len(df_econ):,}")
    col2.metric("Ore Blocks", f"{df_econ['above_cutoff'].sum():,}")
    col3.metric("Mean Grade", f"{df_econ['grade_au_gt'].mean():.2f} g/t")
    col4.metric("Max Grade", f"{df_econ['grade_au_gt'].max():.2f} g/t")


# ============ TAB 2: Ore Classification ============
with tab2:
    st.subheader("Grade Distribution & Classification")

    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            df_econ[df_econ["grade_au_gt"] > 0],
            x="grade_au_gt", nbins=50, color="ore_class",
            color_discrete_map={"Ore": "#2ecc71", "Waste": "#95a5a6"},
            title="Grade Distribution",
            labels={"grade_au_gt": "Gold Grade (g/t)", "count": "Block Count"},
        )
        fig_hist.add_vline(x=cutoff_grade, line_dash="dash", line_color="red",
                           annotation_text=f"Cutoff: {cutoff_grade:.2f} g/t")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        ore_df = df_econ[df_econ["above_cutoff"]].sort_values("grade_au_gt", ascending=False)
        ore_df["cum_tonnes"] = ore_df["tonnage"].cumsum()
        ore_df["cum_metal"] = ore_df["metal_content_oz"].cumsum()
        ore_df["cum_avg_grade"] = ore_df["cum_metal"] / (ore_df["cum_tonnes"] / 31.1035 * ore_df["density_tpm3"].mean())

        fig_gt = go.Figure()
        fig_gt.add_trace(go.Scatter(
            x=ore_df["cum_tonnes"], y=ore_df["grade_au_gt"],
            mode="lines", name="Grade", line=dict(color="#e74c3c"),
        ))
        fig_gt.update_layout(
            title="Grade-Tonnage Curve",
            xaxis_title="Cumulative Tonnes",
            yaxis_title="Grade (g/t)",
            height=400,
        )
        st.plotly_chart(fig_gt, use_container_width=True)

    st.metric("Breakeven Cutoff Grade", f"{cutoff_grade:.3f} g/t")

    st.subheader("Block Model Cross-Section")
    z_levels = sorted(df_econ["z"].unique())
    selected_z = st.select_slider("Select elevation (m)", options=z_levels, value=z_levels[len(z_levels) // 2])
    section = df_econ[df_econ["z"] == selected_z]

    fig_section = go.Figure(data=go.Heatmap(
        x=section["x"], y=section["y"], z=section["grade_au_gt"],
        colorscale="YlOrRd", colorbar=dict(title="Grade (g/t)"),
    ))
    fig_section.update_layout(
        title=f"Plan View at Elevation {selected_z}m",
        xaxis_title="Easting (m)", yaxis_title="Northing (m)",
        height=500, yaxis=dict(scaleanchor="x"),
    )
    st.plotly_chart(fig_section, use_container_width=True)


# ============ TAB 3: Stope Optimization ============
with tab3:
    st.subheader("Stope Layout Optimization (BILP)")

    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Generating candidate stopes..."):
            candidates, b2s = generate_candidate_stopes(
                df_econ, block_size=block_size,
                min_width=min_width, max_width=max_width,
                min_height=min_height, max_height=max_height,
                min_length=min_length, max_length=max_length,
                step=step_size,
            )
            st.session_state["candidates"] = candidates
            st.session_state["b2s"] = b2s

        st.info(f"Generated {len(candidates)} candidate stopes")

        if candidates:
            with st.spinner("Solving BILP optimization..."):
                result = optimize_stope_layout(candidates, b2s)
                st.session_state["opt_result"] = result

            if result["status"] == "Optimal":
                st.success(f"Optimization complete: {result['n_selected']} stopes selected")
            else:
                st.warning(f"Solver status: {result['status']}")
        else:
            st.warning("No viable candidate stopes found. Try adjusting parameters.")

    if "opt_result" in st.session_state:
        result = st.session_state["opt_result"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Selected Stopes", result["n_selected"])
        col2.metric("Total Value", f"${result['total_value']:,.0f}")
        col3.metric("Total Tonnage", f"{result['total_tonnage']:,.0f} t")
        col4.metric("Metal Content", f"{result['total_metal_oz']:,.0f} oz")

        # 3D visualization of selected stopes
        selected = result["selected"]
        if selected:
            fig = go.Figure()

            # Ore blocks (faded)
            ore_sample = df_econ[df_econ["above_cutoff"]].sample(
                min(2000, len(df_econ[df_econ["above_cutoff"]])), random_state=42
            )
            fig.add_trace(go.Scatter3d(
                x=ore_sample["x"], y=ore_sample["y"], z=ore_sample["z"],
                mode="markers", name="Ore Blocks",
                marker=dict(size=2, color="lightgray", opacity=0.15),
            ))

            colors = px.colors.qualitative.Set3
            for i, s in enumerate(selected):
                fig.add_trace(go.Mesh3d(
                    x=[s["x_min"], s["x_min"], s["x_max"], s["x_max"],
                       s["x_min"], s["x_min"], s["x_max"], s["x_max"]],
                    y=[s["y_min"], s["y_max"], s["y_max"], s["y_min"],
                       s["y_min"], s["y_max"], s["y_max"], s["y_min"]],
                    z=[s["z_min"], s["z_min"], s["z_min"], s["z_min"],
                       s["z_max"], s["z_max"], s["z_max"], s["z_max"]],
                    i=[0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 2, 2],
                    j=[1, 2, 4, 5, 5, 6, 1, 3, 1, 5, 3, 7],
                    k=[2, 3, 5, 6, 6, 7, 5, 7, 2, 6, 7, 6],
                    opacity=0.4,
                    color=colors[i % len(colors)],
                    name=f"Stope {s['stope_id']} (${s['total_value']:,.0f})",
                    hovertext=f"Stope {s['stope_id']}<br>"
                              f"Value: ${s['total_value']:,.0f}<br>"
                              f"Grade: {s['avg_grade']:.2f} g/t<br>"
                              f"Tonnes: {s['total_tonnage']:,.0f}",
                ))

            fig.update_layout(
                scene=dict(
                    xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                    zaxis_title="Elevation (m)", aspectmode="data",
                ),
                height=700, margin=dict(l=0, r=0, t=30, b=0),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stope details table
            st.subheader("Selected Stopes Details")
            stope_table = pd.DataFrame([{
                "Stope ID": s["stope_id"],
                "Value ($)": f"${s['total_value']:,.0f}",
                "Tonnage (t)": f"{s['total_tonnage']:,.0f}",
                "Metal (oz)": f"{s['total_metal_oz']:,.1f}",
                "Avg Grade (g/t)": f"{s['avg_grade']:.2f}",
                "Width (m)": s["width_m"],
                "Height (m)": s["height_m"],
                "Length (m)": s["length_m"],
                "Ore %": f"{s['ore_fraction']*100:.0f}%",
            } for s in selected])
            st.dataframe(stope_table, use_container_width=True, hide_index=True)


# ============ TAB 4: Mine Layout ============
with tab4:
    st.subheader("Mine Development Network")

    if "opt_result" in st.session_state and st.session_state["opt_result"]["selected"]:
        selected = st.session_state["opt_result"]["selected"]
        dev = generate_development_network(selected, block_size=block_size)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Development", f"{dev['total_length_m']:,.0f} m")
        col2.metric("Development Cost", f"${dev['total_cost']:,.0f}")
        col3.metric("Mining Levels", dev["n_levels"])

        fig = go.Figure()

        # Stopes
        colors_s = px.colors.qualitative.Pastel
        for i, s in enumerate(selected):
            fig.add_trace(go.Mesh3d(
                x=[s["x_min"], s["x_min"], s["x_max"], s["x_max"],
                   s["x_min"], s["x_min"], s["x_max"], s["x_max"]],
                y=[s["y_min"], s["y_max"], s["y_max"], s["y_min"],
                   s["y_min"], s["y_max"], s["y_max"], s["y_min"]],
                z=[s["z_min"], s["z_min"], s["z_min"], s["z_min"],
                   s["z_max"], s["z_max"], s["z_max"], s["z_max"]],
                i=[0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 2, 2],
                j=[1, 2, 4, 5, 5, 6, 1, 3, 1, 5, 3, 7],
                k=[2, 3, 5, 6, 6, 7, 5, 7, 2, 6, 7, 6],
                opacity=0.3, color=colors_s[i % len(colors_s)],
                name=f"Stope {s['stope_id']}",
                showlegend=False,
            ))

        # Development segments
        type_colors = {"Decline": "red", "Level Drive": "blue", "Crosscut": "green"}
        for seg in dev["segments"]:
            fig.add_trace(go.Scatter3d(
                x=[seg["x1"], seg["x2"]], y=[seg["y1"], seg["y2"]], z=[seg["z1"], seg["z2"]],
                mode="lines", line=dict(color=type_colors.get(seg["type"], "gray"), width=4),
                name=seg["type"], showlegend=False,
                hovertext=f"{seg['type']}: {seg['length_m']:.0f}m",
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                zaxis_title="Elevation (m)", aspectmode="data",
            ),
            height=700, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Development table
        seg_df = pd.DataFrame(dev["segments"])
        summary = seg_df.groupby("type").agg(
            count=("seg_id", "count"),
            total_length=("length_m", "sum"),
        ).reset_index()
        summary["cost"] = summary["total_length"] * 5000
        summary.columns = ["Type", "Count", "Total Length (m)", "Cost ($)"]
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Run stope optimization first (Tab 3) to generate the mine layout.")


# ============ TAB 5: Economics ============
with tab5:
    st.subheader("Economic Analysis")

    if "opt_result" in st.session_state and st.session_state["opt_result"]["selected"]:
        result = st.session_state["opt_result"]
        selected = result["selected"]
        dev = generate_development_network(selected, block_size=block_size)

        total_revenue = sum(s["total_metal_oz"] for s in selected) * gold_price
        total_mining_cost = result["total_tonnage"] * mining_cost
        total_proc_cost = result["total_tonnage"] * processing_cost
        total_ga_cost = result["total_tonnage"] * ga_cost
        dev_cost = dev["total_cost"]
        npv = total_revenue - total_mining_cost - total_proc_cost - total_ga_cost - dev_cost

        col1, col2, col3 = st.columns(3)
        col1.metric("Net Present Value", f"${npv:,.0f}")
        col2.metric("Total Revenue", f"${total_revenue:,.0f}")
        col3.metric("Total Cost", f"${total_mining_cost + total_proc_cost + total_ga_cost + dev_cost:,.0f}")

        # Cost breakdown
        cost_data = pd.DataFrame({
            "Category": ["Mining", "Processing", "G&A", "Development"],
            "Cost ($)": [total_mining_cost, total_proc_cost, total_ga_cost, dev_cost],
        })
        fig_cost = px.pie(cost_data, values="Cost ($)", names="Category",
                          title="Cost Breakdown", hole=0.4)
        st.plotly_chart(fig_cost, use_container_width=True)

        # Sensitivity analysis: NPV vs gold price
        st.subheader("Sensitivity Analysis")
        prices = np.arange(1200, 2800, 100)
        npvs = []
        for p in prices:
            rev = sum(s["total_metal_oz"] for s in selected) * p
            costs = result["total_tonnage"] * (mining_cost + processing_cost + ga_cost) + dev_cost
            npvs.append(rev - costs)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=prices, y=npvs, mode="lines+markers", name="NPV",
            line=dict(color="#3498db", width=2),
        ))
        fig_sens.add_hline(y=0, line_dash="dash", line_color="red")
        fig_sens.add_vline(x=gold_price, line_dash="dash", line_color="green",
                           annotation_text=f"Current: ${gold_price}")
        fig_sens.update_layout(
            title="NPV vs Gold Price",
            xaxis_title="Gold Price ($/oz)",
            yaxis_title="NPV ($)",
            height=400,
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        # Stope value distribution
        stope_values = [s["total_value"] for s in selected]
        fig_sv = px.histogram(x=stope_values, nbins=20,
                              title="Distribution of Selected Stope Values",
                              labels={"x": "Stope Value ($)", "count": "Count"})
        st.plotly_chart(fig_sv, use_container_width=True)
    else:
        st.info("Run stope optimization first (Tab 3) to see economic analysis.")


# ============ TAB 6: Monte Carlo ============
with tab6:
    st.subheader("Monte Carlo Sensitivity Analysis")

    if "opt_result" in st.session_state and st.session_state["opt_result"]["selected"]:
        result = st.session_state["opt_result"]
        selected = result["selected"]
        dev = generate_development_network(selected, block_size=block_size)

        col1, col2, col3 = st.columns(3)
        with col1:
            n_sims = st.number_input("Number of simulations", 100, 10000, 1000, 100)
        with col2:
            price_std = st.slider("Price uncertainty (%)", 5, 30, 15) / 100
        with col3:
            grade_std = st.slider("Grade uncertainty (%)", 5, 25, 10) / 100

        if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
            with st.spinner(f"Running {n_sims} simulations..."):
                mc = run_monte_carlo(
                    selected, dev["total_cost"],
                    base_gold_price=gold_price,
                    base_mining_cost=mining_cost,
                    base_processing_cost=processing_cost,
                    base_ga_cost=ga_cost,
                    recovery=recovery,
                    n_simulations=n_sims,
                    price_std_pct=price_std,
                    grade_std_pct=grade_std,
                )
                st.session_state["mc_result"] = mc

        if "mc_result" in st.session_state:
            mc = st.session_state["mc_result"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean NPV", f"${mc['mean_npv']:,.0f}")
            col2.metric("Std Dev", f"${mc['std_npv']:,.0f}")
            col3.metric("P(NPV > 0)", f"{mc['prob_positive_pct']:.1f}%")
            col4.metric("P50 NPV", f"${mc['percentiles']['P50']:,.0f}")

            col1, col2 = st.columns(2)
            with col1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=mc["npvs"], nbinsx=50, name="NPV",
                    marker_color="steelblue", opacity=0.7,
                ))
                fig_hist.add_vline(x=mc["base_npv"], line_dash="dash", line_color="red",
                                   annotation_text="Deterministic NPV")
                fig_hist.add_vline(x=mc["mean_npv"], line_dash="dash", line_color="green",
                                   annotation_text="Mean NPV")
                fig_hist.add_vline(x=0, line_dash="solid", line_color="black")
                fig_hist.update_layout(
                    title="NPV Distribution", xaxis_title="NPV ($)",
                    yaxis_title="Frequency", height=450,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                tornado = mc["tornado"]
                fig_tornado = go.Figure()
                for t in tornado:
                    fig_tornado.add_trace(go.Bar(
                        y=[t["variable"]], x=[t["npv_high"] - mc["base_npv"]],
                        orientation="h", name=f"{t['variable']} +",
                        marker_color="#2ecc71", showlegend=False,
                        text=[f"${t['npv_high']:,.0f}"], textposition="auto",
                    ))
                    fig_tornado.add_trace(go.Bar(
                        y=[t["variable"]], x=[t["npv_low"] - mc["base_npv"]],
                        orientation="h", name=f"{t['variable']} -",
                        marker_color="#e74c3c", showlegend=False,
                        text=[f"${t['npv_low']:,.0f}"], textposition="auto",
                    ))
                fig_tornado.update_layout(
                    title="Tornado Chart (NPV Sensitivity)",
                    xaxis_title="Change in NPV ($)", barmode="relative",
                    height=450,
                )
                st.plotly_chart(fig_tornado, use_container_width=True)

            # Percentiles table
            st.subheader("NPV Percentiles")
            perc_df = pd.DataFrame([
                {"Percentile": k, "NPV ($)": f"${v:,.0f}"}
                for k, v in mc["percentiles"].items()
            ])
            st.dataframe(perc_df, use_container_width=True, hide_index=True)

            # NPV vs Price scatter
            fig_scatter = go.Figure(data=go.Scatter(
                x=mc["prices_used"], y=mc["npvs"],
                mode="markers", marker=dict(size=3, opacity=0.3, color="steelblue"),
            ))
            fig_scatter.update_layout(
                title="NPV vs Simulated Gold Price",
                xaxis_title="Gold Price ($/oz)", yaxis_title="NPV ($)",
                height=400,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Run stope optimization first (Tab 3) to perform Monte Carlo analysis.")


# ============ TAB 7: ML Grade Prediction ============
with tab7:
    st.subheader("ML-Based Grade Prediction")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Model", ["random_forest", "gradient_boosting"])
    with col2:
        test_pct = st.slider("Test set %", 10, 40, 20)

    if st.button("🤖 Train Model", type="primary"):
        with st.spinner("Training grade prediction model..."):
            ml_result = train_grade_model(
                df_blocks, model_type=model_type, test_size=test_pct / 100
            )
            st.session_state["ml_result"] = ml_result

    if "ml_result" in st.session_state:
        ml = st.session_state["ml_result"]

        st.success(f"Model trained: {ml['model_type']}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train R²", f"{ml['train_metrics']['r2']:.4f}")
        col2.metric("Test R²", f"{ml['test_metrics']['r2']:.4f}")
        col3.metric("Test RMSE", f"{ml['test_metrics']['rmse']:.4f} g/t")
        col4.metric("Test MAE", f"{ml['test_metrics']['mae']:.4f} g/t")

        col1, col2 = st.columns(2)
        with col1:
            # Actual vs Predicted scatter
            fig_ap = go.Figure()
            sample_n = min(2000, len(ml["y_test"]))
            idx = np.random.default_rng(42).choice(len(ml["y_test"]), sample_n, replace=False)
            y_t = np.array(ml["y_test"])[idx]
            y_p = np.array(ml["y_pred_test"])[idx]
            fig_ap.add_trace(go.Scatter(
                x=y_t, y=y_p, mode="markers",
                marker=dict(size=3, opacity=0.4, color="steelblue"),
                name="Predictions",
            ))
            max_v = max(y_t.max(), y_p.max())
            fig_ap.add_trace(go.Scatter(
                x=[0, max_v], y=[0, max_v], mode="lines",
                line=dict(color="red", dash="dash"), name="Perfect",
            ))
            fig_ap.update_layout(
                title="Actual vs Predicted Grade",
                xaxis_title="Actual (g/t)", yaxis_title="Predicted (g/t)",
                height=450,
            )
            st.plotly_chart(fig_ap, use_container_width=True)

        with col2:
            # Feature importance
            imp = ml["feature_importances"]
            imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            fig_imp = go.Figure(data=go.Bar(
                x=[v for _, v in imp_sorted],
                y=[k for k, _ in imp_sorted],
                orientation="h", marker_color="steelblue",
            ))
            fig_imp.update_layout(
                title="Feature Importance",
                xaxis_title="Importance", yaxis=dict(autorange="reversed"),
                height=450,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        # Prediction uncertainty map (RF only)
        if ml["uncertainty"] is not None:
            st.subheader("Prediction Uncertainty Map")
            unc = np.array(ml["uncertainty"])
            unc_df = df_blocks.copy()
            unc_df["uncertainty"] = unc
            unc_df["predicted_grade"] = ml["y_pred_all"]

            z_levels = sorted(unc_df["z"].unique())
            sel_z = st.select_slider(
                "Elevation for uncertainty view (m)",
                options=z_levels, value=z_levels[len(z_levels) // 2],
                key="ml_z_slider"
            )
            section = unc_df[unc_df["z"] == sel_z]

            col1, col2 = st.columns(2)
            with col1:
                fig_pred = go.Figure(data=go.Heatmap(
                    x=section["x"], y=section["y"], z=section["predicted_grade"],
                    colorscale="YlOrRd", colorbar=dict(title="Grade (g/t)"),
                ))
                fig_pred.update_layout(
                    title=f"Predicted Grade at {sel_z}m",
                    xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                    height=450, yaxis=dict(scaleanchor="x"),
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            with col2:
                fig_unc = go.Figure(data=go.Heatmap(
                    x=section["x"], y=section["y"], z=section["uncertainty"],
                    colorscale="Blues", colorbar=dict(title="Std Dev (g/t)"),
                ))
                fig_unc.update_layout(
                    title=f"Prediction Uncertainty at {sel_z}m",
                    xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                    height=450, yaxis=dict(scaleanchor="x"),
                )
                st.plotly_chart(fig_unc, use_container_width=True)

        # Residuals histogram
        fig_res = go.Figure(data=go.Histogram(
            x=ml["residuals"], nbinsx=50,
            marker_color="steelblue", opacity=0.7,
        ))
        fig_res.update_layout(
            title="Residual Distribution (Actual - Predicted)",
            xaxis_title="Residual (g/t)", yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig_res, use_container_width=True)


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Automated Underground Mine Design**  \n"
    "Research Prototype  \n"
    "IIT Mining Engineering"
)
