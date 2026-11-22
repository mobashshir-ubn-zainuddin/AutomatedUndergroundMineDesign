# MineOptima — Automated Underground Mine Design

**🌐 Live Demo: [https://mineoptima.onrender.com](https://mineoptima.onrender.com)**

> ⚡ **Cloud tip:** The free Render host has a 30-second HTTP timeout. Use the default small grid (20×15×10, step 3, 300 simulations) on the hosted version. Run locally for full-size models and heavier modules (stochastic optimisation, SGS, implicit geology).

An integrated computational platform that automates the full underground mine design pipeline — from raw drillhole data through to JORC reserve reporting, risk-adjusted stope optimization, ventilation design, and backfill scheduling — all in a single browser-based interface.

Built with a **Flask + Python** backend and an interactive **Plotly.js** frontend (19 tabs).

---

## What Has Been Implemented

### Phase 1 — Geological Block Model
- Synthetic 3D block model generator calibrated to Archean lode gold deposits
- Log-normal grade distribution (CV ≈ 180%), nugget effect, quartz vein structures, high-grade shoots, alteration zonation
- Per-block economic valuation: `V = Tonnage × (Grade × Recovery × Price − Cost)`
- Breakeven cutoff grade calculation
- Interactive 3D block model viewer with ore/waste classification overlay

### Phase 2 — Stope Optimization (BILP)
- Floating Stope Algorithm: exhaustive candidate stope generation with configurable geometry constraints (min/max width, height, length)
- **Binary Integer Linear Programming** via PuLP 4.x + HiGHS solver — provably optimal non-overlapping stope selection
- 3D stope box visualisation coloured by value

### Phase 3 — Mine Layout & Development
- Algorithmic generation of decline, level drives, and crosscuts
- Per-segment cost estimation ($/m)
- Interactive 3D development network viewer alongside stope boxes

### Phase 4 — Economics & Sensitivity
- Full NPV calculation (revenue − mining − processing − G&A − development)
- Waterfall cost breakdown chart
- Gold price sensitivity curve ($1,200–$2,800/oz)

### Phase 5 — Monte Carlo Risk Analysis
- Configurable NPV simulations (default 300 on cloud, up to 1,000+ locally)
- Price, grade, and cost uncertainty modelled independently
- NPV distribution histogram with P10/P50/P90 percentiles
- Tornado sensitivity chart identifying the dominant risk driver

### Phase 6 — ML Grade Prediction
- Random Forest and Gradient Boosting models with 11 spatial features
- Per-block prediction uncertainty via tree variance
- Actual vs predicted scatter, feature importance bar chart
- Uncertainty heatmap at any elevation slice

### Phase 7 — Production Scheduling & Geotechnical
- **Production Scheduling:** grade-priority stope sequencing, annual mining rate control, DCF NPV, cumulative cash flow
- **Geotechnical Assessment:** Mathews stability graph, in-situ stress, pillar Factor of Safety, risk classification (Low / Medium / High / Critical)

### Phase 7b — Geostatistics & Scenario Management
- **Ordinary Kriging (OK):** experimental variogram, Spherical / Exponential / Gaussian model fitting, kriging variance, moving neighbourhood, cross-validation R² and RMSE
- **Multi-Scenario Comparison:** save multiple optimisation runs with different economic parameters and compare NPV, tonnage, grade, and stope count side-by-side

### Phase 8 — Advanced Modules (Latest)

#### Bolani Iron Ore Dataset
- Real Surpac drillhole data (collar, survey, assay, geology) for the Bolani iron ore deposit, Jharkhand, India
- 29 drillholes, 585 composites (Fe%, SiO₂%, Al₂O₃%)
- **Drillhole Depth Profile Viewer:** select any individual drillhole or overlay all; choose Fe / SiO₂ / Al₂O₃; colour by lithology; depth log, box plot by lithocode, mean±σ bar chart across drillholes
- IDW block model from composites with transparent 3D viewer and grade-range filter
- Fe% normal and log-normal distribution histograms, Fe–SiO₂ and Fe–Al₂O₃ scatter plots, experimental variogram

#### Advanced Kriging Variants
- **Simple Kriging (SK):** uses a known global mean; cross-validated with R², RMSE, 3D grade map
- **Indicator Kriging (IK):** estimates P(grade > threshold) → Measured / Indicated / Inferred classification; 3D classified block viewer and summary bar chart
- **Sequential Gaussian Simulation (SGS):** K equiprobable grade realisations; E-type (mean) and uncertainty (σ) maps; P10/P90 band scatter; uncertainty histogram

#### JORC / NI 43-101 Reserve Reporting
- Measured / Indicated / Inferred block classification by drill proximity or rock type
- Summary table: blocks, tonnes, mean grade, contained metal, in-situ value
- Cutoff grade sensitivity table (0.5×–2× base cutoff)
- Grade–tonnage curve and classification bar chart

#### Stochastic Stope Optimization
- Generates K SGS grade simulations per block
- Selects stopes only where P(Economic Value > 0) ≥ p_min (user-controlled)
- 3D confidence scatter coloured by P(V > 0); P(V > 0) histogram with p_min threshold line

#### Multi-Objective Stope Optimization
- ε-constraint method sweeping geotechnical risk threshold
- Objective: `Maximize α₁·NPV − α₂·R_geo − α₃·C_dev`
- 2D Pareto frontier (NPV vs Risk) and 3D Pareto space (NPV × Risk × Cost)
- Recommended trade-off point highlighted; user-configurable α weights

#### 3D Implicit Geological Modeling
- Radial Basis Function (RBF) signed distance field delineates ore zone boundaries
- Ore zone, waste zone, iso-surface at SDF = 0, and fault zones displayed simultaneously in 3D
- Block classification: ore / waste based on SDF value

#### Ventilation Network Design
- Hardy Cross iterative airflow balancing across development network branches
- Minimum airflow check: ≥ 0.06 m³/s per kW diesel (regulatory)
- Wet-bulb temperature at depth check: ≤ 27°C
- Fan sizing (kW and ΔP), booster fan flagging, level-by-level compliance chart

#### Backfill Optimization
- Primary / secondary stope classification by spatial arrangement
- Network flow sequencing: fill primaries before mining secondaries
- CRF ($85/m³) and paste fill ($120/m³) support
- Per-period schedule bar chart, 3D dependency network, economics waterfall

---

## Project Structure

```
Automated_Underground_Mine_design/
│
├── flask_app.py                  # Flask backend — all REST API endpoints (25+)
├── Procfile                      # Gunicorn start command for Render/Railway
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── implementation_plan.txt       # Phase-by-phase implementation notes
│
├── src/                          # Core computational modules
│   ├── block_model_generator.py  # Synthetic gold block model (log-normal, nugget effect)
│   ├── economic_evaluator.py     # Block valuation & cutoff grade
│   ├── stope_generator.py        # Floating stope candidate generation
│   ├── stope_optimizer.py        # BILP optimisation (PuLP 4.x + HiGHS)
│   ├── mine_layout.py            # Development network generation
│   ├── monte_carlo.py            # Monte Carlo NPV & tornado analysis
│   ├── grade_prediction.py       # Random Forest / GBM grade prediction
│   ├── production_scheduler.py   # DCF production scheduling
│   ├── geotechnical.py           # Mathews method, pillar FoS
│   ├── kriging.py                # OK, SK, IK, SGS geostatistical estimation
│   ├── bolani_loader.py          # Surpac drillhole data parser + IDW block model
│   ├── reserve_reporting.py      # JORC/NI 43-101 reserve classification
│   ├── stochastic_optimizer.py   # SGS-based probabilistic stope selection
│   ├── multi_objective.py        # ε-constraint Pareto optimisation
│   ├── implicit_geology.py       # RBF implicit geological modelling
│   ├── ventilation.py            # Hardy Cross ventilation circuit design
│   └── backfill.py               # Network flow backfill scheduling
│
├── bolani_data/                  # Real Surpac export (Bolani iron ore, India)
│   ├── collar.csv
│   ├── survey.csv
│   ├── assay.csv
│   └── geology.csv
│
├── bolani_results/               # Surpac output images for reference
│
├── templates/
│   └── index.html                # 19-tab dashboard (landing page + app)
│
├── static/
│   ├── css/style.css             # Dark-themed responsive CSS
│   └── js/app.js                 # Frontend logic — API calls, Plotly charts
│
└── report/
    ├── report.tex                # LaTeX source (22-page technical report)
    └── report.pdf                # Compiled PDF
```

---

## Tech Stack

| Technology | Role |
|---|---|
| Python 3.x | Core language for all backend modules |
| Flask | Web framework — RESTful API server |
| Gunicorn | Production WSGI server |
| HTML5 / CSS3 / JavaScript | Responsive dark-themed frontend |
| Plotly.js | Interactive 3D and 2D charts (Scatter3d, Waterfall, Box) |
| NumPy / Pandas | Numerical computation and data handling |
| SciPy | Spatial filtering, RBF interpolation, normal-score transforms |
| PuLP 4.x | BILP mathematical programming interface |
| HiGHS | Open-source MILP solver (provably optimal) |
| scikit-learn | Random Forest & Gradient Boosting grade prediction |

---

## Running Locally (Step-by-Step)

### 1. Clone the repository

```bash
git clone https://github.com/mobashshir-ubn-zainuddin/AutomatedUndergroundMineDesign
cd Automated_Underground_Mine_design
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Linux / macOS
source venv/bin/activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

> If you see a `highspy` import error, install it separately:
> ```bash
> pip install highspy
> ```

### 4. Start the Flask development server

```bash
python flask_app.py
```

The app starts on **http://localhost:5050**. Open that URL in your browser.

> To use the production server locally (same as Render):
> ```bash
> gunicorn flask_app:app --bind 0.0.0.0:5050 --timeout 120 --workers 1 --threads 4
> ```

### 5. Using the dashboard

Follow this recommended workflow through the 19 tabs:

| Step | Tab | Action |
|---|---|---|
| 1 | **Block Model** | Set grid size (up to 40×30×20 locally), click **Generate Block Model** |
| 2 | **Ore Classification** | Auto-populated; view ore/waste 3D map |
| 3 | **Stope Optimization** | Click **Run Optimization** (BILP — takes 5–30s) |
| 4 | **Mine Layout** | Click **Generate Layout** to build development network |
| 5 | **Economics** | Click **Calculate Economics** |
| 6 | **Monte Carlo** | Set simulations (up to 1,000 locally), click **Run** |
| 7 | **ML Prediction** | Select Random Forest or GBM, click **Train Model** |
| 8 | **Production Schedule** | Click **Generate Schedule** |
| 9 | **Geotechnical** | Click **Run Assessment** |
| 10 | **Kriging** | Select variogram model, click **Run Kriging** |
| 11 | **Scenarios** | Save and compare multiple runs |
| 12 | **Adv. Kriging** | Choose SK / IK / SGS variant, click **Run** |
| 13 | **Bolani Data** | Click **Load Bolani Dataset**, then **Build Block Model** |
| 14 | **Reserve Report** | Click **Generate Reserve Report** (JORC/NI 43-101) |
| 15 | **Stochastic Opt.** | Set K simulations and p_min, click **Run** |
| 16 | **Multi-Objective** | Set α weights, click **Run Multi-Objective Opt.** |
| 17 | **Implicit Geology** | Click **Build Implicit Model** |
| 18 | **Ventilation** | Set diesel power and ambient temp, click **Design Ventilation** |
| 19 | **Backfill** | Select fill type, click **Optimize Backfill** |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Landing page + dashboard |
| `/healthz` | GET | Health check for deployment monitoring |
| `/api/generate-block-model` | POST | Generate synthetic gold block model |
| `/api/cross-section` | POST | Grade heatmap at given elevation |
| `/api/optimize` | POST | Floating stope + BILP optimization |
| `/api/mine-layout` | POST | Generate development network |
| `/api/economics` | POST | NPV, cost breakdown, sensitivity |
| `/api/monte-carlo` | POST | Monte Carlo NPV simulation |
| `/api/ml-predict` | POST | Train ML grade prediction model |
| `/api/ml-uncertainty` | POST | Prediction uncertainty at elevation |
| `/api/schedule` | POST | Production schedule with DCF NPV |
| `/api/geotechnical` | POST | Geotechnical stability assessment |
| `/api/kriging` | POST | Ordinary Kriging (OK) |
| `/api/kriging/simple` | POST | Simple Kriging (SK) |
| `/api/kriging/indicator` | POST | Indicator Kriging (IK) — resource classification |
| `/api/kriging/sgs` | POST | Sequential Gaussian Simulation (SGS) |
| `/api/scenarios` | GET | List saved scenarios |
| `/api/scenarios/save` | POST | Save current run as named scenario |
| `/api/scenarios/compare` | POST | Compare all saved scenarios |
| `/api/scenarios/delete/<id>` | DELETE | Remove a scenario |
| `/api/bolani/load` | POST | Load Bolani drillhole composites |
| `/api/bolani/drillhole-profile` | POST | Depth profile for selected element/drillhole |
| `/api/bolani/block-model` | POST | IDW block model from composites |
| `/api/reserve-report` | POST | JORC/NI 43-101 reserve table |
| `/api/stochastic-optimize` | POST | SGS-based probabilistic stope selection |
| `/api/multi-objective` | POST | ε-constraint Pareto optimization |
| `/api/implicit-geology` | POST | RBF implicit geological model |
| `/api/ventilation` | POST | Hardy Cross ventilation circuit |
| `/api/backfill` | POST | Network flow backfill optimizer |

---

## Key Results (Default Parameters — Gold Model)

| Metric | Value |
|---|---|
| Blocks modeled | 24,000 |
| Candidate stopes generated | ~3,400 |
| Optimal stopes selected (BILP) | ~420 |
| Total stope value | ~$380M |
| Mean NPV (Monte Carlo) | ~$231M |
| P(NPV > 0) | 95.8% |
| ML grade prediction R² | 0.978 |
| Total development | ~39,600 m |
| Bolani composites (real data) | 585 (29 drillholes) |
| Bolani mean Fe% | 63.4% |

---

## Deployment

**Live:** [https://mineoptima.onrender.com](https://mineoptima.onrender.com)

### Render (current hosting)

1. Push this repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com) and connect the repo
3. Build command: `pip install -r requirements.txt`
4. Start command: auto-detected from `Procfile`:
   ```
   gunicorn flask_app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 4
   ```
5. Add environment variable: `PORT=10000`
6. Deploy — Render rebuilds automatically on every `git push`

> **Free tier limit:** Render's free plan has a 30-second HTTP timeout. The heavy modules (BILP, ML, SGS, stochastic optimisation) may hit this limit. Use the default small grid (20×15×10) and 300 simulations on the live site. All features run without limitation locally.

### Railway / other platforms
Same setup — Railway auto-detects the `Procfile`.

---

## Data Sources & Calibration

### Synthetic Gold Model
Calibrated against published parameters from Archean lode gold operations:

| Parameter | Value | Reference |
|---|---|---|
| Grade distribution | Log-normal, mean ~2–5 g/t Au | Archean lode gold deposits (Canadian Shield) |
| High-grade shoots | Up to 20+ g/t Au | NI 43-101 reports, Red Lake (Evolution Mining) |
| Block size | 5 m × 5 m × 5 m | Industry standard for underground stope design |
| Gold price | US$1,900/oz | World Gold Council benchmark |
| Mining cost | $45/t | Canadian underground gold AISC benchmarks |
| Processing cost | $25/t | Typical CIL/CIP gold processing |
| Recovery | 92% | LaRonde Mine (Agnico Eagle) technical reports |
| UCS | 120 MPa (σ = 25 MPa) | Hard-rock Precambrian geology |
| Mathews stability | Mathews et al. (1981), Potvin (1988) | Global standard for open stope stability |

### Bolani Real Drillhole Data
Iron ore deposit, Jharkhand, India — assay data exported from Surpac. Composites include Fe%, SiO₂%, Al₂O₃%, P, and S at 0.3–3 m intervals across 29 vertical drillholes.

---

## Author

**Mobashshir Zainuddin**  
Roll No: 23MI3PE01  
Department of Mining Engineering  
Indian Institute of Technology (IIT) Kharagpur

---

## License

This project was developed for academic and research purposes.
