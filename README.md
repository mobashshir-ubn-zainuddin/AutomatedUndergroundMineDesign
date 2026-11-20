# MineOptima — Automated Underground Mine Design

An integrated computational platform that combines **Operations Research**, **Machine Learning**, and **Monte Carlo simulation** to automate underground stope layout optimization. The system replaces weeks of manual mine design iteration with an intelligent pipeline that runs in under 60 seconds.

Built with a **Flask + Python** backend and an interactive **Plotly.js** frontend.

---

## Key Features

- **3D Block Model Generation** — Synthetic gold deposit with two ore zones and spatially correlated grade fields (Gaussian filtering)
- **Economic Evaluation** — Breakeven cutoff grade calculation, per-block valuation (V = Tonnage × (Grade × Recovery × Price − Cost))
- **Floating Stope Algorithm** — Exhaustive candidate stope generation with configurable geometry constraints
- **BILP Stope Optimization** — Binary Integer Linear Programming via PuLP 4.x + HiGHS solver with provably optimal non-overlapping stope selection
- **Development Network** — Algorithmic generation of decline, level drives, and crosscuts with cost estimation
- **Monte Carlo Simulation** — 1,000+ NPV simulations under price/grade/cost uncertainty with tornado sensitivity analysis
- **ML Grade Prediction** — Random Forest / Gradient Boosting with 11 spatial features, uncertainty quantification via tree variance
- **Interactive Web Dashboard** — 7-tab interface with 3D Plotly visualization, real-time parameter controls, responsive dark-themed UI

---

## Project Structure

```
Automated_Underground_Mine_design/
│
├── flask_app.py                  # Flask backend — REST API endpoints
├── app.py                        # Streamlit app (legacy, optional)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── Technology_Report.docx        # Technology implementation report
├── implementation_plan.txt       # Original implementation plan
│
├── src/                          # Core computational modules
│   ├── block_model_generator.py  # Synthetic 3D block model with spatial correlation
│   ├── economic_evaluator.py     # Block valuation & ore/waste classification
│   ├── stope_generator.py        # Floating stope algorithm (candidate generation)
│   ├── stope_optimizer.py        # BILP optimization (PuLP 4.x + HiGHS)
│   ├── mine_layout.py            # Development network (decline, drives, crosscuts)
│   ├── monte_carlo.py            # Monte Carlo NPV simulation & tornado analysis
│   └── grade_prediction.py       # ML grade prediction (Random Forest / GBM)
│
├── templates/
│   └── index.html                # Main HTML template (landing page + dashboard)
│
├── static/
│   ├── css/
│   │   └── style.css             # Dark-themed responsive CSS
│   └── js/
│       └── app.js                # Frontend logic — API calls, Plotly charts, tabs
│
└── data/                         # Generated datasets
    ├── block_model.csv           # Raw block model (24,000 blocks)
    └── economic_block_model.csv  # Block model with economic values
```

---

## Tech Stack

| Technology | Role |
|---|---|
| Python 3.x | Core language for all backend modules |
| Flask | Web framework serving RESTful APIs |
| HTML5 / CSS3 / JS | Responsive frontend with dark theme |
| Plotly.js | Interactive 3D visualization (Scatter3d, Mesh3d, heatmaps) |
| NumPy / Pandas | Numerical computation and data management |
| SciPy | Gaussian spatial filtering for block model generation |
| PuLP 4.x | Mathematical programming interface for BILP |
| HiGHS | Open-source mixed-integer linear programming solver |
| scikit-learn | Random Forest & Gradient Boosting for grade prediction |

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Automated_Underground_Mine_design.git
cd Automated_Underground_Mine_design
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `highspy` is not in requirements.txt, also install it:

```bash
pip install highspy
```

### 4. Run the Flask application

```bash
python flask_app.py
```

The app will start on **http://localhost:5050**. Open this URL in your browser.

### 5. Using the application

1. **Scroll down** or click **"Launch App"** to reach the dashboard
2. Click **"Generate Block Model"** in the sidebar to create the 3D block model
3. Switch to the **Stope Optimization** tab and click **"Run Optimization"** to solve the BILP
4. The **Mine Layout** and **Economics** tabs auto-populate after optimization
5. Run **Monte Carlo** and **ML Prediction** from their respective tabs

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Landing page + dashboard |
| `/api/generate-block-model` | POST | Generate block model & evaluate economics |
| `/api/cross-section` | POST | Get grade heatmap at a given elevation |
| `/api/optimize` | POST | Run floating stope + BILP optimization |
| `/api/mine-layout` | POST | Generate development network |
| `/api/economics` | POST | Calculate NPV, cost breakdown, sensitivity |
| `/api/monte-carlo` | POST | Run Monte Carlo NPV simulation |
| `/api/ml-predict` | POST | Train ML grade prediction model |
| `/api/ml-uncertainty` | POST | Get prediction uncertainty at an elevation |

---

## Key Results (Default Parameters)

| Metric | Value |
|---|---|
| Blocks modeled | 24,000 |
| Candidate stopes | 3,358 |
| Optimal stopes selected | 424 |
| Total stope value | $380M |
| Mean NPV (Monte Carlo) | $231M |
| P(NPV > 0) | 95.8% |
| ML grade prediction R² | 0.978 |
| Total development | 39,600m |

---

## How It Differs from Traditional Practice

| Aspect | Traditional | MineOptima |
|---|---|---|
| Stope layout | Manual trial-and-error in Surpac/Deswik (days–weeks) | BILP with provably optimal solution (seconds) |
| Grade estimation | Kriging/IDW with limited uncertainty | Random Forest with per-block uncertainty |
| Risk analysis | Deterministic single-scenario NPV | 1,000-iteration Monte Carlo with full distribution |
| Workflow | Disconnected tools requiring specialists | Integrated pipeline, single web interface |
| Visualization | Expensive CAD licenses | Browser-based, accessible to any stakeholder |

---

## Deployment

To deploy online, you can use:

- **Render** — connect your GitHub repo, set build command `pip install -r requirements.txt`, start command `python flask_app.py`
- **Railway** — similar setup, auto-detects Python
- **AWS / GCP / Azure** — deploy as a Docker container or on a VM

For production, use a WSGI server:

```bash
pip install gunicorn
gunicorn flask_app:app --bind 0.0.0.0:5050
```

---

## License

This project is a research prototype developed for academic purposes.
