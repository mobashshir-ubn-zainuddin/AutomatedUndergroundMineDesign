// ===== GLOBAL STATE =====
let blockData = null;
let zLevels = [];
let mlZLevels = [];
let optimized = false;

const plotlyDarkLayout = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#94a3b8', family: 'Inter, system-ui, sans-serif', size: 12 },
  margin: { l: 50, r: 20, t: 40, b: 50 },
  xaxis: { gridcolor: '#1e293b', zerolinecolor: '#2d3a4f' },
  yaxis: { gridcolor: '#1e293b', zerolinecolor: '#2d3a4f' },
};

const plotly3dScene = {
  xaxis: { title: 'Easting (m)', backgroundcolor: '#0a0e17', gridcolor: '#1e293b', color: '#94a3b8' },
  yaxis: { title: 'Northing (m)', backgroundcolor: '#0a0e17', gridcolor: '#1e293b', color: '#94a3b8' },
  zaxis: { title: 'Elevation (m)', backgroundcolor: '#0a0e17', gridcolor: '#1e293b', color: '#94a3b8' },
  aspectmode: 'data',
};

const plotlyConfig = { responsive: true, displayModeBar: true, displaylogo: false };

// ===== TABS =====
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
  });
});

// ===== RANGE DISPLAY =====
document.getElementById('p_rec').addEventListener('input', e => {
  document.getElementById('p_rec_val').textContent = e.target.value + '%';
});

// ===== INLINE LOADING (per-tab) =====
function showTabLoading(tabId, text) {
  let el = document.getElementById(tabId + '-loader');
  if (!el) {
    el = document.createElement('div');
    el.id = tabId + '-loader';
    el.className = 'tab-loader';
    el.innerHTML = '<div class="spinner"></div><div class="loading-text"></div>';
    const tab = document.getElementById(tabId);
    tab.insertBefore(el, tab.firstChild);
  }
  el.querySelector('.loading-text').textContent = text;
  el.style.display = 'flex';
}
function hideTabLoading(tabId) {
  const el = document.getElementById(tabId + '-loader');
  if (el) el.style.display = 'none';
}
// legacy wrappers (unused but safe)
function showLoading(text) {}
function hideLoading() {}

// ===== UTILITY =====
function fmt(n) {
  if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return '$' + (n / 1e3).toFixed(0) + 'K';
  return '$' + n.toFixed(0);
}

function fmtNum(n) {
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

async function apiCall(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Server error');
  }
  return res.json();
}

// ===== 1. GENERATE BLOCK MODEL =====
async function generateBlockModel() {
  showTabLoading('tab-block', 'Generating block model & evaluating economics...');
  try {
    const data = await apiCall('/api/generate-block-model', {
      nx: +document.getElementById('p_nx').value,
      ny: +document.getElementById('p_ny').value,
      nz: +document.getElementById('p_nz').value,
      block_size: +document.getElementById('p_bs').value,
      seed: +document.getElementById('p_seed').value,
      gold_price: +document.getElementById('p_gold').value,
      mining_cost: +document.getElementById('p_mining').value,
      processing_cost: +document.getElementById('p_proc').value,
      ga_cost: +document.getElementById('p_ga').value,
      recovery: +document.getElementById('p_rec').value,
    });

    blockData = data;
    zLevels = data.z_levels;
    optimized = false;

    // Metrics
    document.getElementById('block-metrics').style.display = '';
    document.getElementById('m_total_blocks').textContent = fmtNum(data.total_blocks);
    document.getElementById('m_ore_blocks').textContent = fmtNum(data.ore_blocks);
    document.getElementById('m_mean_grade').textContent = data.mean_grade.toFixed(3) + ' g/t';
    document.getElementById('m_max_grade').textContent = data.max_grade.toFixed(3) + ' g/t';
    document.getElementById('block-status').style.display = 'none';
    hideTabLoading('tab-block');

    // Ore tab
    document.getElementById('ore-status').style.display = 'none';
    document.getElementById('ore-content').style.display = '';
    document.getElementById('m_cutoff').textContent = data.cutoff_grade.toFixed(3) + ' g/t';

    // Z slider
    const zSlider = document.getElementById('zSlider');
    zSlider.min = 0;
    zSlider.max = zLevels.length - 1;
    zSlider.value = Math.floor(zLevels.length / 2);
    document.getElementById('zSliderVal').textContent = zLevels[zSlider.value] + 'm';

    updateBlockPlot();
    plotGradeHistogram(data);
    plotGradeTonnage(data);
    updateCrossSection();

    // Update dependent tab statuses
    document.getElementById('stope-status').innerHTML = 'Block model ready. Click "Run Optimization" to proceed.';
    document.getElementById('stope-status').className = 'status-msg success';
    document.getElementById('ml-status').innerHTML = 'Block model ready. Click "Train Model" to proceed.';
    document.getElementById('ml-status').className = 'status-msg success';

  } catch (e) {
    document.getElementById('block-status').textContent = 'Error: ' + e.message;
    document.getElementById('block-status').className = 'status-msg error';
    document.getElementById('block-status').style.display = '';
    hideTabLoading('tab-block');
  }
}

function updateBlockPlot() {
  if (!blockData) return;
  const mode = document.getElementById('blockViewMode').value;
  let d;
  if (mode === 'ore') d = blockData.ore_blocks_data;
  else if (mode === 'high') d = blockData.high_grade_data;
  else d = blockData.all_blocks;

  Plotly.newPlot('chart-block3d', [{
    type: 'scatter3d', mode: 'markers',
    x: d.x, y: d.y, z: d.z,
    marker: { size: 2.5, color: d.grade, colorscale: 'YlOrRd', colorbar: { title: 'Grade (g/t)', tickfont: { color: '#94a3b8' } }, opacity: 0.6 },
    text: d.grade.map((g, i) => 'Grade: ' + g.toFixed(2) + ' g/t<br>Value: ' + fmt(d.value[i])),
    hoverinfo: 'text',
  }], { ...plotlyDarkLayout, scene: plotly3dScene, height: 600, margin: { l: 0, r: 0, t: 10, b: 0 } }, plotlyConfig);
}

function plotGradeHistogram(data) {
  Plotly.newPlot('chart-grade-hist', [
    { x: data.grade_histogram.ore_grades, type: 'histogram', nbinsx: 50, name: 'Ore', marker: { color: '#10b981', opacity: 0.7 } },
    { x: data.grade_histogram.waste_grades.filter(g => g > 0), type: 'histogram', nbinsx: 50, name: 'Waste', marker: { color: '#64748b', opacity: 0.5 } },
  ], {
    ...plotlyDarkLayout, barmode: 'overlay', title: { text: 'Grade Distribution', font: { size: 14 } },
    xaxis: { ...plotlyDarkLayout.xaxis, title: 'Gold Grade (g/t)' },
    yaxis: { ...plotlyDarkLayout.yaxis, title: 'Block Count' },
    shapes: [{ type: 'line', x0: data.cutoff_grade, x1: data.cutoff_grade, y0: 0, y1: 1, yref: 'paper', line: { color: '#ef4444', dash: 'dash', width: 2 } }],
    annotations: [{ x: data.cutoff_grade, y: 1, yref: 'paper', text: 'Cutoff: ' + data.cutoff_grade.toFixed(2), showarrow: false, font: { color: '#ef4444', size: 11 }, yanchor: 'bottom' }],
    legend: { x: 0.75, y: 0.95, bgcolor: 'rgba(0,0,0,0)' },
    height: 400,
  }, plotlyConfig);
}

function plotGradeTonnage(data) {
  Plotly.newPlot('chart-grade-tonnage', [{
    x: data.grade_tonnage.cum_tonnes, y: data.grade_tonnage.grades,
    mode: 'lines', line: { color: '#ef4444', width: 2 }, name: 'Grade',
  }], {
    ...plotlyDarkLayout, title: { text: 'Grade-Tonnage Curve', font: { size: 14 } },
    xaxis: { ...plotlyDarkLayout.xaxis, title: 'Cumulative Tonnes' },
    yaxis: { ...plotlyDarkLayout.yaxis, title: 'Grade (g/t)' },
    height: 400,
  }, plotlyConfig);
}

async function updateCrossSection() {
  const idx = +document.getElementById('zSlider').value;
  const z = zLevels[idx];
  document.getElementById('zSliderVal').textContent = z + 'm';

  try {
    const data = await apiCall('/api/cross-section', { z });
    Plotly.newPlot('chart-cross-section', [{
      x: data.x, y: data.y, z: data.grade, type: 'heatmap',
      colorscale: 'YlOrRd', colorbar: { title: 'Grade (g/t)', tickfont: { color: '#94a3b8' } },
    }], {
      ...plotlyDarkLayout,
      title: { text: 'Plan View at ' + z + 'm', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Easting (m)', scaleanchor: 'y' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Northing (m)' },
      height: 450,
    }, plotlyConfig);
  } catch (e) { /* ignore */ }
}

// ===== 3. STOPE OPTIMIZATION =====
async function runOptimization() {
  if (!blockData) { alert('Generate block model first'); return; }
  showTabLoading('tab-stope', 'Generating candidates & solving BILP optimization...');
  try {
    const data = await apiCall('/api/optimize', {
      min_width: +document.getElementById('p_minw').value,
      max_width: +document.getElementById('p_maxw').value,
      min_height: +document.getElementById('p_minh').value,
      max_height: +document.getElementById('p_maxh').value,
      min_length: +document.getElementById('p_minl').value,
      max_length: +document.getElementById('p_maxl').value,
      step: +document.getElementById('p_step').value,
    });

    optimized = true;
    document.getElementById('stope-status').style.display = 'none';
    document.getElementById('stope-content').style.display = '';
    document.getElementById('m_n_selected').textContent = data.n_selected;
    document.getElementById('m_stope_value').textContent = fmt(data.total_value);
    document.getElementById('m_stope_tonnage').textContent = fmtNum(Math.round(data.total_tonnage)) + ' t';
    document.getElementById('m_stope_metal').textContent = fmtNum(Math.round(data.total_metal_oz)) + ' oz';

    // 3D stopes
    const traces = [{
      type: 'scatter3d', mode: 'markers', name: 'Ore Blocks',
      x: data.ore_background.x, y: data.ore_background.y, z: data.ore_background.z,
      marker: { size: 1.5, color: '#475569', opacity: 0.1 },
    }];

    const colors = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#ec4899','#84cc16','#f97316','#6366f1'];
    data.selected.forEach((s, i) => {
      const c = colors[i % colors.length];
      traces.push({
        type: 'mesh3d',
        x: [s.x_min, s.x_min, s.x_max, s.x_max, s.x_min, s.x_min, s.x_max, s.x_max],
        y: [s.y_min, s.y_max, s.y_max, s.y_min, s.y_min, s.y_max, s.y_max, s.y_min],
        z: [s.z_min, s.z_min, s.z_min, s.z_min, s.z_max, s.z_max, s.z_max, s.z_max],
        i: [0,0,0,0,4,4,0,0,0,0,2,2], j: [1,2,4,5,5,6,1,3,1,5,3,7], k: [2,3,5,6,6,7,5,7,2,6,7,6],
        opacity: 0.4, color: c, name: 'Stope ' + s.stope_id,
        hovertext: 'Stope ' + s.stope_id + '<br>Value: ' + fmt(s.total_value) + '<br>Grade: ' + s.avg_grade.toFixed(2) + ' g/t',
      });
    });

    Plotly.newPlot('chart-stope3d', traces, {
      ...plotlyDarkLayout, scene: plotly3dScene, height: 650, margin: { l: 0, r: 0, t: 10, b: 0 }, showlegend: true, legend: { bgcolor: 'rgba(0,0,0,0)' },
    }, plotlyConfig);

    // Stope table
    const thead = document.querySelector('#stope-table thead');
    thead.innerHTML = '<tr><th>ID</th><th>Value</th><th>Tonnage</th><th>Metal (oz)</th><th>Avg Grade</th><th>W×H×L (m)</th><th>Ore %</th></tr>';
    const tbody = document.querySelector('#stope-table tbody');
    tbody.innerHTML = data.selected.map(s =>
      '<tr><td>' + s.stope_id + '</td><td>' + fmt(s.total_value) + '</td><td>' + fmtNum(Math.round(s.total_tonnage)) +
      '</td><td>' + s.total_metal_oz.toFixed(1) + '</td><td>' + s.avg_grade.toFixed(2) + ' g/t</td><td>' +
      s.width_m + '×' + s.height_m + '×' + s.length_m + '</td><td>' + (s.ore_fraction * 100).toFixed(0) + '%</td></tr>'
    ).join('');

    // Update dependent tabs
    ['layout-status','econ-status','mc-status'].forEach(id => {
      document.getElementById(id).innerHTML = 'Optimization complete. Data ready.';
      document.getElementById(id).className = 'status-msg success';
    });

    hideTabLoading('tab-stope');

    // Auto-load layout & economics
    loadMineLayout();
    loadEconomics();

  } catch (e) {
    document.getElementById('stope-status').textContent = 'Error: ' + e.message;
    document.getElementById('stope-status').className = 'status-msg error';
    document.getElementById('stope-status').style.display = '';
    hideTabLoading('tab-stope');
  }
}

// ===== 4. MINE LAYOUT =====
async function loadMineLayout() {
  try {
    const data = await apiCall('/api/mine-layout', {});
    document.getElementById('layout-status').style.display = 'none';
    document.getElementById('layout-content').style.display = '';
    document.getElementById('m_dev_length').textContent = fmtNum(Math.round(data.total_length_m)) + ' m';
    document.getElementById('m_dev_cost').textContent = fmt(data.total_cost);
    document.getElementById('m_n_levels').textContent = data.n_levels;

    const traces = [];
    const stopeColors = ['#475569','#64748b','#334155'];
    data.stope_boxes.forEach((s, i) => {
      traces.push({
        type: 'mesh3d',
        x: [s.x_min,s.x_min,s.x_max,s.x_max,s.x_min,s.x_min,s.x_max,s.x_max],
        y: [s.y_min,s.y_max,s.y_max,s.y_min,s.y_min,s.y_max,s.y_max,s.y_min],
        z: [s.z_min,s.z_min,s.z_min,s.z_min,s.z_max,s.z_max,s.z_max,s.z_max],
        i:[0,0,0,0,4,4,0,0,0,0,2,2],j:[1,2,4,5,5,6,1,3,1,5,3,7],k:[2,3,5,6,6,7,5,7,2,6,7,6],
        opacity: 0.2, color: stopeColors[i % 3], showlegend: false,
      });
    });

    const typeColors = { 'Decline': '#ef4444', 'Level Drive': '#3b82f6', 'Crosscut': '#10b981' };
    const added = {};
    data.segments.forEach(seg => {
      traces.push({
        type: 'scatter3d', mode: 'lines',
        x: [seg.x1, seg.x2], y: [seg.y1, seg.y2], z: [seg.z1, seg.z2],
        line: { color: typeColors[seg.type] || '#94a3b8', width: 5 },
        name: seg.type, showlegend: !added[seg.type],
        hovertext: seg.type + ': ' + seg.length_m + 'm',
      });
      added[seg.type] = true;
    });

    Plotly.newPlot('chart-layout3d', traces, {
      ...plotlyDarkLayout, scene: plotly3dScene, height: 650, margin: { l: 0, r: 0, t: 10, b: 0 }, legend: { bgcolor: 'rgba(0,0,0,0)' },
    }, plotlyConfig);

    const thead = document.querySelector('#layout-table thead');
    thead.innerHTML = '<tr><th>Type</th><th>Count</th><th>Total Length (m)</th><th>Cost</th></tr>';
    const tbody = document.querySelector('#layout-table tbody');
    tbody.innerHTML = data.summary.map(s =>
      '<tr><td>' + s.type + '</td><td>' + s.count + '</td><td>' + fmtNum(Math.round(s.total_length)) + '</td><td>' + fmt(s.cost) + '</td></tr>'
    ).join('');

  } catch (e) { /* ignore */ }
}

// ===== 5. ECONOMICS =====
async function loadEconomics() {
  try {
    const data = await apiCall('/api/economics', {});
    document.getElementById('econ-status').style.display = 'none';
    document.getElementById('econ-content').style.display = '';
    document.getElementById('m_npv').textContent = fmt(data.npv);
    document.getElementById('m_revenue').textContent = fmt(data.total_revenue);
    document.getElementById('m_total_cost').textContent = fmt(data.total_cost);

    // Cost pie
    Plotly.newPlot('chart-cost-pie', [{
      labels: data.cost_breakdown.categories, values: data.cost_breakdown.values,
      type: 'pie', hole: 0.45,
      marker: { colors: ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'] },
      textinfo: 'label+percent', textfont: { color: '#e2e8f0' },
    }], {
      ...plotlyDarkLayout, title: { text: 'Cost Breakdown', font: { size: 14 } }, height: 400, showlegend: false,
    }, plotlyConfig);

    // NPV sensitivity
    Plotly.newPlot('chart-npv-sens', [{
      x: data.sensitivity.prices, y: data.sensitivity.npvs,
      mode: 'lines+markers', line: { color: '#3b82f6', width: 2 }, marker: { size: 4 },
    }], {
      ...plotlyDarkLayout, title: { text: 'NPV vs Gold Price', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Gold Price ($/oz)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'NPV ($)' },
      shapes: [
        { type: 'line', x0: data.sensitivity.prices[0], x1: data.sensitivity.prices[data.sensitivity.prices.length - 1], y0: 0, y1: 0, line: { color: '#ef4444', dash: 'dash' } },
        { type: 'line', x0: data.current_price, x1: data.current_price, y0: Math.min(...data.sensitivity.npvs), y1: Math.max(...data.sensitivity.npvs), line: { color: '#10b981', dash: 'dash' } },
      ],
      annotations: [{ x: data.current_price, y: Math.max(...data.sensitivity.npvs), text: 'Current: $' + data.current_price, showarrow: false, font: { color: '#10b981', size: 11 }, yanchor: 'bottom' }],
      height: 400,
    }, plotlyConfig);

    // Stope value distribution
    Plotly.newPlot('chart-stope-dist', [{
      x: data.stope_values, type: 'histogram', nbinsx: 25, marker: { color: '#3b82f6', opacity: 0.7 },
    }], {
      ...plotlyDarkLayout, title: { text: 'Stope Value Distribution', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Stope Value ($)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Count' },
      height: 350,
    }, plotlyConfig);

  } catch (e) { /* ignore */ }
}

// ===== 6. MONTE CARLO =====
async function runMonteCarlo() {
  if (!optimized) { alert('Run stope optimization first'); return; }
  showTabLoading('tab-mc', 'Running Monte Carlo simulation...');
  try {
    const data = await apiCall('/api/monte-carlo', {
      n_simulations: +document.getElementById('p_nsims').value,
      price_std: +document.getElementById('p_price_std').value,
      grade_std: +document.getElementById('p_grade_std').value,
    });

    document.getElementById('mc-status').style.display = 'none';
    document.getElementById('mc-content').style.display = '';
    document.getElementById('m_mc_mean').textContent = fmt(data.mean_npv);
    document.getElementById('m_mc_std').textContent = fmt(data.std_npv);
    document.getElementById('m_mc_prob').textContent = data.prob_positive_pct.toFixed(1) + '%';
    document.getElementById('m_mc_p50').textContent = fmt(data.percentiles.P50);

    // NPV histogram
    Plotly.newPlot('chart-mc-hist', [{
      x: data.npvs, type: 'histogram', nbinsx: 50, marker: { color: '#3b82f6', opacity: 0.7 },
    }], {
      ...plotlyDarkLayout, title: { text: 'NPV Distribution', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'NPV ($)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Frequency' },
      shapes: [
        { type: 'line', x0: data.base_npv, x1: data.base_npv, y0: 0, y1: 1, yref: 'paper', line: { color: '#ef4444', dash: 'dash' } },
        { type: 'line', x0: data.mean_npv, x1: data.mean_npv, y0: 0, y1: 1, yref: 'paper', line: { color: '#10b981', dash: 'dash' } },
        { type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper', line: { color: '#94a3b8', width: 1 } },
      ],
      annotations: [
        { x: data.base_npv, y: 1, yref: 'paper', text: 'Deterministic', showarrow: false, font: { color: '#ef4444', size: 10 }, yanchor: 'bottom' },
        { x: data.mean_npv, y: 0.95, yref: 'paper', text: 'Mean', showarrow: false, font: { color: '#10b981', size: 10 }, yanchor: 'bottom' },
      ],
      height: 420,
    }, plotlyConfig);

    // Tornado chart
    const tTraces = [];
    data.tornado.forEach(t => {
      tTraces.push({
        y: [t.variable], x: [t.npv_high - data.base_npv], type: 'bar', orientation: 'h',
        marker: { color: '#10b981' }, showlegend: false, text: [fmt(t.npv_high)], textposition: 'auto', textfont: { size: 10 },
      });
      tTraces.push({
        y: [t.variable], x: [t.npv_low - data.base_npv], type: 'bar', orientation: 'h',
        marker: { color: '#ef4444' }, showlegend: false, text: [fmt(t.npv_low)], textposition: 'auto', textfont: { size: 10 },
      });
    });
    Plotly.newPlot('chart-tornado', tTraces, {
      ...plotlyDarkLayout, barmode: 'relative',
      title: { text: 'Tornado Chart (NPV Sensitivity)', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Change in NPV ($)' },
      height: 420,
    }, plotlyConfig);

    // Percentiles table
    const thead = document.querySelector('#mc-perc-table thead');
    thead.innerHTML = '<tr><th>Percentile</th><th>NPV</th></tr>';
    const tbody = document.querySelector('#mc-perc-table tbody');
    tbody.innerHTML = Object.entries(data.percentiles).map(([k, v]) =>
      '<tr><td>' + k + '</td><td>' + fmt(v) + '</td></tr>'
    ).join('');

    // NPV vs Price scatter
    Plotly.newPlot('chart-mc-scatter', [{
      x: data.prices_used, y: data.npvs, mode: 'markers',
      marker: { size: 3, opacity: 0.3, color: '#3b82f6' },
    }], {
      ...plotlyDarkLayout, title: { text: 'NPV vs Simulated Gold Price', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Gold Price ($/oz)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'NPV ($)' },
      height: 380,
    }, plotlyConfig);

    hideTabLoading('tab-mc');

  } catch (e) {
    document.getElementById('mc-status').textContent = 'Error: ' + e.message;
    document.getElementById('mc-status').className = 'status-msg error';
    document.getElementById('mc-status').style.display = '';
    hideTabLoading('tab-mc');
  }
}

// ===== 7. ML GRADE PREDICTION =====
async function runMLPrediction() {
  if (!blockData) { alert('Generate block model first'); return; }
  showTabLoading('tab-ml', 'Training grade prediction model...');
  try {
    const data = await apiCall('/api/ml-predict', {
      model_type: document.getElementById('p_model_type').value,
      test_pct: +document.getElementById('p_test_pct').value,
    });

    document.getElementById('ml-status').style.display = 'none';
    document.getElementById('ml-content').style.display = '';
    document.getElementById('m_train_r2').textContent = data.train_metrics.r2.toFixed(4);
    document.getElementById('m_test_r2').textContent = data.test_metrics.r2.toFixed(4);
    document.getElementById('m_test_rmse').textContent = data.test_metrics.rmse.toFixed(4) + ' g/t';
    document.getElementById('m_test_mae').textContent = data.test_metrics.mae.toFixed(4) + ' g/t';

    mlZLevels = data.z_levels;

    // Actual vs Predicted
    const maxV = Math.max(Math.max(...data.actual_vs_predicted.actual), Math.max(...data.actual_vs_predicted.predicted));
    Plotly.newPlot('chart-actual-pred', [
      { x: data.actual_vs_predicted.actual, y: data.actual_vs_predicted.predicted, mode: 'markers', marker: { size: 3, opacity: 0.4, color: '#3b82f6' }, name: 'Predictions' },
      { x: [0, maxV], y: [0, maxV], mode: 'lines', line: { color: '#ef4444', dash: 'dash' }, name: 'Perfect' },
    ], {
      ...plotlyDarkLayout, title: { text: 'Actual vs Predicted Grade', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Actual (g/t)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Predicted (g/t)' },
      height: 420, legend: { bgcolor: 'rgba(0,0,0,0)' },
    }, plotlyConfig);

    // Feature importance
    Plotly.newPlot('chart-feat-imp', [{
      x: data.feature_importances.values, y: data.feature_importances.names,
      type: 'bar', orientation: 'h', marker: { color: '#3b82f6' },
    }], {
      ...plotlyDarkLayout, title: { text: 'Feature Importance', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Importance' },
      yaxis: { ...plotlyDarkLayout.yaxis, autorange: 'reversed' },
      height: 420,
    }, plotlyConfig);

    // Uncertainty section
    if (data.has_uncertainty) {
      document.getElementById('ml-uncertainty-section').style.display = '';
      const mlSlider = document.getElementById('mlZSlider');
      mlSlider.min = 0;
      mlSlider.max = mlZLevels.length - 1;
      mlSlider.value = Math.floor(mlZLevels.length / 2);
      document.getElementById('mlZSliderVal').textContent = mlZLevels[mlSlider.value] + 'm';
      updateMLUncertainty();
    } else {
      document.getElementById('ml-uncertainty-section').style.display = 'none';
    }

    // Residuals
    Plotly.newPlot('chart-residuals', [{
      x: data.residuals, type: 'histogram', nbinsx: 50, marker: { color: '#3b82f6', opacity: 0.7 },
    }], {
      ...plotlyDarkLayout, title: { text: 'Residual Distribution (Actual - Predicted)', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Residual (g/t)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Count' },
      height: 320,
    }, plotlyConfig);

    hideTabLoading('tab-ml');

  } catch (e) {
    document.getElementById('ml-status').textContent = 'Error: ' + e.message;
    document.getElementById('ml-status').className = 'status-msg error';
    document.getElementById('ml-status').style.display = '';
    hideTabLoading('tab-ml');
  }
}

async function updateMLUncertainty() {
  const idx = +document.getElementById('mlZSlider').value;
  const z = mlZLevels[idx];
  document.getElementById('mlZSliderVal').textContent = z + 'm';

  try {
    const data = await apiCall('/api/ml-uncertainty', { z });

    Plotly.newPlot('chart-pred-grade', [{
      x: data.x, y: data.y, z: data.predicted_grade, type: 'heatmap',
      colorscale: 'YlOrRd', colorbar: { title: 'Grade (g/t)', tickfont: { color: '#94a3b8' } },
    }], {
      ...plotlyDarkLayout, title: { text: 'Predicted Grade at ' + z + 'm', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Easting (m)', scaleanchor: 'y' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Northing (m)' },
      height: 420,
    }, plotlyConfig);

    Plotly.newPlot('chart-pred-unc', [{
      x: data.x, y: data.y, z: data.uncertainty, type: 'heatmap',
      colorscale: 'Blues', colorbar: { title: 'Std Dev (g/t)', tickfont: { color: '#94a3b8' } },
    }], {
      ...plotlyDarkLayout, title: { text: 'Prediction Uncertainty at ' + z + 'm', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Easting (m)', scaleanchor: 'y' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Northing (m)' },
      height: 420,
    }, plotlyConfig);
  } catch (e) { /* ignore */ }
}

// ===== PRODUCTION SCHEDULE =====
async function runSchedule() {
  if (!optimized) { document.getElementById('sched-status').textContent = 'Run optimization first.'; return; }
  showTabLoading('tab-schedule', 'Generating production schedule...');
  try {
    const res = await fetch('/api/schedule', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        mining_rate: parseInt(document.getElementById('p_mining_rate').value),
        discount_rate: parseFloat(document.getElementById('p_discount_rate').value),
      })
    });
    const d = await res.json();
    if (d.error) { document.getElementById('sched-status').textContent = d.error; hideTabLoading('tab-schedule'); return; }

    document.getElementById('sched-status').style.display = 'none';
    document.getElementById('sched-content').style.display = 'block';

    const fmt = v => (v >= 1e6) ? '$' + (v/1e6).toFixed(1) + 'M' : '$' + (v/1e3).toFixed(0) + 'K';
    document.getElementById('m_dcf_npv').textContent = fmt(d.total_dcf_npv);
    document.getElementById('m_undisc_npv').textContent = fmt(d.undiscounted_npv);
    document.getElementById('m_mine_life').textContent = d.mine_life_years + ' years';
    document.getElementById('m_mining_rate').textContent = (d.mining_rate_tpa/1000).toFixed(0) + 'K t/yr';

    const periods = d.periods.map(p => 'Year ' + p.period);

    Plotly.newPlot('chart-sched-cashflow', [
      { x: periods, y: d.periods.map(p => p.revenue), name: 'Revenue', type: 'bar', marker: { color: '#22d3ee' } },
      { x: periods, y: d.periods.map(p => -p.opex), name: 'OpEx', type: 'bar', marker: { color: '#f43f5e' } },
      { x: periods, y: d.periods.map(p => -p.dev_cost), name: 'Dev Cost', type: 'bar', marker: { color: '#f59e0b' } },
      { x: periods, y: d.periods.map(p => p.net_cash_flow), name: 'Net CF', type: 'scatter', mode: 'lines+markers', line: { color: '#a78bfa', width: 3 } },
    ], { ...plotlyDarkLayout, barmode: 'relative', title: { text: 'Period Cash Flow', font: { size: 14 } }, height: 420 }, plotlyConfig);

    Plotly.newPlot('chart-sched-cumulative', [
      { x: periods, y: d.periods.map(p => p.cumulative_dcf), type: 'scatter', mode: 'lines+markers',
        fill: 'tozeroy', line: { color: '#10b981', width: 3 }, fillcolor: 'rgba(16,185,129,0.15)', name: 'Cumulative DCF' },
    ], { ...plotlyDarkLayout, title: { text: 'Cumulative DCF NPV', font: { size: 14 } }, height: 420 }, plotlyConfig);

    Plotly.newPlot('chart-sched-tonnage', [
      { x: periods, y: d.periods.map(p => p.tonnage), type: 'bar', marker: { color: '#3b82f6' }, name: 'Tonnage' },
    ], { ...plotlyDarkLayout, title: { text: 'Period Tonnage', font: { size: 14 } }, yaxis: { ...plotlyDarkLayout.yaxis, title: 'Tonnes' }, height: 380 }, plotlyConfig);

    Plotly.newPlot('chart-sched-grade', [
      { x: periods, y: d.periods.map(p => p.avg_grade), type: 'scatter', mode: 'lines+markers',
        line: { color: '#fbbf24', width: 3 }, marker: { size: 8 }, name: 'Avg Grade' },
    ], { ...plotlyDarkLayout, title: { text: 'Period Average Grade (g/t)', font: { size: 14 } }, height: 380 }, plotlyConfig);

  } catch (e) { document.getElementById('sched-status').textContent = 'Error: ' + e.message; }
  hideTabLoading('tab-schedule');
}

// ===== GEOTECHNICAL ASSESSMENT =====
async function runGeotech() {
  if (!optimized) { document.getElementById('geo-status').textContent = 'Run optimization first.'; return; }
  showTabLoading('tab-geotech', 'Assessing geotechnical stability...');
  try {
    const res = await fetch('/api/geotechnical', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({})
    });
    const d = await res.json();
    if (d.error) { document.getElementById('geo-status').textContent = d.error; hideTabLoading('tab-geotech'); return; }

    document.getElementById('geo-status').style.display = 'none';
    document.getElementById('geo-content').style.display = 'block';

    const s = d.summary;
    document.getElementById('m_geo_stopes').textContent = s.total_stopes;
    document.getElementById('m_geo_stability').textContent = s.avg_stability_number;
    document.getElementById('m_geo_pillars').textContent = s.total_pillars;
    document.getElementById('m_geo_fos').textContent = s.avg_fos;

    // Risk distribution pie
    const riskLabels = Object.keys(s.risk_distribution);
    const riskValues = Object.values(s.risk_distribution);
    const riskColors = { Low: '#10b981', Medium: '#f59e0b', High: '#f43f5e', Critical: '#dc2626' };
    Plotly.newPlot('chart-geo-risk', [{
      labels: riskLabels, values: riskValues, type: 'pie', hole: 0.45,
      marker: { colors: riskLabels.map(l => riskColors[l] || '#6366f1') },
      textinfo: 'label+percent', textfont: { color: '#e2e8f0' },
    }], { ...plotlyDarkLayout, title: { text: 'Stope Risk Distribution', font: { size: 14 } }, height: 380, showlegend: false }, plotlyConfig);

    // Pillar status bar
    const pillarStatuses = ['Adequate', 'Marginal', 'Inadequate'];
    const pillarCounts = [s.adequate_pillars, s.marginal_pillars, s.inadequate_pillars];
    const pillarColors = ['#10b981', '#f59e0b', '#f43f5e'];
    Plotly.newPlot('chart-geo-pillar', [{
      x: pillarStatuses, y: pillarCounts, type: 'bar',
      marker: { color: pillarColors },
    }], { ...plotlyDarkLayout, title: { text: 'Pillar Assessment', font: { size: 14 } }, height: 380 }, plotlyConfig);

    // Stability number scatter (depth vs stability number)
    const sa = d.stope_assessments;
    const riskColorMap = { Low: '#10b981', Medium: '#f59e0b', High: '#f43f5e', Critical: '#dc2626' };
    Plotly.newPlot('chart-geo-stability-scatter', [{
      x: sa.map(s => s.depth_m), y: sa.map(s => s.stability_number),
      mode: 'markers', type: 'scatter',
      marker: { size: 10, color: sa.map(s => riskColorMap[s.risk_level] || '#6366f1'), opacity: 0.8 },
      text: sa.map(s => `Stope ${s.stope_id}<br>Risk: ${s.risk_level}<br>UCS: ${s.ucs} MPa`),
      hoverinfo: 'text',
    }], {
      ...plotlyDarkLayout, title: { text: 'Stability Number vs Depth', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Depth (m)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Stability Number' },
      height: 420,
    }, plotlyConfig);

    // Mathews zone classification
    const zones = { Stable: 0, Transitional: 0, Caving: 0 };
    sa.forEach(s => { if (zones[s.mathews_zone] !== undefined) zones[s.mathews_zone]++; });
    Plotly.newPlot('chart-geo-mathews', [{
      labels: Object.keys(zones), values: Object.values(zones), type: 'pie', hole: 0.45,
      marker: { colors: ['#10b981', '#f59e0b', '#f43f5e'] },
      textinfo: 'label+percent', textfont: { color: '#e2e8f0' },
    }], { ...plotlyDarkLayout, title: { text: 'Mathews Stability Zones', font: { size: 14 } }, height: 420, showlegend: false }, plotlyConfig);

  } catch (e) { document.getElementById('geo-status').textContent = 'Error: ' + e.message; }
  hideTabLoading('tab-geotech');
}

// ===== SMOOTH SCROLL =====
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(a.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth' });
    document.getElementById('navLinks').classList.remove('open');
  });
});

// ===== NAVBAR SCROLL EFFECT =====
window.addEventListener('scroll', () => {
  document.getElementById('navbar').style.boxShadow =
    window.scrollY > 10 ? '0 4px 20px rgba(0,0,0,0.3)' : 'none';
});
