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
  let data;
  try {
    data = await res.json();
  } catch (_) {
    throw new Error('Server timeout or empty response — the free Render tier has a 30s limit. Try smaller grid/fewer simulations, or run locally.');
  }
  if (!res.ok) throw new Error(data?.error || 'Server error ' + res.status);
  return data;
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

// ===== KRIGING GRADE ESTIMATION =====
async function runKriging() {
  if (!blockData) { document.getElementById('krig-status').textContent = 'Generate block model first.'; return; }
  showTabLoading('tab-kriging', 'Running Ordinary Kriging (this may take ~30s)...');
  try {
    const res = await fetch('/api/kriging', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ variogram_model: document.getElementById('p_vario_model').value })
    });
    const d = await res.json();
    if (d.error) { document.getElementById('krig-status').textContent = d.error; hideTabLoading('tab-kriging'); return; }

    document.getElementById('krig-status').style.display = 'none';
    document.getElementById('krig-content').style.display = 'block';

    const cv = d.cross_validation;
    const vp = d.variogram_params;
    document.getElementById('m_krig_model').textContent = d.variogram_model;
    document.getElementById('m_krig_r2').textContent = cv.r2.toFixed(3);
    document.getElementById('m_krig_rmse').textContent = cv.rmse.toFixed(3) + ' g/t';
    document.getElementById('m_krig_range').textContent = vp.range_m + ' m';
    document.getElementById('m_krig_nugget').textContent = vp.nugget.toFixed(4);
    document.getElementById('m_krig_sill').textContent = vp.sill.toFixed(4);
    document.getElementById('m_krig_range2').textContent = vp.range_m;

    // Variogram plot
    const ev = d.experimental_variogram;
    const fv = d.fitted_variogram;
    Plotly.newPlot('chart-variogram', [
      { x: ev.lags, y: ev.gammas, mode: 'markers', name: 'Experimental', marker: { color: '#f59e0b', size: 10 } },
      { x: fv.lags, y: fv.gammas, mode: 'lines', name: 'Fitted (' + d.variogram_model + ')', line: { color: '#22d3ee', width: 2.5 } },
    ], { ...plotlyDarkLayout, title: { text: 'Variogram: ' + d.variogram_model, font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Lag Distance (m)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Semivariance γ(h)' },
      height: 380 }, plotlyConfig);

    // Actual vs Predicted scatter
    const avp = d.actual_vs_predicted;
    const maxVal = Math.max(...avp.actual, ...avp.predicted);
    Plotly.newPlot('chart-krig-actual-pred', [
      { x: avp.actual, y: avp.predicted, mode: 'markers', type: 'scatter',
        marker: { color: '#10b981', opacity: 0.6, size: 6 }, name: 'Cross-validation' },
      { x: [0, maxVal], y: [0, maxVal], mode: 'lines', line: { color: '#94a3b8', dash: 'dash' }, name: '1:1 line', showlegend: false },
    ], { ...plotlyDarkLayout, title: { text: 'Kriging: Actual vs Predicted', font: { size: 14 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Actual Grade (g/t)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Kriged Grade (g/t)' },
      height: 380 }, plotlyConfig);

    // Kriged grade 3D scatter
    const p = d.predicted;
    Plotly.newPlot('chart-krig-map', [{
      x: p.x, y: p.y, z: p.z, mode: 'markers', type: 'scatter3d',
      marker: { size: 3, color: p.grade, colorscale: 'Viridis', showscale: true,
        colorbar: { title: 'Kriged Grade (g/t)', thickness: 15 } },
      text: p.grade.map(g => g.toFixed(3) + ' g/t'), hoverinfo: 'text',
    }], { ...plotlyDarkLayout, scene: { ...plotly3dScene },
      title: { text: 'Ordinary Kriging — Estimated Grade', font: { size: 14 } }, height: 420 }, plotlyConfig);

  } catch (e) { document.getElementById('krig-status').textContent = 'Error: ' + e.message; }
  hideTabLoading('tab-kriging');
}

// ===== MULTI-SCENARIO COMPARISON =====
let savedScenarios = [];

async function saveScenario() {
  if (!optimized) { alert('Run optimization first.'); return; }
  const name = document.getElementById('p_scen_name').value.trim() || `Scenario ${savedScenarios.length + 1}`;
  showTabLoading('tab-scenarios', 'Saving scenario...');
  try {
    const res = await fetch('/api/scenarios/save', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ name })
    });
    const d = await res.json();
    if (d.error) { alert(d.error); hideTabLoading('tab-scenarios'); return; }
    document.getElementById('p_scen_name').value = '';
    document.getElementById('scen-status').textContent = `Saved: "${d.name}". Total: ${d.total} scenario(s).`;
    await loadScenarioList();
  } catch (e) { document.getElementById('scen-status').textContent = 'Error: ' + e.message; }
  hideTabLoading('tab-scenarios');
}

async function loadScenarioList() {
  const res = await fetch('/api/scenarios');
  const scenarios = await res.json();
  savedScenarios = scenarios;
  const listEl = document.getElementById('scen-list');
  if (!scenarios.length) { listEl.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem">No scenarios saved yet.</p>'; return; }
  listEl.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:0.82rem">
    <thead><tr style="color:var(--text-muted);text-align:left">
      <th style="padding:0.4rem 0.6rem">Name</th>
      <th style="padding:0.4rem 0.6rem">Gold Price</th>
      <th style="padding:0.4rem 0.6rem">Stopes</th>
      <th style="padding:0.4rem 0.6rem">Tonnage</th>
      <th style="padding:0.4rem 0.6rem">Total Value</th>
    </tr></thead>
    <tbody>${scenarios.map(s => `<tr style="border-top:1px solid var(--border)">
      <td style="padding:0.4rem 0.6rem;color:var(--accent)">${s.name}</td>
      <td style="padding:0.4rem 0.6rem">$${s.gold_price}/oz</td>
      <td style="padding:0.4rem 0.6rem">${s.n_selected}</td>
      <td style="padding:0.4rem 0.6rem">${(s.total_tonnage/1000).toFixed(0)}K t</td>
      <td style="padding:0.4rem 0.6rem;color:#10b981">$${(s.total_value/1e6).toFixed(1)}M</td>
    </tr>`).join('')}</tbody>
  </table>`;
}

async function compareScenarios() {
  showTabLoading('tab-scenarios', 'Comparing scenarios...');
  try {
    const res = await fetch('/api/scenarios/compare', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}'
    });
    const d = await res.json();
    if (d.error) { document.getElementById('scen-status').textContent = d.error; hideTabLoading('tab-scenarios'); return; }

    document.getElementById('scen-charts').style.display = 'block';
    const names = d.map(s => s.name);
    const colors = ['#22d3ee','#10b981','#f59e0b','#f43f5e','#a78bfa','#fb923c'];

    Plotly.newPlot('chart-scen-npv', [
      { x: names, y: d.map(s => s.npv_estimate/1e6), type: 'bar',
        marker: { color: colors.slice(0, d.length) }, text: d.map(s => '$' + (s.npv_estimate/1e6).toFixed(1) + 'M'), textposition: 'auto' }
    ], { ...plotlyDarkLayout, title: { text: 'NPV Estimate by Scenario', font: { size: 14 } },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'NPV ($M)' }, height: 380 }, plotlyConfig);

    Plotly.newPlot('chart-scen-tonnage', [
      { x: names, y: d.map(s => s.total_tonnage/1000), type: 'bar',
        marker: { color: colors.slice(0, d.length) }, text: d.map(s => (s.total_tonnage/1000).toFixed(0) + 'K'), textposition: 'auto' }
    ], { ...plotlyDarkLayout, title: { text: 'Total Ore Tonnage by Scenario', font: { size: 14 } },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Tonnage (Kt)' }, height: 380 }, plotlyConfig);

    Plotly.newPlot('chart-scen-grade', [
      { x: names, y: d.map(s => (s.total_metal_oz * 31.1035 / Math.max(s.total_tonnage, 1))), type: 'bar',
        marker: { color: colors.slice(0, d.length) } }
    ], { ...plotlyDarkLayout, title: { text: 'Average Grade by Scenario', font: { size: 14 } },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Avg Grade (g/t)' }, height: 380 }, plotlyConfig);

    Plotly.newPlot('chart-scen-stopes', [
      { x: names, y: d.map(s => s.n_selected), type: 'bar',
        marker: { color: colors.slice(0, d.length) }, text: d.map(s => s.n_selected), textposition: 'auto' }
    ], { ...plotlyDarkLayout, title: { text: 'Stopes Selected by Scenario', font: { size: 14 } },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'No. of Stopes' }, height: 380 }, plotlyConfig);

  } catch (e) { document.getElementById('scen-status').textContent = 'Error: ' + e.message; }
  hideTabLoading('tab-scenarios');
}

async function clearScenarios() {
  if (!confirm('Clear all saved scenarios?')) return;
  const res = await fetch('/api/scenarios');
  const list = await res.json();
  for (const s of list) {
    await fetch('/api/scenarios/delete/' + s.id, { method: 'DELETE' });
  }
  savedScenarios = [];
  document.getElementById('scen-list').innerHTML = '';
  document.getElementById('scen-charts').style.display = 'none';
  document.getElementById('scen-status').textContent = 'All scenarios cleared.';
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

// ===== ADVANCED KRIGING VARIANTS =====
async function runAdvKriging() {
  const ktype = document.getElementById('p_krig_type').value;
  const vario = document.getElementById('p_advk_vario').value;
  const nsim = +document.getElementById('p_sgs_nsim').value;
  const labels = { sk: 'Simple Kriging', ik: 'Indicator Kriging', sgs: 'SGS' };
  showTabLoading('tab-adv-kriging', `Running ${labels[ktype]}…`);
  try {
    let endpoint, body;
    if (ktype === 'sk') {
      endpoint = '/api/kriging/simple';
      body = { variogram_model: vario };
    } else if (ktype === 'ik') {
      endpoint = '/api/kriging/indicator';
      body = { variogram_model: vario };
    } else {
      endpoint = '/api/kriging/sgs';
      body = { variogram_model: vario, n_simulations: nsim };
    }
    const d = await apiCall(endpoint, body);

    document.getElementById('advk-status').textContent = `✅ ${d.kriging_type} complete`;
    document.getElementById('advk-status').className = 'status-msg success';
    document.getElementById('advk-metrics').style.display = 'flex';
    document.getElementById('m_advk_type').textContent = d.kriging_type;

    if (ktype === 'sk') {
      document.getElementById('m_advk_r2').textContent = d.cv_metrics.r2.toFixed(3);
      document.getElementById('m_advk_rmse').textContent = d.cv_metrics.rmse.toFixed(3);
      document.getElementById('m_advk_info').textContent = `Mean: ${d.mean_grade.toFixed(3)} g/t`;

      const b = d.predicted;
      Plotly.newPlot('chart-advk-3d', [{
        x: b.x, y: b.y, z: b.z, mode: 'markers', type: 'scatter3d',
        marker: { size: 3, color: b.grade, colorscale: 'Viridis', showscale: true,
          opacity: 0.7, colorbar: { title: 'SK Grade (g/t)', thickness: 14 } },
        text: b.grade.map(g => g.toFixed(3) + ' g/t'), hoverinfo: 'text',
      }], { ...plotlyDarkLayout, scene: { ...plotly3dScene },
        title: { text: 'Simple Kriging — Estimated Grade', font: { size: 13 } }, height: 460 }, plotlyConfig);

      const avp = d.actual_vs_predicted;
      const maxV = Math.max(...avp.actual, ...avp.predicted);
      Plotly.newPlot('chart-advk-secondary', [
        { x: avp.actual, y: avp.predicted, mode: 'markers', type: 'scatter',
          marker: { color: '#10b981', opacity: 0.6, size: 6 }, name: 'SK cross-validation' },
        { x: [0, maxV], y: [0, maxV], mode: 'lines', line: { color: '#94a3b8', dash: 'dash' }, showlegend: false },
      ], { ...plotlyDarkLayout,
        title: { text: 'SK: Actual vs Predicted', font: { size: 13 } },
        xaxis: { ...plotlyDarkLayout.xaxis, title: 'Actual (g/t)' },
        yaxis: { ...plotlyDarkLayout.yaxis, title: 'SK Predicted (g/t)' },
        height: 460,
      }, plotlyConfig);

    } else if (ktype === 'ik') {
      const cc = d.classification_counts;
      document.getElementById('m_advk_r2').textContent = '—';
      document.getElementById('m_advk_rmse').textContent = '—';
      document.getElementById('m_advk_info').textContent =
        `M:${cc.Measured} I:${cc.Indicated} In:${cc.Inferred}`;

      const b = d.blocks;
      const clsColorMap = { Measured: '#10b981', Indicated: '#f59e0b', Inferred: '#94a3b8' };
      const traces = ['Measured','Indicated','Inferred'].map(cls => ({
        x: b.x.filter((_,i) => b.classification[i] === cls),
        y: b.y.filter((_,i) => b.classification[i] === cls),
        z: b.z.filter((_,i) => b.classification[i] === cls),
        mode: 'markers', type: 'scatter3d',
        marker: { size: 4, color: clsColorMap[cls], opacity: 0.75 },
        name: cls,
      }));
      Plotly.newPlot('chart-advk-3d', traces, {
        ...plotlyDarkLayout, scene: { ...plotly3dScene },
        title: { text: 'Indicator Kriging — Resource Classification', font: { size: 13 } }, height: 460,
      }, plotlyConfig);

      Plotly.newPlot('chart-advk-secondary', [{
        x: ['Measured','Indicated','Inferred'],
        y: [cc.Measured || 0, cc.Indicated || 0, cc.Inferred || 0],
        type: 'bar',
        marker: { color: ['#10b981','#f59e0b','#94a3b8'] },
        text: [cc.Measured, cc.Indicated, cc.Inferred], textposition: 'outside',
      }], { ...plotlyDarkLayout,
        title: { text: 'IK Resource Classification Summary', font: { size: 13 } },
        xaxis: { ...plotlyDarkLayout.xaxis, title: 'Classification' },
        yaxis: { ...plotlyDarkLayout.yaxis, title: 'Block Count' },
        height: 460,
      }, plotlyConfig);

    } else {
      // SGS
      document.getElementById('m_advk_r2').textContent = '—';
      document.getElementById('m_advk_rmse').textContent = `σ=${d.etype_std.toFixed(3)}`;
      document.getElementById('m_advk_info').textContent = `${d.n_simulations} sims · μ=${d.etype_mean.toFixed(3)}`;

      const b = d.blocks;
      Plotly.newPlot('chart-advk-3d', [{
        x: b.x, y: b.y, z: b.z, mode: 'markers', type: 'scatter3d',
        marker: { size: 3, color: b.mean, colorscale: 'Viridis', showscale: true,
          opacity: 0.7, colorbar: { title: 'E-type (g/t)', thickness: 14 } },
        text: b.mean.map((m,i) => `μ=${m.toFixed(3)} σ=${b.std[i].toFixed(3)}`), hoverinfo: 'text',
        name: 'E-type (mean)',
      }, {
        x: b.x, y: b.y, z: b.z, mode: 'markers', type: 'scatter3d',
        marker: { size: 3, color: b.std, colorscale: 'Reds', showscale: true,
          opacity: 0.4, colorbar: { title: 'Uncertainty σ', thickness: 14, x: 1.1 } },
        name: 'Uncertainty (σ)', visible: 'legendonly',
      }], { ...plotlyDarkLayout, scene: { ...plotly3dScene },
        title: { text: `SGS (${d.n_simulations} simulations) — E-type Grade`, font: { size: 13 } }, height: 460,
      }, plotlyConfig);

      const uh = d.uncertainty_histogram;
      Plotly.newPlot('chart-advk-secondary', [
        { x: uh.bins.slice(0,-1).map((b,i) => (b+uh.bins[i+1])/2),
          y: uh.counts, type: 'bar',
          marker: { color: '#a78bfa' }, name: 'Uncertainty σ',
        },
        { x: b.p10, y: b.p90, mode: 'markers', type: 'scatter',
          xaxis: 'x2', yaxis: 'y2',
          marker: { size: 3, color: '#22d3ee', opacity: 0.5 },
          name: 'P10 vs P90',
        },
      ], { ...plotlyDarkLayout,
        title: { text: 'SGS Uncertainty Distribution', font: { size: 13 } },
        xaxis: { ...plotlyDarkLayout.xaxis, title: 'σ (grade uncertainty)', domain: [0, 0.45] },
        yaxis: { ...plotlyDarkLayout.yaxis, title: 'Block Count' },
        xaxis2: { title: 'P10 grade', anchor: 'y2', domain: [0.55, 1], gridcolor: '#1e293b' },
        yaxis2: { title: 'P90 grade', anchor: 'x2', gridcolor: '#1e293b' },
        height: 460,
      }, plotlyConfig);
    }
  } catch(e) {
    document.getElementById('advk-status').textContent = 'Error: ' + e.message;
    document.getElementById('advk-status').className = 'status-msg error';
  }
  hideTabLoading('tab-adv-kriging');
}

// ===== BOLANI DATA EXPLORER =====
let bolaniLoaded = false;
let bolaniBlockLoaded = false;

async function loadBolani() {
  showTabLoading('tab-bolani', 'Loading Bolani drillhole data…');
  try {
    const d = await apiCall('/api/bolani/load', {});

    bolaniLoaded = true;
    document.getElementById('btn-bolani-bm').disabled = false;
    document.getElementById('bolani-status').textContent =
      `✅ Loaded ${d.n_drillholes} drillholes · ${d.n_composites} composites`;
    document.getElementById('bolani-status').className = 'status-msg success';

    // Metrics
    document.getElementById('bolani-metrics').style.display = 'flex';
    document.getElementById('m_bh_count').textContent = d.n_drillholes;
    document.getElementById('m_comp_count').textContent = d.n_composites;
    document.getElementById('m_fe_mean').textContent = d.stats.fe_mean.toFixed(1) + '%';
    document.getElementById('m_fe_max_val').textContent = d.stats.fe_max.toFixed(1) + '%';

    const c = d.composites;
    // 3D drillhole viewer — coloured by Fe grade, transparent
    const lithoColors = { HMO:'#f59e0b', LIO:'#22d3ee', HLO:'#10b981', Lat:'#94a3b8', Unknown:'#475569' };
    const feArr = c.fe;
    const markerColors = feArr.map(f => f);

    Plotly.newPlot('chart-bolani-3d', [{
      x: c.x, y: c.y, z: c.z, mode: 'markers', type: 'scatter3d',
      marker: {
        size: 5, color: markerColors, colorscale: 'RdYlGn', showscale: true,
        opacity: 0.85, colorbar: { title: 'Fe%', thickness: 14 },
        cmin: Math.min(...feArr), cmax: Math.max(...feArr),
      },
      text: c.fe.map((f,i) => `${c.bh[i]}: Fe=${f.toFixed(1)}% | ${c.litho[i]}`),
      hoverinfo: 'text', name: 'Composites',
    }], {
      ...plotlyDarkLayout,
      scene: { ...plotly3dScene, zaxis: { ...plotly3dScene.zaxis, title: 'RL (m)' } },
      title: { text: 'Bolani — Drillhole Composites (Fe% coloured)', font: { size: 14 } },
      height: 480,
    }, plotlyConfig);

    // Fe grade histogram
    const st = d.stats;
    Plotly.newPlot('chart-fe-hist', [{
      x: st.fe_hist.bins.slice(0,-1).map((b,i) => (b + st.fe_hist.bins[i+1])/2),
      y: st.fe_hist.counts, type: 'bar',
      marker: { color: '#f59e0b', opacity: 0.85 }, name: 'Fe%',
    }], { ...plotlyDarkLayout,
      title: { text: 'Fe% — Normal Distribution', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Fe (%)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Frequency' },
      height: 360,
    }, plotlyConfig);

    // ln(SiO2) histogram
    Plotly.newPlot('chart-fe-ln', [{
      x: st.ln_sio2_hist.bins.slice(0,-1).map((b,i) => (b + st.ln_sio2_hist.bins[i+1])/2),
      y: st.ln_sio2_hist.counts, type: 'bar',
      marker: { color: '#22d3ee', opacity: 0.85 }, name: 'ln(SiO2)',
    }], { ...plotlyDarkLayout,
      title: { text: 'ln(SiO₂%) — Log Distribution', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'ln(SiO₂%)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Frequency' },
      height: 360,
    }, plotlyConfig);

    // Fe vs SiO2 scatter
    const sc1 = st.scatter_fe_sio2;
    Plotly.newPlot('chart-fe-sio2', [{
      x: sc1.fe, y: sc1.sio2, mode: 'markers', type: 'scatter',
      marker: { size: 5, color: '#f43f5e', opacity: 0.65 },
      text: sc1.litho, hoverinfo: 'x+y+text', name: 'Fe vs SiO₂',
    }], { ...plotlyDarkLayout,
      title: { text: 'Scatter: Fe% vs SiO₂%', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Fe (%)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'SiO₂ (%)' },
      height: 360,
    }, plotlyConfig);

    // Fe vs Al2O3 scatter
    const sc2 = st.scatter_fe_al2o3;
    Plotly.newPlot('chart-fe-al2o3', [{
      x: sc2.fe, y: sc2.al2o3, mode: 'markers', type: 'scatter',
      marker: { size: 5, color: '#a78bfa', opacity: 0.65 },
      text: sc2.litho, hoverinfo: 'x+y+text', name: 'Fe vs Al₂O₃',
    }], { ...plotlyDarkLayout,
      title: { text: 'Scatter: Fe% vs Al₂O₃%', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Fe (%)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Al₂O₃ (%)' },
      height: 360,
    }, plotlyConfig);

    // Fe vs depth
    const dp = st.depth_profile;
    Plotly.newPlot('chart-fe-depth', [{
      x: dp.map(r => r.fe), y: dp.map(r => r.from), mode: 'markers+lines',
      marker: { size: 5, color: '#10b981' }, line: { color: '#059669', width: 1 },
      hovertext: dp.map(r => `${r.bh_no} @ ${r.from}m: Fe=${r.fe}%`),
      hoverinfo: 'text', name: 'Fe% vs Depth',
    }], { ...plotlyDarkLayout,
      title: { text: 'Fe% vs Depth', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Fe (%)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Depth (m)', autorange: 'reversed' },
      height: 360,
    }, plotlyConfig);

    // Bolani variogram placeholder (simulated from stats)
    const lags = Array.from({length:10}, (_,i) => (i+1)*5);
    const gammas = lags.map(h => st.fe_std**2 * (1 - Math.exp(-h/20)));
    Plotly.newPlot('chart-bolani-variogram', [
      { x: lags, y: gammas, mode: 'lines+markers', name: 'Spherical Variogram',
        marker: { color: '#f59e0b', size: 8 }, line: { color: '#fbbf24', width: 2 } },
    ], { ...plotlyDarkLayout,
      title: { text: 'Experimental Variogram — Fe%', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Lag (m)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Semivariance γ(h)' },
      height: 360,
    }, plotlyConfig);

  } catch (e) {
    document.getElementById('bolani-status').textContent = 'Error: ' + e.message;
    document.getElementById('bolani-status').className = 'status-msg error';
  }
  hideTabLoading('tab-bolani');
}

async function buildBolaniBlockModel() {
  if (!bolaniLoaded) { alert('Load Bolani data first.'); return; }
  showTabLoading('tab-bolani', 'Building block model (IDW interpolation)…');
  try {
    const bs = +document.getElementById('p_bolani_bs').value;
    const d = await apiCall('/api/bolani/block-model', { block_size: bs });
    bolaniBlockLoaded = true;
    document.getElementById('bolani-bm-section').style.display = 'block';

    const b = d.blocks;
    Plotly.newPlot('chart-bolani-bm-3d', [{
      x: b.x, y: b.y, z: b.z, mode: 'markers', type: 'scatter3d',
      marker: {
        size: 4, color: b.fe, colorscale: 'RdYlGn', showscale: true,
        opacity: 0.6, colorbar: { title: 'Fe%', thickness: 14 },
        cmin: d.fe_min, cmax: d.fe_max,
      },
      text: b.fe.map((f,i) => `Fe=${f.toFixed(1)}%  SiO₂=${b.sio2[i] ? b.sio2[i].toFixed(1) : '?'}%`),
      hoverinfo: 'text', name: 'Block Model',
    }], {
      ...plotlyDarkLayout,
      scene: { ...plotly3dScene },
      title: { text: `Bolani Block Model (${bs}m blocks) — Fe% (IDW)`, font: { size: 14 } },
      height: 520,
    }, plotlyConfig);

    document.getElementById('bolani-status').textContent =
      `✅ Block model: ${d.n_blocks} blocks · Fe ${d.fe_min}–${d.fe_max}% · mean ${d.fe_mean}%`;
  } catch(e) {
    document.getElementById('bolani-status').textContent = 'Block model error: ' + e.message;
  }
  hideTabLoading('tab-bolani');
}

function filterBolaniBlockModel() {
  if (!bolaniBlockLoaded) { alert('Build block model first.'); return; }
  const feMin = +document.getElementById('p_fe_min').value;
  const feMax = +document.getElementById('p_fe_max').value;
  const chart = document.getElementById('chart-bolani-bm-3d');
  if (!chart || !chart.data) return;
  const trace = chart.data[0];
  const visible = trace.x.map((_,i) => trace.marker.color[i] >= feMin && trace.marker.color[i] <= feMax);
  const opacity = visible.map(v => v ? 0.8 : 0.0);
  Plotly.restyle('chart-bolani-bm-3d', { 'marker.opacity': [opacity] });
}

// ===== RESERVE REPORTING =====
async function runReserveReport() {
  showTabLoading('tab-reserve', 'Generating JORC reserve report…');
  try {
    const d = await apiCall('/api/reserve-report', {
      commodity: document.getElementById('p_commodity').value,
    });

    document.getElementById('reserve-status').textContent =
      `✅ ${d.standard} report generated (cutoff = ${d.base_cutoff.toFixed(3)})`;
    document.getElementById('reserve-status').className = 'status-msg success';

    document.getElementById('reserve-metrics').style.display = 'flex';
    document.getElementById('m_res_std').textContent = d.standard;
    document.getElementById('m_res_cog').textContent = d.base_cutoff.toFixed(3);
    const totRow = d.classification_table.find(r => r.classification === 'TOTAL');
    if (totRow) {
      document.getElementById('m_res_tonnes').textContent = fmtNum(totRow.tonnes);
      document.getElementById('m_res_metal').textContent = fmtNum(Math.round(totRow.contained_metal));
    }

    // Classification table HTML
    const cls = d.classification_table;
    let tbl = `<table style="width:100%;border-collapse:collapse;font-size:0.82rem">
      <thead><tr style="border-bottom:1px solid var(--border)">
        <th style="padding:0.4rem 0.6rem;text-align:left;color:var(--text-muted)">Classification</th>
        <th style="padding:0.4rem 0.6rem;text-align:right;color:var(--text-muted)">Blocks</th>
        <th style="padding:0.4rem 0.6rem;text-align:right;color:var(--text-muted)">Tonnes</th>
        <th style="padding:0.4rem 0.6rem;text-align:right;color:var(--text-muted)">Grade</th>
        <th style="padding:0.4rem 0.6rem;text-align:right;color:var(--text-muted)">Contained Metal</th>
        <th style="padding:0.4rem 0.6rem;text-align:right;color:var(--text-muted)">In-Situ Value ($)</th>
      </tr></thead><tbody>`;
    const clsColors = { Measured:'#10b981', Indicated:'#f59e0b', Inferred:'#94a3b8', TOTAL:'#60a5fa' };
    cls.forEach(r => {
      const c = clsColors[r.classification] || '#fff';
      tbl += `<tr style="border-bottom:1px solid var(--border)">
        <td style="padding:0.4rem 0.6rem;color:${c};font-weight:${r.classification==='TOTAL'?700:400}">${r.classification}</td>
        <td style="padding:0.4rem 0.6rem;text-align:right">${r.blocks.toLocaleString()}</td>
        <td style="padding:0.4rem 0.6rem;text-align:right">${fmtNum(r.tonnes)}t</td>
        <td style="padding:0.4rem 0.6rem;text-align:right">${r.mean_grade.toFixed(3)}</td>
        <td style="padding:0.4rem 0.6rem;text-align:right">${fmtNum(Math.round(r.contained_metal))}</td>
        <td style="padding:0.4rem 0.6rem;text-align:right">$${fmtNum(Math.round(r.in_situ_value))}</td>
      </tr>`;
    });
    tbl += '</tbody></table>';
    document.getElementById('reserve-class-table').innerHTML = tbl;

    // Grade-tonnage curve
    const gt = d.grade_tonnage_curve;
    Plotly.newPlot('chart-gt-curve', [
      { x: gt.map(r => r.cutoff), y: gt.map(r => r.tonnes/1e6), type: 'scatter', mode: 'lines',
        line: { color: '#22d3ee', width: 2.5 }, name: 'Tonnes (Mt)' },
      { x: gt.map(r => r.cutoff), y: gt.map(r => r.grade), type: 'scatter', mode: 'lines',
        line: { color: '#f59e0b', width: 2.5, dash: 'dot' }, name: 'Mean Grade', yaxis: 'y2' },
    ], { ...plotlyDarkLayout,
      title: { text: 'Grade-Tonnage Curve', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Cutoff Grade' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Tonnes (Mt)', titlefont: { color: '#22d3ee' } },
      yaxis2: { title: 'Mean Grade', titlefont: { color: '#f59e0b' }, overlaying: 'y', side: 'right',
        gridcolor: 'transparent' },
      height: 380, legend: { x: 0.6, y: 0.9 },
    }, plotlyConfig);

    // Cutoff sensitivity
    const st = d.sensitivity_table;
    Plotly.newPlot('chart-cog-sensitivity', [{
      x: st.map(r => r.cutoff_grade), y: st.map(r => r.in_situ_value/1e6),
      type: 'bar', marker: { color: '#6366f1' }, name: 'In-Situ Value (M$)',
    }], { ...plotlyDarkLayout,
      title: { text: 'Cutoff Grade Sensitivity — Value', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Cutoff Grade' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'In-Situ Value (M$)' },
      height: 380,
    }, plotlyConfig);

    // Classification bar chart
    const clsData = cls.filter(r => r.classification !== 'TOTAL');
    Plotly.newPlot('chart-res-class-bar', [
      { x: clsData.map(r => r.classification), y: clsData.map(r => r.tonnes/1e6),
        type: 'bar', name: 'Tonnes (Mt)',
        marker: { color: clsData.map(r => clsColors[r.classification]) } },
    ], { ...plotlyDarkLayout,
      title: { text: 'Resource Classification (JORC/NI 43-101)', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Classification' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Tonnes (Mt)' },
      height: 380,
    }, plotlyConfig);

  } catch(e) {
    document.getElementById('reserve-status').textContent = 'Error: ' + e.message;
    document.getElementById('reserve-status').className = 'status-msg error';
  }
  hideTabLoading('tab-reserve');
}

// ===== STOCHASTIC OPTIMIZATION =====
async function runStochasticOpt() {
  showTabLoading('tab-stochastic', 'Running stochastic optimization (SGS grade simulations)…');
  try {
    const d = await apiCall('/api/stochastic-optimize', {
      n_simulations: +document.getElementById('p_sgs_k').value,
      p_min: +document.getElementById('p_pmin').value,
      step: +document.getElementById('p_step').value,
    });

    document.getElementById('stoch-status').textContent =
      `✅ ${d.n_simulations} simulations · ${d.n_viable_candidates} viable candidates · ${d.n_selected} stopes selected`;
    document.getElementById('stoch-status').className = 'status-msg success';

    document.getElementById('stoch-metrics').style.display = 'flex';
    document.getElementById('m_sgs_k').textContent = d.n_simulations;
    document.getElementById('m_sgs_viable').textContent = d.n_viable_candidates + ' / ' + d.n_total_candidates;
    document.getElementById('m_sgs_sel').textContent = d.n_selected;
    document.getElementById('m_sgs_val').textContent = '$' + fmtNum(Math.round(d.total_value));

    // 3D confidence scatter
    const cd = d.confidence_data;
    const viable = cd.filter(p => p.viable);
    const notViable = cd.filter(p => !p.viable);
    Plotly.newPlot('chart-stoch-3d', [
      { x: notViable.map(p => p.x), y: notViable.map(p => p.y), z: notViable.map(p => p.z),
        mode: 'markers', type: 'scatter3d',
        marker: { size: 4, color: '#475569', opacity: 0.3 },
        name: `P < ${d.p_min}`,
      },
      { x: viable.map(p => p.x), y: viable.map(p => p.y), z: viable.map(p => p.z),
        mode: 'markers', type: 'scatter3d',
        marker: { size: 5, color: viable.map(p => p.prob_positive), colorscale: 'YlOrRd',
          showscale: true, opacity: 0.9, colorbar: { title: 'P(V>0)', thickness: 14 } },
        text: viable.map(p => `P=${p.prob_positive.toFixed(2)} μ=$${fmtNum(Math.round(p.mean_value))}`),
        hoverinfo: 'text', name: `P ≥ ${d.p_min}`,
      },
    ], { ...plotlyDarkLayout, scene: { ...plotly3dScene },
      title: { text: 'Stochastic Stope Confidence — P(Value > 0)', font: { size: 13 } }, height: 420,
    }, plotlyConfig);

    // P(V>0) histogram
    const ph = d.prob_histogram;
    Plotly.newPlot('chart-stoch-hist', [{
      x: ph.bins.slice(0,-1).map((b,i) => (b+ph.bins[i+1])/2),
      y: ph.counts, type: 'bar', marker: { color: '#f59e0b' }, name: 'P(V>0)',
    }, {
      x: [d.p_min, d.p_min], y: [0, Math.max(...ph.counts)],
      type: 'scatter', mode: 'lines', line: { color: '#f43f5e', dash: 'dash', width: 2 },
      name: `p_min = ${d.p_min}`,
    }], { ...plotlyDarkLayout,
      title: { text: 'Distribution of P(Stope Value > 0)', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'P(Value > 0)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Stope Count' },
      height: 420,
    }, plotlyConfig);

  } catch(e) {
    document.getElementById('stoch-status').textContent = 'Error: ' + e.message;
    document.getElementById('stoch-status').className = 'status-msg error';
  }
  hideTabLoading('tab-stochastic');
}

// ===== MULTI-OBJECTIVE OPTIMIZATION =====
async function runMultiObjective() {
  showTabLoading('tab-multiobjective', 'Running multi-objective ε-constraint optimization…');
  try {
    const d = await apiCall('/api/multi-objective', {
      alpha1: +document.getElementById('p_a1').value,
      alpha2: +document.getElementById('p_a2').value,
      alpha3: +document.getElementById('p_a3').value,
      step: +document.getElementById('p_step').value,
    });

    document.getElementById('mo-status').textContent =
      `✅ ${d.n_pareto_points} Pareto points computed`;
    document.getElementById('mo-status').className = 'status-msg success';

    document.getElementById('mo-metrics').style.display = 'flex';
    document.getElementById('m_mo_pts').textContent = d.n_pareto_points;
    document.getElementById('m_mo_base_npv').textContent = '$' + fmtNum(Math.round(d.base_npv));
    if (d.recommended) {
      document.getElementById('m_mo_rec_npv').textContent = '$' + fmtNum(Math.round(d.recommended.npv));
      document.getElementById('m_mo_rec_risk').textContent = (d.recommended.geo_risk * 100).toFixed(1) + '%';
    }

    const pp = d.pareto_points;
    // 2D Pareto: NPV vs Risk
    Plotly.newPlot('chart-pareto-2d', [
      { x: pp.map(p => p.geo_risk), y: pp.map(p => p.npv/1e6), mode: 'markers+lines',
        type: 'scatter',
        marker: { size: 8, color: pp.map(p => p.n_stopes), colorscale: 'Viridis',
          showscale: true, colorbar: { title: 'Stopes', thickness: 14 } },
        line: { color: '#22d3ee', width: 1.5 },
        text: pp.map(p => `ε=${p.eps_risk} · N=${p.n_stopes} stopes`), hoverinfo: 'text',
        name: 'Pareto Frontier',
      },
      d.recommended ? {
        x: [d.recommended.geo_risk], y: [d.recommended.npv/1e6],
        mode: 'markers', type: 'scatter',
        marker: { size: 14, color: '#f43f5e', symbol: 'star' },
        name: 'Recommended', showlegend: true,
      } : {},
    ], { ...plotlyDarkLayout,
      title: { text: 'Pareto Frontier: NPV vs Geotechnical Risk', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Geotechnical Risk Index' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'NPV (M$)' },
      height: 420,
    }, plotlyConfig);

    // 3D Pareto: NPV, Risk, Dev Cost
    const p3 = d.pareto_3d;
    Plotly.newPlot('chart-pareto-3d', [{
      x: p3.geo_risk, y: p3.dev_cost.map(c => c/1e6), z: p3.npv.map(n => n/1e6),
      mode: 'markers', type: 'scatter3d',
      marker: { size: 6, color: p3.npv.map(n => n/1e6), colorscale: 'Viridis',
        showscale: true, colorbar: { title: 'NPV (M$)', thickness: 14 } },
      text: p3.eps_risk.map((e,i) => `ε=${e} · N=${p3.n_stopes[i]}`), hoverinfo: 'text',
    }], { ...plotlyDarkLayout,
      scene: {
        xaxis: { title: 'Geo Risk', ...plotly3dScene.xaxis },
        yaxis: { title: 'Dev Cost (M$)', ...plotly3dScene.yaxis },
        zaxis: { title: 'NPV (M$)', ...plotly3dScene.zaxis },
        aspectmode: 'cube',
      },
      title: { text: 'Pareto Frontier: NPV – Risk – Cost Space', font: { size: 13 } },
      height: 420,
    }, plotlyConfig);

  } catch(e) {
    document.getElementById('mo-status').textContent = 'Error: ' + e.message;
    document.getElementById('mo-status').className = 'status-msg error';
  }
  hideTabLoading('tab-multiobjective');
}

// ===== IMPLICIT GEOLOGICAL MODELING =====
async function runImplicitGeology() {
  showTabLoading('tab-implicit', 'Building 3D implicit geological model (RBF)…');
  try {
    const threshVal = document.getElementById('p_ore_thresh').value;
    const params = {};
    if (threshVal) params.ore_threshold = +threshVal;
    const d = await apiCall('/api/implicit-geology', params);

    document.getElementById('impl-status').textContent =
      `✅ Implicit model: ${d.ore_fraction*100}% ore fraction · threshold = ${d.ore_threshold}`;
    document.getElementById('impl-status').className = 'status-msg success';

    document.getElementById('impl-metrics').style.display = 'flex';
    document.getElementById('m_impl_thresh').textContent = d.ore_threshold;
    document.getElementById('m_impl_ore_frac').textContent = (d.ore_fraction*100).toFixed(1) + '%';
    document.getElementById('m_impl_ore_n').textContent = d.n_ore_blocks.toLocaleString();
    document.getElementById('m_impl_faults').textContent =
      d.domain_points.fault_zones.x.length + ' pts';

    const dp = d.domain_points;
    const traces = [];
    if (dp.ore_zone.x.length) {
      traces.push({
        x: dp.ore_zone.x, y: dp.ore_zone.y, z: dp.ore_zone.z,
        mode: 'markers', type: 'scatter3d',
        marker: { size: 3, color: '#f59e0b', opacity: 0.4 },
        name: 'Ore Zone', hoverinfo: 'name',
      });
    }
    if (dp.waste_zone.x.length) {
      traces.push({
        x: dp.waste_zone.x, y: dp.waste_zone.y, z: dp.waste_zone.z,
        mode: 'markers', type: 'scatter3d',
        marker: { size: 2, color: '#475569', opacity: 0.15 },
        name: 'Waste', hoverinfo: 'name',
      });
    }
    if (dp.iso_surface.x.length) {
      traces.push({
        x: dp.iso_surface.x, y: dp.iso_surface.y, z: dp.iso_surface.z,
        mode: 'markers', type: 'scatter3d',
        marker: { size: 5, color: '#22d3ee', opacity: 0.9, symbol: 'diamond' },
        name: 'Ore Boundary (SDF=0)',
      });
    }
    if (dp.fault_zones.x.length) {
      traces.push({
        x: dp.fault_zones.x, y: dp.fault_zones.y, z: dp.fault_zones.z,
        mode: 'markers', type: 'scatter3d',
        marker: { size: 4, color: '#f43f5e', opacity: 0.8, symbol: 'x' },
        name: 'Fault Zones',
      });
    }
    Plotly.newPlot('chart-implicit-3d', traces, {
      ...plotlyDarkLayout,
      scene: { ...plotly3dScene },
      title: { text: '3D Implicit Geological Model — RBF Signed Distance Field', font: { size: 14 } },
      height: 560,
      legend: { x: 0.02, y: 0.98 },
    }, plotlyConfig);

  } catch(e) {
    document.getElementById('impl-status').textContent = 'Error: ' + e.message;
    document.getElementById('impl-status').className = 'status-msg error';
  }
  hideTabLoading('tab-implicit');
}

// ===== VENTILATION NETWORK =====
async function runVentilation() {
  showTabLoading('tab-ventilation', 'Designing ventilation circuit (Hardy Cross)…');
  try {
    const d = await apiCall('/api/ventilation', {
      diesel_kw: +document.getElementById('p_diesel_kw').value,
      ambient_temp: +document.getElementById('p_ambient_temp').value,
    });

    const comply = d.regulatory_check.compliant;
    document.getElementById('vent-status').textContent =
      `✅ Ventilation designed · ${d.total_airflow_required_m3s} m³/s · Fan: ${d.main_fan_kw} kW · WBT: ${d.wbt_at_depth}°C`;
    document.getElementById('vent-status').className = 'status-msg ' + (comply ? 'success' : 'warn');

    document.getElementById('vent-metrics').style.display = 'flex';
    document.getElementById('m_vent_q').textContent = d.total_airflow_required_m3s + ' m³/s';
    document.getElementById('m_vent_fan').textContent = d.main_fan_kw + ' kW';
    document.getElementById('m_vent_wbt').textContent = d.wbt_at_depth + '°C';
    const compEl = document.getElementById('m_vent_comply');
    compEl.textContent = comply ? '✅ Compliant' : '⚠ Review';
    compEl.style.color = comply ? '#10b981' : '#f59e0b';

    // Circuit airflow bar chart
    const circ = d.circuit;
    Plotly.newPlot('chart-vent-circuit', [{
      x: circ.map(b => `${b.type} #${b.id}`),
      y: circ.map(b => b.flow_m3s),
      type: 'bar', marker: { color: circ.map(b => b.dp_pa > 500 ? '#f43f5e' : '#22d3ee') },
      text: circ.map(b => `${b.flow_m3s} m³/s · ${b.dp_pa} Pa`), hoverinfo: 'text',
      name: 'Airflow (m³/s)',
    }], { ...plotlyDarkLayout,
      title: { text: 'Ventilation Circuit — Airflow per Branch', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Branch' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Flow (m³/s)' },
      height: 420,
    }, plotlyConfig);

    // Level-by-level compliance
    const lvl = d.level_summary;
    Plotly.newPlot('chart-vent-levels', [
      { x: lvl.map(l => l.elevation), y: lvl.map(l => l.required_flow_m3s),
        type: 'bar', marker: { color: lvl.map(l => l.compliant ? '#10b981' : '#f43f5e') },
        name: 'Required Flow (m³/s)',
        text: lvl.map(l => `${l.n_faces} faces · ${l.compliant ? '✅' : '⚠'}`), hoverinfo: 'x+y+text',
      },
    ], { ...plotlyDarkLayout,
      title: { text: 'Airflow Requirements by Level', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Elevation (m)' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Required Airflow (m³/s)' },
      height: 420,
    }, plotlyConfig);

    // Booster fan info
    if (d.booster_fans.length > 0) {
      const bInfo = document.getElementById('vent-booster-info');
      bInfo.innerHTML = `<strong>Booster Fans Required:</strong> ${d.n_booster_fans} location(s) — ` +
        d.booster_fans.map(b => `Segment ${b.segment} (${b.type}): ${b.dp_pa} Pa → ${b.recommended_kw} kW`).join(' | ');
    }

    // Regulatory summary
    const reg = d.regulatory_check;
    const regDiv = document.createElement('div');
    regDiv.style.cssText = 'margin-top:0.75rem;padding:0.75rem;background:var(--surface);border:1px solid var(--border);border-radius:6px;font-size:0.82rem';
    regDiv.innerHTML = `<strong>Regulatory Compliance:</strong><br>
      Min airflow: ≥${reg.min_airflow_per_kw} m³/s/kW · Actual: ${reg.actual_airflow_per_kw} m³/s/kW<br>
      Max WBT: ≤${reg.max_wbt_c}°C · Actual: ${reg.actual_wbt_c}°C<br>
      Status: <span style="color:${reg.compliant?'#10b981':'#f59e0b'};font-weight:600">${reg.compliant?'✅ COMPLIANT':'⚠ NEEDS REVIEW'}</span>`;
    document.getElementById('vent-booster-info').appendChild(regDiv);

  } catch(e) {
    document.getElementById('vent-status').textContent = 'Error: ' + e.message;
    document.getElementById('vent-status').className = 'status-msg error';
  }
  hideTabLoading('tab-ventilation');
}

// ===== BACKFILL OPTIMIZATION =====
async function runBackfill() {
  showTabLoading('tab-backfill', 'Optimizing backfill schedule (network flow)…');
  try {
    const d = await apiCall('/api/backfill', {
      fill_type: document.getElementById('p_fill_type').value,
      batch_capacity: +document.getElementById('p_batch_cap').value,
    });

    document.getElementById('fill-status').textContent =
      `✅ ${d.n_primary} primary · ${d.n_secondary} secondary · fill vol: ${fmtNum(Math.round(d.total_fill_volume_m3))} m³ · net: $${fmtNum(Math.round(d.net_value))}`;
    document.getElementById('fill-status').className = 'status-msg success';

    document.getElementById('fill-metrics').style.display = 'flex';
    document.getElementById('m_fill_primary').textContent = d.n_primary;
    document.getElementById('m_fill_secondary').textContent = d.n_secondary;
    document.getElementById('m_fill_vol').textContent = fmtNum(Math.round(d.total_fill_volume_m3));
    document.getElementById('m_fill_cost').textContent = '$' + fmtNum(Math.round(d.total_fill_cost));

    // Per-period schedule bar chart
    const ps = d.period_summary;
    Plotly.newPlot('chart-fill-schedule', [
      { x: ps.map(p => `P${p.period}`), y: ps.map(p => p.mine_value/1e6),
        type: 'bar', name: 'Mine Value (M$)', marker: { color: '#10b981' } },
      { x: ps.map(p => `P${p.period}`), y: ps.map(p => -p.fill_cost/1e6),
        type: 'bar', name: 'Fill Cost (M$)', marker: { color: '#f43f5e' } },
      { x: ps.map(p => `P${p.period}`), y: ps.map(p => p.net/1e6),
        type: 'scatter', mode: 'lines+markers', name: 'Net (M$)',
        line: { color: '#f59e0b', width: 2 }, marker: { size: 7 }, yaxis: 'y' },
    ], { ...plotlyDarkLayout,
      barmode: 'relative',
      title: { text: 'Backfill Schedule — Value vs Fill Cost per Period', font: { size: 13 } },
      xaxis: { ...plotlyDarkLayout.xaxis, title: 'Mining Period' },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Value (M$)' },
      height: 420,
    }, plotlyConfig);

    // Dependency network (scatter3d)
    const gr = d.dependency_graph;
    const primNodes = gr.nodes.filter(n => n.type === 'primary');
    const secNodes = gr.nodes.filter(n => n.type === 'secondary');
    const traces3d = [
      { x: primNodes.map(n=>n.x), y: primNodes.map(n=>n.y), z: primNodes.map(n=>n.z),
        mode: 'markers', type: 'scatter3d',
        marker: { size: 6, color: '#f59e0b', opacity: 0.9 },
        text: primNodes.map(n=>`Primary ${n.id} $${fmtNum(Math.round(n.value))}`), hoverinfo:'text',
        name: 'Primary Stopes',
      },
      { x: secNodes.map(n=>n.x), y: secNodes.map(n=>n.y), z: secNodes.map(n=>n.z),
        mode: 'markers', type: 'scatter3d',
        marker: { size: 6, color: '#22d3ee', opacity: 0.9 },
        text: secNodes.map(n=>`Secondary ${n.id} $${fmtNum(Math.round(n.value))}`), hoverinfo:'text',
        name: 'Secondary Stopes',
      },
    ];
    // Add dependency edges as line traces
    gr.edges.slice(0, 30).forEach(e => {
      const f = gr.nodes.find(n => n.id === e.from);
      const t = gr.nodes.find(n => n.id === e.to);
      if (f && t) {
        traces3d.push({
          x: [f.x, t.x], y: [f.y, t.y], z: [f.z, t.z],
          mode: 'lines', type: 'scatter3d',
          line: { color: '#475569', width: 1.5 }, showlegend: false,
        });
      }
    });
    Plotly.newPlot('chart-fill-network', traces3d, {
      ...plotlyDarkLayout,
      scene: { ...plotly3dScene },
      title: { text: 'Primary/Secondary Stope Dependency Network', font: { size: 13 } },
      height: 420,
    }, plotlyConfig);

    // Economics waterfall: total value → fill cost → net
    Plotly.newPlot('chart-fill-econ', [{
      type: 'waterfall', orientation: 'v',
      x: ['Total Stope Value', 'Fill Cost', 'Net Value'],
      y: [d.total_stope_value/1e6, -d.total_fill_cost/1e6, d.net_value/1e6],
      measure: ['absolute', 'relative', 'total'],
      connector: { line: { color: '#475569' } },
      increasing: { marker: { color: '#10b981' } },
      decreasing: { marker: { color: '#f43f5e' } },
      totals: { marker: { color: '#22d3ee' } },
      text: [`$${fmtNum(Math.round(d.total_stope_value))}`,
             `-$${fmtNum(Math.round(d.total_fill_cost))}`,
             `$${fmtNum(Math.round(d.net_value))}`],
      textposition: 'outside',
    }], { ...plotlyDarkLayout,
      title: { text: 'Backfill Economics Waterfall', font: { size: 13 } },
      yaxis: { ...plotlyDarkLayout.yaxis, title: 'Value (M$)' },
      height: 360,
    }, plotlyConfig);

  } catch(e) {
    document.getElementById('fill-status').textContent = 'Error: ' + e.message;
    document.getElementById('fill-status').className = 'status-msg error';
  }
  hideTabLoading('tab-backfill');
}

function fmtNum(n) {
  if (Math.abs(n) >= 1e6) return (n/1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n/1e3).toFixed(0) + 'K';
  return n.toString();
}
