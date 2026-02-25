"""
dashboard.py - Precision Horticulture Orchard Dashboard

Single-page FastAPI + Jinja2 dashboard displaying:
  - Bar chart of fruit count per wire zone (Chart.js)
  - Predicted yield (tons/ha) and estimated leaf year
  - 3D scatter plot of fruit-labeled point cloud colored by zone (Plotly.js)

Serves on http://localhost:8000

Usage:
    pip install fastapi uvicorn jinja2
    python dashboard.py
"""

import csv
import json
import os
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:
    raise ImportError("Run: pip install fastapi uvicorn jinja2")

ZONE_FRUIT_CSV = "outputs/zone_fruit_counts.csv"
YIELD_JSON = "outputs/yield_prediction.json"
FRUIT_LABELED_CSV = "outputs/fruit_labeled.csv"

ZONE_COLORS = [
    "#6c757d",  # Zone 0 ground - grey
    "#28a745",  # Zone 1 low wire - green
    "#ffc107",  # Zone 2 mid wire - amber
    "#fd7e14",  # Zone 3 high wire - orange
    "#dc3545",  # Zone 4 top wire - red
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MT-MVSNet Orchard Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e0e0e0; }
  header {
    background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
    padding: 20px 32px; border-bottom: 2px solid #38a169;
    display: flex; align-items: center; gap: 16px;
  }
  header h1 { font-size: 1.5rem; color: #68d391; }
  header p { font-size: 0.85rem; color: #a0aec0; margin-top: 2px; }
  .badge { background: #38a169; color: #fff; padding: 3px 10px;
           border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .kpi-card {
    background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
    padding: 20px; text-align: center;
  }
  .kpi-card .value { font-size: 2rem; font-weight: 700; color: #68d391; }
  .kpi-card .label { font-size: 0.8rem; color: #a0aec0; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi-card .sub { font-size: 0.75rem; color: #718096; margin-top: 4px; }
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  .chart-card {
    background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px; padding: 20px;
  }
  .chart-card h3 { font-size: 0.95rem; color: #a0aec0; margin-bottom: 16px;
                   text-transform: uppercase; letter-spacing: 0.05em; }
  .full-width { grid-column: 1 / -1; }
  .chart-wrapper { position: relative; height: 300px; }
  #scatter3d { height: 500px; }
  .zone-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .zone-table th { background: #2d3748; padding: 10px 14px; text-align: left;
                   color: #a0aec0; font-weight: 600; text-transform: uppercase;
                   font-size: 0.75rem; letter-spacing: 0.05em; }
  .zone-table td { padding: 10px 14px; border-bottom: 1px solid #2d3748; }
  .zone-table tr:last-child td { border-bottom: none; }
  .zone-dot { display: inline-block; width: 10px; height: 10px;
              border-radius: 50%; margin-right: 8px; vertical-align: middle; }
  .progress-bar-bg { background: #2d3748; border-radius: 4px; height: 8px; }
  .progress-bar-fill { height: 8px; border-radius: 4px; }
  footer { text-align: center; padding: 20px; color: #4a5568; font-size: 0.8rem;
           border-top: 1px solid #2d3748; }
</style>
</head>
<body>
<header>
  <div>
    <h1>MT-MVSNet Orchard Dashboard</h1>
    <p>Precision horticulture · Guyot multi-leader system · Fruit 3D reconstruction</p>
  </div>
  <span class="badge">LIVE</span>
</header>

<div class="container">

  <!-- KPI Cards -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="value" id="kpi-fruits">—</div>
      <div class="label">Fruits Detected</div>
      <div class="sub">from 3D point cloud</div>
    </div>
    <div class="kpi-card">
      <div class="value" id="kpi-yield">—</div>
      <div class="label">Predicted Yield</div>
      <div class="sub">tons / hectare</div>
    </div>
    <div class="kpi-card">
      <div class="value" id="kpi-year">—</div>
      <div class="label">Estimated Season</div>
      <div class="sub">leaf year (Pink Lady ref)</div>
    </div>
    <div class="kpi-card">
      <div class="value" id="kpi-r2">—</div>
      <div class="label">Model R²</div>
      <div class="sub">season regression fit</div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts-grid">

    <div class="chart-card">
      <h3>Fruit Count per Wire Zone</h3>
      <div class="chart-wrapper">
        <canvas id="zoneBarChart"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <h3>Zone Distribution</h3>
      <table class="zone-table">
        <thead>
          <tr><th>Zone</th><th>Height</th><th>Fruits</th><th>Share</th></tr>
        </thead>
        <tbody id="zone-table-body"></tbody>
      </table>
    </div>

    <div class="chart-card full-width">
      <h3>3D Fruit Point Cloud — coloured by wire zone</h3>
      <div id="scatter3d"></div>
    </div>

  </div>
</div>

<footer>MT-MVSNet · Depth reconstruction + fruit segmentation · outputs auto-loaded</footer>

<script>
const ZONE_COLORS = {{ zone_colors | safe }};
const zoneData = {{ zone_data | safe }};
const yieldData = {{ yield_data | safe }};
const cloudData = {{ cloud_data | safe }};

// --- KPI Cards ---
if (yieldData) {
  document.getElementById('kpi-fruits').textContent =
    yieldData.inputs.total_fruits_detected.toLocaleString();
  document.getElementById('kpi-yield').textContent =
    yieldData.yield.predicted_tons_ha.toFixed(1);
  document.getElementById('kpi-year').textContent =
    yieldData.season_estimation.estimated_year;
  document.getElementById('kpi-r2').textContent =
    yieldData.season_estimation.r_squared.toFixed(3);
}

// --- Bar Chart ---
if (zoneData && zoneData.length > 0) {
  const ctx = document.getElementById('zoneBarChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: zoneData.map(z => z.zone_label),
      datasets: [{
        label: 'Fruit Count',
        data: zoneData.map(z => z.fruit_count),
        backgroundColor: ZONE_COLORS,
        borderColor: ZONE_COLORS,
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.raw.toLocaleString()} fruits`
          }
        }
      },
      scales: {
        x: { ticks: { color: '#a0aec0', font: { size: 11 } },
             grid: { color: '#2d3748' } },
        y: { ticks: { color: '#a0aec0' },
             grid: { color: '#2d3748' },
             title: { display: true, text: 'Fruit Count', color: '#718096' } }
      }
    }
  });

  // --- Zone Table ---
  const total = zoneData.reduce((s, z) => s + z.fruit_count, 0);
  const tbody = document.getElementById('zone-table-body');
  zoneData.forEach((z, i) => {
    const pct = total > 0 ? (z.fruit_count / total * 100).toFixed(1) : '0.0';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><span class="zone-dot" style="background:${ZONE_COLORS[i]}"></span>Zone ${z.zone_id}</td>
      <td style="color:#a0aec0">${z.z_min_m}–${z.z_max_m}m</td>
      <td><strong>${z.fruit_count.toLocaleString()}</strong></td>
      <td style="min-width:120px">
        <div style="display:flex;align-items:center;gap:8px">
          <div class="progress-bar-bg" style="flex:1">
            <div class="progress-bar-fill" style="width:${pct}%;background:${ZONE_COLORS[i]}"></div>
          </div>
          <span style="font-size:0.8rem;color:#a0aec0;width:38px">${pct}%</span>
        </div>
      </td>`;
    tbody.appendChild(row);
  });
}

// --- 3D Scatter Plot ---
if (cloudData && cloudData.x && cloudData.x.length > 0) {
  const traces = [];
  const zoneIds = [...new Set(cloudData.zone)];
  const zoneNames = {{ zone_label_map | safe }};

  zoneIds.forEach(zid => {
    const mask = cloudData.zone.map((z, i) => z === zid ? i : -1).filter(i => i >= 0);
    traces.push({
      type: 'scatter3d',
      mode: 'markers',
      name: zoneNames[zid] || `Zone ${zid}`,
      x: mask.map(i => cloudData.x[i]),
      y: mask.map(i => cloudData.y[i]),
      z: mask.map(i => cloudData.z[i]),
      marker: {
        size: 2,
        color: ZONE_COLORS[zid] || '#888',
        opacity: 0.7
      }
    });
  });

  Plotly.newPlot('scatter3d', traces, {
    paper_bgcolor: '#1a1f2e',
    plot_bgcolor: '#1a1f2e',
    font: { color: '#a0aec0' },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      xaxis: { title: 'X (m)', gridcolor: '#2d3748', zerolinecolor: '#4a5568' },
      yaxis: { title: 'Y (m)', gridcolor: '#2d3748', zerolinecolor: '#4a5568' },
      zaxis: { title: 'Z height (m)', gridcolor: '#2d3748', zerolinecolor: '#4a5568' },
      bgcolor: '#1a1f2e'
    },
    legend: { bgcolor: '#1a1f2e', bordercolor: '#2d3748', font: { color: '#a0aec0' } }
  }, { responsive: true });
} else {
  document.getElementById('scatter3d').innerHTML =
    '<p style="color:#4a5568;padding:40px;text-align:center">No point cloud data found.<br>Run inference_combined.py first.</p>';
}
</script>
</body>
</html>"""


app = FastAPI(title="MT-MVSNet Orchard Dashboard")


def load_zone_data():
    if not os.path.exists(ZONE_FRUIT_CSV):
        return []
    zones = []
    with open(ZONE_FRUIT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zones.append({
                "zone_id": int(row["zone_id"]),
                "zone_label": row["zone_label"],
                "z_min_m": float(row["z_min_m"]),
                "z_max_m": float(row["z_max_m"]),
                "fruit_count": int(row["fruit_count"]),
            })
    return zones


def load_yield_data():
    if not os.path.exists(YIELD_JSON):
        return None
    with open(YIELD_JSON) as f:
        return json.load(f)


def load_cloud_data(max_points: int = 20_000):
    """Load fruit-labeled point cloud, downsample if large, assign zone ids."""
    if not os.path.exists(FRUIT_LABELED_CSV):
        return {"x": [], "y": [], "z": [], "zone": []}

    ZONE_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    rows = []
    with open(FRUIT_LABELED_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                is_fruit = int(float(row.get("is_fruit", 0)))
                if not is_fruit:
                    continue
                rows.append((
                    float(row["x"]), float(row["y"]), float(row["z"])
                ))
            except (KeyError, ValueError):
                continue

    # Uniform downsample
    if len(rows) > max_points:
        step = len(rows) // max_points
        rows = rows[::step]

    xs, ys, zs, zones = [], [], [], []
    for x, y, z in rows:
        xs.append(round(x, 4))
        ys.append(round(y, 4))
        zs.append(round(z, 4))
        zone_id = 4  # default top
        for i in range(5):
            if ZONE_EDGES[i] <= z < ZONE_EDGES[i + 1]:
                zone_id = i
                break
        zones.append(zone_id)

    return {"x": xs, "y": ys, "z": zs, "zone": zones}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    zone_data = load_zone_data()
    yield_data = load_yield_data()
    cloud_data = load_cloud_data()

    from zone_classifier import ZONE_LABELS
    zone_label_map = {i: ZONE_LABELS[i] for i in range(5)}

    # Fill missing yield data with placeholders
    if yield_data is None:
        yield_data = {
            "inputs": {"total_fruits_detected": 0, "trees_per_ha": 2500,
                       "avg_fruit_weight_g": 185},
            "yield": {"predicted_tons_ha": 0.0, "formula": ""},
            "season_estimation": {"estimated_year": "N/A", "r_squared": 0.0},
            "zone_breakdown": [],
        }

    html = HTML_TEMPLATE.replace(
        "{{ zone_colors | safe }}", json.dumps(ZONE_COLORS)
    ).replace(
        "{{ zone_data | safe }}", json.dumps(zone_data)
    ).replace(
        "{{ yield_data | safe }}", json.dumps(yield_data)
    ).replace(
        "{{ cloud_data | safe }}", json.dumps(cloud_data)
    ).replace(
        "{{ zone_label_map | safe }}", json.dumps(zone_label_map)
    )

    return HTMLResponse(content=html)


@app.get("/api/zones")
async def api_zones():
    return load_zone_data()


@app.get("/api/yield")
async def api_yield():
    return load_yield_data() or {}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
