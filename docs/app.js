/* Coffee Rust Forecaster dashboard: renders docs/rust_risk.json with Chart.js. */
"use strict";

const DATA_URL = "rust_risk.json";
const STALE_AFTER_DAYS = 3;
const LEVEL_STATUS = { Low: "good", Moderate: "warning", High: "serious", Severe: "critical" };

// Trusted constant markup only (never data): status icons drawn in currentColor.
const ICONS = {
  good: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-6.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  warning: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M8 4.6v4.2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="11.2" r="1" fill="currentColor"/></svg>',
  serious: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2l6.5 11.5h-13z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M8 6.3v3.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="11.8" r="0.95" fill="currentColor"/></svg>',
  critical: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5.5 1.8h5l3.7 3.7v5l-3.7 3.7h-5l-3.7-3.7v-5z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M8 4.8v4.2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="11.3" r="1" fill="currentColor"/></svg>',
};

const $ = (id) => document.getElementById(id);
const state = { data: null, charts: [], hover: null, hourlyCharts: [] };

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function tokens() {
  return {
    surface: css("--surface-1"), text: css("--text-primary"), text2: css("--text-secondary"),
    muted: css("--text-muted"), grid: css("--grid"), axis: css("--axis"),
    series: css("--series-1"), wash: css("--wash"),
    risk: [css("--risk-low"), css("--risk-moderate"), css("--risk-high"), css("--risk-severe")],
    status: {
      good: css("--status-good"), warning: css("--status-warning"),
      serious: css("--status-serious"), critical: css("--status-critical"),
    },
  };
}
const statusOf = (level) => LEVEL_STATUS[level] || "good";
const pct = (p) => `${Math.round(p * 100)}%`;
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// Dates arrive as local (America/Sao_Paulo) strings; build Dates from components so the
// viewer's own timezone never shifts them.
function parseLocal(s) {
  const [d, t = "00:00"] = s.split("T");
  const [y, m, day] = d.split("-").map(Number);
  const [hh, mm] = t.split(":").map(Number);
  return new Date(y, m - 1, day, hh, mm);
}
const fmtNight = (s) => parseLocal(s).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
const fmtDayShort = (s) => parseLocal(s).toLocaleDateString("en-GB", { weekday: "short", day: "numeric" });
const fmtHour = (s) => {
  const d = parseLocal(s);
  return `${d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" })}, ${s.slice(11, 16)}`;
};

function statusInline(level) {
  const wrap = el("span", "status-inline");
  const dot = el("span", "dot");
  dot.style.background = `var(--status-${statusOf(level)})`;
  wrap.append(dot, el("span", null, level));
  return wrap;
}

const tip = () => $("tooltip");
function showTooltip(title, rows, x, y) {
  const t = tip();
  t.replaceChildren(el("div", "tt-title", title));
  for (const r of rows) {
    const row = el("div", "tt-row");
    const lbl = el("span", "lbl");
    if (r.key) {
      const k = el("span", "line-key");
      k.style.background = r.key;
      lbl.append(k);
    }
    lbl.append(document.createTextNode(r.label));
    row.append(lbl, el("span", "val", r.value));
    t.append(row);
  }
  t.hidden = false;
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let left = x + pad, top = y + pad;
  if (left + w > window.innerWidth - 8) left = x - w - pad;
  if (top + h > window.innerHeight - 8) top = y - h - pad;
  t.style.left = `${Math.max(8, left)}px`;
  t.style.top = `${Math.max(8, top)}px`;
}
const hideTooltip = () => { tip().hidden = true; };

function renderOverview(d) {
  $("site-name").textContent = `${d.site.name}, ${d.site.state}`;
  const gen = new Date(d.generated_at);
  $("updated").textContent = `Issued ${fmtNight(d.issue_date)} · updated ${gen.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })}`;

  const ageDays = (Date.now() - gen.getTime()) / 86400000;
  if (ageDays > STALE_AFTER_DAYS) {
    $("stale").hidden = false;
    $("stale").textContent = `This forecast is ${Math.floor(ageDays)} days old. The scheduled update may have failed.`;
  }

  const a = d.alert, status = statusOf(a.level);
  const pill = $("alert-pill");
  pill.innerHTML = ICONS[status];
  pill.append(document.createTextNode(a.level));
  pill.style.background = `var(--status-${status})`;
  pill.style.color = "var(--on-status)";
  $("hero-value").textContent = pct(a.max_probability);
  $("hero-caption").textContent = `peak infection probability, night of ${fmtNight(a.peak_night)}`;

  const tonight = d.days[0];
  $("tonight-value").textContent = pct(tonight.probability);
  const cap = $("tonight-caption");
  cap.replaceChildren(statusInline(tonight.level),
    document.createTextNode(` · ${tonight.longest_wet_run_hours} h wet run forecast`));

  $("high-value").textContent = `${a.nights_high_or_above} of ${d.days.length}`;

  const recent = d.recent_nights;
  const n = recent.filter((r) => r.event).length;
  $("recent-value").textContent = `${n} of ${recent.length}`;
  const strip = $("recent-strip");
  strip.replaceChildren(...recent.map((r) => {
    const s = el("span", r.event ? "on" : "");
    const label = `${fmtNight(r.night)}: ${r.event ? "favourable" : "not favourable"}`;
    s.setAttribute("role", "listitem");
    s.setAttribute("aria-label", label);
    s.title = label;
    return s;
  }));
}

// Gradient stops sit at the middle of each alert band (Low 0-25%, Moderate, High, Severe),
// so a bar's colour at any height matches the risk level of that probability.
const RISK_STOPS = [0.125, 0.375, 0.625, 0.875];

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? [...h].map((c) => c + c).join("") : h;
  const n = parseInt(full, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function riskColorAt(p, colors, alpha = 1) {
  const rgb = colors.map(hexToRgb);
  const last = RISK_STOPS.length - 1;
  let c;
  if (p <= RISK_STOPS[0]) c = rgb[0];
  else if (p >= RISK_STOPS[last]) c = rgb[last];
  else {
    let i = 0;
    while (p >= RISK_STOPS[i + 1]) i++;
    const f = (p - RISK_STOPS[i]) / (RISK_STOPS[i + 1] - RISK_STOPS[i]);
    c = rgb[i].map((v, k) => Math.round(v + (rgb[i + 1][k] - v) * f));
  }
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
}

function riskGradient(chart, colors) {
  const { ctx, chartArea, scales: { y } } = chart;
  if (!chartArea || !y) return colors[0];
  const g = ctx.createLinearGradient(0, y.getPixelForValue(0), 0, y.getPixelForValue(100));
  colors.forEach((c, i) => g.addColorStop(RISK_STOPS[i], c));
  return g;
}

// Draws each bar with a soft glow in the colour at its top. Skipped when printing: a blurred
// shadow makes Chrome rasterize the canvas onto an opaque white box in the PDF.
const printing = window.matchMedia("print");
const barGlow = {
  id: "barGlow",
  beforeDatasetDraw(chart, args, opts) {
    if (printing.matches) return;
    const { ctx } = chart;
    const values = chart.data.datasets[args.index].data;
    args.meta.data.forEach((bar, i) => {
      ctx.save();
      ctx.shadowColor = riskColorAt(values[i] / 100, opts.colors, 0.45);
      ctx.shadowBlur = opts.blur;
      bar.draw(ctx);
      ctx.restore();
    });
    return false;
  },
};

const capLabels = {
  id: "capLabels",
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx, chartArea: area, scales: { y } } = chart;
    const meta = chart.getDatasetMeta(0);
    ctx.save();
    ctx.font = `600 12px ${Chart.defaults.font.family}`;
    ctx.fillStyle = opts.color;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    meta.data.forEach((bar, i) => {
      ctx.fillText(`${Math.round(chart.data.datasets[0].data[i])}%`, bar.x, bar.y - 6);
    });
    ctx.font = `11px ${Chart.defaults.font.family}`;
    ctx.fillStyle = opts.muted;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    opts.levels.forEach((lv, i) => {
      const lo = lv.min * 100, hi = opts.levels[i + 1] ? opts.levels[i + 1].min * 100 : 100;
      ctx.fillText(lv.name, area.right + 8, y.getPixelForValue((lo + hi) / 2));
    });
    ctx.restore();
  },
};

function renderRiskChart(d, tk) {
  const days = d.days;
  const chart = new Chart($("risk-chart"), {
    type: "bar",
    data: {
      labels: days.map((x) => [x.lead === 1 ? `Tonight` : fmtDayShort(x.night), x.level]),
      datasets: [{
        data: days.map((x) => x.probability * 100),
        backgroundColor: (c) => riskGradient(c.chart, tk.risk),
        hoverBackgroundColor: (c) => riskGradient(c.chart, tk.risk),
        borderRadius: 6,
        borderSkipped: "start",
        maxBarThickness: 24,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      layout: { padding: { top: 22, right: 72 } },
      interaction: { mode: "index", intersect: false },
      scales: {
        y: {
          min: 0, max: 100,
          ticks: { stepSize: 25, color: tk.muted, callback: (v) => `${v}%` },
          grid: { color: tk.grid, lineWidth: 1 },
          border: { display: false },
        },
        x: {
          grid: { display: false },
          border: { color: tk.axis },
          ticks: { color: tk.text2, maxRotation: 0 },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
        capLabels: { color: tk.text, muted: tk.muted, levels: d.levels },
        barGlow: { colors: tk.risk, blur: 14 },
      },
      onHover(evt, _els, ch) {
        const hits = ch.getElementsAtEventForMode(evt, "index", { intersect: false }, false);
        if (!hits.length) return hideTooltip();
        const x = days[hits[0].index];
        showTooltip(`Night of ${fmtNight(x.night)}`, [
          { label: "Infection probability", value: pct(x.probability) },
          { label: "Alert level", value: x.level },
          { label: "Rule on raw forecast", value: x.rule_event ? "Infection" : "No infection" },
          { label: "Longest wet run", value: `${x.longest_wet_run_hours} h` },
          { label: "Temperature", value: `${x.t_min}–${x.t_max} °C` },
          { label: "Rain", value: `${x.rain_mm} mm` },
        ], evt.native.clientX, evt.native.clientY);
      },
    },
    plugins: [barGlow, capLabels],
  });
  chart.canvas.addEventListener("pointerleave", hideTooltip);
  return chart;
}

const hourlyLayers = {
  id: "hourlyLayers",
  beforeDatasetsDraw(chart, _a, o) {
    const { ctx, chartArea: area, scales: { x } } = chart;
    const step = x.getPixelForValue(1) - x.getPixelForValue(0);
    ctx.save();
    ctx.fillStyle = o.wash;
    let i = 0;
    while (i < o.fav.length) {
      if (!o.fav[i]) { i++; continue; }
      let j = i;
      while (j + 1 < o.fav.length && o.fav[j + 1]) j++;
      const x0 = Math.max(area.left, x.getPixelForValue(i) - step / 2);
      const x1 = Math.min(area.right, x.getPixelForValue(j) + step / 2);
      ctx.fillRect(x0, area.top, x1 - x0, area.bottom - area.top);
      i = j + 1;
    }
    for (const t of o.thresholds || []) {
      const y = chart.scales.y.getPixelForValue(t.y);
      ctx.strokeStyle = o.muted;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(area.left, y); ctx.lineTo(area.right, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = o.muted;
      ctx.font = `11px ${Chart.defaults.font.family}`;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(t.label, area.right + 6, y);
    }
    ctx.restore();
  },
  afterDatasetsDraw(chart, _a, o) {
    const { ctx, chartArea: area, scales: { x } } = chart;
    ctx.save();
    if (o.boundary >= 0 && o.boundary < o.fav.length - 1) {
      const bx = (x.getPixelForValue(o.boundary) + x.getPixelForValue(o.boundary + 1)) / 2;
      ctx.strokeStyle = o.text2;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(bx, area.top); ctx.lineTo(bx, area.bottom); ctx.stroke();
      if (o.labelBoundary) {
        ctx.fillStyle = o.text2;
        ctx.font = `11px ${Chart.defaults.font.family}`;
        ctx.textBaseline = "top";
        ctx.textAlign = "left";
        ctx.fillText("Forecast →", bx + 4, area.top + 2);
        ctx.textAlign = "right";
        ctx.fillText("← Recent", bx - 4, area.top + 2);
      }
    }
    if (state.hover != null) {
      const hx = x.getPixelForValue(state.hover);
      ctx.strokeStyle = o.text2;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(hx, area.top); ctx.lineTo(hx, area.bottom); ctx.stroke();
      const ds = chart.data.datasets[0];
      if (chart.config.type === "line") {
        const v = ds.data[state.hover];
        if (v != null) {
          const hy = chart.scales.y.getPixelForValue(v);
          ctx.beginPath(); ctx.arc(hx, hy, 6, 0, Math.PI * 2); ctx.fillStyle = o.surface; ctx.fill();
          ctx.beginPath(); ctx.arc(hx, hy, 4, 0, Math.PI * 2); ctx.fillStyle = o.series; ctx.fill();
        }
      }
    }
    ctx.restore();
  },
};

function hourlyChart(canvas, type, values, h, tk, { yOpts, thresholds, showX, labelBoundary }) {
  const isMidnight = (i) => h.time[i] && h.time[i].endsWith("T00:00");
  const dataset = type === "line"
    ? { data: values, borderColor: tk.series, borderWidth: 2, pointRadius: 0, tension: 0.25,
        borderCapStyle: "round", borderJoinStyle: "round" }
    : { data: values, backgroundColor: tk.series, borderRadius: 2, borderSkipped: "start",
        barPercentage: 1, categoryPercentage: 0.75 };
  return new Chart(canvas, {
    type,
    data: { labels: h.time, datasets: [dataset] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      events: [],
      layout: { padding: { right: 44 } },
      scales: {
        x: {
          border: { color: tk.axis },
          grid: { color: (c) => (isMidnight(c.index) ? tk.grid : "transparent"), drawTicks: false },
          ticks: {
            display: showX, autoSkip: false, maxRotation: 0, color: tk.muted, padding: 6,
            callback(_v, i) {
              if (!isMidnight(i)) return "";
              if (this.chart.width < 640 && parseLocal(h.time[i]).getDate() % 2) return "";
              return fmtDayShort(h.time[i]);
            },
          },
        },
        y: {
          ...yOpts,
          border: { display: false },
          grid: { color: tk.grid },
          ticks: { color: tk.muted, maxTicksLimit: 3, ...(yOpts.ticks || {}) },
          afterFit: (s) => { s.width = 44; },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
        hourlyLayers: {
          fav: h.favourable, wash: tk.wash, muted: tk.muted, text2: tk.text2,
          surface: tk.surface, series: tk.series, thresholds, labelBoundary,
          boundary: h.time.indexOf(h.observed_until),
        },
      },
    },
    plugins: [hourlyLayers],
  });
}

function hourlyTooltip(i, x, y) {
  const h = state.data.hourly, tk = tokens();
  const observed = i <= h.time.indexOf(h.observed_until);
  showTooltip(`${fmtHour(h.time[i])} · ${observed ? "recent" : "forecast"}`, [
    { label: "Temperature", value: `${h.temperature_2m[i]} °C`, key: tk.series },
    { label: "Humidity", value: `${h.relative_humidity_2m[i]}%`, key: tk.series },
    { label: "Rain", value: `${h.precipitation[i]} mm`, key: tk.series },
    { label: "Leaf wet", value: h.leaf_wet[i] ? "Yes" : "No" },
    { label: "Germination conditions", value: h.favourable[i] ? "Met" : "Not met" },
  ], x, y);
}

function setHover(i) {
  state.hover = i;
  state.hourlyCharts.forEach((c) => c.draw());
}

function renderHourly(d, tk) {
  const h = d.hourly;
  const temps = h.temperature_2m;
  const tLo = Math.floor(Math.min(10, ...temps) / 10) * 10;
  const tHi = Math.ceil(Math.max(30, ...temps) / 10) * 10;
  const charts = [
    hourlyChart($("temp-chart"), "line", temps, h, tk, {
      yOpts: { min: tLo, max: tHi, ticks: { stepSize: 10, maxTicksLimit: 6 } },
      thresholds: [{ y: 15, label: "15 °C" }, { y: 28, label: "28 °C" }],
      showX: false, labelBoundary: true,
    }),
    hourlyChart($("rh-chart"), "line", h.relative_humidity_2m, h, tk, {
      yOpts: { min: 0, max: 100, ticks: { stepSize: 50 } },
      thresholds: [{ y: 90, label: "90 %" }],
      showX: false,
    }),
    hourlyChart($("rain-chart"), "bar", h.precipitation, h, tk, {
      yOpts: { min: 0, suggestedMax: 2 },
      showX: true,
    }),
  ];
  state.hourlyCharts = charts;
  return charts;
}

function wireHourlyInteraction() {
  const box = $("wx-multiples");
  const n = () => state.data.hourly.time.length;
  box.addEventListener("pointermove", (e) => {
    const chart = state.hourlyCharts.find((c) => c.canvas === e.target);
    if (!chart) return;
    const rect = chart.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const { left, right } = chart.chartArea;
    if (px < left || px > right) { setHover(null); return hideTooltip(); }
    const i = Math.max(0, Math.min(n() - 1, Math.round(chart.scales.x.getValueForPixel(px))));
    setHover(i);
    hourlyTooltip(i, e.clientX, e.clientY);
  });
  box.addEventListener("pointerleave", () => { setHover(null); hideTooltip(); });
  box.addEventListener("blur", () => { setHover(null); hideTooltip(); });
  box.addEventListener("keydown", (e) => {
    const keys = { ArrowLeft: -1, ArrowRight: 1, PageUp: -24, PageDown: 24 };
    let i = state.hover ?? state.data.hourly.time.indexOf(state.data.hourly.observed_until);
    if (e.key in keys) i += keys[e.key];
    else if (e.key === "Home") i = 0;
    else if (e.key === "End") i = n() - 1;
    else return;
    e.preventDefault();
    i = Math.max(0, Math.min(n() - 1, i));
    setHover(i);
    const c = state.hourlyCharts[0];
    const r = c.canvas.getBoundingClientRect();
    hourlyTooltip(i, r.left + c.scales.x.getPixelForValue(i), r.top + c.chartArea.top);
  });
}

function fillTable(table, headers, rows) {
  const thead = el("thead"), tr = el("tr");
  headers.forEach(([label, num]) => tr.append(el("th", num ? "num" : null, label)));
  thead.append(tr);
  const tbody = el("tbody");
  for (const cells of rows) {
    const row = el("tr");
    cells.forEach((c, i) => {
      const td = el("td", headers[i][1] ? "num" : null);
      if (c instanceof Node) td.append(c); else td.textContent = c;
      row.append(td);
    });
    tbody.append(row);
  }
  table.replaceChildren(thead, tbody);
}

function renderTables(d) {
  fillTable($("night-table"),
    [["Night"], ["Probability", true], ["Level"], ["Rule on raw forecast"], ["Wet hours", true],
     ["Longest wet run", true], ["Temperature", true], ["Mean RH", true], ["Rain", true]],
    d.days.map((x) => [
      x.lead === 1 ? `${fmtNight(x.night)} (tonight)` : fmtNight(x.night),
      pct(x.probability), statusInline(x.level), x.rule_event ? "Infection" : "No infection",
      `${x.wet_hours} h`, `${x.longest_wet_run_hours} h`, `${x.t_min}–${x.t_max} °C`,
      `${x.rh_mean}%`, `${x.rain_mm} mm`,
    ]));
}

function renderAbout(d) {
  const m = d.model;
  $("skill").textContent = Object.entries(m.cv_auc_by_lead)
    .map(([k, v]) => `Day ${k} ${v.toFixed(2)}`).join(" · ");
  $("model-meta").textContent =
    `${m.type} on ${m.forecast_model} forecasts, trained on issue dates ${m.train_issue_range[0]} to ` +
    `${m.train_issue_range[1]}. Weather data: Open-Meteo. Site ${d.site.latitude}, ${d.site.longitude}.`;
}

function renderCharts() {
  state.charts.forEach((c) => c.destroy());
  const tk = tokens();
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.color = tk.text2;
  state.charts = [renderRiskChart(state.data, tk), ...renderHourly(state.data, tk)];
}

// Each backdrop layer travels its own extra height over the full scroll range, so the
// taller (near) layer moves faster than the far one and neither edge ever shows,
// whatever the page length or screen size.
function initParallax() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const layers = [...document.querySelectorAll(".backdrop .layer")];
  // Browsers already fire scroll events at most once per frame, so update directly.
  const update = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const progress = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
    for (const layer of layers) {
      const travel = Math.max(0, layer.offsetHeight - window.innerHeight);
      layer.style.transform = `translate3d(0, ${(-progress * travel).toFixed(1)}px, 0)`;
    }
  };
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

async function boot() {
  initParallax();
  try {
    const res = await fetch(DATA_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();
    renderOverview(state.data);
    renderCharts();
    renderTables(state.data);
    renderAbout(state.data);
    wireHourlyInteraction();
    // Canvases are sized for the screen; re-measure and redraw when printing starts and ends.
    printing.addEventListener("change", () => state.charts.forEach((c) => { c.resize(); c.draw(); }));
  } catch (err) {
    $("error").hidden = false;
    $("error").textContent = `Could not load the forecast (${err.message}).`;
  } finally {
    $("content").setAttribute("aria-busy", "false");
  }
}

document.addEventListener("DOMContentLoaded", boot);