/* =====================================================================
   Dashboard charts (Chart.js) — theme-aware
   Each chart is wrapped in try/catch so one failure never blocks the rest.
   ===================================================================== */
(function () {
  "use strict";
  if (typeof Chart === "undefined") {
    console.error("Chart.js failed to load — charts cannot render.");
    return;
  }
  if (!window.DASHBOARD_DATA) return;

  const data = window.DASHBOARD_DATA;
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  const styles = getComputedStyle(document.documentElement);
  const textColor = styles.getPropertyValue("--text-2").trim() || "#94a3b8";
  const labelColor = styles.getPropertyValue("--text-0").trim() || (isLight ? "#10163a" : "#f4f7fe");
  const gridColor = isLight ? "rgba(15,23,60,0.08)" : "rgba(255,255,255,0.06)";
  const blue = "#3b82f6", indigo = "#6366f1", violet = "#a855f7", cyan = "#22d3ee",
        green = "#22c55e", red = "#f43f5e", amber = "#f59e0b";
  const blueShades = ["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#2563eb", "#3b82f6", "#60a5fa"];

  Chart.defaults.color = textColor;
  Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.responsive = true;

  const commonGrid = { grid: { color: gridColor, drawTicks: false }, ticks: { color: textColor } };

  function safeChart(label, fn) {
    try { fn(); } catch (err) { console.error("Chart render failed (" + label + "):", err); }
  }

  /* Draws a "NN%" label at the end of every bar — works in both themes
     since it reads the current theme's text color at render time. */
  const percentLabelPlugin = {
    id: "percentLabels",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      const isHorizontal = chart.config.options.indexAxis === "y";
      chart.data.datasets.forEach((dataset, di) => {
        const meta = chart.getDatasetMeta(di);
        meta.data.forEach((bar, i) => {
          const value = dataset.data[i];
          if (value === undefined || value === null) return;
          ctx.save();
          ctx.fillStyle = labelColor;
          ctx.font = "700 11px 'Plus Jakarta Sans', sans-serif";
          ctx.textBaseline = "middle";
          if (isHorizontal) {
            ctx.textAlign = "left";
            ctx.fillText(value + "%", bar.x + 8, bar.y);
          } else {
            ctx.textAlign = "center";
            ctx.fillText(value + "%", bar.x, bar.y - 10);
          }
          ctx.restore();
        });
      });
    },
  };

  /* ---- Approval vs Rejection Doughnut ---- */
  safeChart("approval-split", () => {
    const c1 = document.getElementById("chart-approval-split");
    if (!c1) return;
    new Chart(c1, {
      type: "doughnut",
      data: {
        labels: ["Approved", "Declined"],
        datasets: [{ data: [data.approved || 0, data.rejected || 0], backgroundColor: [green, red], borderWidth: 0, hoverOffset: 8 }],
      },
      options: { cutout: "72%", plugins: { legend: { position: "bottom" } }, animation: { animateRotate: true, duration: 1200 } },
    });
  });

  /* ---- Income Distribution (bar histogram) ---- */
  safeChart("income-dist", () => {
    const c2 = document.getElementById("chart-income-dist");
    if (!c2 || !data.income_bins) return;
    new Chart(c2, {
      type: "bar",
      data: {
        labels: data.income_bins.labels,
        datasets: [{ label: "Applicants", data: data.income_bins.counts, backgroundColor: blue, borderRadius: 6, maxBarThickness: 36 }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { x: commonGrid, y: { ...commonGrid, beginAtZero: true } },
      },
    });
  });

  /* ---- Education Distribution (pie) ---- */
  safeChart("education-dist", () => {
    const c3 = document.getElementById("chart-education-dist");
    if (!c3 || !data.education) return;
    new Chart(c3, {
      type: "pie",
      data: {
        labels: data.education.labels,
        datasets: [{ data: data.education.counts, backgroundColor: [blue, indigo, violet, cyan, amber, "#10b981"], borderWidth: 0 }],
      },
      options: { plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } } },
    });
  });

  /* ---- Employment / Income type distribution ---- */
  safeChart("employment-dist", () => {
    const c4 = document.getElementById("chart-employment-dist");
    if (!c4 || !data.employment) return;
    new Chart(c4, {
      type: "bar",
      data: {
        labels: data.employment.labels,
        datasets: [{ label: "Applicants", data: data.employment.counts, backgroundColor: cyan, borderRadius: 6, maxBarThickness: 32 }],
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: { x: { ...commonGrid, beginAtZero: true }, y: commonGrid },
      },
    });
  });

  /* ---- Feature Importance (horizontal bar) ---- */
  safeChart("feature-importance", () => {
    const c5 = document.getElementById("chart-feature-importance");
    if (!c5 || !data.importance) return;
    new Chart(c5, {
      type: "bar",
      data: {
        labels: data.importance.labels,
        datasets: [{
          label: "Importance %",
          data: data.importance.values,
          backgroundColor: data.importance.labels.map((_, i) => blueShades[i % blueShades.length]),
          borderRadius: 6,
          maxBarThickness: 24,
        }],
      },
      plugins: [percentLabelPlugin],
      options: {
        indexAxis: "y",
        animation: { duration: 1400, easing: "easeOutQuart" },
        layout: { padding: { right: 36 } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => ctx.parsed.x + "%" } } },
        scales: {
          x: { ...commonGrid, beginAtZero: true, suggestedMax: Math.max(...data.importance.values) * 1.25 },
          y: commonGrid,
        },
      },
    });
  });

  /* ---- Model accuracy comparison ---- */
  safeChart("model-accuracy", () => {
    const c6 = document.getElementById("chart-model-accuracy");
    if (!c6 || !data.model_accuracy) return;
    new Chart(c6, {
      type: "bar",
      data: {
        labels: data.model_accuracy.labels,
        datasets: [{ label: "Accuracy %", data: data.model_accuracy.values, backgroundColor: [blue, indigo, violet, amber], borderRadius: 8, maxBarThickness: 46 }],
      },
      plugins: [percentLabelPlugin],
      options: {
        animation: { duration: 1400, easing: "easeOutQuart" },
        layout: { padding: { top: 24 } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => ctx.parsed.y + "%" } } },
        scales: { x: commonGrid, y: { ...commonGrid, beginAtZero: true, max: 100 } },
      },
    });
  });

  /* ---- Prediction trends (line, last 14 days) ---- */
  safeChart("prediction-trends", () => {
    const c7 = document.getElementById("chart-prediction-trends");
    if (!c7 || !data.trends) return;
    new Chart(c7, {
      type: "line",
      data: {
        labels: data.trends.labels,
        datasets: [
          { label: "Approved", data: data.trends.approved, borderColor: green, backgroundColor: "rgba(34,197,94,0.12)", tension: 0.4, fill: true, pointRadius: 0 },
          { label: "Declined", data: data.trends.declined, borderColor: red, backgroundColor: "rgba(244,63,94,0.10)", tension: 0.4, fill: true, pointRadius: 0 },
        ],
      },
      options: {
        plugins: { legend: { position: "bottom" } },
        scales: { x: commonGrid, y: { ...commonGrid, beginAtZero: true } },
        interaction: { mode: "index", intersect: false },
      },
    });
  });

})();
