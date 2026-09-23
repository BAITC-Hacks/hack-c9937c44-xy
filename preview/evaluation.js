"use strict";

const metricText = value => Number.isFinite(value) ? format(value) : "—";
const splitLabel = value => ({ development: "Development", holdout: "Holdout", purged: "Исключены на границе", all: "Все выпуски" })[value] || "—";

async function loadEvaluation(directory, manifest, origin, rows) {
  const response = await fetch(new URL("evaluation.json", directory), { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error("Не удалось загрузить оценку качества.");
  const report = await response.json();
  if (report.schema_version !== 1 || report.run_created_at !== manifest.created_at || report.source_data_timezone !== manifest.source_data_timezone || report.horizon_hours !== manifest.horizon_hours || JSON.stringify(report.completed_origins) !== JSON.stringify(manifest.completed_origins)) throw new Error("Оценка относится к другому запуску. Повторите evaluate.py.");
  if (![report.rows, report.summary, report.daily_metrics, report.assumptions].every(Array.isArray)) throw new Error("Неполная оценка качества.");
  const paired = report.rows.filter(row => Date.parse(row.forecast_origin) === origin);
  if (paired.length !== rows.length * 2) throw new Error("Оценка не покрывает выбранный выпуск.");
  const seen = new Set();
  for (const item of paired) {
    const hour = (Date.parse(item.valid_time) - origin) / 3600000;
    const id = { turbine_1: 1, turbine_2: 2 }[item.turbine_id];
    const key = `${hour}:${id}`;
    if (!id || !Number.isInteger(hour) || !rows[hour] || seen.has(key)) throw new Error("Некорректные часы или турбины в оценке.");
    seen.add(key);
    for (const column of ["observed", "persistence"]) {
      const value = item[column];
      if (value !== null && (!Number.isFinite(value) || value < 0 || value > 1)) throw new Error("Некорректные значения сравнения.");
      rows[hour][`${column}${id}`] = value;
    }
    if (item.eligible !== (item.observed !== null && item.persistence !== null) || Math.abs(item.power_normalized - rows[hour][`power${id}`]) > 1e-7 || !Number.isFinite(item.power_normalized)) throw new Error("Оценка не совпадает с CSV прогноза.");
  }
  return report;
}

function clearEvaluation(message) {
  $("quality-note").textContent = message;
  $("quality-selected").textContent = "";
  $("quality-rows").replaceChildren();
  $("comparison-download").hidden = true;
  $("comparison-legend").hidden = true;
}

function drawEvaluation() {
  const report = state.evaluation;
  if (!report) {
    clearEvaluation($("source").value === "synthetic" || $("source").value === "demo"
      ? "Демонстрация: фактических измерений и метрик качества нет."
      : "Оценка для этого запуска отсутствует. Прогноз показан без факта и baseline.");
    return;
  }
  $("quality-note").textContent = report.assumptions.join(" ");
  const selected = report.daily_metrics.find(row => Date.parse(row.forecast_origin) === Date.parse(state.rows[0].time)
    && row.horizon_hours === state.horizon && row.turbine_id === (state.turbine === "both" ? "both" : `turbine_${state.turbine}`));
  if (!selected) throw new Error("Нет метрик выбранного горизонта.");
  $("quality-selected").textContent = `Выбранный выпуск · ${splitLabel(selected.split)} · ${selected.n}/${selected.expected} турбино-часов · исключено ${selected.excluded}. Модель: MAE ${metricText(selected.model_mae)}, RMSE ${metricText(selected.model_rmse)}. Persistence: MAE ${metricText(selected.persistence_mae)}, RMSE ${metricText(selected.persistence_rmse)}.`;
  $("quality-rows").replaceChildren(...report.summary.map(row => {
    const tr = document.createElement("tr");
    for (const value of [`${splitLabel(row.split)} · ${shortTime(row.first_origin)} — ${shortTime(row.last_origin)} UTC`, row.turbine_id, row.origins, `${row.n}/${row.expected}`, row.missing_observed, row.missing_persistence,
      metricText(row.model_mae), metricText(row.model_rmse), metricText(row.persistence_mae), metricText(row.persistence_rmse)]) {
      const td = document.createElement("td");
      td.textContent = String(value);
      tr.append(td);
    }
    return tr;
  }));
  $("comparison-download").href = new URL(`../outputs/${$("source").value}/evaluation-hours.csv`, location.href).href;
  $("comparison-download").hidden = false;
  $("comparison-legend").hidden = false;
}
