/* Map-led forecast viewer. Browser demo is synthetic; Python modes load audited artifacts. */
"use strict";

const state = { horizon: 48, turbine: "both", layer: "power", hour: 12, expanded: false, rows: [], timer: null, csv: "", audit: null, request: 0 };
const $ = id => document.getElementById(id);
const format = (value, digits = 3) => value.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const timeLabel = value => new Date(value).toLocaleString("ru-RU", { timeZone: "UTC", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
const shortTime = value => new Date(value).toLocaleString("ru-RU", { timeZone: "UTC", day: "2-digit", month: "2-digit", hour: "2-digit" });
const powerFromWind = wind => wind < 3 || wind > 25 ? 0 : clamp((wind ** 3 - 27) / (12 ** 3 - 27), 0, 1);
const selectedIds = () => state.turbine === "both" ? [1, 2] : [Number(state.turbine)];

function buildRows() {
  const date = $("date").value;
  const start = Date.parse(`${date}T00:00:00Z`);
  const day = Number(date.slice(-2));
  state.rows = Array.from({ length: state.horizon }, (_, hour) => {
    const wind1 = 8.4 + 4.5 * Math.sin(hour / 6 + day / 5) + 1.2 * Math.cos(hour / 2.8) + 0.45 * Math.sin(hour * 1.8);
    const wind2 = Math.max(0, wind1 * 0.93 + 0.55 * Math.sin(hour / 4 + 1));
    return {
      time: new Date(start + hour * 3600000).toISOString(),
      wind1, wind2, power1: powerFromWind(wind1), power2: powerFromWind(wind2),
    };
  });
  state.hour = clamp(state.hour, 0, state.rows.length - 1);
  const origin = state.rows[0].time;
  const lines = ["forecast_origin,valid_time,turbine_id,power_normalized,wind_speed_ms"];
  state.rows.forEach(row => [1, 2].forEach(id => lines.push(`${origin},${row.time},turbine_${id},${row[`power${id}`].toFixed(6)},${row[`wind${id}`].toFixed(3)}`)));
  state.csv = lines.join("\n") + "\n";
  state.stem = `DEMO_forecast_${date}_${state.horizon}h`;
  state.audit = { mode: "synthetic-demo", physical_validation: { wind_limits_applied: true } };
}

async function loadArtifact(requestedHorizon) {
  const date = $("date").value;
  const source = $("source").value;
  const stem = `forecast_${date.replaceAll("-", "")}T0000Z`;
  const base = new URL(`../outputs/${source}/${stem}`, location.href);
  const [csvResponse, auditResponse] = await Promise.all([fetch(`${base}.csv`), fetch(`${base}.json`)]);
  if (!csvResponse.ok || !auditResponse.ok) throw new Error(`Файл ${stem} не найден в outputs/${source}. Запустите Python-расчёт и откройте /preview/ из корня проекта.`);
  const [csv, audit] = await Promise.all([csvResponse.text(), auditResponse.json()]);
  const origin = Date.parse(`${date}T00:00:00Z`);
  const expectedMode = source === "demo" ? "synthetic-demo" : "historical-backtest";
  if (audit.mode !== expectedMode || Date.parse(audit.forecast_origin) !== origin || (source === "backtest" && audit.trained_model !== true)) throw new Error("CSV-аудит не соответствует источнику и дате.");
  const lines = csv.trim().split(/\r?\n/);
  if (lines.shift() !== "forecast_origin,valid_time,turbine_id,power_normalized,wind_speed_ms" || ![24, 48].includes(audit.horizon_hours) || lines.length !== audit.horizon_hours * 2) throw new Error("Неполный CSV или неверный горизонт прогноза.");
  const hours = new Map();
  for (const line of lines) {
    const fields = line.split(",");
    if (fields.length !== 5 || Date.parse(fields[0]) !== origin || !["turbine_1", "turbine_2"].includes(fields[2])) throw new Error("Некорректная строка прогноза.");
    const time = Date.parse(fields[1]), power = Number(fields[3]), wind = Number(fields[4]);
    if (!Number.isFinite(time) || !Number.isFinite(power) || !Number.isFinite(wind) || power < 0 || power > 1 || wind < 0) throw new Error("Некорректные значения прогноза.");
    const hour = (time - origin) / 3600000;
    if (!Number.isInteger(hour) || hour < 0 || hour >= audit.horizon_hours) throw new Error("Время прогноза вне горизонта.");
    const row = hours.get(hour) || { time: new Date(time).toISOString() };
    const id = fields[2].slice(-1);
    if (row[`power${id}`] !== undefined) throw new Error("Повтор турбины в CSV.");
    row[`power${id}`] = power;
    row[`wind${id}`] = wind;
    hours.set(hour, row);
  }
  const horizon = Math.min(requestedHorizon, audit.horizon_hours);
  const rows = Array.from({ length: horizon }, (_, hour) => hours.get(hour));
  if (rows.some(row => !row || row.power1 === undefined || row.power2 === undefined)) throw new Error("В CSV пропущены часы или турбины.");
  return { rows, csv, audit, stem, source, horizon };
}

function windLimitsApplied() { return state.audit?.physical_validation?.wind_limits_applied !== false; }

function setSourceLabels() {
  const source = $("source").value;
  const browserDemo = source === "synthetic";
  const pythonDemo = source === "demo";
  $("mode-badge").innerHTML = `<i></i> ${source === "backtest" ? "МОДЕЛЬ" : "ДЕМО"}`;
  $("inspector-mode").textContent = source === "backtest" ? "МОДЕЛЬ" : "ДЕМО";
  $("footer-status").textContent = source === "backtest" ? "Результат модели · без оценки точности" : "Демонстрационный интерфейс · сентябрь 2026";
  $("source-note").textContent = browserDemo
    ? "Синтетический сценарий интерфейса. Не является результатом модели или оценкой точности."
    : pythonDemo ? "Загружен синтетический офлайн-прогноз Python. Это не результат модели или оценка точности."
      : "Загружен выход модели из CSV и JSON-аудита. Качество за февраль не оценено.";
  $("source-title").textContent = browserDemo ? "Синтетический сценарий интерфейса" : pythonDemo ? "Офлайн-расчёт Python" : "Прогноз модели";
  $("source-detail").textContent = browserDemo ? "Архив погоды и модель не запускались" : pythonDemo ? "Синтетические погодные данные" : `Архивный прогноз погоды · ${state.audit?.history_policy || "история до даты выпуска"}`;
  $("table-description").textContent = `${browserDemo ? "Синтетические значения" : pythonDemo ? "Python демо" : "Выход модели"} · UTC`;
  $("threshold-note").textContent = windLimitsApplied() ? "Пороги: 3 / 25 м/с" : "Ветер 10 м · пороги не применены";
  $("shutdown-detail").textContent = windLimitsApplied() ? "турбино-часов в горизонте" : "ветер 10 м · порог не применён";
  $("physics-rule").textContent = windLimitsApplied() ? "0–1 p.u. · ветер <3 или >25 м/с → 0" : "0–1 p.u. · к ветру 10 м пороги турбины не применяются";
  $("download").disabled = false;
  document.dispatchEvent(new Event("forecast-updated"));
}

function clearData(message) {
  state.rows = [];
  state.csv = "";
  state.audit = null;
  $("download").disabled = true;
  $("source-note").textContent = message;
  $("mode-badge").textContent = "НЕТ ДАННЫХ";
  $("inspector-mode").textContent = "—";
  $("footer-status").textContent = "Прогноз не загружен";
  $("source-title").textContent = "Прогноз не загружен";
  $("source-detail").textContent = message;
  $("table-description").textContent = "Нет данных · UTC";
  $("condition-title").textContent = "Нет данных";
  $("condition-copy").textContent = "Выберите доступную дату и источник.";
  $("threshold-note").textContent = "—";
  $("physics-rule").textContent = "Нет данных";
  for (const id of ["selected-time", "dock-time", "current-value", "asset-1-value", "asset-2-value", "pin-1-value", "pin-2-value", "avg-power", "peak-power", "peak-time", "avg-wind", "shutdown-hours", "t1-mean", "t2-mean", "range-label", "origin-label", "range-start", "range-middle", "range-end"]) $(id).textContent = "—";
  $("power-chart").textContent = "Нет данных для выбранной даты";
  $("wind-chart").textContent = "Нет данных для выбранной даты";
  $("rows").textContent = "";
  $("toggle-table").textContent = "Нет данных";
  $("status-text").textContent = message;
  document.dispatchEvent(new Event("forecast-updated"));
}

async function loadSelection() {
  stopPlayback();
  const request = ++state.request;
  const date = $("date");
  if (!date.value || !date.checkValidity()) { clearData("Выберите дату с января по февраль 2026 года."); return; }
  const requestedHorizon = state.horizon;
  if ($("source").value === "synthetic") {
    buildRows();
    setSourceLabels();
    render();
    return;
  }
  clearData("Загрузка прогноза…");
  try {
    const { rows, csv, audit, stem, horizon } = await loadArtifact(requestedHorizon);
    if (request !== state.request) return;
    state.rows = rows;
    state.csv = csv;
    state.audit = audit;
    state.stem = stem;
    state.horizon = horizon;
    state.hour = clamp(state.hour, 0, rows.length - 1);
    setSourceLabels();
    render();
  } catch (error) {
    if (request === state.request) clearData(error.message);
  }
}

function setPressed(selector, attribute, value) {
  document.querySelectorAll(selector).forEach(button => {
    const active = button.dataset[attribute] === String(value);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function drawMap() {
  const row = state.rows[state.hour];
  const ids = selectedIds();
  const metric = state.layer === "power" ? "power" : "wind";
  const values = ids.map(id => row[`${metric}${id}`]);
  const value = mean(values);
  const unit = state.layer === "power" ? "p.u." : "м/с";
  const stopped = windLimitsApplied() ? ids.filter(id => row[`wind${id}`] < 3 || row[`wind${id}`] > 25) : [];

  $("selected-time").textContent = `${timeLabel(row.time)} UTC`;
  $("dock-time").textContent = `${timeLabel(row.time)} UTC · час ${state.hour + 1} из ${state.horizon}`;
  $("current-value").textContent = format(value, state.layer === "power" ? 3 : 1);
  $("current-unit").textContent = unit;
  $("reading-title").textContent = state.layer === "power" ? "Прогноз мощности" : "Прогноз ветра";
  $("reading-subtitle").textContent = state.turbine === "both" ? "Обе турбины" : `Турбина 0${state.turbine}`;
  $("reading-note").textContent = state.layer === "power" ? "Среднее значение · нормализованная мощность" : "Средняя скорость ветра · прогноз 10 м";
  $("reading-ring").style.setProperty("--ring-fill", `${clamp((state.layer === "power" ? value : value / 25) * 100, 0, 100)}%`);
  $("reading-ring").style.setProperty("--ring-color", state.layer === "power" ? "#158b79" : "#4c78ad");
  $("site-map").classList.toggle("wind-layer", state.layer === "wind");
  $("map-legend-label").textContent = state.layer === "power" ? "Нормализованная мощность · p.u." : "Прогноз ветра на 10 м · м/с";
  $("condition-title").textContent = !windLimitsApplied() ? "Мощность в диапазоне 0–1" : stopped.length ? "Сработал порог ветра" : "Физическая проверка пройдена";
  $("condition-copy").textContent = !windLimitsApplied() ? "Ветер на высоте 10 м: пороги турбины не применяются" : stopped.length
    ? `${stopped.map(id => `T${id}: ${format(row[`wind${id}`], 1)} м/с`).join(" · ")} · соответствующая мощность 0 p.u.`
    : "Порог включения 3 м/с · отключения 25 м/с";
  $("condition-dot").parentElement.classList.toggle("warn", stopped.length > 0);

  [1, 2].forEach(id => {
    const number = `${format(row[`${metric}${id}`], state.layer === "power" ? 3 : 1)} ${unit}`;
    $(`asset-${id}-value`).textContent = number;
    $(`pin-${id}-value`).textContent = number;
    const isSelected = state.turbine === String(id);
    $(`pin-${id}`).classList.toggle("selected", isSelected);
    document.querySelector(`.asset-row[data-select="${id}"]`).classList.toggle("selected", isSelected);
    document.querySelectorAll(`[data-select="${id}"]`).forEach(button => button.setAttribute("aria-pressed", String(isSelected)));
    const intensity = state.layer === "power" ? row[`power${id}`] : row[`wind${id}`] / 25;
    $(`pin-${id}`).style.opacity = String(0.72 + 0.28 * clamp(intensity, 0, 1));
  });
}

function drawSummary() {
  const ids = selectedIds();
  const powers = state.rows.flatMap(row => ids.map(id => row[`power${id}`]));
  const winds = state.rows.flatMap(row => ids.map(id => row[`wind${id}`]));
  const peak = Math.max(...powers);
  const peakRow = state.rows.find(row => ids.some(id => row[`power${id}`] === peak));
  $("avg-power").textContent = format(mean(powers));
  $("peak-power").textContent = format(peak);
  $("peak-time").textContent = `${timeLabel(peakRow.time)} UTC`;
  $("avg-wind").textContent = format(mean(winds), 1);
  $("shutdown-hours").textContent = windLimitsApplied() ? String(winds.filter(wind => wind < 3 || wind > 25).length) : "—";
  $("t1-mean").textContent = format(mean(state.rows.map(row => row.power1)));
  $("t2-mean").textContent = format(mean(state.rows.map(row => row.power2)));
  $("range-label").textContent = `${timeLabel(state.rows[0].time)} — ${timeLabel(state.rows.at(-1).time)} · UTC`;
  $("origin-label").textContent = `${timeLabel(state.rows[0].time)} UTC`;
}

function drawChart(kind) {
  const isWind = kind === "wind";
  const host = $(isWind ? "wind-chart" : "power-chart");
  const width = 900, height = isWind ? 132 : 225;
  const left = 43, right = 14, top = 14, bottom = 27;
  const plotW = width - left - right, plotH = height - top - bottom;
  const max = isWind ? 26 : 1;
  const x = index => left + index * plotW / (state.horizon - 1);
  const y = value => top + (1 - value / max) * plotH;
  const key = id => `${kind}${id}`;
  const line = id => state.rows.map((row, index) => `${index ? "L" : "M"}${x(index).toFixed(1)} ${y(row[key(id)]).toFixed(1)}`).join(" ");
  const grid = isWind ? (windLimitsApplied() ? [0, 3, 10, 20, 25] : [0, 5, 10, 15, 20, 25]) : [0, 0.25, 0.5, 0.75, 1];
  let svg = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">`;
  grid.forEach(level => {
    const important = isWind && windLimitsApplied() && (level === 3 || level === 25);
    svg += `<line x1="${left}" y1="${y(level)}" x2="${width - right}" y2="${y(level)}" stroke="${important ? "#dfc18d" : "#e7eee9"}" stroke-dasharray="${important ? "4 4" : "3 5"}"/><text x="${left - 9}" y="${y(level) + 3}" text-anchor="end" font-size="9" fill="${important ? "#a47a37" : "#94a49b"}">${isWind ? level : level.toFixed(2)}</text>`;
  });
  const ticks = state.horizon === 48 ? [0, 12, 24, 36, 47] : [0, 6, 12, 18, 23];
  ticks.forEach(index => {
    svg += `<text x="${x(index)}" y="${height - 7}" text-anchor="${index === 0 ? "start" : index === state.horizon - 1 ? "end" : "middle"}" font-size="9" fill="#94a49b">${shortTime(state.rows[index].time)}</text>`;
  });
  if (state.horizon === 48) svg += `<line x1="${x(24)}" y1="${top}" x2="${x(24)}" y2="${height - bottom}" stroke="#c9d9d0" stroke-dasharray="5 5"/>`;
  if (state.turbine !== "2") svg += `<path d="${line(1)}" fill="none" stroke="#158b79" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`;
  if (state.turbine !== "1") svg += `<path d="${line(2)}" fill="none" stroke="#4c78ad" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round"/>`;
  svg += `<line x1="${x(state.hour)}" y1="${top}" x2="${x(state.hour)}" y2="${height - bottom}" stroke="#718a7d" stroke-width="1.4" stroke-dasharray="4 4"/>`;
  selectedIds().forEach(id => {
    const color = id === 1 ? "#158b79" : "#4c78ad";
    svg += `<circle cx="${x(state.hour)}" cy="${y(state.rows[state.hour][key(id)])}" r="5" fill="${color}" stroke="#fff" stroke-width="2.5"/>`;
  });
  svg += `</svg>`;
  host.innerHTML = svg;
  host.setAttribute("aria-label", `${isWind ? "Ветер, м/с" : "Мощность, p.u."}; выбран ${timeLabel(state.rows[state.hour].time)} UTC. Значения есть в таблице ниже.`);
}

function drawTable() {
  const start = state.expanded ? 0 : clamp(state.hour - 2, 0, state.horizon - 5);
  const visible = state.expanded ? state.rows : state.rows.slice(start, start + 5);
  $("rows").innerHTML = visible.map((row, offset) => {
    const hour = state.expanded ? offset : start + offset;
    const stopped = windLimitsApplied() ? [1, 2].filter(id => row[`wind${id}`] < 3 || row[`wind${id}`] > 25) : [];
    const condition = !windLimitsApplied() ? "Порог не применён" : stopped.length ? `Порог: T${stopped.join(", T")}` : "В пределах";
    return `<tr class="${hour === state.hour ? "selected-row" : ""}"><td><button type="button" class="row-hour" data-hour="${hour}" aria-label="Выбрать ${timeLabel(row.time)} UTC">${timeLabel(row.time)}</button></td><td>${format(row.power1)}</td><td>${format(row.power2)}</td><td>${format(row.wind1, 1)}</td><td>${format(row.wind2, 1)}</td><td class="${stopped.length ? "constraint-off" : ""}">${condition}</td></tr>`;
  }).join("");
  $("toggle-table").textContent = state.expanded ? "Свернуть таблицу ↑" : `Показать все ${state.horizon} ${state.horizon === 24 ? "часа" : "часов"} ↓`;
  $("toggle-table").setAttribute("aria-expanded", String(state.expanded));
}

function render() {
  if (!state.rows.length) return;
  state.hour = clamp(state.hour, 0, state.rows.length - 1);
  $("time-range").max = String(state.rows.length - 1);
  $("time-range").value = String(state.hour);
  $("time-range").setAttribute("aria-valuetext", `${timeLabel(state.rows[state.hour].time)} UTC`);
  $("range-start").textContent = shortTime(state.rows[0].time);
  $("range-middle").textContent = shortTime(state.rows[Math.floor(state.horizon / 2)].time);
  $("range-end").textContent = shortTime(state.rows.at(-1).time);
  setPressed("[data-layer]", "layer", state.layer);
  setPressed("[data-horizon]", "horizon", state.horizon);
  setPressed("[data-turbine]", "turbine", state.turbine);
  drawMap();
  drawSummary();
  drawChart("power");
  drawChart("wind");
  drawTable();
}

function selectHour(index, announce = false) {
  if (!state.rows.length) return;
  const next = clamp(Math.round(index), 0, state.rows.length - 1);
  if (next === state.hour) return;
  state.hour = next;
  render();
  if (announce) $("status-text").textContent = `Выбран ${timeLabel(state.rows[state.hour].time)} UTC`;
}

function stopPlayback() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $("play-toggle").textContent = "▶";
  $("play-toggle").setAttribute("aria-label", "Запустить почасовое воспроизведение");
}

function chartHour(event) {
  if (!state.rows.length) return;
  const box = event.currentTarget.getBoundingClientRect();
  const proportion = ((event.clientX - box.left) / box.width * 900 - 43) / (900 - 43 - 14);
  selectHour(proportion * (state.horizon - 1));
}

document.querySelectorAll("[data-horizon]").forEach(button => button.addEventListener("click", () => {
  state.horizon = Number(button.dataset.horizon);
  loadSelection();
}));
document.querySelectorAll("[data-layer]").forEach(button => button.addEventListener("click", () => {
  state.layer = button.dataset.layer;
  render();
}));
document.querySelectorAll("[data-turbine]").forEach(button => button.addEventListener("click", () => {
  state.turbine = button.dataset.turbine;
  render();
}));
document.querySelectorAll("[data-select]").forEach(button => button.addEventListener("click", () => {
  state.turbine = button.dataset.select;
  render();
}));
$("source").addEventListener("change", () => { state.hour = 0; loadSelection(); });
$("date").addEventListener("change", () => { state.hour = 0; loadSelection(); });
$("time-range").addEventListener("input", event => selectHour(Number(event.target.value), true));
$("previous-hour").addEventListener("click", () => { stopPlayback(); selectHour(state.hour - 1, true); });
$("next-hour").addEventListener("click", () => { stopPlayback(); selectHour(state.hour + 1, true); });
$("play-toggle").addEventListener("click", () => {
  if (!state.rows.length) return;
  if (state.timer) { stopPlayback(); return; }
  if (state.hour === state.horizon - 1) selectHour(0);
  $("play-toggle").textContent = "Ⅱ";
  $("play-toggle").setAttribute("aria-label", "Остановить почасовое воспроизведение");
  state.timer = setInterval(() => {
    if (state.hour >= state.horizon - 1) { stopPlayback(); return; }
    selectHour(state.hour + 1);
  }, 650);
});
$("power-chart").addEventListener("pointermove", chartHour);
$("wind-chart").addEventListener("pointermove", chartHour);
$("power-chart").addEventListener("click", chartHour);
$("wind-chart").addEventListener("click", chartHour);
$("rows").addEventListener("click", event => {
  const button = event.target.closest("[data-hour]");
  if (button) selectHour(Number(button.dataset.hour), true);
});
$("toggle-table").addEventListener("click", () => { if (!state.rows.length) return; state.expanded = !state.expanded; drawTable(); });
$("download").addEventListener("click", () => {
  if (!state.csv) return;
  const url = URL.createObjectURL(new Blob([state.csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.stem}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $("status-text").textContent = "CSV загружен";
});

loadSelection();
