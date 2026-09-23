"use strict";
(() => {
  const form = $("economics-form");
  const status = $("economics-status");
  const money = value => new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 }).format(value);
  const energy = value => new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 }).format(value);
  const fields = ["capacity_1_mw", "capacity_2_mw", "plan_mw", "tariff_kzt_kwh", "advance_kzt_kwh", "balancing_kzt_kwh"];
  let generation = 0, timer, result = null;

  window.economicSettings = () => {
    if (!$("eco-enabled").checked) return undefined;
    if (!form.checkValidity()) throw new Error("Заполните параметры экономического сценария корректными числами.");
    const settings = Object.fromEntries(fields.map(key => [key, $("eco-" + key).value === "" ? null : $("eco-" + key).valueAsNumber]));
    settings.tariff_note = $("eco-tariff_note").value.trim();
    if ((settings.advance_kzt_kwh === null) !== (settings.balancing_kzt_kwh === null)) throw new Error("Для сравнения закупки укажите обе цены или оставьте обе пустыми.");
    if (settings.plan_mw > settings.capacity_1_mw + settings.capacity_2_mw) throw new Error("План отпуска превышает суммарную номинальную мощность.");
    return settings;
  };

  function clear() {
    ++generation;
    clearTimeout(timer);
    result = null;
    $("economics-result").hidden = true;
    $("economics-download").disabled = true;
    document.dispatchEvent(new Event("economics-updated"));
  }

  async function calculate() {
    clear();
    if (!$("eco-enabled").checked) { status.textContent = "Экономический сценарий отключён; в отчёт он не включается."; return; }
    if (!state.rows.length) { status.textContent = "Сначала загрузите прогноз."; return; }
    const request = generation;
    try {
      const economics = window.economicSettings();
      status.textContent = "Считаем почасовой недобор относительно плана…";
      const response = await fetch("/api/economics", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: $("source").value, date: $("date").value,
          origin: $("source").value === "synthetic" ? undefined : state.audit?.forecast_origin,
          horizon: state.horizon, rows: $("source").value === "synthetic" ? state.rows : undefined, economics }),
      });
      if (!(response.headers.get("content-type") || "").includes("application/json")) throw new Error("Для расчёта запустите python serve_preview.py --port 8767 и откройте панель на этом порту.");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Не удалось рассчитать экономику.");
      if (request !== generation) return;
      result = data;
      const totals = data.totals;
      $("eco-forecast").textContent = energy(totals.forecast_mwh) + " МВт·ч";
      $("eco-plan").textContent = "План: " + energy(totals.planned_mwh) + " МВт·ч";
      $("eco-shortfall").textContent = energy(totals.shortfall_mwh) + " МВт·ч";
      $("eco-kwh").textContent = money(totals.shortfall_kwh) + " кВт·ч · " + totals.shortfall_hours + " ч недобора";
      $("eco-revenue").textContent = money(totals.lost_energy_revenue_kzt) + " ₸";
      $("eco-agent").textContent = data.recommendation;
      $("eco-procurement").textContent = data.procurement
        ? `Закупка заранее: ${money(data.procurement.advance_cost_kzt)} ₸. Балансирование: ${money(data.procurement.balancing_cost_kzt)} ₸. Разница: ${money(data.procurement.potential_saving_kzt)} ₸ (без комиссий, без НДС).`
        : "Добавьте две цены закупки, чтобы сравнить затраты. Более низкая цена заранее не гарантируется.";
      $("eco-hourly").replaceChildren(...data.rows.map(row => {
        const tr = document.createElement("tr");
        [row.valid_time.replace("T", " ").slice(0, 16), energy(row.forecast_mw), energy(row.plan_mw), energy(row.shortfall_mwh), money(row.lost_energy_revenue_kzt)].forEach(value => {
          const td = document.createElement("td"); td.textContent = value; tr.append(td);
        });
        return tr;
      }));
      $("economics-result").hidden = false;
      $("economics-download").disabled = false;
      status.textContent = `Сценарий · обе турбины · ${data.horizon_hours} ч · выпуск ${data.forecast_origin.replace("T", " ").slice(0, 16)} UTC. Параметры включаются в PDF и ZIP отчёта.`;
    } catch (error) {
      if (request === generation) status.textContent = error instanceof TypeError
        ? "Нет соединения с сервисом экономики. Проверьте локальный сервер и нажмите «Пересчитать»." : error.message;
    }
  }

  form.addEventListener("input", () => { clear(); status.textContent = "Параметры изменены…"; timer = setTimeout(calculate, 300); });
  form.addEventListener("submit", event => { event.preventDefault(); calculate(); });
  document.addEventListener("forecast-updated", calculate);
  $("economics-download").addEventListener("click", () => {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "alem-wind-economics.json";
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  calculate();
})();
